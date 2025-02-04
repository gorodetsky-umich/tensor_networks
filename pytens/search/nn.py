"""Neural network modules for branch selection."""

import itertools
from typing import List
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
import matplotlib.pyplot as plt

from pytens.search.state import Split, Merge, SearchState
from pytens.algs import TensorNetwork

# each split has three possible ways to split errors
predefined_splits = [
    Split(i, list(comb))
    for i in range(10)
    for comb in itertools.chain(
        itertools.combinations(range(5), 1),
        itertools.combinations(range(5), 2),
    )
]
predefined_merges = [Merge(i, j) for i in range(10) for j in range(5)]
N_ACTIONS = len(predefined_splits) * 3 + len(predefined_merges)


class ValueNet(nn.Module):
    """The critic role in actor-critic mode"""

    def __init__(self, node_num=10, state_dim=64, hidden_dim=16):
        super().__init__()

        self.node_num = node_num
        self.hidden = nn.Linear((node_num + 1) * state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, graphs):
        """Forward pass of the value network."""
        if len(graphs.shape) == 1:
            graphs = graphs.unsqueeze(0)

        outs = self.hidden(graphs)
        outs = F.relu(outs)
        value = self.output(outs)
        return value


class OpPicker(nn.Module):
    """The actor network for operations."""

    def __init__(self, node_num=10, state_dim=64, hidden_dim=16):
        super().__init__()

        ac_num = len(predefined_splits) * 3 + len(predefined_merges)
        self.ac_num = ac_num
        self.node_num = node_num
        self.hidden = nn.Linear((node_num + 1) * state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, ac_num)

    def forward(self, graphs):
        """Return the probability for picking one action based on node and current state"""
        # print(nodes.shape, edges.shape, node.shape)
        if len(graphs.shape) == 1:
            graphs = graphs.unsqueeze(0)
        # graphs shape: batch_size x hidden_dim
        outs = F.relu(self.hidden(graphs))
        logits = self.output(outs)
        return logits


class StateEncoder(nn.Module):
    """Embedding a tensor network."""

    def __init__(
        self, node_num=10, node_dim=2**16, state_dim=64, hidden_dim=64
    ):
        super().__init__()

        self.node_num = node_num
        self.node_dim = node_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        # we can use a small LSTM to encode the matrix chunk by chunk
        self.node_embedding = nn.LSTM(node_dim, hidden_dim, batch_first=True)
        self.edge_embedding = nn.Linear(node_num * node_num, state_dim)
        self.node_out = nn.Linear(hidden_dim, state_dim)
        self.edge_out = nn.Linear(node_num * node_num, state_dim)

    def forward(self, st: SearchState):
        """Forward pass to encode a given tensor network"""
        # st.network.draw()
        # plt.show()
        # extract features
        hout = torch.zeros(self.hidden_dim)
        cout = torch.zeros(self.hidden_dim)
        for _, d in st.network.network.nodes(data=True):
            value = d["tensor"].value.reshape(-1)
            chunks = torch.tensor([])
            # divide them into chunks of 512 for encoding
            for ci in range(0, len(value), self.node_dim):
                if len(value) - ci < self.node_dim:
                    chunk = torch.tensor(value[ci:])
                    chunk = F.pad(chunk, (0, self.node_dim - len(chunk)))
                else:
                    chunk = torch.tensor(value[ci : ci + self.node_dim])

                chunks = torch.cat([chunks, chunk.unsqueeze(0)])

            hout, cout = self.node_embedding(chunks, (hout, cout))
            hout = hout[-1]
            cout = cout[-1]

        # node_embed = hout

        # extract edge features
        adjacency_matrix = nx.to_numpy_array(st.network.network)
        node_num = len(adjacency_matrix)
        edges = F.pad(
            torch.tensor(adjacency_matrix, dtype=torch.float32),
            (0, self.node_num - node_num, 0, self.node_num - node_num),
        )

        n_out = None  # self.node_out(nodes) # node_num x state_dim
        e_out = self.edge_out(edges.reshape(-1))  # state_dim
        return torch.cat([n_out.reshape(-1), e_out])

    def masks(self, st: SearchState):
        """Mask out invalid actions for a given state."""
        nodes = st.network.network.nodes
        masks = torch.ones(N_ACTIONS)
        for i, n in enumerate(nodes):
            ndims = len(st.network.network.nodes[n]["tensor"].indices)
            # split n
            for j in range(15):
                # print("dealing with", n, predefined_splits[i * 15 + j])
                if all(
                    ind < ndims
                    for ind in predefined_splits[i * 15 + j].left_indices
                ):
                    masks[i * 45 + j * 3 : i * 45 + j * 3 + 3] = 0
                # else:
                #     print("masking out", n, predefined_splits[i * 15 + j])
                # print(masks[i * 15 + j : i * 15 + j + 3])

            # merge n
            n_nbrs = len(list(st.network.network.neighbors(n)))
            masks[
                len(predefined_splits) * 3 + i * 5 : len(predefined_splits) * 3
                + i * 5
                + n_nbrs
            ] = 0

        return masks


class RLTrainer:
    """The reinforcement learning engine."""

    def __init__(self, params):
        self.params = params
        self.gamma = 0.99
        self.value_net = ValueNet()
        self.op_picker = OpPicker()
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005
        self.iters = 600
        self.lam = 0.95
        self.eps = params["eps"]
        self.max_ops = params["max_ops"]
        self.state_to_torch = StateEncoder()
        self.ent_coef = 0.01

    def select_action(
        self, states, random_sample=True, sample_size=1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        n_envs = len(states)
        state_dim = 64
        node_num = 10
        graphs = torch.zeros(n_envs, (node_num + 1) * state_dim)
        masks = torch.zeros(n_envs, N_ACTIONS)
        for i, state in enumerate(states):
            graphs[i] = self.state_to_torch(state)
            masks[i] = self.state_to_torch.masks(state)

        state_values = self.value_net(graphs)  # n_envs
        action_logits = self.op_picker(graphs.detach())  # n_envs x n_actions
        # mask out invalid actions
        # print(action_logits.shape, masks.shape)
        action_logits[masks == 1] = -1e8
        # print(masks)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        # print(action_pd.probs)
        if random_sample:
            actions = action_pd.sample((sample_size,))
        else:
            _, actions = torch.topk(action_logits, sample_size, dim=-1)
            actions = actions.transpose(0, 1)

        action_log_probs = action_pd.log_prob(actions)
        action_log_probs = action_log_probs.transpose(0, 1)
        actions = actions.transpose(0, 1)
        entropy = action_pd.entropy()
        return actions, action_log_probs, state_values, entropy

    def step(self, states: List[SearchState], actions: torch.Tensor):
        """Perform the given actions over the given states."""
        rewards = torch.zeros_like(actions, dtype=torch.float32)
        done = torch.zeros_like(actions)
        new_states = []
        for i, (prev_st, action) in enumerate(zip(states, actions)):
            core_cost = np.prod(
                [i.size for i in prev_st.network.free_indices()]
            )
            if action < len(predefined_splits) * 3:
                split_ac = action // 3
                split_cnt = action % 3
                curr_st = None
                ac = copy.deepcopy(predefined_splits[split_ac])
                ac.node = list(prev_st.network.network.nodes)[ac.node]
                # print(ac)
                for ac_result in prev_st.take_action(
                    ac, split_errors=5, no_heuristic=True
                ):
                    curr_st = ac_result

                    if split_cnt == 0:
                        break

                    split_cnt -= 1

                # print("!!! find split for", action, split_ac, predefined_splits[split_ac], action % 3)
                rewards[i] = 1 - curr_st.network.cost() / core_cost
                # print(curr_st.network.cost(), curr_st.network.cost() / core_cost, rewards[i])
                # curr_st.network.draw()
                # plt.show()
                done[i] = 0
            else:
                ac = copy.deepcopy(
                    predefined_merges[action - 3 * len(predefined_splits)]
                )
                ac.node1 = list(prev_st.network.network.nodes)[ac.node1]
                nbrs = list(prev_st.network.network.neighbors(ac.node1))
                ac.node2 = nbrs[ac.node2]
                curr_st = next(prev_st.take_action(ac))
                rewards[i] = 1 - curr_st.network.cost() / core_cost
                done[i] = 0

            new_states.append(curr_st)

        return new_states, rewards, done

    def get_losses(
        self,
        rewards,
        action_log_probs: torch.Tensor,
        value_preds,
        entropy: torch.Tensor,
        masks,
    ):
        T, n_envs = rewards.shape
        advantages = torch.zeros(T, n_envs)
        # print(rewards)
        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t]
                + self.gamma * masks[t] * value_preds[t + 1]
                - value_preds[t]
            )
            gae = td_error + self.gamma * self.lam * masks[t] * gae
            advantages[t] = gae

        # print(advantages)

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean()
            - self.ent_coef * entropy.mean()
        )
        return critic_loss, actor_loss

    def train(self, nets: List[TensorNetwork]):
        """Run the training of the network"""
        writer = SummaryWriter()
        critic_opt = torch.optim.AdamW(
            self.value_net.parameters(), lr=self.critic_lr
        )
        actor_opt = torch.optim.RMSprop(
            self.op_picker.parameters(), lr=self.actor_lr
        )

        n_steps_per_update = self.params["max_ops"]
        n_envs = len(nets)
        for it in range(self.iters):
            ep_value_preds = torch.zeros(n_steps_per_update, n_envs)
            ep_rewards = torch.zeros(n_steps_per_update, n_envs)
            ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs)
            masks = torch.zeros(n_steps_per_update, n_envs)

            # reset environments
            states = []
            for net in nets:
                s = SearchState(net, net.norm() * self.eps)
                states.append(s)

            # collect trajectories
            for step in range(n_steps_per_update):
                actions, action_log_probs, state_value_preds, entropy = (
                    self.select_action(states)
                )
                # print(actions.shape)
                states, rewards, done = self.step(states, actions)
                # print("step rewards", rewards)

                ep_value_preds[step] = state_value_preds.squeeze(-1)
                ep_rewards[step] = rewards.squeeze(-1)
                ep_action_log_probs[step] = action_log_probs.squeeze(-1)
                masks[step] = 1 - done.squeeze(-1)

            # calculate the losses and optimize
            critic_loss, actor_loss = self.get_losses(
                ep_rewards, ep_action_log_probs, ep_value_preds, entropy, masks
            )

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            # critic_scheduler.step()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            # actor_scheduler.step()

            writer.add_scalar("Loss/critic", critic_loss.detach().item(), it)
            writer.add_scalar("Loss/actor", actor_loss.detach().item(), it)
            writer.add_scalar(
                "Episode return", ep_rewards[-1].mean().detach().item(), it
            )
            print(
                f"Iteration: {it}, critic_loss: {critic_loss}, actor_loss: {actor_loss}"
            )

            if (it + 1) % 100 == 0:
                # save the model
                with open("models/value.pkl", "wb") as value_model:
                    torch.save(self.value_net.state_dict(), value_model)

                with open("models/action.pkl", "wb") as action_model:
                    torch.save(self.op_picker.state_dict(), action_model)

                with open("models/state.pkl", "wb") as state_model:
                    torch.save(self.state_to_torch.state_dict(), state_model)

    def sample_rollout(self, net: TensorNetwork):
        """Get a rollout by random sampling."""

    def greedy_rollout(self, net: TensorNetwork):
        """Get a rollout by greedy sampling."""
        s = SearchState(net, net.norm() * self.eps)
        states = [s]
        best_network = None
        start = time.time()
        with torch.no_grad():
            for _ in range(self.params["max_ops"]):
                actions, _, _, _ = self.select_action(
                    states, random_sample=False, sample_size=1
                )
                states, _, _ = self.step(states, actions)
                if (
                    best_network is None
                    or best_network.cost() > states[0].network.cost()
                ):
                    best_network = states[0].network

        print(best_network.cost(), time.time() - start)
        best_network.draw()
        plt.show()

    def smc_rollout(self, net: TensorNetwork):
        """Get a rollout by sequential monte carlo."""
        s = SearchState(net, net.norm() * self.eps)
        states = [s]
        best_network = None
        start = time.time()

        with torch.no_grad():
            for _ in range(self.params["max_ops"]):
                # sample some actions from the policy
                actions, _, _, _ = self.select_action(states, sample_size=10)
                candidates = []
                for s, acs in zip(states, actions):
                    s_nexts, _, _ = self.step(
                        [copy.deepcopy(s) for _ in range(10)], acs
                    )
                    for s_next in s_nexts:
                        graph_embed = self.state_to_torch(s_next)
                        s_val = self.value_net(graph_embed)
                        candidates.append((s_next, s_val))
                # reweight the samples by their values
                vals, probs = zip(*candidates)
                probs = torch.tensor(probs, dtype=torch.float32)
                max_x = torch.max(probs)
                min_x = torch.min(probs)
                probs = (probs - min_x) / (max_x - min_x)
                # print(probs)
                # resample by new weights
                val_indices = torch.multinomial(probs, 10, True)
                states = [vals[i] for i in val_indices]
                for s in states:
                    if (
                        best_network is None
                        or best_network.cost() > s.network.cost()
                    ):
                        best_network = s.network

        print(best_network.cost(), time.time() - start)
        best_network.draw()
        plt.show()

    def beam_rollout(self, net: TensorNetwork):
        """Get a rollout by beam search."""
        s = SearchState(net, net.norm() * self.eps)
        states = [s]
        best_network = None
        start = time.time()
        with torch.no_grad():
            for _ in range(self.params["max_ops"]):
                actions, _, _, _ = self.select_action(
                    states, random_sample=False, sample_size=5
                )
                candidates = []
                for s, acs in zip(states, actions):
                    s_nexts, _, _ = self.step(
                        [copy.deepcopy(s) for _ in range(5)], acs
                    )
                    for s_next in s_nexts:
                        if (
                            best_network is None
                            or best_network.cost() > s_next.network.cost()
                        ):
                            best_network = s_next.network

                        if len(candidates) < 5:
                            candidates.append(s_next)
                        else:
                            for i, candidate in enumerate(candidates):
                                if (
                                    candidate.network.cost()
                                    > s_next.network.cost()
                                ):
                                    candidates.pop(i)
                                    candidates.append(s_next)
                                    break

                states = candidates

        best_network.draw()
        print(best_network.cost(), time.time() - start)
        plt.show()

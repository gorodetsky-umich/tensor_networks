"""Search algorithsm for tensor networks."""

import itertools
import time
import copy
import heapq
import math
import random
import argparse
import pickle
from typing import Sequence, Dict, List, Tuple, Self, Generator, Optional
from pydantic.dataclasses import dataclass

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from pytens.algs import NodeName, TensorNetwork, Index, Tensor


parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str)
parser.add_argument("--log", type=str)


class Action:
    """Base action."""
    def __lt__(self, other) -> bool:
        return str(self) < str(other)
    
    def __hash__(self) -> int:
        return hash(self.__str__())

class Split(Action):
    """Split action."""

    def __init__(
        self,
        node: NodeName,
        left_indices: Sequence[int],
    ):
        self.node = node
        self.left_indices = left_indices

    def __str__(self) -> str:
        return f"Split({self.node}, {self.left_indices})"

    def execute(self, network: TensorNetwork) -> Tuple[NodeName, NodeName, NodeName]:
        """Execute a split action."""
        node_indices = network.network.nodes[self.node]["tensor"].indices
        # print("node indices", node_indices)
        # left_dims = [node_indices[i].size for i in self.left_indices]
        # right_dims = [node_indices[i].size for i in self.right_indices]
        right_indices = [i for i in range(len(node_indices)) if i not in self.left_indices]
        return network.split(
            self.node, self.left_indices, right_indices
        )


class Merge(Action):
    """Merge action."""

    def __init__(self, node1: NodeName, node2: NodeName):
        self.node1 = node1
        self.node2 = node2

    def __str__(self) -> str:
        return f"Merge({self.node1}, {self.node2})"

    def execute(self, network: TensorNetwork):
        """Execute a merge action."""
        network.merge(self.node1, self.node2)
        return network


class SearchState:
    """Class for representation of intermediate search states."""

    def __init__(
        self, net: TensorNetwork, delta: float, threshold: float = 0.1, max_ops: int = 5
    ):
        self.network = net
        self.curr_delta = delta
        self.last_action = None  # Last action taken to reach this state
        self.max_ops = max_ops
        self.threshold = threshold
        self.is_noop = False
        self.used_ops = 0

    def get_legal_actions(self):
        """Return a list of all legal actions in this state."""
        actions = []
        for n in self.network.network.nodes:
            indices = self.network.network.nodes[n]["tensor"].indices
            indices = range(len(indices))
            # get all partitions of indices
            for sz in range(1, len(indices) // 2 + 1):
                combs = list(itertools.combinations(indices, sz))
                if len(indices) % 2 == 0 and sz == len(indices) // 2:
                    combs = combs[: len(combs) // 2]

                for comb in combs:
                    left_indices = comb
                    ac = Split(n, left_indices)
                    actions.append(ac)

        for n in self.network.network.nodes:
            for m in self.network.network.neighbors(n):
                if str(n) < str(m):
                    n_indices = self.network.network.nodes[n]["tensor"].indices
                    m_indices = self.network.network.nodes[m]["tensor"].indices
                    if len(set(n_indices).union(set(m_indices))) <= 5:
                        ac = Merge(n, m)
                        actions.append(ac)

        return actions

    def take_action(self, action: Action, split_errors: int = 0, no_heuristic: bool = False) -> Generator["SearchState", None, None]:
        """Return a new GameState after taking the specified action."""
        if isinstance(action, Split) and split_errors != 0:
            new_net = copy.deepcopy(self.network)
            try:
                indices = new_net.network.nodes[action.node]["tensor"].indices
                # print(indices, action.left_indices)
                left_sz = np.prod([indices[i].size for i in action.left_indices])
                right_sz = np.prod([indices[i].size for i in range(len(indices)) if i not in action.left_indices])
                max_sz = min(left_sz, right_sz)
                u, s, v = action.execute(new_net)
                # print(u, new_net.network.nodes[u]["tensor"].indices)
                # print(s, new_net.network.nodes[s]["tensor"].indices)
                # print(v, new_net.network.nodes[v]["tensor"].indices)
                # new_net.draw()
                # plt.show()
                u_val = new_net.network.nodes[u]["tensor"].value
                v_val = new_net.network.nodes[v]["tensor"].value
                # This should produce a lot of new states
                s_val = np.diag(new_net.network.nodes[s]["tensor"].value)
                
                slist = list(s_val * s_val)
                slist.reverse()
                truncpost = []
                for elem in np.cumsum(slist):
                    if elem <= self.curr_delta ** 2:
                        truncpost.append(elem)
                    else:
                        break

                if not no_heuristic and (len(truncpost) == 0 and max_sz == len(s_val)):
                    return

                split_num = min(split_errors, len(truncpost))
                # print("split_num", split_num)
                if no_heuristic and split_num == 0:
                    tmp_net = copy.deepcopy(new_net)
                    tmp_net.merge(v, s)
                    new_state = SearchState(
                        tmp_net, self.curr_delta, max_ops=self.max_ops, threshold=self.threshold
                    )
                    new_state.last_action = action
                    new_state.used_ops = self.used_ops + 1
                    yield new_state

                for idx, elem in enumerate(reversed(truncpost[-split_num:])):
                    truncation_rank = max(max_sz - len(truncpost) + idx, 1)
                    used_delta = elem

                    # it is possible to do the truncation at this point
                    tmp_net = copy.deepcopy(new_net)
                    # truncate u, s, v according to idx
                    
                    tmp_net.network.nodes[u]["tensor"].update_val_size(u_val[..., :truncation_rank])
                    tmp_net.network.nodes[s]["tensor"].update_val_size(np.diag(s_val[:truncation_rank]))
                    tmp_net.network.nodes[v]["tensor"].update_val_size(v_val[:truncation_rank, ...])
                    # tmp_net.draw()
                    # plt.show()
                    # print(tmp_net.network.nodes[u]["tensor"].indices)
                    # print(tmp_net.network.nodes[s]["tensor"].indices)
                    # print(tmp_net.network.nodes[v]["tensor"].indices)
                    tmp_net.merge(v, s)
                    # print("merging", v, s)
                    # tmp_net.draw()
                    # plt.show()

                    # print(idx, self.curr_delta ** 2, cum_slist[idx-1])
                    remaining_delta = float(np.sqrt(self.curr_delta**2 - used_delta))
                    # we cannot afford to put this into a list, so generator
                    new_state = SearchState(
                        tmp_net, remaining_delta, max_ops=self.max_ops, threshold=self.threshold
                    )
                    new_state.last_action = action
                    new_state.used_ops = self.used_ops + 1

                    yield new_state
            except np.linalg.LinAlgError:
                pass

        elif isinstance(action, Split) and split_errors == 0:
            new_net = copy.deepcopy(self.network)
            indices = new_net.network.nodes[action.node]["tensor"].indices
            left_sz = np.prod([indices[i].size for i in action.left_indices])
            right_indices = [i for i in range(len(indices)) if i not in action.left_indices]
            right_sz = np.prod([indices[i].size for i in right_indices])
            max_sz = min(left_sz, right_sz)
            (u, v), new_delta = new_net.delta_split(action.node, action.left_indices, right_indices, delta=self.curr_delta)
            new_state = SearchState(
                new_net, new_delta, max_ops=self.max_ops, threshold=self.threshold
            )
            new_state.last_action = action
            new_state.used_ops = self.used_ops + 1
            index_sz = new_net.get_contraction_index(u, v)[0].size
            new_state.is_noop = max_sz == index_sz
            yield new_state

        elif isinstance(action, Merge):
            new_net = copy.deepcopy(self.network)
            action.execute(new_net)
            # new_net.draw()
            # plt.show()
            new_state = SearchState(
                new_net, self.curr_delta, max_ops=self.max_ops, threshold=self.threshold
            )
            new_state.last_action = action
            new_state.used_ops = self.used_ops + 1
            # new_state.is_noop = len(new_net.network.nodes) == 1

            yield new_state

        else:
            raise TypeError("Unrecognized action type")

    def optimize(self):
        """Optimize the current structure."""
        free_indices = self.network.free_indices()
        root = None
        for n, t in self.network.network.nodes(data=True):
            if free_indices[0] in t["tensor"].indices:
                root = n
                break

        root = self.network.orthonormalize(root)
        _, self.curr_delta = self.network.optimize(root, self.curr_delta)

    def is_terminal(self) -> bool:
        """Whether the current state is a terminal state."""
        return self.is_noop or len(self.network.network.nodes) >= self.max_ops

    def get_result(self, total_cost: float) -> float:
        """Whether the current state succeeds or not."""
        if self.is_noop:
            return 0

        return float(self.network.cost() <= self.threshold * total_cost)

    def __lt__(self, other: Self) -> bool:
        return self.network.cost() > other.network.cost()

# each split has three possible ways to split errors
predefined_splits = [Split(i, list(comb)) for i in range(10) for comb in itertools.chain(itertools.combinations(range(5), 1), itertools.combinations(range(5), 2))]
predefined_merges = [Merge(i, j) for i in range(10) for j in range(5)]
N_ACTIONS = len(predefined_splits) * 3 + len(predefined_merges)

class ValueNet(nn.Module):
    """The critic role in actor-critic mode"""
    def __init__(self, node_num = 10, state_dim = 64, hidden_dim = 16):
        super().__init__()

        self.node_num = node_num
        self.hidden = nn.Linear((node_num + 1) * state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, graphs):
        if len(graphs.shape) == 1:
            graphs = graphs.unsqueeze(0)

        outs = self.hidden(graphs)
        outs = F.relu(outs)
        value = self.output(outs)
        return value

class OpPicker(nn.Module):
    """The actor network for operations."""
    def __init__(self, node_num = 10, state_dim = 64, hidden_dim = 16):
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
    def __init__(self, node_num = 10, node_dim = 2**16, state_dim = 64, hidden_dim = 64):
        super().__init__()

        self.node_num = node_num
        self.node_dim = node_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        # we can use a small LSTM to encode the matrix chunk by chunk
        self.node_embedding = nn.LSTM(node_dim, hidden_dim, batch_first = True)
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
        for i, (_, d) in enumerate(st.network.network.nodes(data=True)):
            value = d["tensor"].value.reshape(-1)
            chunks = torch.tensor([])
            # divide them into chunks of 512 for encoding
            for ci in range(0, len(value), self.node_dim):
                if len(value) - ci < self.node_dim:
                    chunk = torch.tensor(value[ci:])
                    chunk = F.pad(chunk, (0, self.node_dim - len(chunk)))
                else:
                    chunk = torch.tensor(value[ci:ci+self.node_dim])

                chunks = torch.cat([chunks, chunk.unsqueeze(0)])

            hout, cout = self.node_embedding(chunks, (hout, cout))
            hout = hout[-1]
            cout = cout[-1]

        node_embed = hout

        # extract edge features
        adjacency_matrix = nx.to_numpy_array(st.network.network)
        node_num = len(adjacency_matrix)
        edges = F.pad(torch.tensor(adjacency_matrix, dtype=torch.float32), (0, self.node_num - node_num, 0, self.node_num - node_num))

        n_out = self.node_out(nodes) # node_num x state_dim
        e_out = self.edge_out(edges.reshape(-1)) # state_dim
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
                if all(ind < ndims for ind in predefined_splits[i * 15 + j].left_indices):
                    masks[i * 45 + j * 3 : i * 45 + j * 3 + 3] = 0
                # else:
                #     print("masking out", n, predefined_splits[i * 15 + j])
                # print(masks[i * 15 + j : i * 15 + j + 3])
            
            # merge n
            n_nbrs = len(list(st.network.network.neighbors(n)))
            masks[len(predefined_splits) * 3 + i * 5 : len(predefined_splits) * 3 + i * 5 + n_nbrs] = 0

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

        state_values = self.value_net(graphs) # n_envs
        action_logits = self.op_picker(graphs.detach()) # n_envs x n_actions
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
            core_cost = np.prod([i.size for i in prev_st.network.free_indices()])
            if action < len(predefined_splits) * 3:
                split_ac = action // 3
                split_cnt = action % 3
                curr_st = None
                ac = copy.deepcopy(predefined_splits[split_ac])
                ac.node = list(prev_st.network.network.nodes)[ac.node]
                # print(ac)
                for ac_result in prev_st.take_action(ac, split_errors = 5, no_heuristic = True):
                    curr_st = ac_result

                    if split_cnt == 0:
                        break

                    split_cnt -= 1

                # print("!!! find split for", action, split_ac, predefined_splits[split_ac], action % 3)
                rewards[i] = 1-curr_st.network.cost() / core_cost
                # print(curr_st.network.cost(), curr_st.network.cost() / core_cost, rewards[i])
                # curr_st.network.draw()
                # plt.show()
                done[i] = 0
            else:
                ac = copy.deepcopy(predefined_merges[action - 3 * len(predefined_splits)])
                ac.node1 = list(prev_st.network.network.nodes)[ac.node1]
                nbrs = list(prev_st.network.network.neighbors(ac.node1))
                ac.node2 = nbrs[ac.node2]
                curr_st = next(prev_st.take_action(ac))
                rewards[i] = 1-curr_st.network.cost() / core_cost
                done[i] = 0

            new_states.append(curr_st)

        return new_states, rewards, done

    def get_losses(self, rewards, action_log_probs: torch.Tensor, value_preds, entropy: torch.Tensor, masks):
        T, n_envs = rewards.shape
        advantages = torch.zeros(T, n_envs)
        # print(rewards)
        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + self.gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + self.gamma * self.lam * masks[t] * gae
            advantages[t] = gae

        # print(advantages)

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - self.ent_coef * entropy.mean()
        )
        return critic_loss, actor_loss

    def train(self, nets: List[TensorNetwork]):
        writer = SummaryWriter()
        critic_opt = torch.optim.AdamW(self.value_net.parameters(), lr = self.critic_lr)
        actor_opt = torch.optim.RMSprop(self.op_picker.parameters(), lr = self.actor_lr)

        n_steps_per_update = self.params["max_ops"]
        n_envs = len(nets)
        for iter in range(self.iters):
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
                actions, action_log_probs, state_value_preds, entropy = self.select_action(states)
                # print(actions.shape)
                states, rewards, done = self.step(states, actions)
                # print("step rewards", rewards)

                ep_value_preds[step] = state_value_preds.squeeze(-1)
                ep_rewards[step] = rewards.squeeze(-1)
                ep_action_log_probs[step] = action_log_probs.squeeze(-1)
                masks[step] = 1 - done.squeeze(-1)
            
            # calculate the losses and optimize
            critic_loss, actor_loss = self.get_losses(ep_rewards, ep_action_log_probs, ep_value_preds, entropy, masks)

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            # critic_scheduler.step()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            # actor_scheduler.step()

            writer.add_scalar("Loss/critic", critic_loss.detach().item(), iter)
            writer.add_scalar("Loss/actor", actor_loss.detach().item(), iter)
            writer.add_scalar("Episode return", ep_rewards[-1].mean().detach().item(), iter)
            print(f"Iteration: {iter}, critic_loss: {critic_loss}, actor_loss: {actor_loss}")

            if (iter + 1) % 100 == 0:
                # save the model
                with open("models/value.pkl", "wb") as value_model:
                    torch.save(self.value_net.state_dict(), value_model)

                with open("models/action.pkl", "wb") as action_model:
                    torch.save(self.op_picker.state_dict(), action_model)

                with open("models/state.pkl", "wb") as state_model:
                    torch.save(self.state_to_torch.state_dict(), state_model)

    def sample_rollout(self, net: TensorNetwork):
        pass

    def greedy_rollout(self, net: TensorNetwork):
        s = SearchState(net, net.norm() * self.eps)
        states = [s]
        best_network = None
        start = time.time()
        with torch.no_grad():
            for _ in range(self.params["max_ops"]):
                actions, _, _, _ = self.select_action(states, random_sample=False, sample_size=1)
                states, _, _ = self.step(states, actions)
                if best_network is None or best_network.cost() > states[0].network.cost():
                    best_network = states[0].network

        print(best_network.cost(), time.time() - start)
        best_network.draw()
        plt.show()

    def smc_rollout(self, net: TensorNetwork):
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
                    s_nexts, _, _ = self.step([copy.deepcopy(s) for _ in range(10)], acs)
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
                    if best_network is None or best_network.cost() > s.network.cost():
                        best_network = s.network

        print(best_network.cost(), time.time() - start)
        best_network.draw()
        plt.show()

    def beam_rollout(self, net: TensorNetwork):
        s = SearchState(net, net.norm() * self.eps)
        states = [s]
        best_network = None
        start = time.time()
        with torch.no_grad():
            for _ in range(self.params["max_ops"]):
                actions, _, _, _ = self.select_action(states, random_sample=False, sample_size=5)
                candidates = []
                for s, acs in zip(states, actions):
                    s_nexts, _, _ = self.step([copy.deepcopy(s) for _ in range(5)], acs)
                    for s_next in s_nexts:
                        if best_network is None or best_network.cost() > s_next.network.cost():
                            best_network = s_next.network

                        if len(candidates) < 5:
                            candidates.append(s_next)
                        else:
                            for i, candidate in enumerate(candidates):
                                if candidate.network.cost() > s_next.network.cost():
                                    candidates.pop(i)
                                    candidates.append(s_next)
                                    break

                states = candidates

        best_network.draw()
        print(best_network.cost(), time.time() - start)
        plt.show()


class Node:
    """Representation of one node in MCTS."""

    def __init__(self, state: SearchState, parent: Self = None):
        self.state = state  # Game state for this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node was visited
        self.wins = 0  # Number of wins after visiting this node

    def is_fully_expanded(self):
        """Check if all possible moves have been expanded."""
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight: float = 1.41) -> Self:
        """Use UCB1 to select the best child node."""
        choices_weights = []
        for child in self.children:
            if child.state.is_noop:
                weight = 0
            elif child.state.is_terminal() and child.wins == 0:
                weight = 0
            else:
                weight = (child.wins / child.visits) + exploration_weight * math.sqrt(
                    math.log(self.visits) / child.visits
                )

            choices_weights.append(weight)

        max_weight = max(choices_weights)
        # candidates = [self.children[i] for i, w in enumerate(choices_weights) if w == max_weight]
        return self.children[choices_weights.index(max_weight)]

    def expand(self):
        """Expand by creating a new child node for a random untried action."""
        legal_actions = self.state.get_legal_actions()
        tried_actions = [child.state.last_action for child in self.children]
        untried_actions = [
            action for action in legal_actions if action not in tried_actions
        ]

        action = random.choice(untried_actions)
        start = time.time()
        next_state = self.state.take_action(action)
        if isinstance(action, Split):
            print(
                "completing the action",
                action,
                self.state.network.network.nodes[action.node]["tensor"].indices,
                "takes",
                time.time() - start,
            )
        else:
            print("completing the action", action, "takes", time.time() - start)
        child_node = Node(state=next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result: int):
        """Backpropagate the result of the simulation up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    """The MCTS search engine."""

    def __init__(self, exploration_weight: float = 1.41):
        self.exploration_weight = exploration_weight
        self.initial_cost = 0
        self.best_network = None

    def search(self, initial_state: SearchState, simulations: int = 1000):
        """Perform the mcts search."""
        root = Node(initial_state)
        self.initial_cost = initial_state.network.cost()
        self.best_network = initial_state.network

        for _ in range(simulations):
            node = self.select(root)
            if not node.state.is_terminal():
                node = node.expand()
            result = self.simulate(node)
            node.backpropagate(result)
            # print("one simulation time", time.time() - start)

    def select(self, node: Node) -> Node:
        """Select a leaf node."""
        while not node.state.is_terminal():
            if node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            else:
                return node
        return node

    def simulate(self, node: Node) -> float:
        """Run a random simulation from the given node to a terminal state."""
        curr_state = node.state
        prev_state = node.state
        step = 0
        while not curr_state.is_terminal():
            prev_state = curr_state
            action = random.choice(curr_state.get_legal_actions())
            curr_state = curr_state.take_action(action)
            step += 1

        # print("complete", step, "steps in", time.time() - start)
        best_candidate = curr_state
        if curr_state.is_noop:
            best_candidate = prev_state

        if best_candidate.network.cost() < self.best_network.cost():
            self.best_network = best_candidate.network

        return curr_state.get_result(self.initial_cost)


class BeamSearch:
    """Beam search with a given beam size."""

    def __init__(self, params):
        self.params = params
        self.heap: List[SearchState] = None
        self.initial_cost = 0
        self.best_network = None
        self.stats = {
            "split": 0,
            "merge": 0,
            "split time": 0,
            "count": 0,
            "unique": set(),
        }

    def search(self, initial_state, guided: bool = False):
        """Perform the beam search from the given initial state."""
        self.initial_cost = initial_state.network.cost()
        self.best_network = initial_state.network
        self.heap = [(-self.initial_cost, initial_state)] # the initial state has a very bad score
        # trainer = RLTrainer(self.params)
        trainer = None
        if guided:
            trainer = RLTrainer(self.params)
            with open("models/value.pkl", "rb") as value_model:
                trainer.value_net.load_state_dict(torch.load(value_model, weights_only=True))

            with open("models/action.pkl", "rb") as action_model:
                trainer.op_picker.load_state_dict(torch.load(action_model, weights_only=True))
            
            with open("models/state.pkl", "rb") as state_model:
                trainer.state_to_torch.load_state_dict(torch.load(state_model, weights_only=True))

        for _ in range(self.params["max_ops"]):
            # start = time.time()
            # maintain a set of networks of at most k
            self.step(trainer)
            # print("one step time", time.time() - start)

        # self.stats["unique"] = len(self.stats["unique"])
        # print(self.stats)

    def get_score(self, st, trainer: Optional[RLTrainer] = None):
        if trainer is None:
            return -st.network.cost()
        else:
            st_encoding = trainer.state_to_torch(st)
            return trainer.value_net(st_encoding)

    def step(self, trainer: Optional[RLTrainer] = None):
        """Make a step in a beam search."""
        next_level = []
        
        while len(self.heap) > 0:
            _, state = heapq.heappop(self.heap)
            for ac in state.get_legal_actions():
                action_start = time.time()
                for new_state in state.take_action(ac, split_errors=self.params["split_errors"], no_heuristic=self.params["no_heuristic"]):
                    if new_state.is_noop:
                        continue

                    self.stats["count"] += 1
                    # self.stats["unique"].add(new_state.network.canonical_structure())
                    new_score = self.get_score(new_state, trainer)
                    if len(next_level) < self.params["beam_size"]:
                        heapq.heappush(next_level, (new_score, new_state))
                    elif self.get_score(next_level[0][1], trainer) < new_score: # we decide whether to add a network by its value network
                        heapq.heappushpop(next_level, (new_score, new_state))

                    if new_state.network.cost() < self.best_network.cost():
                        self.best_network = new_state.network

                if isinstance(ac, Split):
                    self.stats["split time"] += time.time() - action_start
                    self.stats["split"] += 1
                else:
                    self.stats["merge"] += 1

        self.heap = next_level


def approx_error(tensor: Tensor, net: TensorNetwork) -> float:
    """Compute the reconstruction error.

    Given a tensor network TN and the target tensor X,
    it returns ||X - TN|| / ||X||.
    """
    try:
        target_free_indices = tensor.indices
        net_free_indices = net.free_indices()
        net_value = net.contract().value
        perm = [net_free_indices.index(i) for i in target_free_indices]
        net_value = net_value.transpose(perm)
        error = float(
            np.linalg.norm(net_value - tensor.value) / np.linalg.norm(tensor.value)
        )
    except Exception:
        net.draw()
        plt.show()
    return error


def log_stats(
    search_stats: dict,
    target_tensor: np.ndarray,
    ts: float,
    st: SearchState,
    bn: TensorNetwork,
):
    """Log statistics of a given state."""
    search_stats["ops"].append((ts, st.used_ops))
    search_stats["costs"].append((ts, st.network.cost()))
    err = approx_error(target_tensor, st.network)
    search_stats["errors"].append((ts, err))
    search_stats["best_cost"].append((ts, bn.cost()))


class MyConfig:
    """Configuring data classes"""

    arbitrary_types_allowed = True


@dataclass(config=MyConfig)
class EnumState:
    """Enumeration state."""

    network: TensorNetwork
    ops: int
    delta: float

    def __lt__(self, other):
        if self.network != other.network:
            return self.network < other.network
        elif self.ops != other.ops:
            return self.ops < other.ops
        else:
            return self.delta < other.delta


class StructureFactory:
    """Pure structure generation."""

    def __init__(self):
        self.space = nx.DiGraph()

    def initialize(self, network: TensorNetwork, max_ops: int = 6):
        """Initial the factory with all possible structures 
        up to a given maximum number of operations.
        """
        if max_ops == 0:
            return

        curr_node = network.canonical_structure()
        for n in network.network.nodes:
            curr_indices = network.network.nodes[n]["tensor"].indices
            indices = range(len(curr_indices))
            # get all partitions of indices
            for sz in range(1, len(indices) // 2 + 1):
                combs = list(itertools.combinations(indices, sz))
                if len(indices) % 2 == 0 and sz == len(indices) // 2:
                    combs = combs[: len(combs) // 2]

                for comb in combs:
                    new_net = copy.deepcopy(network)
                    new_net.split(
                        n,
                        comb,
                        tuple(j for j in indices if j not in comb),
                        preview=True,
                    )

                    new_node = new_net.canonical_structure()
                    # print("checking", new_node)
                    if new_node not in self.space.nodes:
                        self.space.add_node(new_node, network=new_net)
                    self.space.add_edge(
                        curr_node,
                        new_node,
                        action=Split(n, comb),
                    )
                    self.initialize(new_net, max_ops - 1)

            # can we perform merge?
            for m in network.network.neighbors(n):
                if n < m:
                    new_net = copy.deepcopy(network)
                    new_net.merge(n, m, preview=True)
                    new_node = new_net.canonical_structure()
                    # print("checking", new_node)
                    if new_node not in self.space.nodes:
                        self.space.add_node(new_node, network=new_net)
                    self.space.add_edge(curr_node, new_node, action=Merge(n, m))
                    self.initialize(new_net, max_ops - 1)


class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, params: Dict):
        self.params = params

    def add_wodup(
        self,
        best_network: TensorNetwork,
        new_st: SearchState,
        worked: set,
        worklist: List[SearchState],
    ) -> TensorNetwork:
        """Add a network to a worked set to remove duplicates."""
        # new_net.draw()
        # plt.show()
        # new_net_hash = hash(new_net)
        # if new_net_hash not in worked:
        if best_network is None or best_network.cost() > new_st.network.cost():
            best_network = new_st.network

        h = new_st.network.canonical_structure(
            consider_ranks=self.params["consider_ranks"]
        )
        if self.params["prune"]:
            if h in worked:
                return best_network
            else:
                worked.add(h)

        if new_st.used_ops < self.params["max_ops"]:
            worklist.append(new_st)

        return best_network

    def a_star(self, max_ops: int = 5, timeout: float = 3600):
        """Perform the A-star search with a priority queue"""

    def mcts(self, net: TensorNetwork, budget: int = 10000):
        """Run the MCTS as a search engine."""
        engine = MCTS()
        delta = self.params["eps"] * net.norm()
        initial_state = SearchState(net, delta, max_ops=8, threshold=0.2)

        start = time.time()
        engine.search(initial_state, simulations=budget)
        end = time.time()

        best = engine.best_network

        stats = {}
        stats["time"] = end - start
        target_tensor = net.contract().value
        stats["cr_core"] = np.prod(target_tensor.shape) / best.cost()
        stats["cr_start"] = net.cost() / best.cost()
        stats["reconstruction_error"] = np.linalg.norm(
            best.contract().value - target_tensor
        ) / np.linalg.norm(target_tensor)
        stats["best_network"] = best

        # best.draw()
        # plt.show()
        return stats

    def beam(self, net: TensorNetwork, target_tensor: np.ndarray):
        """Run the beam search as a search engine."""
        engine = BeamSearch(self.params)
        # FIXME: temporarily sort by size, the actual solution should give each index a name and sort by names
        indices = [i for i in net.free_indices() if i.size != 1]
        # print(indices)
        # if len(net.network.nodes) == 1 and len(target_tensor.shape) == 5:
        #     correct_permutes = [0,1,2,3,4]
        # elif len(net.network.nodes) > 1 and len(target_tensor.shape) == 5:
        #     correct_permutes = [4,0,1,2,3]
        # elif len(net.network.nodes) == 1 and len(target_tensor.shape) == 4:
        #     correct_permutes = [0,1,2,3]
        # elif len(net.network.nodes) > 1 and len(target_tensor.shape) == 4:
        #     correct_permutes = [2,3,1,0]
        # ordered_indices = [indices[i] for i in correct_permutes]
        # ordered_indices = range(len(target_tensor.shape))
        # target_tensor = target_tensor.transpose([1,2,3,4,0])
        delta = np.sqrt((self.params["eps"] * np.linalg.norm(target_tensor)) ** 2 - np.linalg.norm(net.contract().value.squeeze() - target_tensor) ** 2)
        # print(delta, self.params["eps"] * np.linalg.norm(target_tensor), np.linalg.norm(net.contract().value.squeeze().transpose([2,3,1,0]) - target_tensor) / np.linalg.norm(target_tensor))
        # print(delta, np.linalg.norm(target_tensor), np.linalg.norm(target_tensor.reshape(-1)))
        if self.params["single_core_start"]:
            target_tensor = net.contract()
            net = TensorNetwork()
            net.add_node(0, target_tensor)
        
        initial_state = SearchState(net, delta)

        start = time.time()
        engine.search(initial_state, guided=self.params["guided"])
        end = time.time()

        best = engine.best_network

        stats = {}
        stats["time"] = end - start
        stats["cr_core"] = np.prod(target_tensor.shape) / best.cost()
        stats["cr_start"] = net.cost() / best.cost()
        # best_indices = [i for i in best.free_indices() if i.size != 1]
        # permutes = [best_indices.index(i) for i in ordered_indices]
        stats["reconstruction_error"] = np.linalg.norm(
            best.contract().value.squeeze() - target_tensor
        ) / np.linalg.norm(target_tensor)
        stats["best_network"] = best

        # best.draw()
        # plt.show()
        return stats

    def dfs(
        self,
        net: TensorNetwork,
    ):
        """Perform an exhaustive enumeration with the DFS algorithm."""
        target_tensor = net.contract()

        search_stats = {
            "networks": [],
            "best_networks": [],
            "best_cost": [],
            "costs": [],
            "errors": [],
            "ops": [],
        }
        logging_time = 0
        start = time.time()

        if self.params["single_core_start"]:
            network = TensorNetwork()
            network.add_node("G0", target_tensor)
        else:
            network = copy.deepcopy(net)
        delta = self.params["eps"] * net.norm()
        best_network = net
        worked = set([network.canonical_structure()])
        count = 0

        # with open("output/BigEarthNet-v1_0_stack/stack_18_test_0_ht_eps_010/010/beam_010.pkl", "rb") as f:
        #     desired_net = pickle.load(f)

        # net_g2 = desired_net.network.nodes["G2"]["tensor"]
        # desired_net.network.nodes["G2"]["tensor"] = Tensor(net_g2.value.squeeze(), [net_g2.indices[0], net_g2.indices[2]])
        # interested_hash = desired_net.canonical_structure()

        def helper(curr_st: SearchState):
            # plt.figure(curr_net.canonical_structure())
            nonlocal best_network
            nonlocal logging_time
            nonlocal search_stats
            nonlocal start
            nonlocal count

            count += 1

            # curr_st.network.draw()
            # plt.show()

            if self.params["prune"]:
                h = curr_st.network.canonical_structure(
                    consider_ranks=self.params["consider_ranks"]
                )
                if h in worked:
                    return
                else:
                    worked.add(h)

            if curr_st.used_ops >= self.params["max_ops"]:
                # print("max op")
                return

            if self.params["timeout"] is not None and time.time() - start > self.params["timeout"]:
                return

            for ac in curr_st.get_legal_actions():
                # print(ac)
                for new_st in curr_st.take_action(ac, split_errors = self.params["split_errors"], no_heuristic = self.params["no_heuristic"]):
                    # new_st.network.draw()
                    # plt.show()
                    # if new_st.network.canonical_structure() == interested_hash:
                    #     plt.figure(figsize=(12, 5))
                    #     plt.subplot(121)
                    #     curr_st.network.draw()
                    #     plt.subplot(122)
                    #     new_st.network.draw()
                    #     # plt.show()
                    #     plt.savefig(f"same_hash_{time.time()}_{ac}.png")
                    #     plt.close()

                    if not self.params["no_heuristic"] and new_st.is_noop:
                        # print("noop")
                        continue
                

                    if new_st.network.cost() < best_network.cost():
                        best_network = new_st.network

                    ts = time.time() - start - logging_time
                    verbose_start = time.time()
                    if self.params["verbose"]:
                        log_stats(search_stats, target_tensor, ts, new_st, best_network)
                    verbose_end = time.time()
                    logging_time += verbose_end - verbose_start

                    helper(new_st)

                # plt.close(curr_net.canonical_structure())

        helper(SearchState(network, delta))
        end = time.time()

        search_stats["time"] = end - start - logging_time
        search_stats["best_network"] = best_network
        search_stats["cr_core"] = (
            np.prod([i.size for i in net.free_indices()]) / best_network.cost()
        )
        search_stats["cr_start"] = net.cost() / best_network.cost()
        err = approx_error(target_tensor, best_network)
        search_stats["reconstruction_error"] = err
        search_stats["count"] = count

        return search_stats

    def bfs(self, net: TensorNetwork):
        """Perform an exhaustive enumeration with the BFS algorithm."""
        target_tensor = net.contract()

        search_stats = {
            "networks": [],
            "best_networks": [],
            "best_cost": [],
            "costs": [],
            "errors": [],
            "ops": [],
        }
        logging_time = 0
        start = time.time()

        network = copy.deepcopy(net)
        delta = self.params["eps"] * net.norm()

        worked = set()
        worklist = [SearchState(network, delta)]
        worked.add(network.canonical_structure())
        best_network = None
        count = 0

        while len(worklist) != 0:
            st = worklist.pop(0)

            if time.time() - start >= self.params["timeout"]:
                break

            for ac in st.get_legal_actions():
                # plt.subplot(2,1,1)
                # st.network.draw()
                for new_st in st.take_action(ac):
                    # plt.subplot(2,1,2)
                    # new_st.network.draw()
                    # plt.show()
                    if not self.params["no_heuristic"] and new_st.is_noop:
                        continue

                    if self.params["optimize"]:
                        new_st.optimize()

                    ts = time.time() - start - logging_time
                    best_network = self.add_wodup(
                        best_network,
                        new_st,
                        worked,
                        worklist,
                    )
                    count += 1

                    verbose_start = time.time()
                    if self.params["verbose"]:
                        log_stats(search_stats, target_tensor, ts, new_st, best_network)
                    verbose_end = time.time()
                    logging_time += verbose_end - verbose_start

        end = time.time()

        search_stats["time"] = end - start - logging_time
        search_stats["best_network"] = best_network
        search_stats["cr_core"] = (
            np.prod([i.size for i in net.free_indices()]) / best_network.cost()
        )
        search_stats["cr_start"] = net.cost() / best_network.cost()
        err = approx_error(target_tensor, best_network)
        search_stats["reconstruction_error"] = err
        search_stats["count"] = count

        return search_stats


def test_case_3():
    """Test exhaustive search.

    Target size: 14 x 16 x 18 x 20 x22
    Ranks:
    R12 = 3
    R23 = 4
    R34 = 3
    R45 = 2
    """
    target_net = TensorNetwork()

    g1 = np.random.randn(14, 3)
    g1_indices = [Index("I1", 14), Index("r12", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(3, 16, 4)
    g2_indices = [Index("r12", 3), Index("I2", 16), Index("r23", 4)]
    target_net.add_node("G2", Tensor(g2, g2_indices))

    g3 = np.random.randn(4, 18, 3)
    g3_indices = [Index("r23", 4), Index("I3", 18), Index("r34", 3)]
    target_net.add_node("G3", Tensor(g3, g3_indices))

    g4 = np.random.randn(3, 20, 2)
    g4_indices = [Index("r34", 3), Index("I4", 20), Index("r45", 2)]
    target_net.add_node("G4", Tensor(g4, g4_indices))

    g5 = np.random.randn(2, 22)
    g5_indices = [Index("r45", 2), Index("I5", 22)]
    target_net.add_node("G5", Tensor(g5, g5_indices))

    target_net.add_edge("G1", "G2")
    target_net.add_edge("G2", "G3")
    target_net.add_edge("G3", "G4")
    target_net.add_edge("G4", "G5")

    return target_net


def test_case_4():
    """Test exhaustive search.

    Target size: 40 x 60 x 3 x 9 x 9
    Ranks:
    R12 = 3
    R13 = 3
    R34 = 3
    R35 = 3
    """
    target_net = TensorNetwork()

    g1 = np.random.randn(40, 3, 3)
    g1_indices = [Index("I1", 40), Index("r12", 3), Index("r13", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(3, 60)
    g2_indices = [Index("r12", 3), Index("I2", 60)]
    target_net.add_node("G2", Tensor(g2, g2_indices))

    g3 = np.random.randn(3, 3, 3, 3)
    g3_indices = [
        Index("r13", 3),
        Index("I3", 3),
        Index("r34", 3),
        Index("r35", 3),
    ]
    target_net.add_node("G3", Tensor(g3, g3_indices))

    g4 = np.random.randn(3, 9)
    g4_indices = [Index("r34", 3), Index("I4", 9)]
    target_net.add_node("G4", Tensor(g4, g4_indices))

    g5 = np.random.randn(3, 9)
    g5_indices = [Index("r35", 3), Index("I5", 9)]
    target_net.add_node("G5", Tensor(g5, g5_indices))

    target_net.add_edge("G1", "G2")
    target_net.add_edge("G1", "G3")
    target_net.add_edge("G3", "G4")
    target_net.add_edge("G3", "G5")

    return target_net


def test_case_5():
    """Test exhaustive search.

    Target size: 14 x 16 x 18 x 20 x22
    Ranks:
    R12 = 3
    R1i1 = 3
    R1i1 = 3
    i1i2 = 2
    i1R4 = 4
    i2R3 = 3
    i2R5 = 3
    """
    target_net = TensorNetwork()

    g1 = np.random.randn(14, 4, 3)
    g1_indices = [Index("I1", 14), Index("r12", 4), Index("r1i1", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(4, 16)
    g2_indices = [Index("r12", 4), Index("I2", 16)]
    target_net.add_node("G2", Tensor(g2, g2_indices))

    i1 = np.random.randn(3, 2, 4)
    i1_indices = [Index("r1i1", 3), Index("i1i2", 2), Index("i1r4", 4)]
    target_net.add_node("i1", Tensor(i1, i1_indices))

    i2 = np.random.randn(2, 3, 3)
    i2_indices = [Index("i1i2", 2), Index("i2r3", 3), Index("i2r5", 3)]
    target_net.add_node("i2", Tensor(i2, i2_indices))

    g3 = np.random.randn(3, 18)
    g3_indices = [Index("i2r3", 3), Index("I3", 18)]
    target_net.add_node("G3", Tensor(g3, g3_indices))

    g4 = np.random.randn(20, 4)
    g4_indices = [Index("I4", 20), Index("i1r4", 4)]
    target_net.add_node("G4", Tensor(g4, g4_indices))

    g5 = np.random.randn(22, 3)
    g5_indices = [Index("I5", 22), Index("i2r5", 3)]
    target_net.add_node("G5", Tensor(g5, g5_indices))

    target_net.add_edge("G1", "G2")
    target_net.add_edge("G1", "i1")
    target_net.add_edge("i1", "i2")
    target_net.add_edge("i1", "G4")
    target_net.add_edge("i2", "G3")
    target_net.add_edge("i2", "G5")

    return target_net


if __name__ == "__main__":
    factory = StructureFactory()
    a = np.random.randn(14, 16, 18, 20, 22)
    initial_net = TensorNetwork()
    initial_net.add_node(
        "G",
        Tensor(
            a,
            [
                Index("a", 14),
                Index("I0", 16),
                Index("I1", 18),
                Index("I2", 20),
                Index("I3", 22),
            ],
        ),
    )
    factory.initialize(initial_net, max_ops=5)
    print(len(factory.space.nodes))
    print(len(factory.space.edges))

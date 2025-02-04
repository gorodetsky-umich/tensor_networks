"""Abstraction based search"""

from typing import List
import heapq

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pytens.algs import TensorNetwork, Tensor, Index
from pytens.search.state import SearchState, Action


class HierarchicalSearch:
    def __init__(self, params):
        self.params = params

    def dfs(self, st: SearchState, replay_actions: List[List[Action]] = []):
        if len(st.past_actions) >= self.params["max_ops"]:
            return

        for ac in st.get_legal_actions():
            for new_st in st.take_action(ac):
                heapq.heappush(
                    replay_actions,
                    (-new_st.network.cost(), new_st.past_actions),
                )

                if len(replay_actions) > 100:
                    heapq.heappop()

                self.dfs(new_st, replay_actions)

    def abstract_search(self, target_tensor: Tensor):
        """Perform the search on an abstracted tensor"""
        # average pooling of the tensor
        step_size = 6
        pooled_value = F.avg_pool3d(
            torch.tensor(target_tensor.value), step_size, stride=step_size
        )
        pooled_indices = [
            Index(i.name, i.size // step_size) for i in target_tensor.indices
        ]
        # do the dfs over this abstracted tensor and record their action sequence
        replay_actions = []

        init_net = TensorNetwork()
        init_net.add_node("G0", Tensor(pooled_value.numpy(), pooled_indices))
        delta = torch.norm(pooled_value) * self.params["eps"]
        self.dfs(SearchState(init_net, delta), replay_actions)

        return replay_actions

    def replay(
        self,
        st: SearchState,
        action_seq: List[Action],
        best_network: TensorNetwork = None,
    ):
        """Replay a given sequence of actions and get the best network"""
        if len(action_seq) == 0:
            return

        ac, acs = action_seq
        for new_st in st.take_action(
            ac, split_errors=self.params["split_errors"]
        ):
            if (
                best_network is None
                or new_st.network.cost() < best_network.cost()
            ):
                best_network = new_st.network

            self.replay(new_st, acs)

    def search(self, target_tensor: Tensor):
        # create one abstraction and filter out several action sequences
        replay_actions = self.abstract_search(target_tensor)
        best_network = None
        for ac_seq in replay_actions:
            net = TensorNetwork()
            net.add_node("G0", target_tensor)
            delta = self.params["eps"] * net.norm()
            st = SearchState(net, delta)
            self.replay(st, ac_seq, best_network)

        print(best_network.cost())
        best_network.draw()
        plt.show()

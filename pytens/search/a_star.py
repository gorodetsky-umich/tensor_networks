"""Class for A-Star search."""

import heapq
import copy
import time

import numpy as np

from pytens.algs import TensorNetwork
from pytens.search.state import SearchState


class AStar:
    """Basic class for A-star search"""

    def __init__(self, params):
        self.params = params

    def score(self, st: SearchState):
        """Get the score for a given search state"""
        # score of epsilon / cost (larger is better)
        return st.curr_delta / st.network.cost()

        # TODO: score of distribution of singular values (entropy?)
        # TODO: score of look ahead

    def search(self, net: TensorNetwork, target_tensor: np.ndarray):
        """Search by priority queue"""
        init_net = copy.deepcopy(net)
        delta = np.sqrt(
            (self.params["eps"] * np.linalg.norm(target_tensor)) ** 2
            - np.linalg.norm(net.contract().value.squeeze() - target_tensor)
            ** 2
        )
        init_st = SearchState(init_net, delta)

        # maintain a priority queue
        pq = heapq.heapify([(self.score(init_st), init_st)])

        stats = {
            "compression": [],
            "unique": {},
        }

        best_network = net
        tic = time.time()
        while len(pq) != 0:
            if (
                self.params["timeout"] is not None
                and time.time() - tic >= self.params["timeout"]
            ):
                break

            st = heapq.heappop(pq)
            if len(st.past_actions) >= self.params["max_ops"]:
                continue

            for ac in st.get_legal_actions():
                for new_st in st.take_action(ac):
                    if new_st.network.cost() < best_network.cost():
                        best_network = new_st.network

                    heapq.heappush(pq, (self.score(new_st), new_st))
                    stats["compression"].append(
                        (time.time() - tic, best_network.cost())
                    )

                    ukey = new_st.network.canonical_structure()
                    if ukey not in stats["unique"]:
                        stats["unique"][ukey] = 0

                    stats["unique"][ukey] += 1

        stats["time"] = time.time() - tic
        stats["best_network"] = best_network
        stats["cr_core"] = (
            np.prod([i.size for i in net.free_indices()]) / best_network.cost()
        )
        stats["cr_start"] = net.cost() / best_network.cost()

        return stats

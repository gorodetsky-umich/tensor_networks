
from typing import Dict, List
import itertools
import time
import copy
import multiprocessing
import queue

import numpy as np
import matplotlib.pyplot as plt

from pytens.algs import TensorNetwork
from pytens.search.state import SearchState, Action, SplitIndex
from pytens.search.constraint import ConstraintSearch, BAD_SCORE

class PartitionSearch:
    """Search by partitions free indices"""
    def __init__(self, params):
        self.params = params
        self.best_network = None
        self.tic = 0
        self.stats = {
            "unique": {},
            "compression": []
        }
        self.constraint_engine = ConstraintSearch(params)
        self.costs = {}
        self.ranks = {}
        self.conflicts = []

    def fill_holes(self, st: SearchState, queue: multiprocessing.Queue, estimate_cost: bool = True):
        """Enumerate all possible splits up to the maximum number of ops."""
        sts = [st]
        # best_costs = [st.network.cost()]
        best_cost = st.network.cost()
        for _ in range(1, self.params["max_ops"] + 1):
            next_sts = []
            for curr_st in sts:
                for action in curr_st.get_legal_actions(index_actions=True):
                    if isinstance(action, SplitIndex):
                        split_ac = action.to_split(curr_st.network)
                    else:
                        split_ac = action

                    new_net = copy.deepcopy(curr_st.network)
                    node_indices = new_net.network.nodes[split_ac.node]["tensor"].indices
                    right_indices = [i for i in range(len(node_indices)) if i not in split_ac.left_indices]
                    [u, v], _ = new_net.delta_split(
                        split_ac.node,
                        split_ac.left_indices,
                        right_indices,
                        preview=True)
                    new_st = SearchState(new_net, curr_st.curr_delta)
                    new_link = new_net.get_contraction_index(u, v)[0]
                    new_st.past_actions = curr_st.past_actions + [action]

                    if estimate_cost:
                        new_st.ac_to_link = copy.deepcopy(curr_st.ac_to_link)
                        new_st.ac_to_link[action] = new_link.name
                        # print("Getting cost for", [str(ac) for ac in new_st.past_actions])
                        rank, cost = self.constraint_engine.get_cost(new_st, best_cost)
                        best_cost = min(best_cost, cost)
                        self.costs[tuple(new_st.past_actions)] = cost
                        self.ranks[tuple(new_st.past_actions)] = rank
                    else:
                        # replay for all possible sketch completions
                        ranks = {}
                        ac_sizes = []
                        for ac in new_st.past_actions:
                            if isinstance(action, SplitIndex):
                                index_ac = ac
                            else:
                                index_ac = ac.to_index(st)

                            # we consider the same number of split points
                            _, ac_size = self.constraint_engine.split_actions[index_ac]
                            ac_sizes.append(ac_size)

                        for sz_comb in itertools.product(ac_sizes):
                            ranks = dict(zip(new_st.past_actions, sz_comb))
                            self.replay(st, new_st.past_actions, ranks, queue)

                    next_sts.append(new_st)

            sts = next_sts

        if estimate_cost:
            # get the smallest and replay with error splits around the estimated ranks
            costs = sorted([(v, k) for k, v in self.costs.items()])
            # print(costs)
            # try the top 10?
            for c, acs in costs[:1]:
                print("expect cost", c)
                self.stats["best_acs"] = acs
                # we separate out the first action to reuse previous results
                # first_ac = SplitIndex(acs[0].indices, target_size = self.ranks[acs][acs[0]])
                # for new_st in st.take_action(first_ac, svd=self.constraint_engine.first_steps[acs[0]], split_errors = self.params["split_errors"]):
                #     self.replay(new_st, acs[1:], self.ranks[acs], queue)
                # take the result and do the rounding
                self.replay(st, acs, self.ranks[acs], queue)

    def replay(self, st: SearchState, actions: List[Action], ranks: Dict[Action, int], queue: multiprocessing.Queue):
        """Apply the given actions around the given ranks."""
        if not actions:
            print("replayed network cost", st.network.cost())
            for n in st.network.network.nodes:
                net = copy.deepcopy(st.network)
                net.round(n, st.curr_delta)
                if net.cost() < self.best_network.cost():
                    self.best_network = net

            queue.put({"best_network": self.best_network, "stats": self.stats})
            return

        # print("replaying actions:", [str(ac) for ac in actions])
        ac = actions[0]
        ac.target_size = ranks[ac]
        # print("new action", new_ac)
        for new_st in st.take_action(ac, split_errors = self.params["split_errors"]):
            # print(ac)
            # if self.best_network.cost() > new_st.network.cost():
            #     self.best_network = new_st.network

            self.stats["compression"].append(
                (
                    time.time() - self.tic,
                    new_st.network.cost(),
                )
            )
            ukey = new_st.network.canonical_structure()
            self.stats["unique"][ukey] = self.stats["unique"].get(ukey,0) + 1
            self.replay(new_st, actions[1:], ranks, queue)

    def search(self, net: TensorNetwork) -> Dict:
        """Start the search from a given network. We can only support single core now."""


        if "core" not in self.params["start_from"]:
            raise ValueError("Only starting from single cores is supported")

        self.best_network = net

        delta = net.norm() * self.params["eps"]
        init_st = SearchState(net, delta)
        free_indices = net.free_indices()

        start = time.time()
        self.constraint_engine.preprocess(net.contract())
        print("preprocessing time", time.time() - start)
        toc1 = time.time()

        # with open("constraint_engine.pkl", "rb") as f:
        #     import pickle
        #     # pickle.dump(self.constraint_engine, f)
        #     self.constraint_engine = pickle.load(f)

        # tmp_indices = [SplitIndex([Index("I3", 120), Index("I4", 12)]),
        #                SplitIndex([Index("I2", 120)]),
        #                SplitIndex([Index("I0", 3), Index("I1", 1122)])]
        # replay_ranks = {
        #     tmp_indices[0]: 120,
        #     tmp_indices[1]: 40,
        #     tmp_indices[2]: 792,
        # }
        # for ind in tmp_indices:
        #     sums, sizes = self.constraint_engine.split_actions[ind]
        #     print(sums)
        #     print(sizes)

        # self.replay(init_st, tmp_indices, replay_ranks)

        self.tic = time.time()
        q = multiprocessing.Queue()
        # import threading
        # thread = threading.Thread(target=self.fill_holes, args=(init_st, queue))
        # thread.start()
        # thread.join(timeout=self.params["timeout"])
        p = multiprocessing.Process(target=self.fill_holes, args=(init_st, q))
        p.start()
        try:
            result = q.get(timeout=self.params["timeout"])
            self.best_network = result["best_network"]
            self.stats = result["stats"]
            p.join(timeout=self.params["timeout"])
        except (multiprocessing.TimeoutError, queue.Empty):
            pass
        finally:
            if p.is_alive():
                p.kill()
        # self.fill_holes(init_st, queue)
        # result = queue.get(timeout=self.params["timeout"])
        toc2 = time.time()

        self.stats["time"] = toc2 - start
        self.stats["preprocess"] = toc1 - start
        self.stats["best_network"] = self.best_network
        self.stats["cr_core"] = float(np.prod([i.size for i in free_indices])) / self.best_network.cost()
        self.stats["cr_start"] = net.cost() / self.best_network.cost()
        self.stats["reconstruction_error"] = float(np.linalg.norm(
            self.best_network.contract().value - net.contract().value
        ) / np.linalg.norm(net.contract().value))
        return self.stats

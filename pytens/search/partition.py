
from typing import Dict, List
import itertools
import time
import copy
import multiprocessing
import queue
import pickle
import heapq

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
            "compression": [],
            "count": 0,
        }
        self.constraint_engine = ConstraintSearch(params)
        self.costs = {}
        self.ranks = {}
        self.delta = 0

    def get_cost(self, init_st: SearchState, new_st: SearchState, best_cost: List[int], result_queue: multiprocessing.Queue):
        if self.params["fit_mode"] == "topk":
            # print("Getting cost for", [str(ac) for ac in new_st.past_actions])
            # new_st.network.draw()
            # plt.show()
            rank, cost = self.constraint_engine.get_cost(new_st, best_cost[-1])
            if "Split(['I3', 'I5'])" in [str(ac) for ac in new_st.past_actions] and "Split(['I4'])" in [str(ac) for ac in new_st.past_actions]:
                new_st.network.draw()
                plt.show()

            if cost != BAD_SCORE:
                best_cost.append(cost)
                best_cost = sorted(best_cost)
                if len(best_cost) > self.params["k"]:
                    best_cost = best_cost[:self.params["k"]]
            self.costs[tuple(new_st.past_actions)] = cost
            self.ranks[tuple(new_st.past_actions)] = rank

            return best_cost
        else:
            # equally distribute the errors between steps
            delta = self.delta / np.sqrt(len(new_st.past_actions))
            for idx, ac in enumerate(new_st.past_actions):
                # index_ac = ac.to_index(st, idx)
                index_ac = ac
                index_ac.delta = delta

            self.replay(init_st, new_st.past_actions, result_queue, True)
            return best_cost

    def pseudo_action_execution(self, curr_st: SearchState, action: Action):
        if isinstance(action, SplitIndex):
            split_ac = action.to_split(curr_st.network)
        else:
            split_ac = action

        new_net = copy.deepcopy(curr_st.network)
        node_indices = new_net.network.nodes[split_ac.node]["tensor"].indices
        right_indices = [i for i in range(len(node_indices)) if i not in split_ac.left_indices]
        u, s, v = new_net.split(
            split_ac.node,
            split_ac.left_indices,
            right_indices,
            preview=True)
        new_net.merge(v, s, preview=True)
        new_st = SearchState(new_net, curr_st.curr_delta)
        new_link = new_net.get_contraction_index(u, v)[0]
        new_st.past_actions = curr_st.past_actions + [action]
        new_st.links = copy.deepcopy(curr_st.links)
        new_st.links.append(new_link.name)
        return new_st

    def fill_holes(self, st: SearchState, result_queue: multiprocessing.Queue):
        """Enumerate all possible splits up to the maximum number of ops."""
        sts = [st]
        # best_costs = [st.network.cost()]
        best_cost = [st.network.cost()]
        for _ in range(1, self.params["max_ops"] + 1):
            next_sts = []
            for curr_st in sts:
                for action in curr_st.get_legal_actions(index_actions=self.params["action_type"] == "output"):
                    new_st = self.pseudo_action_execution(curr_st, action)
                    self.stats["count"] += 1
                    best_cost = self.get_cost(st, new_st, best_cost, result_queue)
                    next_sts.append(new_st)

            sts = next_sts

        if self.params["fit_mode"] == "topk":
            # get the smallest and replay with error splits around the estimated ranks
            costs = sorted([(v, k) for k, v in self.costs.items()])
            # print(costs[:5])
            # try the top 10?
            for c, acs in costs[:self.params["k"]]:
                print("expect cost", c)
                print([str(ac) for ac in acs], self.ranks[acs])
                for k, ac in enumerate(acs):
                    ac.target_size = self.ranks[acs][k]

                self.stats["best_acs"] = acs
                self.replay(st, acs, result_queue, True)

        
        result_queue.put({"best_network": self.best_network, "stats": self.stats})

    def replay(self, st: SearchState, actions: List[Action], result_queue: multiprocessing.Queue, first_iter = False):
        """Apply the given actions around the given ranks."""
        if not actions:
            # st.network.draw()
            # plt.show()
            # print("replayed network cost", st.network.cost(), "remaining delta", st.curr_delta)
            for n in st.network.network.nodes:
                net = copy.deepcopy(st.network)
                net.round(n, st.curr_delta)
                # print("round",n)
                # net.draw()
                # plt.show()
                # plt.savefig(f"debug_{n}.png")
                # plt.close()
                if net.cost() < self.best_network.cost():
                    self.best_network = net

            return

        # print("replaying actions:", [str(ac) for ac in actions])
        ac = actions[0]
        if first_iter and self.params["fit_mode"] == "all":
            svd_file = self.constraint_engine.first_steps.get(ac, None)
            svd_data = np.load(svd_file)
            svd = (svd_data['u'], svd_data['s'], svd_data['v'])
        else:
            svd = None
        # print(ac, [str(x) for x in self.constraint_engine.first_steps.keys()])
        # print("new action", new_ac)
        for new_st in st.take_action(ac, svd=svd, split_errors = self.params["split_errors"]):
            # print(ac)
            # if self.best_network.cost() > new_st.network.cost():
            #     self.best_network = new_st.network
            # new_st.network.draw()
            # plt.show()
            
            self.stats["compression"].append(
                (
                    time.time() - self.tic,
                    new_st.network.cost(),
                )
            )
            ukey = new_st.network.canonical_structure()
            self.stats["unique"][ukey] = self.stats["unique"].get(ukey,0) + 1
            self.replay(new_st, actions[1:], result_queue)

    def rank_search_and_replay(self, net: TensorNetwork, acs: List[Action]):
        """Replay actions on the given tensor network."""
        preprocess_end = time.time()
        delta = net.norm() * self.params["eps"]
        self.delta = delta
        init_st = SearchState(net, delta)
        free_indices = net.free_indices()
        new_st = init_st
        for ac in acs:
            ac.target_size = None
            new_st = self.pseudo_action_execution(new_st, ac)

        _ = self.get_cost(init_st, new_st, net.cost(), None)

        self.best_network = net
        # get the smallest and replay with error splits around the estimated ranks
        costs = sorted([(v, k) for k, v in self.costs.items()])
        # print(costs[:5])
        # try the top 10?
        for c, acs in costs[:1]:
            print("expect cost", c)
            print([str(ac) for ac in acs], self.ranks[acs])
            for k, ac in enumerate(acs):
                ac.target_size = self.ranks[acs][k]

            self.stats["best_acs"] = acs
            self.replay(init_st, acs, None, True)

        self.stats["time"] = time.time() - self.tic
        self.stats["preprocess"] = preprocess_end - self.tic
        self.stats["best_network"] = self.best_network
        self.stats["cr_core"] = float(np.prod([i.size for i in free_indices])) / self.best_network.cost()
        self.stats["cr_start"] = net.cost() / self.best_network.cost()
        self.stats["reconstruction_error"] = float(np.linalg.norm(
            self.best_network.contract().value - net.contract().value
        ) / np.linalg.norm(net.contract().value))
        return self.stats

    def search(self, net: TensorNetwork) -> Dict:
        """Start the search from a given network. We can only support single core now."""
        if self.params["replay_from"] is not None:
            start = time.time()
            self.tic = start
            with open(self.params["replay_from"], "rb") as ac_file:
                acs = pickle.load(ac_file)

            # preprocess only for the given actions
            self.constraint_engine.preprocess(net.contract(), acs)
            return self.rank_search_and_replay(net, acs)

        if "core" not in self.params["start_from"]:
            raise ValueError("Only starting from single cores is supported")

        self.best_network = net

        delta = net.norm() * self.params["eps"]
        self.delta = delta
        init_st = SearchState(net, delta)
        free_indices = net.free_indices()
        
        start = time.time()
        # print(start)
        self.constraint_engine.preprocess(net.contract(), compute_uv=self.params["fit_mode"]=="all")
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

        print(toc2, start)
        self.stats["time"] = toc2 - start
        self.stats["preprocess"] = toc1 - start
        self.stats["best_network"] = self.best_network
        self.stats["cr_core"] = float(np.prod([i.size for i in free_indices])) / self.best_network.cost()
        self.stats["cr_start"] = net.cost() / self.best_network.cost()
        self.stats["reconstruction_error"] = float(np.linalg.norm(
            self.best_network.contract().value - net.contract().value
        ) / np.linalg.norm(net.contract().value))
        return self.stats

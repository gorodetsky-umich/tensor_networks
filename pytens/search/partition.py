
from typing import Dict, List
import itertools
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

from pytens.algs import TensorNetwork, Index
from pytens.search.state import SearchState, SplitIndex, SplitIndexAround
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

    def enumerate(self, st: SearchState, actions: List[SplitIndex]):
        """Enumerate all possible splits up to the maximum number of ops."""
        sts = [st]
        # best_costs = [st.network.cost()]
        best_cost = st.network.cost()
        for _ in range(1, self.params["max_ops"] + 1):
            next_sts = []
            for curr_st in sts:
                for action in actions:
                    if curr_st.past_actions and (action < curr_st.past_actions[-1] or not action.is_valid(curr_st.past_actions)):
                        # print(action, [str(ac) for ac in curr_st.past_actions], action < curr_st.past_actions[-1], action in curr_st.past_actions)
                        continue

                    # bad_action = False
                    # for conflict in self.conflicts:
                    #     if set(conflict).issubset(set(curr_st.past_actions + [action])):
                    #         bad_action = True
                    #         break

                    # if bad_action:
                    #     continue
                    
                    split_ac = action.to_split(curr_st.network)
                    new_net = copy.deepcopy(curr_st.network)
                    node_indices = new_net.network.nodes[split_ac.node]["tensor"].indices
                    right_indices = [i for i in range(len(node_indices)) if i not in split_ac.left_indices]
                    [u, v], _ = new_net.delta_split(split_ac.node, split_ac.left_indices, right_indices, preview=True)
                    new_st = SearchState(new_net, curr_st.curr_delta)
                    new_link = new_net.get_contraction_index(u, v)[0]
                    new_st.ac_to_link = copy.deepcopy(curr_st.ac_to_link)
                    new_st.ac_to_link[action] = new_link.name
                    new_st.past_actions = curr_st.past_actions + [action]
                    # print("Getting cost for", [str(ac) for ac in new_st.past_actions])
                    rank, cost = self.constraint_engine.get_cost(new_st, best_cost)
                    # print(cost)
                    best_cost = min(best_cost, cost)
                    # if cost != BAD_SCORE:
                    #     if len(best_costs) > 10:
                    #         best_costs.pop()

                    #     best_costs.append(cost)
                    #     best_costs = sorted(best_costs)

                    self.costs[tuple(new_st.past_actions)] = cost
                    self.ranks[tuple(new_st.past_actions)] = rank

                    next_sts.append(new_st)
            
            sts = next_sts


    def dfs(self, st: SearchState, actions: List[SplitIndex]):
        """DFS search to select whether one action is performed or not."""
        # print(actions)
        if len(st.past_actions) >= self.params["max_ops"]:
            return

        if not actions:
            return

        action = actions[0]
        
        # second branch, take this action
        # print("taking action", action, "past actions:", st.past_actions)
        # st.network.draw()
        # plt.show()
        # for new_st in st.take_action(action, split_errors = self.params["split_errors"]):
        #     if new_st.network.cost() < self.best_network.cost():
        #         self.best_network = new_st.network

        #     self.stats["compression"].append(
        #         (
        #             time.time() - self.tic,
        #             new_st.network.cost(),
        #         )
        #     )
        #     ukey = new_st.network.canonical_structure()
        #     self.stats["unique"][ukey] = self.stats["unique"].get(ukey,0) + 1

        #     self.dfs(new_st, actions[1:])
        if not action.is_valid(st.past_actions):
            # first branch, do not take this action
            self.dfs(st, actions[1:])
        else:
            split_ac = action.to_split(st.network)
            new_net = copy.deepcopy(st.network)
            node_indices = new_net.network.nodes[split_ac.node]["tensor"].indices
            right_indices = [i for i in range(len(node_indices)) if i not in split_ac.left_indices]
            [u, v], _ = new_net.delta_split(split_ac.node, split_ac.left_indices, right_indices, preview=True)
            new_st = SearchState(new_net, st.curr_delta)
            new_link = new_net.get_contraction_index(u, v)[0]
            new_st.ac_to_link = copy.deepcopy(st.ac_to_link)
            new_st.ac_to_link[action] = new_link.name
            new_st.past_actions = st.past_actions + [action]
            # print("Getting cost for", [str(ac) for ac in new_st.past_actions])
            rank, cost = self.constraint_engine.get_cost(new_st, BAD_SCORE)
            # print(cost)
            self.costs[tuple(new_st.past_actions)] = cost
            self.ranks[tuple(new_st.past_actions)] = rank
            self.dfs(new_st, actions[1:])

            self.dfs(st, actions[1:])

    def replay(self, st: SearchState, actions: List[SplitIndex], ranks: Dict[SplitIndex, int]):
        """Apply the given actions around the given ranks."""
        if not actions:
            print("replayed network cost", st.network.cost())
            for n in st.network.network.nodes:
                net = copy.deepcopy(st.network)
                net.round(n, st.curr_delta)
                if net.cost() < self.best_network.cost():
                    self.best_network = net

            return
        
        # print("replaying actions:", [str(ac) for ac in actions])
        ac = actions[0]
        new_ac = SplitIndexAround(ac.indices, ranks[ac])
        # print("new action", new_ac)
        for new_st in st.take_action(new_ac, split_errors = self.params["split_errors"]):
            if self.best_network.cost() > new_st.network.cost():
                self.best_network = new_st.network

            self.stats["compression"].append(
                (
                    time.time() - self.tic,
                    new_st.network.cost(),
                )
            )
            ukey = new_st.network.canonical_structure()
            self.stats["unique"][ukey] = self.stats["unique"].get(ukey,0) + 1
            self.replay(new_st, actions[1:], ranks)

    def search(self, net: TensorNetwork) -> Dict:
        """Start the search from a given network. We can only support single core now."""
        

        if "core" not in self.params["start_from"]:
            raise ValueError("Only starting from single cores is supported")

        self.best_network = net

        delta = net.norm() * self.params["eps"]
        init_st = SearchState(net, delta)
        free_indices = net.free_indices()
        
        start = time.time()
        split_actions = []
        for k in range(1, len(free_indices) // 2 + 1):
            combs = list(itertools.combinations(free_indices, k))
            if len(free_indices) % 2 == 0 and k == len(free_indices) // 2:
                combs = combs[:len(combs) // 2]

            for comb in combs:
                split_actions.append(SplitIndex(comb))

        self.constraint_engine.preprocess(net.contract())
        print("preprocessing time", time.time() - start)

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
        self.enumerate(init_st, split_actions)
        toc1 = time.time()
        
        # get the smallest and replay with error splits around the estimated ranks
        costs = sorted([(v, k) for k, v in self.costs.items()])
        # print(costs)
        # try the top 10?
        for c, acs in costs[:1]:
            print("expect cost", c)
            ranks = self.ranks[acs]
            self.replay(init_st, acs, ranks)
            # take the result and do the rounding
        
        toc2 = time.time()

        self.stats["time"] = toc2 - start
        self.stats["preprocess"] = toc1 - self.tic
        self.stats["best_network"] = self.best_network
        self.stats["cr_core"] = float(np.prod([i.size for i in free_indices])) / self.best_network.cost()
        self.stats["cr_start"] = net.cost() / self.best_network.cost()
        self.stats["reconstruction_error"] = float(np.linalg.norm(
            self.best_network.contract().value.squeeze() - net.contract().value
        ) / np.linalg.norm(net.contract().value))
        return self.stats

"""Structure search with output-directed splits."""

from typing import Dict, List
import time
import copy
import multiprocessing
import queue
import pickle
import atexit

import numpy as np

from pytens.algs import TensorNetwork, SVDConfig
from pytens.search.configuration import SearchConfig
from pytens.search.state import SearchState, Action, OSplit
from pytens.search.constraint import ConstraintSearch, BAD_SCORE
from pytens.search.utils import remove_temp_dir


class PartitionSearch:
    """Search by partitions free indices"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.stats = {
            "unique": {},
            "compression": [],
            "count": 0,
            "tic": 0,
            "best_network": None,
        }
        self.constraint_engine = ConstraintSearch(config)
        atexit.register(remove_temp_dir, config.output.output_dir)
        self.costs = {}
        self.ranks = {}
        self.delta = 0

    def get_cost(
        self,
        init_st: SearchState,
        new_st: SearchState,
        best_cost: List[int],
        result_queue: multiprocessing.Queue,
    ) -> List[int]:
        """Call a constraint solver to estimate the cost of a given network."""
        if self.config.rank_search.fit_mode == "topk":
            rank, cost = self.constraint_engine.get_cost(new_st, best_cost[-1])
            if cost != BAD_SCORE:
                best_cost.append(cost)
                best_cost = sorted(best_cost)
                if len(best_cost) > self.config.rank_search.k:
                    best_cost = best_cost[: self.config.rank_search.k]
            self.costs[tuple(new_st.past_actions)] = cost
            self.ranks[tuple(new_st.past_actions)] = rank

            return best_cost

        if self.config.rank_search.fit_mode == "all":
            # equally distribute the errors between steps
            delta = self.delta / np.sqrt(len(new_st.past_actions))
            for ac in new_st.past_actions:
                # index_ac = ac.to_osplit(st, idx)
                index_ac = ac
                index_ac.delta = delta

            self.replay(init_st, new_st.past_actions, result_queue, True)
            return best_cost

        return best_cost

    def pseudo_action_execution(self, curr_st: SearchState, action: Action):
        """Perform a split without actual data computation."""
        if isinstance(action, OSplit):
            split_ac = action.to_isplit(curr_st.network)
        else:
            split_ac = action

        new_net = copy.deepcopy(curr_st.network)

        (u, s, v), _ = new_net.svd(
            split_ac.node,
            split_ac.left_indices,
            SVDConfig(compute_data=False),
        )
        new_net.merge(v, s, compute_data=False)
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
        for _ in range(1, self.config.engine.max_ops + 1):
            next_sts = []
            for curr_st in sts:
                is_osplit = self.config.synthesizer.action_type == "osplit"
                for action in curr_st.get_legal_actions(
                    index_actions=is_osplit
                ):
                    new_st = self.pseudo_action_execution(curr_st, action)
                    self.stats["count"] += 1
                    best_cost = self.get_cost(
                        st, new_st, best_cost, result_queue
                    )
                    next_sts.append(new_st)

            sts = next_sts

        if self.config.rank_search.fit_mode == "topk":
            # get the smallest and
            # replay with error splits around the estimated ranks
            costs = sorted([(v, k) for k, v in self.costs.items()])
            # print(costs[:5])
            # try the top 10?
            for _, acs in costs[: self.config.rank_search.k]:
                for k, ac in enumerate(acs):
                    ac.target_size = self.ranks[acs][k]

                self.stats["best_acs"] = acs
                self.replay(st, acs, result_queue, True)

        result_queue.put(self.stats)

    def replay(
        self,
        st: SearchState,
        actions: List[Action],
        result_queue: multiprocessing.Queue,
        first_iter=False,
    ):
        """Apply the given actions around the given ranks."""
        if not actions:
            # st.network.draw()
            # plt.show()
            for n in st.network.network.nodes:
                net = copy.deepcopy(st.network)
                net.round(n, st.curr_delta)
                # print("round",n)
                # net.draw()
                # plt.show()
                # plt.savefig(f"debug_{n}.png")
                # plt.close()
                if net.cost() < self.stats["best_network"].cost():
                    self.stats["best_network"] = net

            return

        # print("replaying actions:", [str(ac) for ac in actions])
        ac = actions[0]
        if first_iter and self.config.rank_search.fit_mode == "all":
            svd_file = self.constraint_engine.first_steps.get(ac, None)
            svd_data = np.load(svd_file)
            svd = (svd_data["u"], svd_data["s"], svd_data["v"])
        else:
            svd = None
        # print("new action", new_ac)
        for new_st in st.take_action(ac, svd=svd, config=self.config):
            # print(ac)
            # if self.stats["best_network"].cost() > new_st.network.cost():
            #     self.stats["best_network"] = new_st.network
            # new_st.network.draw()
            # plt.show()

            self.stats["compression"].append(
                (
                    time.time() - self.stats["tic"],
                    new_st.network.cost(),
                )
            )
            ukey = new_st.network.canonical_structure()
            self.stats["unique"][ukey] = self.stats["unique"].get(ukey, 0) + 1
            self.replay(new_st, actions[1:], result_queue)

    def rank_search_and_replay(self, net: TensorNetwork, acs: List[Action]):
        """Replay actions on the given tensor network."""
        preprocess_end = time.time()
        delta = net.norm() * self.config.engine.eps
        self.delta = delta
        init_st = SearchState(net, delta)
        free_indices = net.free_indices()
        new_st = init_st
        for ac in acs:
            ac.target_size = None
            new_st = self.pseudo_action_execution(new_st, ac)

        _ = self.get_cost(init_st, new_st, net.cost(), None)

        self.stats["best_network"] = net
        # get the smallest
        costs = sorted([(v, k) for k, v in self.costs.items()])
        for _, actions in costs[:1]:
            for k, ac in enumerate(actions):
                ac.target_size = self.ranks[actions][k]

            self.stats["best_acs"] = actions
            self.replay(init_st, actions, None, True)

        self.stats["time"] = time.time() - self.stats["tic"]
        self.stats["preprocess"] = preprocess_end - self.stats["tic"]
        self.stats["best_network"] = self.stats["best_network"]
        self.stats["cr_core"] = (
            float(np.prod([i.size for i in free_indices]))
            / self.stats["best_network"].cost()
        )
        self.stats["cr_start"] = net.cost() / self.stats["best_network"].cost()
        self.stats["reconstruction_error"] = float(
            np.linalg.norm(
                self.stats["best_network"].contract().value
                - net.contract().value
            )
            / np.linalg.norm(net.contract().value)
        )
        return self.stats

    def search(self, net: TensorNetwork) -> Dict:
        """Start the search from a given network.
        Only support single core now.
        """
        if self.config.synthesizer.replay_from is not None:
            start = time.time()
            self.stats["tic"] = start
            with open(self.config.synthesizer.replay_from, "rb") as ac_file:
                acs = pickle.load(ac_file)

            # preprocess only for the given actions
            self.constraint_engine.preprocess(net.contract(), acs)
            return self.rank_search_and_replay(net, acs)

        self.stats["best_network"] = net

        delta = net.norm() * self.config.engine.eps
        self.delta = delta
        init_st = SearchState(net, delta)
        free_indices = net.free_indices()

        start = time.time()
        # print(start)
        self.constraint_engine.preprocess(
            net.contract(),
            compute_uv=self.config.rank_search.fit_mode == "all",
        )
        toc1 = time.time()

        self.stats["tic"] = time.time()
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.fill_holes, args=(init_st, q))
        p.start()
        try:
            self.stats = q.get(timeout=self.config.engine.timeout)
            p.join(timeout=self.config.engine.timeout)
        except (multiprocessing.TimeoutError, queue.Empty):
            pass
        finally:
            if p.is_alive():
                p.kill()
        toc2 = time.time()

        self.stats["time"] = toc2 - start
        self.stats["preprocess"] = toc1 - start
        self.stats["cr_core"] = (
            float(np.prod([i.size for i in free_indices]))
            / self.stats["best_network"].cost()
        )
        self.stats["cr_start"] = net.cost() / self.stats["best_network"].cost()
        self.stats["reconstruction_error"] = float(
            np.linalg.norm(
                self.stats["best_network"].contract().value
                - net.contract().value
            )
            / np.linalg.norm(net.contract().value)
        )
        return self.stats

"""Structure search with output-directed splits."""

from typing import List, Optional, Generator, Sequence
import time
import copy
import pickle
import atexit

import numpy as np

from pytens.algs import TreeNetwork, SVDConfig
from pytens.cross.cross import TensorFunc
from pytens.search.configuration import SearchConfig
from pytens.search.state import SearchState, Action, OSplit, ISplit
from pytens.search.constraint import ConstraintSearch, BAD_SCORE
from pytens.search.utils import remove_temp_dir, SearchStats, SearchResult


class PartitionSearch:
    """Search by partitions free indices"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.constraint_engine = ConstraintSearch(config)
        self.best_network = None
        self.best_acs = []
        self.unused_delta = 0.0
        self.search_stats = SearchStats()

        self._costs = {}
        self._ranks = {}
        self._delta = 0
        self._tic = 0

    def reset(self):
        """Reset the search states."""
        self.best_network = None
        self.best_acs = []
        self._costs = {}
        self._ranks = {}
        self.unused_delta = 0.0
        self.search_stats = SearchStats()

    def get_cost(
        self,
        init_st: SearchState,
        new_st: SearchState,
        best_cost: List[int],
        tensor_func: Optional[TensorFunc] = None,
    ) -> List[int]:
        """Call a constraint solver to estimate the cost of a given network."""
        if self.config.rank_search.fit_mode == "topk":
            rank, cost = self.constraint_engine.get_cost(new_st, best_cost[-1])
            # print(new_st.network)
            # print(cost)
            if cost != BAD_SCORE:
                best_cost.append(cost)
                best_cost = sorted(best_cost)
                if len(best_cost) > self.config.rank_search.k:
                    best_cost = best_cost[: self.config.rank_search.k]
            self._costs[tuple(new_st.past_actions)] = cost
            self._ranks[tuple(new_st.past_actions)] = rank

            return best_cost

        if self.config.rank_search.fit_mode == "all":
            # equally distribute the errors between steps
            delta = self._delta / np.sqrt(len(new_st.past_actions))
            for ac in new_st.past_actions:
                # index_ac = ac.to_osplit(st, idx)
                index_ac = ac
                index_ac.delta = delta

            self.replay(init_st, new_st.past_actions, True, tensor_func)
            return best_cost

        return best_cost

    def pseudo_action_execution(self, curr_st: SearchState, action: Action):
        """Perform a split without actual data computation."""
        if isinstance(action, OSplit):
            split_ac = action.to_isplit(curr_st.network)
        elif isinstance(action, ISplit):
            split_ac = action
        else:
            raise ValueError(
                f"unknown type for {action}with type {type(action)}"
            )

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
        # print("adding action", action)
        return new_st

    def fill_holes(
        self,
        st: SearchState,
        actions: Optional[Sequence[Action]] = None,
        budget: int = 2,
        tensor_func: Optional[TensorFunc] = None,
    ) -> Generator[SearchResult, None, None]:
        """Enumerate all possible splits up to the maximum number of ops."""
        sts = [st]
        # best_costs = [st.network.cost()]
        best_cost = [st.network.cost()]
        for _ in range(1, self.config.engine.max_ops + 1):
            next_sts = []
            for curr_st in sts:
                is_osplit = self.config.synthesizer.action_type == "osplit"
                if actions is not None:
                    legal_actions = actions
                else:
                    legal_actions = curr_st.get_legal_actions(
                        index_actions=is_osplit
                    )
                for action in legal_actions:
                    if (
                        is_osplit
                        and curr_st.past_actions
                        and (
                            action < curr_st.past_actions[-1]
                            or not action.is_valid(curr_st.past_actions)
                        )
                    ):
                        continue

                    # we want to see at most two size two actions
                    if (
                        actions is not None
                        and len(curr_st.past_actions) >= budget
                    ):
                        continue

                    new_st = self.pseudo_action_execution(curr_st, action)
                    self.search_stats.count += 1
                    best_cost = self.get_cost(
                        st, new_st, best_cost, tensor_func
                    )
                    # print("cost for new action", action, "is", best_cost)
                    # print(
                    #     "cost for new action",
                    #     action,
                    #     "is",
                    #     self.best_network.cost(),
                    # )
                    next_sts.append(new_st)

            sts = next_sts

        if self.config.rank_search.fit_mode == "topk":
            # get the smallest and
            # replay with error splits around the estimated ranks
            costs = sorted([(v, k) for k, v in self._costs.items()])
            # print(costs[:5])
            # try the top 10?
            for c, acs in costs[: self.config.rank_search.k]:
                if c == BAD_SCORE:
                    acs = []

                # print("best actions")
                for k, ac in enumerate(acs):
                    ac.target_size = self._ranks[acs][k]
                    # print(ac, ac.target_size)

                self.best_acs = acs
                if actions is None:
                    self.replay(
                        st, acs, first_iter=True, tensor_func=tensor_func
                    )

                result = SearchResult(
                    stats=self.search_stats,
                    best_network=self.best_network,
                    best_actions=self.best_acs,
                )
                result.best_solver_cost = c
                # result_queue.put(result)
                yield result
        else:
            yield SearchResult(
                stats=self.search_stats,
                best_network=self.best_network,
                best_actions=self.best_acs,
            )

    def _round(
        self, st: SearchState, tensor_func: Optional[TensorFunc] = None
    ):
        if tensor_func is not None:
            # st.curr_delta = self.config.engine.eps * st.network.norm()
            self.best_network = st.network
            return

        assert self.best_network is not None

        for n in st.network.network.nodes:
            net = copy.deepcopy(st.network)
            _, unused_delta = net.round(n, st.curr_delta)
            # print("round",n)
            # net.draw()
            # plt.show()
            # plt.savefig(f"debug_{n}.png")
            # plt.close()
            if net.cost() < self.best_network.cost():
                self.best_network = net
                self.unused_delta = unused_delta
                self.best_acs = st.past_actions

    def replay(
        self,
        st: SearchState,
        actions: List[Action],
        first_iter: bool = False,
        tensor_func: Optional[TensorFunc] = None,
    ):
        """Apply the given actions around the given ranks."""
        if not actions:
            # st.network.draw()
            # plt.show()
            self._round(st, tensor_func)
            return

        # print("replaying actions:", [str(ac) for ac in actions])
        ac = actions[0]
        if first_iter and self.config.rank_search.fit_mode == "all":
            svd_file = self.constraint_engine.first_steps.get(ac, None)
            if svd_file is None:
                raise ValueError("get no svd file in the mode 'all'")

            svd_data = np.load(svd_file)
            svd = (svd_data["u"], svd_data["s"], svd_data["v"])
        else:
            svd = None
        # print("new action", new_ac)

        # in cross approximation, we need to get the error bound
        for new_st in st.take_action(
            ac,
            svd=svd,
            config=self.config,
            tensor_func=tensor_func,
        ):
            timestamp = time.time() - self._tic
            self.search_stats.costs.append((timestamp, new_st.network.cost()))
            ukey = new_st.network.canonical_structure()
            self.search_stats.incr_unique(ukey)
            self.replay(new_st, actions[1:], tensor_func=tensor_func)
            # print(new_st.network)

    def rank_search_and_replay(
        self,
        net: TreeNetwork,
        acs: List[Action],
        delta: Optional[float] = None,
    ) -> SearchResult:
        """Replay actions on the given tensor network."""
        preprocess_end = time.time()
        if delta is None:
            delta = net.norm() * self.config.engine.eps
        self._delta = delta

        init_st = SearchState(net, delta)
        new_st = init_st
        for ac in acs:
            ac.target_size = None
            new_st = self.pseudo_action_execution(new_st, ac)

        _ = self.get_cost(init_st, new_st, [net.cost()], None)

        self.best_network = net
        # get the smallest
        costs = sorted([(v, k) for k, v in self._costs.items()])
        for _, actions in costs[:1]:
            for k, ac in enumerate(actions):
                ac.target_size = self._ranks[actions][k]

            self.best_acs = actions
            self.replay(init_st, actions, True)

        result = SearchResult(
            stats=self.search_stats,
            best_network=self.best_network,
            best_actions=self.best_acs,
        )
        result.stats.time = time.time() - self._tic
        result.stats.preprocess_time = preprocess_end - self._tic
        result.unused_delta = self.unused_delta
        return result

    def search(
        self,
        net: TreeNetwork,
        delta: Optional[float] = None,
        actions: Optional[List[OSplit]] = None,
        budget: int = 2,
        tensor_func: Optional[TensorFunc] = None,
    ) -> Generator[SearchResult, None, None]:
        """Start the search from a given network.
        Only support single core now.
        """
        if self.config.synthesizer.replay_from is not None:
            start = time.time()
            self._tic = start
            with open(self.config.synthesizer.replay_from, "rb") as ac_file:
                acs = pickle.load(ac_file)

            # preprocess only for the given actions
            self.constraint_engine.preprocess(
                net.contract(), acs, cross_func=tensor_func
            )
            if self.config.output.remove_temp_after_run:
                atexit.register(
                    remove_temp_dir,
                    self.config.output.output_dir,
                    self.constraint_engine.temp_files,
                )
            yield self.rank_search_and_replay(net, acs, delta)
            return

        if self.best_network is None:
            self.best_network = net

        if tensor_func is not None:
            delta = self.config.engine.eps

        if delta is None:
            delta = net.norm() * self.config.engine.eps

        self.unused_delta = delta
        self._delta = delta
        init_st = SearchState(net, delta)

        start = time.time()
        # print(start)
        self.constraint_engine.preprocess(
            net.contract(),
            compute_uv=self.config.rank_search.fit_mode == "all",
            acs=actions,
            delta=delta,
            cross_func=tensor_func,
        )
        if self.config.output.remove_temp_after_run:
            atexit.register(
                remove_temp_dir,
                self.config.output.output_dir,
                self.constraint_engine.temp_files,
            )
        toc1 = time.time()

        self._tic = time.time()
        for result in self.fill_holes(
            init_st, actions, budget, tensor_func=tensor_func
        ):
            toc2 = time.time()

            result.stats.time = toc2 - start
            result.stats.preprocess_time = toc1 - start
            result.unused_delta = self.unused_delta
            yield result

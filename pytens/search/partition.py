"""Structure search with output-directed splits."""

import atexit
import copy
import heapq
import pickle
import time
from typing import List, Optional, Sequence

import numpy as np

from pytens.algs import TreeNetwork
from pytens.cross.cross import TensorFunc
from pytens.search.configuration import SearchConfig
from pytens.search.constraint import ConstraintSearch
from pytens.search.state import Action, ISplit, OSplit, SearchState
from pytens.search.utils import (
    DataTensor,
    SearchResult,
    SearchStats,
    init_state,
    remove_temp_dir,
)
from pytens.types import Index


class PartitionSearch:
    """Search by partitions free indices"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.constraint_engine = ConstraintSearch(config)
        self.unused_delta = 0.0
        self.stats = SearchStats()

        self._delta = 0.0

    def reset(self):
        """Reset the search states."""
        self.unused_delta = 0.0
        self.stats = SearchStats()

    def get_cost(
        self,
        data_tensor: DataTensor,
        new_st: SearchState,
        best_costs: List[int],
    ) -> Optional[SearchState]:
        """Call a constraint solver to estimate the cost of a given network."""
        if self.config.rank_search.search_mode == "topk":
            return self.constraint_engine.solve(new_st, best_costs[-1])

        if self.config.rank_search.search_mode == "all":
            # equally distribute the errors between steps
            delta = self._delta / np.sqrt(len(new_st.past_actions))
            for ac in new_st.past_actions:
                ac.delta = delta

            res = self.replay(data_tensor, new_st.past_actions, True)
            assert res.best_state is not None
            return res.best_state

        raise ValueError("unknown fit mode for rank search")

    def _sketch_execution(self, curr_st: SearchState, action: Action):
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
        (u, s, v), _ = split_ac.svd(new_net, compute_data=False)
        new_net.merge(v, s, compute_data=False)
        new_st = SearchState(new_net, curr_st.curr_delta)
        new_link = new_net.get_contraction_index(u, v)[0]
        new_st.past_actions = curr_st.past_actions + [action]
        new_st.links = copy.deepcopy(curr_st.links)
        new_st.links.append(new_link.name)
        return new_st

    def _enumerate(
        self,
        data_tensor: DataTensor,
        exclusions: Optional[Sequence[Index]],
    ) -> Sequence[SearchState]:
        """Enumerate all possible splits up to the maximum number of ops."""
        sts = [init_state(data_tensor, self._delta)]
        curr_sts = [init_state(data_tensor, self._delta)]

        for _ in range(1, self.config.engine.max_ops + 1):
            next_sts = []
            for curr_st in curr_sts:
                is_osplit = self.config.synthesizer.action_type == "osplit"
                for action in curr_st.get_legal_actions(is_osplit):
                    if (
                        is_osplit
                        and exclusions is not None
                        and len(action.indices) == 1
                        and action.indices[0] in exclusions
                    ):
                        continue

                    new_st = self._sketch_execution(curr_st, action)
                    next_sts.append(new_st)
                    self.stats.count += 1
                    smallest_sts = heapq.nsmallest(
                        min(self.config.rank_search.k, len(sts)),
                        sts,
                        lambda s: s.network.cost(),
                    )
                    best_costs = [st.network.cost() for st in smallest_sts]
                    best_st = self.get_cost(data_tensor, new_st, best_costs)
                    if best_st is not None:
                        heapq.heappush(sts, best_st)

                    if len(sts) > self.config.rank_search.k:
                        # pop the largest element
                        # Find the index of the largest element
                        largest_idx = max(
                            range(len(sts)),
                            key=lambda i: sts[i].network.cost(),
                        )
                        # Remove it from the heap
                        sts.pop(largest_idx)

            curr_sts = next_sts

        return heapq.nsmallest(
            min(self.config.rank_search.k, len(sts)),
            sts,
            lambda s: s.network.cost(),
        )

    def _top_k(
        self,
        data_tensor: DataTensor,
        sts: Sequence[SearchState],
    ) -> SearchResult:
        result = SearchResult()
        for st in sts:
            if self.config.rank_search.search_mode == "topk":
                for k, ac in enumerate(st.past_actions):
                    for ind in st.network.all_indices():
                        if ind.name == st.links[k]:
                            ac.target_size = ind.size
                            break

                replay_res = self.replay(data_tensor, st.past_actions, True)
                result = result.update_best_state(replay_res)
            else:
                if result.best_state is None or st < result.best_state:
                    result.best_state = st

        return result

    def _round(
        self, st: SearchState, tensor_func: Optional[TensorFunc] = None
    ) -> SearchResult:
        res = SearchResult()
        if tensor_func is not None:
            cross_res = st.network.cross(tensor_func, self._delta)
            # st.curr_delta = self.config.engine.eps * st.network.norm()
            res.best_state = st
            res.best_dim_tree = cross_res.dim_tree
            res.unused_delta = np.sqrt(
                self.config.engine.eps**2
                - cross_res.ranks_and_errors[-1][1] ** 2
            )
            return res

        best_state = st
        unused_delta = 0.0
        for n in st.network.network.nodes:
            tmp_st = copy.deepcopy(st)
            _, unused_delta = tmp_st.network.round(n, st.curr_delta)
            if tmp_st.network.cost() < best_state.network.cost():
                best_state = tmp_st

        res.best_state = best_state
        res.unused_delta = unused_delta
        return res

    def _replay_impl(
        self,
        st: SearchState,
        actions: List[Action],
        first_iter: bool = False,
        tensor_func: Optional[TensorFunc] = None,
    ) -> SearchResult:
        if not actions:
            return self._round(st, tensor_func)

        ac = actions[0]
        svd = None
        if first_iter and self.config.rank_search.search_mode == "all":
            svd_file = self.constraint_engine.first_steps.get(ac, None)
            if svd_file is None:
                raise ValueError("get no svd file in the mode 'all'")

            svd_data = np.load(svd_file)
            svd = (svd_data["u"], svd_data["s"], svd_data["v"])

        new_st = st.take_action(ac, svd=svd, tensor_func=tensor_func)
        if new_st is None:
            raise RuntimeError("cannot replay the given actions")

        timestamp = time.time() - self.stats.search_start
        self.stats.costs.append((timestamp, new_st.network.cost()))
        ukey = new_st.network.canonical_structure()
        self.stats.incr_unique(ukey)
        return self._replay_impl(new_st, actions[1:], tensor_func=tensor_func)

    def replay(
        self,
        data_tensor: DataTensor,
        actions: List[Action],
        first_iter: bool = False,
    ) -> SearchResult:
        """Apply the given actions around the given ranks."""
        st = init_state(data_tensor, self._delta)
        if isinstance(data_tensor, TreeNetwork):
            return self._replay_impl(st, actions, first_iter)

        if isinstance(data_tensor, TensorFunc):
            return self._replay_impl(st, actions, first_iter, data_tensor)

        raise TypeError("unsupported data tensor type")

    def rank_search(
        self, data_tensor: DataTensor, acs: List[Action]
    ) -> Optional[SearchState]:
        """Search for ranks for the given set of split actions."""
        st = init_state(data_tensor, self._delta)

        for ac in acs:
            ac.target_size = None
            st = self._sketch_execution(st, ac)

        return self.get_cost(data_tensor, st, [data_tensor.cost()])

    def preprocess(
        self, data_tensor: DataTensor, exclusions: Optional[Sequence[Index]]
    ) -> float:
        """Precompute the pair of ranks and errors for the given data tensor"""
        preprocess_start = time.time()
        if self.config.synthesizer.replay_from is not None:
            with open(self.config.synthesizer.replay_from, "rb") as ac_file:
                acs = pickle.load(ac_file)

            ind_combs = [ac.indices for ac in acs]
        else:
            ind_combs = SearchState.all_index_combs(data_tensor.free_indices())

        for comb in ind_combs:
            if (
                exclusions is not None
                and len(comb) == 1
                and comb[0] in exclusions
            ):
                continue

            self.constraint_engine.preprocess_comb(
                data_tensor,
                comb,
                compute_uv=self.config.rank_search.search_mode == "all",
            )

        if self.config.output.remove_temp_after_run:
            atexit.register(
                remove_temp_dir,
                self.config.output.output_dir,
                self.constraint_engine.temp_files,
            )
        preprocess_end = time.time()
        return preprocess_end - preprocess_start

    def search(
        self,
        data_tensor: DataTensor,
        delta: Optional[float] = None,
        exclusions: Optional[Sequence[Index]] = None,
    ) -> SearchResult:
        """Start the search from a given network.
        Only support single core now.
        """
        result = SearchResult()

        if delta is None:
            if isinstance(data_tensor, TreeNetwork):
                self._delta = data_tensor.norm() * self.config.engine.eps
            elif isinstance(data_tensor, TensorFunc):
                self._delta = self.config.engine.eps
            else:
                raise TypeError("Unsupported data tensor type")
        else:
            self._delta = delta

        self.constraint_engine.delta = self._delta
        self.stats.preprocess_time = self.preprocess(data_tensor, exclusions)
        self.stats.search_start = time.time()

        if self.config.synthesizer.replay_from is not None:
            with open(self.config.synthesizer.replay_from, "rb") as ac_file:
                acs = pickle.load(ac_file)

            best_st = self.rank_search(data_tensor, acs)
            if best_st is None:
                print("No better structure is found")
                return result

            # there should be only one value for costs
            assert len(best_st.links) == len(acs)
            # set the target ranks for actions
            for k, ind in enumerate(best_st.links):
                for i in best_st.network.all_indices():
                    if i.name == ind:
                        acs[k].target_size = i.size

            search_res = self.replay(data_tensor, acs, True)
        else:
            self.unused_delta = self._delta
            sts = self._enumerate(data_tensor, exclusions)
            search_res = self._top_k(data_tensor, sts)

        result = result.update_best_state(search_res)
        self.stats.search_end = time.time()
        result.stats = self.stats
        return result

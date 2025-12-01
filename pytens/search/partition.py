"""Structure search with output-directed splits."""

import atexit
import copy
import heapq
import logging
import pickle
import time
from typing import List, Optional, Sequence

import numpy as np
from line_profiler import profile

from pytens.algs import FoldedTensorTrain, Tensor, TensorTrain, TreeNetwork
from pytens.cross.cross import cross
from pytens.cross.funcs import FuncTensorNetwork, TensorFunc
from pytens.search.configuration import SearchConfig
from pytens.search.constraint import ConstraintSearch
from pytens.search.state import Action, ISplit, OSplit, SearchState
from pytens.search.utils import (
    DataTensor,
    SearchResult,
    SearchStats,
    get_conflicts,
    init_state,
    remove_temp_dir,
    to_splits,
)
from pytens.types import Index, IndexMerge, FoldDir, NodeName

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PartitionSearch:
    """Search by partitions free indices"""

    def __init__(self, config: SearchConfig, data_tensor: TreeNetwork):
        self.config = config
        self.constraint_engine = ConstraintSearch(config)
        self.unused_delta = 0.0
        self.stats = SearchStats()

        self._delta = 0.0
        self._data_tensor = data_tensor

    def reset(self):
        """Reset the search states."""
        self.unused_delta = 0.0
        self.stats = SearchStats()

    def get_cost(
        self,
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

            res = self.replay(new_st.past_actions, True)
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
        merge_ops: Sequence[IndexMerge],
        exclusions: Optional[Sequence[Index]],
    ) -> Sequence[SearchState]:
        """Enumerate all possible splits up to the maximum number of ops."""
        sts = [init_state(data_tensor, self._delta)]
        curr_sts = [init_state(data_tensor, self._delta)]
        # merged_indices = []
        # for merge_op in merge_ops:
        #     merged_indices.append(merge_op.result)

        # for ind in data_tensor.free_indices():
        #     found = False
        #     for merge_op in merge_ops:
        #         if ind in merge_op.indices:
        #             found = True
        #             break

        #     if not found:
        #         merged_indices.append(ind)

        for _ in range(1, self.config.engine.max_ops + 1):
            next_sts = []
            for curr_st in curr_sts:
                is_osplit = self.config.synthesizer.action_type == "osplit"
                for action in curr_st.get_legal_actions(is_osplit, merge_ops):
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
                    best_st = self.get_cost(new_st, best_costs)
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

                # print([str(ac) for ac in st.past_actions])
                replay_res = self.replay(st.past_actions, True)
                result = result.update_best_state(replay_res)
            else:
                if result.best_state is None or st < result.best_state:
                    result.best_state = st

        return result

    def _round(self, st: SearchState) -> SearchResult:
        res = SearchResult()
        best_state = st
        unused_delta = 0.0
        for n in st.network.network.nodes:
            tmp_st = copy.deepcopy(st)
            _, unused_delta = tmp_st.network.round(n, st.curr_delta)
            if tmp_st.network.cost() < best_state.network.cost():
                best_state = tmp_st

        best_state.network.compress()
        res.best_state = best_state
        res.unused_delta = unused_delta
        return res

    @profile
    def _replay_impl(
        self,
        st: SearchState,
        actions: List[Action],
        first_iter: bool = False,
    ) -> SearchResult:
        if not actions:
            # # undo everything else
            # while True:
            #     modified = False
            #     for ac in to_splits(st.network):
            #         # print("checking action", ac)
            #         # for pac in st.past_actions:
            #         # print("past action", pac)

            #         if ac not in st.past_actions:
            #             # print("revert", ac)
            #             modified = True
            #             st.network.merge(*ac.reverse_edge)
            #             break
            #     # print("-------")
            #     if not modified:
            #         break

            # print(st.network)
            return self._round(st)

        ac = actions[0]
        # st = copy.deepcopy(st)
        # if isinstance(st.network, TensorTrain):
        #     st.network.fold(ac.indices)
        # if isinstance(st.network, TensorTrain):
        #     st = copy.deepcopy(st)
        #     st.network, _ = st.network.swap(ac.indices)
        conflict_ac = get_conflicts(ac, to_splits(st.network))
        st = copy.deepcopy(st)
        while conflict_ac is not None:
            assert conflict_ac.reverse_edge is not None
            st.network.merge(*conflict_ac.reverse_edge)
            conflict_ac = get_conflicts(ac, to_splits(st.network))

        svd = None
        if first_iter and self.config.rank_search.search_mode == "all":
            svd_file = self.constraint_engine.first_steps.get(ac, None)
            if svd_file is None:
                raise ValueError("get no svd file in the mode 'all'")

            svd_data = np.load(svd_file)
            svd = (svd_data["u"], svd_data["s"], svd_data["v"])

        # print(st.network)
        # print(ac, ac.target_size)
        # print("applying", ac)
        new_st = st.take_action(ac, svd=svd)
        if new_st is None:
            raise RuntimeError("cannot replay the given actions")

        new_st.network.compress()
        timestamp = time.time() - self.stats.search_start
        self.stats.costs.append((timestamp, new_st.network.cost()))
        ukey = new_st.network.canonical_structure()
        self.stats.incr_unique(ukey)

        return self._replay_impl(new_st, actions[1:])

    def replay(
        self,
        actions: List[Action],
        first_iter: bool = False,
    ) -> SearchResult:
        """Apply the given actions around the given ranks."""
        st = init_state(self._data_tensor, self._delta)
        self._data_tensor.replay_preprocess(actions)
        return self._replay_impl(st, actions, first_iter)

    def rank_search(self, acs: List[Action]) -> Optional[SearchState]:
        """Search for ranks for the given set of split actions."""
        st = init_state(self._data_tensor, self._delta)

        for ac in acs:
            ac.target_size = None
            st = self._sketch_execution(st, ac)

        return self.get_cost(st, [self._data_tensor.cost()])

    def _cross_to_tt_to_ftt(self, merge_ops: Sequence[IndexMerge]):
        """Handle index merging during preprocessing for cross results."""
        # after we know how indices are merged, we create a permuted function
        new_indices = []
        for mop in merge_ops:
            new_indices.extend(mop.indices)

        # there exists some free indices are not merged
        for ind in self._data_tensor.free_indices():
            found = False
            for mop in merge_ops:
                if ind in mop.indices:
                    found = True
                    break

            if not found:
                new_indices.append(ind)

        assert len(new_indices) == len(self._data_tensor.free_indices())
        # print("reorder the indices into", new_indices)
        if self._data_tensor.size() < 1e8:
            tt = self._data_tensor.reorder_by_svd(new_indices, 0)
        else:
            tt = self._data_tensor.reorder_by_cross(
                new_indices, self.config.engine.eps * 0.5
            )
        # if not ok:
        #     print("warning: cross didn't reach the target eps")
        # tt = data_tensor.reorder_by_svd(new_indices, self.config.engine.eps)

        # fold the indices according to merge ops
        for mop in merge_ops:
            start_node = tt.node_by_free_index(mop.indices[0].name)
            end_node = tt.node_by_free_index(mop.indices[-1].name)
            tt, _ = tt.fold_nodes(start_node, end_node, FoldDir.IN_BOUND)

        # there exists some free indices are not merged
        for ind in self._data_tensor.free_indices():
            found = False
            for mop in merge_ops:
                if ind in mop.indices:
                    found = True
                    break

            if not found:
                tt.backbone_nodes.append(tt.node_by_free_index(ind.name))

        # orthonormalize before proceeding
        if isinstance(tt, FoldedTensorTrain):
            tt.orthonormalize(tt.backbone_nodes[0])
        else:
            tt.orthonormalize(list(tt.network.nodes)[0])

        self._data_tensor = tt

    def _cross_to_ftt(self, merge_ops: Sequence[IndexMerge]):
        """Handle index merging during preprocessing for cross results."""
        # after we know how indices are merged, we create a permuted function
        new_indices = []
        for mop in merge_ops:
            new_indices.append(mop.indices)

        # there exists some free indices are not merged
        for ind in self._data_tensor.free_indices():
            found = False
            for mop in merge_ops:
                if ind in mop.indices:
                    found = True
                    break

            if not found:
                new_indices.append([ind])

        tt = FoldedTensorTrain.rand_ftt(new_indices)
        finds = [
            ind.with_new_rng(range(ind.size)) for ind in tt.free_indices()
        ]
        func = FuncTensorNetwork(finds, self._data_tensor)
        cross(
            func,
            tt,
            tt.backbone_nodes[0],
            eps=self.config.engine.eps,
            max_iters=100,
            kickrank=1,
        )

        # orthonormalize before proceeding
        if isinstance(tt, FoldedTensorTrain):
            tt.orthonormalize(tt.backbone_nodes[0])
        else:
            tt.orthonormalize(list(tt.network.nodes)[0])

        self._data_tensor = tt

    # def _preprocess_ftt(self, data_tensor: FoldedTensorTrain) -> FoldedTensorTrain:
    #     data_tensor.orthonormalize(data_tensor.backbone_nodes[0])
    #     return data_tensor

    def preprocess(
        self,
        merge_ops: Sequence[IndexMerge],
        exclusions: Optional[Sequence[Index]],
    ) -> None:
        """Precompute the pair of ranks and errors for the given data tensor"""
        if self.config.synthesizer.replay_from is not None:
            with open(self.config.synthesizer.replay_from, "rb") as ac_file:
                acs = pickle.load(ac_file)

            ind_combs = [ac.indices for ac in acs]
        else:
            indices = []
            for mop in merge_ops:
                indices.append(mop.result)

            for ind in self._data_tensor.free_indices():
                found = False
                for mop in merge_ops:
                    if ind in mop.indices:
                        found = True
                        break

                if not found:
                    indices.append(ind)

            ind_combs = SearchState.all_index_combs(indices)

        for comb in ind_combs:
            if (
                exclusions is not None
                and len(comb) == 1
                and comb[0] in exclusions
            ):
                continue

            # restore comb to the original indices
            restored_comb = []
            for ind in comb:
                found = False
                for mop in merge_ops:
                    if ind == mop.result:
                        found = True
                        restored_comb.extend(mop.indices)

                if not found:
                    restored_comb.append(ind)

            comb_complement = [
                ind
                for ind in self._data_tensor.free_indices()
                if ind not in restored_comb
            ]
            comb_ac = OSplit(restored_comb)
            complement_ac = OSplit(comb_complement)
            ac = min(comb_ac, complement_ac)

            self.constraint_engine.preprocess_comb(
                self._data_tensor,
                ac.indices,
                compute_uv=self.config.rank_search.search_mode == "all",
            )

        if self.config.output.remove_temp_after_run:
            atexit.register(
                remove_temp_dir,
                self.config.output.output_dir,
                self.constraint_engine.temp_files,
            )

        # if isinstance(data_tensor, TensorFunc):
        #     self.stats.search_cross_evals += data_tensor.stats

    @profile
    def search(
        self,
        merge_ops: Sequence[IndexMerge],
        delta: Optional[float] = None,
        exclusions: Optional[Sequence[Index]] = None,
    ) -> SearchResult:
        """Start the search from a given network.
        Only support single core now.
        """
        result = SearchResult()

        if delta is None:
            self._delta = self._data_tensor.norm() * self.config.engine.eps
        else:
            self._delta = delta

        logger.debug(
            "**delta: %s, data norm: %s", self._delta, self._data_tensor.norm()
        )
        self.constraint_engine.delta = self._delta

        transform_start = time.time()
        if isinstance(self._data_tensor, TensorTrain):
            if self.config.cross.init_struct == "ftt":
                self._cross_to_ftt(merge_ops)
            else:
                self._cross_to_tt_to_ftt(merge_ops)
        self.stats.merge_transform_time = time.time() - transform_start

        # print(data_tensor)
        # if isinstance(data_tensor, FoldedTensorTrain):
        #     data_tensor = self._preprocess_ftt(data_tensor)
        preprocess_start = time.time()
        self.preprocess(merge_ops, exclusions)
        self.stats.preprocess_time = time.time() - preprocess_start
        self.stats.search_start = time.time()

        if self.config.synthesizer.replay_from is not None:
            with open(self.config.synthesizer.replay_from, "rb") as ac_file:
                acs = pickle.load(ac_file)

            best_st = self.rank_search(acs)
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

            search_res = self.replay(acs, True)
        else:
            self.unused_delta = self._delta
            empty_net = TreeNetwork()
            empty_net.add_node(
                "G", Tensor(np.empty(0), self._data_tensor.free_indices())
            )
            # print("starting enumeration")
            # print(empty_net)
            sts = self._enumerate(empty_net, merge_ops, exclusions)
            # if isinstance(data_tensor, TensorTrain) and len(merge_ops) > 0:
            #     data_tensor = data_tensor.reorder(merge_ops, 0)
            search_res = self._top_k(sts)

        result = result.update_best_state(search_res)
        self.stats.search_end = time.time()
        result.stats = self.stats
        return result
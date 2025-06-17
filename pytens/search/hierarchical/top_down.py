"""Top down reshaping search"""

import random
import math
import copy
from typing import List, Optional, Union, Tuple, Sequence, Self
import itertools

import sympy
import numpy as np

from pytens.search.configuration import SearchConfig
from pytens.search.partition import PartitionSearch
from pytens.algs import TreeNetwork, NodeName
from pytens.types import IndexSplit, IndexMerge, Index
from pytens.search.hierarchical.error_dist import BaseErrorDist
from pytens.cross.funcs import TensorFunc
from pytens.search.hierarchical.utils import (
    trigger_merge,
    corr,
    select_factors,
)


class HSearchState:
    """Hierarchical search state"""

    def __init__(
        self,
        free_indices: List[Index],
        reshape_history: List[Union[IndexMerge, IndexSplit]],
        network: TreeNetwork,
        unused_delta: float,
    ):
        self.free_indices = free_indices
        self.reshape_history = reshape_history
        self.network = network
        self.unused_delta = unused_delta

    def merge_index(self, merge_op: IndexMerge) -> Self:
        """Perform a merge operation on the given node."""
        new_st = copy.deepcopy(self)
        new_net = new_st.network
        new_net.merge_index(merge_op)

        new_indices = []
        for i in new_net.free_indices():
            if i not in self.network.free_indices():
                new_indices.append(i)

        for mi in merge_op.indices:
            new_st.free_indices.remove(mi)

        merge_op.result = new_indices[0]
        new_st.reshape_history.append(merge_op)
        new_st.free_indices.extend(new_indices)

        return new_st

    def split_index(self, split_op: IndexSplit) -> Self:
        """Perform a split operation on the given node."""
        # print("applying split", split_op)
        new_st = copy.deepcopy(self)
        new_net = new_st.network
        new_net.split_index(split_op)
        # print(node)

        old_indices = self.network.free_indices()
        new_indices = []
        for i in new_net.free_indices():
            if i not in old_indices:
                new_indices.append(i)

        ind = split_op.index
        new_st.free_indices.remove(ind)
        new_st.free_indices.extend(new_indices)
        new_st.reshape_history.append(split_op)

        return new_st


def _create_split_target(
    factors: List[int], selected_factors: List[int]
) -> List[int]:
    remaining_factors = factors[:]
    for f in selected_factors:
        remaining_factors.remove(f)

    if len(remaining_factors) == 0:
        return list(selected_factors)

    remaining_size = math.prod(remaining_factors)
    return list(selected_factors) + [remaining_size]


class TopDownSearch:
    """Search for reshaped structures from top to bottom"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.error_dist = BaseErrorDist()

    def search(
        self,
        data_tensor: TreeNetwork,
        error_dist: BaseErrorDist,
    ) -> Tuple[TreeNetwork, HSearchState]:
        """Perform the topdown search starting from the given net"""
        if isinstance(data_tensor, TreeNetwork):
            delta = data_tensor.norm() * self.config.engine.eps
            free_indices = data_tensor.free_indices()
        elif isinstance(data_tensor, TensorFunc):
            delta = self.config.engine.eps
            free_indices = data_tensor.indices
        else:
            raise TypeError("unknown data tensor type")

        init_st = HSearchState(free_indices, [], copy.deepcopy(data_tensor), 0)
        best_network = data_tensor
        best_st = init_st
        # print(net.free_indices())
        st = self._search_at_level(init_st, delta, None)
        nodes = list(st.network.network.nodes)
        for n in nodes:
            network = copy.deepcopy(st.network)
            network.round(n, delta=math.sqrt(st.unused_delta))
            if network.cost() < best_network.cost():
                best_network = network
                best_st = st

        return best_network, best_st

    def _search_at_level(
        self,
        st: HSearchState,
        remaining_delta: float,
        parent_indices: Optional[List[Index]],
    ) -> HSearchState:
        search_engine = PartitionSearch(self.config)
        # decrease the delta budget exponentially
        delta, remaining_delta = self.error_dist.split_delta(remaining_delta)
        merge_ops, split_ops = [], []
        perform_merge = trigger_merge(self.config, parent_indices)
        if perform_merge:
            merge_ops, split_ops = self._merge_by_correlation(st.network)
            for merge_op in merge_ops:
                st.network.merge_index(merge_op)

        result = search_engine.search(st.network, delta=delta)
        if result.best_state is None:
            return st

        bn = result.best_state.network

        if perform_merge:
            # restore the hint indices
            for split_op in split_ops:
                bn.split_index(split_op)

        next_nodes = list(bn.network.nodes)
        # distribute delta equally to all subnets
        remaining_delta = remaining_delta / math.sqrt(len(next_nodes))
        unused_delta = result.unused_delta**2 + st.unused_delta
        best_st = HSearchState(st.free_indices, st.reshape_history, bn, 0)

        # enumerate nodes in the order of their scores
        for node in next_nodes:
            optimize_res = self._optimize_node(best_st, node, remaining_delta)
            best_res = optimize_res[0]
            for res in optimize_res:
                if res[1].network.cost() < best_res[1].network.cost():
                    best_res = res

            if best_res is not None:
                best_st.network = best_res[0]
                best_sn_st = best_res[1]
                best_st.network.replace_with(
                    node, best_sn_st.network, best_sn_st.reshape_history
                )
                best_st.free_indices = best_sn_st.free_indices
                best_st.reshape_history = best_sn_st.reshape_history
                unused_delta += best_sn_st.unused_delta
            else:
                unused_delta += remaining_delta**2

        best_st.unused_delta = unused_delta
        # self.memoization[tuple(st.network.free_indices())] = best_st
        return best_st

    def _merge_by_correlation(
        self,
        net: TreeNetwork,
        threshold: int = 4,
    ) -> Tuple[Sequence[IndexMerge], Sequence[IndexSplit]]:
        """Consider all possible combinations of indices.

        For each combination, we calculate the correlation matrix of
        the reshaped tensor. If the correlation is high enough,
        we merge the indices.
        """

        tensor = list(net.network.nodes(data=True))[0][1]["tensor"]
        value = tensor.value
        indices = tensor.indices

        if len(indices) <= threshold:
            return [], []

        shape = [ind.size for ind in indices]
        comb_corr = {}
        for comb in itertools.combinations(range(len(indices)), 2):
            rights = [x for x in range(len(indices)) if x not in comb]
            value_perm = value.transpose(list(comb) + rights)
            others = [x for i, x in enumerate(shape) if i not in comb]
            value_group = value_perm.reshape(-1, *others)
            corr_data = value_group.reshape(-1, np.prod(others))
            comb_corr[comb] = corr(corr_data, self.config.topdown.aggregation)

        # consider different cases to merge indices from high to low
        # until it reaches the target threshold
        comb_corr = sorted(comb_corr.items(), key=lambda x: x[1])
        merged_indices = set()
        groups = 0
        merge_ops, split_ops = [], []
        for comb, _ in comb_corr:
            assert len(comb) == 2
            if comb[0] in merged_indices or comb[1] in merged_indices:
                continue  # do not allow overlapping indices

            print("adding", comb)
            merged_indices.add(comb[0])
            merged_indices.add(comb[1])

            m_indices = [indices[i] for i in comb]
            m_shape = [ind.size for ind in m_indices]
            m_ind_size = np.prod(m_shape, dtype=int)
            m_ind = Index(f"tmp_merge_{groups}", m_ind_size)

            merge_op = IndexMerge(indices=m_indices, result=m_ind)
            merge_ops.append(merge_op)

            split_op = IndexSplit(index=m_ind, shape=m_shape, result=m_indices)
            split_ops.append(split_op)

            ind_cnt = len(indices) - len(merged_indices) + groups
            if ind_cnt <= threshold:
                break

        return merge_ops, split_ops

    def _optimize_node(
        self, st: HSearchState, node: NodeName, remaining_delta: float
    ) -> List[Tuple[TreeNetwork, HSearchState]]:
        """Optimize the children nodes in a given network"""
        results = []
        for split_result in self._split_indices(st, node):
            split_result.network.orthonormalize(node)
            new_sn = TreeNetwork()
            new_sn.add_node(node, split_result.network.node_tensor(node))
            new_st = HSearchState(
                split_result.free_indices,
                split_result.reshape_history,
                new_sn,
                split_result.unused_delta,
            )
            sn_st = self._search_at_level(
                new_st,
                remaining_delta,
                st.network.node_tensor(node).indices,
            )
            results.append((split_result.network, sn_st))

        return results

    def _split_indices(
        self, st: HSearchState, node: NodeName
    ) -> Sequence[HSearchState]:
        net = st.network
        indices = net.network.nodes[node]["tensor"].indices
        index_splits = self._split_indices_on_budget(st, indices)

        # if self.config.topdown.search_algo == "enumerate":
        #     index_splits = self._split_indices_enum(st, indices)
        # elif self.config.topdown.search_algo == "correlation":
        #     index_splits = self._split_indices_correlation(st, indices)

        # index_splits = sorted(index_splits, key=score_split, reverse=True)
        # index_splits = filter(score_split, index_splits)
        seen = set()
        result_sts = []
        for index_split in index_splits:
            if tuple(index_split) in seen:
                continue

            # print(index_split, "not seen previously in", seen)
            seen.add(tuple(index_split))

            # we need to evaluate how long it takes in the scoring function
            # if not score_split(index_split):
            #     print("Skipping split op", index_split)
            #     continue

            refactored = False
            new_st = copy.deepcopy(st)
            # print(splits_allowed, index_split)
            for split_op in index_split:
                split_op = copy.deepcopy(split_op)

                tmp_net = new_st.network
                tmp_indices = tmp_net.node_tensor(node).indices
                ndims = len(tmp_indices) + len(split_op.shape) - 1
                if (
                    self.config.topdown.search_algo == "enumerate"
                    and ndims > self.config.topdown.group_threshold
                ):
                    continue

                new_st = new_st.split_index(split_op)
                refactored = True
                # To avoid merge in the middle, we need to ensure that
                # none of the splits goes beyond the threshold
                # new_net = new_st.network
                # new_indices = new_net.network.nodes[node]["tensor"].indices
                # ndims = len(new_indices)
                # if ndims > self.config.topdown.group_threshold:
                #     for merged_st in self._merge_indices(new_st, node):
                #         yield refactored, merged_st

                #     return

            if refactored:
                result_sts.append(new_st)

        return result_sts

    def _split_indices_on_budget(
        self, st: HSearchState, indices: List[Index]
    ) -> List[List[IndexSplit]]:
        # distribute the allowed splits between indices

        maxs = []
        for ind in indices:
            if ind in st.free_indices:
                factors = sympy.factorint(ind.size)
                maxs.append(sum(factors.values()) - 1)
            else:
                maxs.append(0)

        if self.config.topdown.search_algo in ("enumerate", "random"):
            budget = self.config.topdown.group_threshold - len(indices)
        elif self.config.topdown.search_algo == "correlation":
            budget = sum(maxs)
        else:
            raise ValueError("unknown search algorithm in top down search")

        budget = min(sum(maxs), budget)  # exhaust the budget as possible
        all_splits = []
        for ind_budget in itertools.product(*[range(x + 1) for x in maxs]):
            if sum(ind_budget) != budget:
                continue

            splits = []
            for i, ind in enumerate(indices):
                ind_splits = self._get_split_op(st, ind, ind_budget[i])
                if len(ind_splits) != 0:
                    splits.append(ind_splits)

            all_splits.extend(itertools.product(*splits))

        return all_splits

    # def _split_indices_corr(
    #     self, st: HSearchState, indices: List[Index]
    # ) -> Iterable[Sequence[IndexSplit]]:
    #     # split each index into most smaller parts
    #     index_splits = []

    #     for ind in indices:
    #         if ind not in st.free_indices:
    #             continue

    #         factors = sympy.factorint(ind.size)
    #         factor_list = [i for i, n in factors.items() for _ in range(n)]
    #         if len(factor_list) == 1:
    #             continue

    #         split_ops = [
    #             IndexSplit(index=ind, shape=factor_perm)
    #             for factor_perm in permute_unique(factor_list)
    #         ]
    #         index_splits.append(split_ops)

    #     return itertools.product(*index_splits)

    def _get_split_op(
        self, st: HSearchState, index: Index, budget: int
    ) -> Sequence[IndexSplit]:
        if index not in st.free_indices or budget <= 0:
            return []

        res = sympy.factorint(index.size)
        factors = [i for i, n in res.items() for _ in range(n)]
        if len(factors) == 1:
            return []

        if self.config.topdown.search_algo == "random":
            k = random.randint(0, len(factors))
            selected = random.sample(factors, k=k)
            shape = _create_split_target(factors, selected)
            return [IndexSplit(index=index, shape=shape)]
        else:
            # we always try our best to decompose the indices to
            # the maximum number and they subsume higher level reshapes
            split_ops = []
            for shape in select_factors(res, budget):
                # for selected in itertools.combinations(factors, r=k):
                split_ops.append(IndexSplit(index=index, shape=shape))

            return split_ops

    def _get_merge_op(
        self, merge_candidates: List[Index]
    ) -> Sequence[IndexMerge]:
        if self.config.topdown.random_algorithm == "random":
            # yield one possible result
            merge_indices = sorted(random.sample(merge_candidates, k=2))
            return [IndexMerge(indices=merge_indices)]
        else:
            merge_len = len(merge_candidates) - 1
            if merge_len < 2:
                return []

            merge_ops = []
            for i in range(2, merge_len):
                for comb in itertools.combinations(merge_candidates, i):
                    merge_ops.append(IndexMerge(indices=comb))

            return merge_ops

    def _merge_indices(
        self, st: HSearchState, node: NodeName
    ) -> Sequence[HSearchState]:
        indices = st.network.network.nodes[node]["tensor"].indices
        if len(indices) > self.config.topdown.group_threshold:
            merge_candidates = []
            for ind in indices:
                if ind in st.free_indices:
                    merge_candidates.append(ind)

            results = []
            for merge_op in self._get_merge_op(merge_candidates):
                if merge_op is not None:
                    new_st = st.merge_index(merge_op)
                    results.extend(self._merge_indices(new_st, node))

            return results
        else:
            return [st]

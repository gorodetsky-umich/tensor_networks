"""Top down reshaping search"""

import copy
import itertools
import math
import random
from typing import List, Optional, Self, Sequence, Tuple, Union, Literal

import numpy as np
import sympy

from pytens.algs import TreeNetwork
from pytens.cross.funcs import TensorFunc, FuncData
from pytens.search.configuration import SearchConfig
from pytens.search.hierarchical.error_dist import BaseErrorDist
from pytens.search.hierarchical.utils import (
    corr,
    select_factors,
    trigger_merge,
)
from pytens.search.partition import PartitionSearch
from pytens.search.utils import DataTensor, init_state
from pytens.types import Index, IndexMerge, IndexSplit, NodeName


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

    # def node_func(self, node: NodeName):
    #     """Get the tensor function for the specified node in the parent net."""
    #     tensor = self.network.node_tensor(node)
    #     indices = [ind.with_new_rng(range(0, ind.size)) for ind in tensor.indices[:]]
    #     return FuncData(indices, tensor.value)

    # def split_func(self, split_ops: Sequence[IndexSplit], node: Optional[NodeName] = None):
    #     """Get the tensor function for the sequence of split operations."""
    #     if node is None:
    #         free_indices = self.network.free_indices()
    #     else:
    #         free_indices = self.network.node_tensor(node).indices

    #     old_func = self.tensor_func
    #     assert old_func is not None, "tensor_func is missing"
    #     old_free = old_func.indices
    #     var_mapping = {}
    #     for split_op in split_ops:
    #         split_out = split_op.result
    #         if split_out is None:
    #             continue

    #         split_inds, split_sizes = [], []
    #         for ind in split_out:
    #             split_inds.append(free_indices.index(ind))
    #             split_sizes.append(int(ind.size))

    #         before_split = old_free.index(split_op.index)
    #         var_mapping[before_split] = (split_inds, split_sizes)

    #     return SplitFunc(free_indices, old_func, var_mapping)

    # def merge_func(
    #     self,
    #     merge_ops: Sequence[IndexMerge],
    # ):
    #     """Create the merged function after index merging."""
    #     assert self.tensor_func is not None, "tensor_func is missing"

    #     old_func = self.tensor_func
    #     old_free = old_func.indices
    #     free_indices = sorted(self.network.free_indices())
    #     var_mapping = {}
    #     for merge_op in merge_ops:
    #         assert merge_op.result is not None, (
    #             "no merge result after index merging"
    #         )
    #         result_ind = free_indices.index(merge_op.result)
    #         old_sizes = [ind.size for ind in merge_op.indices]
    #         old_inds = [old_free.index(ind) for ind in merge_op.indices]
    #         var_mapping[result_ind] = (old_sizes, old_inds)

    #     return MergeFunc(free_indices, old_func, var_mapping)


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

    def cross(
        self, f: TensorFunc, struct: Literal["tucker", "ht", "tt"]
    ) -> TreeNetwork:
        """Compress f into a fixed structure."""
        if struct == "tucker":
            net = TreeNetwork.tucker(f.indices)
        else:
            raise NotImplementedError

        net.cross(f, eps=self.config.engine.eps * self.config.cross.init_eps)
        return net

    def search(
        self,
        data_tensor: DataTensor,
    ) -> HSearchState:
        """Perform the topdown search starting from the given net"""
        if isinstance(data_tensor, TreeNetwork):
            delta = data_tensor.norm() * self.config.engine.eps
            free_indices = data_tensor.free_indices()
        elif isinstance(data_tensor, TensorFunc):
            delta = self.config.engine.eps
            free_indices = data_tensor.indices
            # run the cross approximation to avoid expensive error querying
            data_tensor = self.cross(
                data_tensor, self.config.cross.init_struct
            )
        else:
            raise TypeError("unknown data tensor type")

        init_st = init_state(data_tensor, delta)
        init_st = HSearchState(free_indices[:], [], init_st.network, 0)
        best_st = init_st

        st = self._search_at_level(init_st, delta, True)
        nodes = list(st.network.network.nodes)
        for n in nodes:
            network = copy.deepcopy(st.network)
            # network.round(n, delta=math.sqrt(st.unused_delta))
            if network.cost() < best_st.network.cost():
                best_st = st

        return best_st

    def _search_at_level(
        self,
        st: HSearchState,
        remaining_delta: float,
        is_top: bool = False,
        exclusions: Optional[Sequence[Index]] = None,
        parent_ind: Optional[Index] = None,
    ) -> HSearchState:
        search_engine = PartitionSearch(self.config)
        # decrease the delta budget exponentially
        delta, remaining_delta = self.error_dist.split_delta(remaining_delta)

        merge_ops, split_ops = self._merge_by_corr(st.network, is_top)
        for merge_op in merge_ops:
            st.network.merge_index(merge_op)

        if self.config.engine.decomp_algo == "svd":
            result = search_engine.search(
                st.network,
                delta=delta,
                exclusions=exclusions,
                parent_ind=parent_ind,
            )
        elif self.config.engine.decomp_algo == "cross":
            tensor = st.network.contract()
            indices = []
            for ind in tensor.indices[:]:
                indices.append(ind.with_new_rng(range(0, ind.size)))
            result = search_engine.search(
                FuncData(indices, tensor.value),
                delta=delta,
                exclusions=exclusions,
                parent_ind=parent_ind,
            )
        else:
            raise ValueError("unknown decomposition algorithm")

        if result.best_state is None:
            return st

        bn = result.best_state.network

        for split_op in split_ops:
            bn.split_index(split_op)

        next_nodes = list(bn.network.nodes)
        # distribute delta equally to all subnets
        remaining_delta = remaining_delta / math.sqrt(len(next_nodes))
        unused_delta = result.unused_delta**2 + st.unused_delta
        best_st = HSearchState(st.free_indices, st.reshape_history, bn, 0)

        # print(bn)
        # enumerate nodes in the order of their scores
        for node in next_nodes:
            # print(node)

            optimize_res = self._optimize_node(
                best_st, node, remaining_delta, None
            )
            best_res = None
            for res in optimize_res:
                if (
                    best_res is None
                    or res[1].network.cost() < best_res[1].network.cost()
                ):
                    best_res = res

            if best_res is not None:
                best_st.network = best_res[0]
                best_sn_st = best_res[1]
                # print("best subnet for", node)
                # print(best_sn_st.network)
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

    def _merge_by_corr(
        self,
        net: TreeNetwork,
        is_top: bool = False,
        threshold: int = 4,
    ) -> Tuple[Sequence[IndexMerge], Sequence[IndexSplit]]:
        """Consider all possible combinations of indices.

        For each combination, we calculate the correlation matrix of
        the reshaped tensor. If the correlation is high enough,
        we merge the indices.
        """
        if not trigger_merge(self.config, is_top):
            return [], []

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
        self,
        st: HSearchState,
        node: NodeName,
        remaining_delta: float,
        parent_ind: Optional[Index] = None,
    ) -> List[Tuple[TreeNetwork, HSearchState]]:
        """Optimize the children nodes in a given network"""
        results = []
        for split_result, used_splits in self._split_indices(st, node):
            new_sn = TreeNetwork()
            new_sn.add_node(node, split_result.network.node_tensor(node))
            new_st = HSearchState(
                split_result.free_indices,
                split_result.reshape_history,
                new_sn,
                split_result.unused_delta,
            )

            # if st.tensor_func is not None:
            #     new_st.tensor_func = split_result.split_func(used_splits, node)

            # exclude actions that split single internal indices
            exclusions = []
            for ind in new_sn.free_indices():
                if ind not in split_result.free_indices:
                    exclusions.append(ind)

            sn_st = self._search_at_level(
                new_st,
                remaining_delta,
                exclusions=exclusions,
                parent_ind=parent_ind,
            )
            results.append((split_result.network, sn_st))

        return results

    def _split_indices(
        self, st: HSearchState, node: NodeName
    ) -> Sequence[Tuple[HSearchState, Sequence[IndexSplit]]]:
        indices = st.network.node_tensor(node).indices
        index_splits = self._split_indices_on_budget(st, indices)

        seen = set()
        result_sts = []
        for index_split in index_splits:
            if tuple(index_split) in seen:
                continue

            seen.add(tuple(index_split))

            refactored = False
            new_st = copy.deepcopy(st)
            # if st.tensor_func is None:
            new_st.network.orthonormalize(node)

            used_splits = []
            for split_op in index_split:
                split_op = copy.deepcopy(split_op)
                tmp_indices = new_st.network.node_tensor(node).indices

                ndims = len(tmp_indices) + len(split_op.shape) - 1
                if (
                    self.config.topdown.search_algo == "enumerate"
                    and ndims > self.config.topdown.group_threshold
                ):
                    continue

                new_st = new_st.split_index(split_op)
                used_splits.append(split_op)
                refactored = True

            if refactored:
                result_sts.append((new_st, used_splits))

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

            all_splits.extend(list(itertools.product(*splits)))

        return all_splits

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

        return [st]

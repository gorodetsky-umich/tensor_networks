"""Top down reshaping search"""

import copy
import itertools
import math
import random
import time
from typing import List, Optional, Self, Sequence, Tuple, Union, Literal
import logging
import pickle

import numpy as np
import sympy
import torch
import tntorch
from line_profiler import profile

from pytens.search.state import OSplit
from pytens.algs import HierarchicalTucker, TreeNetwork, TensorTrain, Tensor
from pytens.cross.cross import cross
from pytens.cross.funcs import (
    FuncTensorNetwork,
    TensorFunc,
    PermuteFunc,
    SplitFunc,
    CountableFunc,
)
from pytens.search.configuration import SearchConfig
from pytens.search.hierarchical.error_dist import BaseErrorDist
from pytens.search.hierarchical.utils import (
    tntorch_wrapper,
    select_factors,
    trigger_merge,
    split_func,
    tntorch_to_tt,
)
from pytens.search.partition import PartitionSearch
from pytens.search.utils import (
    DataTensor,
    init_state,
    SearchStats,
    reshape_indices,
    seed_all,
)
from pytens.types import (
    Index,
    IndexMerge,
    IndexSplit,
    NodeName,
    IndexPermute,
    IndexOp,
    SVDAlgorithm,
)
from pytens.search.hierarchical.utils import DisjointSet

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class HSearchState:
    """Hierarchical search state"""

    def __init__(
        self,
        free_indices: List[Index],
        reshape_history: List[IndexOp],
        network: TreeNetwork,
        unused_delta: float,
    ):
        self.free_indices = free_indices
        self.reshape_history = reshape_history
        self.network = network
        self.unused_delta = unused_delta

    # after the cross, we do the normal but the data tensor is a tensor network.
    def merge_index(self, merge_op: IndexMerge) -> Self:
        """Perform a merge operation on the given node."""
        new_st = copy.deepcopy(self)
        new_st.network = new_st.network.merge_index(merge_op)
        new_net = new_st.network
        assert new_net is not None, "merge operation failed"

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

    def split_index(
        self, split_op: IndexSplit, compute_data: bool = True
    ) -> Self:
        """Perform a split operation on the given node."""
        # print("applying split", split_op)
        new_st = copy.deepcopy(self)
        new_net = new_st.network
        new_net.split_index(split_op, compute_data=compute_data)
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
        self.stats = SearchStats()
        self.init_splits = 0

    @profile
    def search(
        self,
        data_tensor: DataTensor,
    ) -> HSearchState:
        """Perform the topdown search starting from the given net"""
        if isinstance(data_tensor, TreeNetwork):
            delta = data_tensor.norm() * self.config.engine.eps
            free_indices = data_tensor.free_indices()
            init_st = HSearchState(free_indices[:], [], data_tensor, 0)

        elif isinstance(data_tensor, CountableFunc):
            # we run index split to split as many as possible
            splits, f = self._gen_split_func(data_tensor)
            # run the cross approximation to avoid expensive error querying
            net = self._run_init_cross(data_tensor, f)
            delta = net.norm() * self.config.engine.eps
            init_st = HSearchState(f.indices, splits, net, 0)

        else:
            raise ValueError("unknown decomposition algorithm")

        best_st = init_st

        st = self._search_at_level(init_st, delta, True)
        nodes = list(st.network.network.nodes)
        for n in nodes:
            network = copy.deepcopy(st.network)
            network.round(n, delta=math.sqrt(st.unused_delta))
            if network.cost() < best_st.network.cost():
                best_st = st
                best_st.network = network

        return best_st

    def _gen_split_func(
        self, data_tensor: CountableFunc
    ) -> Tuple[List[IndexOp], TensorFunc]:
        free_indices = data_tensor.indices
        init_st = init_state(data_tensor, self.config.engine.eps)

        splits = []

        # create index splits in the chosen order
        st = HSearchState(free_indices[:], [], init_st.network, 0)
        for n in st.network.network.nodes:
            split_results = self._split_indices(st, n, compute_data=False)
            if len(split_results) == 0:
                continue

            st, used_splits = split_results[0]
            splits.extend(used_splits)

        split_indices = list(sorted(st.network.free_indices()))
        split_f = split_func(data_tensor, split_indices, splits)

        if self.config.topdown.merge_algo == "rand_nbr":
            split_f = self._rand_permute_indices(splits, split_f)

        self.init_splits = len(splits)
        return splits, split_f

    def _rand_permute_indices(
        self, splits: List[IndexOp], split_f: TensorFunc
    ) -> TensorFunc:
        # randomly permute the indices of f
        perm = list(range(len(split_f.indices)))
        random.shuffle(perm)
        perm_indices = [split_f.indices[i] for i in perm]
        unperm = np.argsort(perm)
        splits.append(
            IndexPermute(perm=tuple(perm), unperm=tuple(unperm.tolist()))
        )
        return PermuteFunc(perm_indices, split_f, unperm)

    def _run_init_cross(
        self, data_tensor: CountableFunc, f: TensorFunc
    ) -> TreeNetwork:
        init_eps = self.config.engine.eps * self.config.cross.init_eps
        cross_start = time.time()
        if self.config.cross.init_struct == "tt":
            # construct a tensor train with the split indices
            net = TensorTrain.rand_tt(f.indices)
            cross(f, net, net.end_nodes()[0], eps=init_eps)
        elif self.config.cross.init_struct == "tt_cross":
            domains = [torch.arange(ind.size) for ind in f.indices]
            res = tntorch.cross(
                tntorch_wrapper(f),
                domains,
                eps=init_eps,
                kickrank=2,
                max_iter=100,
                verbose=False,
            )
            net = tntorch_to_tt(res, f.indices)
        elif self.config.cross.init_struct == "ht":
            net = HierarchicalTucker.rand_ht(f.indices, 1)
            cross(f, net, net.root(), eps=init_eps)
        else:
            raise ValueError("unknown initial structure for cross")

        cross_res_file = f"{self.config.output.output_dir}/init_net.pkl"
        with open(cross_res_file, "wb") as cross_writer:
            pickle.dump(net, cross_writer)

        self.stats.cross_time = time.time() - cross_start
        self.stats.init_cross_size = net.cost()
        self.stats.init_cross_evals = data_tensor.num_calls()
        # print("initial cost", net.cost())

        return net

    @profile
    def _search_at_level(
        self,
        st: HSearchState,
        remaining_delta: float,
        is_top: bool = False,
        exclusions: Optional[Sequence[Index]] = None,
    ) -> HSearchState:
        search_engine = PartitionSearch(self.config)
        # decrease the delta budget exponentially
        delta, remaining_delta = self.error_dist.split_delta(remaining_delta)

        before_split = st.network.free_indices()
        merge_ops, split_ops = [], []
        merged_indices = before_split
        while (
            trigger_merge(self.config, is_top)
            and len(merged_indices) > self.config.topdown.group_threshold
        ):
            mops, sops = self._to_lower_dim(st, is_top)
            merge_ops.extend(mops)
            split_ops.extend(sops)
            # before run the merges, let's swap the node to the correct places to avoid very expensive computations
            # net_indices = merged_indices[:]
            merged_indices = []
            for merge_op in mops:
                merged_indices.append(merge_op.result)

            # for ind in net_indices:
            #     found = False
            #     for merge_op in mops:
            #         if ind in merge_op.indices:
            #             found = True
            #             break

            #     if not found:
            #         merged_indices.append(ind)

        result = search_engine.search(
            st.network, merge_ops, delta=delta, exclusions=exclusions
        )

        if result.best_state is None:
            return st

        bn = result.best_state.network
        # print(result.stats.search_cross_evals)
        self.stats.merge(result.stats)

        tmp_bn = copy.deepcopy(bn)
        for split_op in reversed(split_ops):
            tmp_bn.split_index(split_op)

        after_split = tmp_bn.free_indices()

        next_nodes = list(bn.network.nodes)
        # distribute delta equally to all subnets
        remaining_delta = remaining_delta / math.sqrt(len(next_nodes))
        unused_delta = result.unused_delta**2 + st.unused_delta
        best_st = HSearchState(st.free_indices, st.reshape_history, bn, 0)

        if (
            sorted(before_split) == sorted(after_split)
            and len(next_nodes) == 1
        ):
            unused_delta += remaining_delta**2
        else:
            # enumerate nodes in the order of their scores
            for node in next_nodes:
                # print(node)
                logger.debug(
                    "optimizing node %s with indices %s",
                    node,
                    best_st.network.node_tensor(node).indices,
                )

                optimize_res = self._optimize_node(
                    best_st, node, remaining_delta
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

    def _to_lower_dim(self, st: HSearchState, is_top: bool = False):
        merge_start = time.time()
        if not trigger_merge(self.config, is_top):
            return [], []

        if self.config.topdown.merge_algo == "svd":
            index_sets = self._merge_by_corr(st)
        elif self.config.topdown.merge_algo == "rand":
            seed_all(self.config.engine.seed)
            index_sets = self._merge_by_rand(st)
        elif self.config.topdown.merge_algo in ("nbr", "rand_nbr"):
            index_sets = self._merge_by_rand(st, rand=False)
        else:
            raise ValueError("unknown merge algorithm")

        merge_ops, split_ops = [], []
        for m_indices in index_sets:
            if len(m_indices) < 2:
                continue

            m_shape = [ind.size for ind in m_indices]
            m_ind_size = int(np.prod(m_shape, dtype=int))
            m_ind = Index(
                "_".join(str(ind.name) for ind in m_indices), m_ind_size
            )

            merge_op = IndexMerge(indices=m_indices, result=m_ind)
            merge_ops.append(merge_op)

            split_op = IndexSplit(index=m_ind, shape=m_shape, result=m_indices)
            split_ops.append(split_op)

        self.stats.merge_time += time.time() - merge_start
        return merge_ops, split_ops

    def _merge_by_rand(
        self,
        st: HSearchState,
        rand: bool = True,
    ) -> Sequence[Sequence[Index]]:
        # randomly partition the indices into @threshold@ sets
        threshold = self.config.topdown.group_threshold
        indices = st.network.free_indices()
        if rand:
            random.shuffle(indices)
        sublen = len(indices) // threshold
        sublists = []
        while len(indices) > threshold and len(sublists) < threshold:
            used_len = len(sublists) * sublen
            if len(sublists) == threshold - 1:
                ind_set = indices[used_len:]
            else:
                ind_set = indices[used_len : used_len + sublen]

            sublists.append(ind_set)

        return sublists

    @profile
    def _merge_by_corr(
        self,
        st: HSearchState,
    ) -> Sequence[Sequence[Index]]:
        """Consider all possible combinations of indices.

        For each combination, we calculate the correlation matrix of
        the reshaped tensor. If the correlation is high enough,
        we merge the indices.
        """
        net = st.network
        indices = net.free_indices()
        threshold = self.config.topdown.group_threshold
        if len(indices) <= threshold:
            return [], []

        comb_corr = {}
        for comb in itertools.combinations(indices, 2):
            ac = OSplit(comb)
            if len(ac.indices) < 9:
                svd_algo = SVDAlgorithm.MERGE
            else:
                svd_algo = SVDAlgorithm.CROSS
            # svd_algo = SVDAlgorithm.MERGE
            svals = ac.svals(copy.deepcopy(net), max_rank=2, algo=svd_algo)
            # print(svals)
            if len(svals) >= 2:
                comb_corr[comb] = -svals[0] / svals[1]
            else:
                comb_corr[comb] = 0

        comb_corr = sorted(comb_corr.items(), key=lambda x: x[1])

        # start from the largest group and expand until the threshold
        group_size = len(indices) // threshold
        index_sets = []
        visited = set()
        for i in range(threshold):
            group = set()
            for xs, _ in comb_corr:
                if xs[0] in visited and xs[0] not in group:
                    continue
                
                if xs[1] in visited and xs[1] not in group:
                    continue

                group.update(xs)
                visited.update(xs)
            
                if len(group) >= group_size and i != threshold - 1:
                    break

            if group:
                index_sets.append(list(group))

        return index_sets

    def _optimize_node(
        self,
        st: HSearchState,
        node: NodeName,
        remaining_delta: float,
    ) -> List[Tuple[TreeNetwork, HSearchState]]:
        """Optimize the children nodes in a given network"""
        results = []
        for split_result, used_splits in self._split_indices(st, node):
            split_result.network.orthonormalize(node)

            new_sn = TreeNetwork()
            new_sn.add_node(
                node, copy.deepcopy(split_result.network.node_tensor(node))
            )
            new_st = HSearchState(
                split_result.free_indices,
                split_result.reshape_history,
                new_sn,
                split_result.unused_delta,
            )

            # new_st.network.orthonormalize(node)
            logger.debug("splitted network: %s", new_st.network)

            # exclude actions that split single internal indices
            exclusions = []
            for ind in new_sn.free_indices():
                if ind not in split_result.free_indices:
                    exclusions.append(ind)

            sn_st = self._search_at_level(
                new_st,
                remaining_delta,
                exclusions=exclusions,
            )
            results.append((split_result.network, sn_st))

        return results

    def _split_indices(
        self, st: HSearchState, node: NodeName, compute_data: bool = True
    ) -> Sequence[Tuple[HSearchState, Sequence[IndexSplit]]]:
        if self.config.engine.decomp_algo == "cross" and compute_data:
            return [(st, [])]

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

                new_st = new_st.split_index(split_op, compute_data)
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
            k = np.random.randint(0, len(factors))
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

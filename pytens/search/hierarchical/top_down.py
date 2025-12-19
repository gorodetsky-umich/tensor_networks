"""Top down reshaping search"""

import copy
import itertools
import functools
import logging
import math
import operator
import pickle
import random
import time
from abc import abstractmethod
from typing import List, Optional, Sequence, Tuple, Dict

import networkx as nx
import numpy as np
import sympy
from line_profiler import profile
import matplotlib.pyplot as plt

from pytens.algs import TensorTrain, TreeNetwork, tt_svd_round
from pytens.cross.funcs import (
    CountingFunc,
    FuncTensorNetwork,
    FuncNeutron,
    PermuteFunc,
    TensorFunc,
)
from pytens.cross.runner import CrossRunner
from pytens.search.configuration import SearchConfig, ReshapeOption
from pytens.search.hierarchical.error_dist import BaseErrorDist
from pytens.search.hierarchical.index_cluster import IndexCluster
from pytens.search.hierarchical.types import (
    HSearchState,
    IndexSplitResult,
    SubnetResult,
)
from pytens.search.hierarchical.utils import (
    DisjointSet,
    select_factors,
    split_func,
)
from pytens.search.partition import PartitionSearch
from pytens.search.state import SearchState
from pytens.search.utils import (
    SearchStats,
    init_state,
    seed_all,
    to_splits,
    index_partition,
    unravel_indices,
)
from pytens.types import (
    Index,
    IndexMerge,
    IndexOp,
    IndexPermute,
    IndexSplit,
    NodeName,
)

logger = logging.getLogger(__name__)


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
        self._cluster = IndexCluster(threshold=config.topdown.group_threshold)

    def set_cluster(self, cluster: IndexCluster):
        """Set the index clustering method."""
        self._cluster = cluster

    @profile
    def search(self) -> HSearchState:
        """Perform the topdown search starting from the given net"""
        net, splits = self._initialize()

        seed_all(self.config.engine.seed)
        delta = net.norm() * self.config.engine.eps
        # print("available delta:", delta)
        free_indices = net.free_indices()
        init_st = HSearchState(free_indices[:], [], net, 0)
        st = self._search(init_st, delta, splits, True)
        # used_delta = net.norm() ** 2 - st.network.norm() ** 2
        # print("unused delta:", st.unused_delta, "used delta:", used_delta, "actual remaining:", delta**2 - used_delta)

        # best_st = init_st
        best_st = None
        for n in st.network.network.nodes:
            network = copy.deepcopy(st.network)
            try:
                network.round(n, delta=math.sqrt(st.unused_delta))
            except np.linalg.LinAlgError:
                continue

            if best_st is None or network.cost() < best_st.network.cost():
                best_st = st
                best_st.network = network

        return best_st

    @abstractmethod
    def _initialize(self) -> Tuple[TreeNetwork, List[IndexOp]]:
        raise NotImplementedError

    def _trigger_merge(self, ind_cnt: int, is_top: bool) -> bool:
        """Determine whether to trigger the index merge operation before search"""
        return (
            (self.config.topdown.reshape_algo == ReshapeOption.CLUSTER)
            and (not is_top or self.config.topdown.merge_mode == "all")
            and ind_cnt > self.config.topdown.group_threshold
        )

    def _revert_splits(self, splits: Sequence[IndexOp], bn: TreeNetwork):
        # revert the splits
        # print(bn)
        for split_op in splits:
            assert isinstance(split_op, IndexSplit)
            assert split_op.result is not None
            # print(split_op)
            first_ind = split_op.result[0]
            if first_ind not in bn.free_indices():
                continue

            nodes = [bn.node_by_free_index(ind.name) for ind in split_op.result]
            unique_nodes = list(set(nodes))
            base_node = unique_nodes[0]
            unique_nodes.sort(key=functools.partial(bn.distance, base_node))
            for n in unique_nodes[1:]:
                bn.merge(base_node, n)

            bn.merge_index(IndexMerge(indices=split_op.result, result=split_op.index))

    def _apply_splits(self, merges: List[IndexMerge], splits: Sequence[IndexOp], net: TreeNetwork):
        for split_op in splits:
            assert isinstance(split_op, IndexSplit)
            assert split_op.result is not None
            net.split_index(split_op)
            # temporarily modify the merge operations
            inside_merge = False
            for merge_op in merges:
                if split_op.index in merge_op.indices:
                    # print(split_op)
                    pos = merge_op.indices.index(split_op.index)
                    merge_op.indices = list(itertools.chain(merge_op.indices[:pos], split_op.result, merge_op.indices[pos+1:]))
                    # print(merge_op.indices)
                    inside_merge = True

            if not inside_merge:
                merges.append(IndexMerge(indices=split_op.result, result=split_op.index))

    @profile
    def _search(
        self,
        st: HSearchState,
        remaining_delta: float,
        splits: Sequence[IndexOp],
        is_top: bool = False,
        exclusions: Optional[Sequence[Index]] = None,
    ) -> HSearchState:
        # print("search structure for initially")
        # print(st.network)

        # TODO: handle the case when the network is a single node
        if not isinstance(st.network, TensorTrain):
            # tmp_net = TreeNetwork()
            # tmp_net.add_node("G0", st.network.contract())
            # st.network = tmp_net
            print("before process")
            print(st.network)
            # turn it into a tensor train
            st.network = self._preprocess(st.network)
            # print("after transforming to tt")
            # print(st.network)

        # decrease the delta budget exponentially
        delta, remaining_delta = self.error_dist.split_delta(remaining_delta)

        before_split = st.network.free_indices()
        merge_ops, split_ops = [], []
        if self._trigger_merge(len(before_split), is_top):
            merge_ops, split_ops = self._to_lower_dim(st)

        # print(merge_ops)
        # apply the splits
        new_merge_ops = merge_ops[:]
        self._apply_splits(new_merge_ops, splits, st.network)
        search_engine = PartitionSearch(self.config, st.network)
        # print(new_merge_ops)
        print("searching for")
        print(st.network)
        result = search_engine.search(new_merge_ops, splits, delta, exclusions)
        bn = result.best_state.network
        # print(new_merge_ops)
        # print(bn)

        if result.best_state is None:
            self._revert_splits(splits, st.network)
            return st

        self.stats.merge(result.stats)

        next_nets = self._get_next_nets(result.best_state)
        self._revert_splits(splits, bn)
        # print(bn)
        tmp_bn = copy.deepcopy(bn)
        for split_op in reversed(split_ops):
            tmp_bn.split_index(split_op)
        after_split = tmp_bn.free_indices()
        best_st = HSearchState(st.free_indices, st.reshape_history, bn, 0)

        # print("result unused", result.unused_delta ** 2)

        unused_delta = st.unused_delta + result.unused_delta**2
        if sorted(before_split) == sorted(after_split) and len(next_nets) == 1:
            best_st.unused_delta = unused_delta + remaining_delta**2
            # self.memoization[tuple(st.network.free_indices())] = best_st
            return best_st

        # future work: it would be better to dynamically distribute errors
        # distribute delta equally to all subnets
        remaining_delta = remaining_delta / math.sqrt(len(next_nets))
        # enumerate nodes in the order of their scores
        for subnet in next_nets:
            # print("==before revert==")
            # print(subnet)
            self._revert_splits(splits, subnet)
            # print("==after revert==")
            # print(subnet)
            # print("="*20)
            self._search_for_subnet(best_st, remaining_delta, subnet)

        # after we replaced all subnets, we compress the additional edges
        # print(best_st.network.norm() ** 2, sn_unused_delta)
        best_st.unused_delta = unused_delta
        # self.memoization[tuple(st.network.free_indices())] = best_st
        return best_st

    @profile
    def _preprocess(self, net: TreeNetwork) -> TreeNetwork:
        return net

    def _get_next_nets(self, best_st: SearchState) -> List[TreeNetwork]:
        """Get the next level nodes to optimize"""
        subgraph_nodes = DisjointSet()
        for node in best_st.network.network.nodes:
            subgraph_nodes.union(node, node)

        for ac in to_splits(best_st.network):
            if ac not in best_st.past_actions and ac.reverse_edge is not None:
                subgraph_nodes.union(*ac.reverse_edge)

        subnets = []
        for group in sorted(subgraph_nodes.groups().values()):
            # print("group:", group)
            tn = TreeNetwork()
            tn.network = nx.subgraph(best_st.network.network, group).copy()
            subnets.append(tn)

        return subnets

    def _search_for_subnet(
        self,
        best_st: HSearchState,
        remaining_delta: float,
        subnet: TreeNetwork,
    ) -> float:
        logger.debug(
            "optimizing node %s with indices %s", subnet, subnet.free_indices()
        )

        optimize_res = self._optimize_node(
            best_st, copy.deepcopy(subnet), remaining_delta
        )
        if not optimize_res:
            return remaining_delta**2

        best_res = optimize_res[0]
        for res in optimize_res[1:]:
            res_cost = res.subnet_state.network.cost()
            best_cost = best_res.subnet_state.network.cost()
            if best_res is None or res_cost < best_cost:
                best_res = res

        best_st.network = best_res.network
        best_sn_st = best_res.subnet_state

        # if nothing happened in the subnet, we contract the entire subnet
        contraction_size = np.prod(
            [ind.size for ind in best_sn_st.network.free_indices()]
        )
        if contraction_size < best_sn_st.network.cost():
            node = best_sn_st.network.contract()
            tmp_subnet = TreeNetwork()
            tmp_subnet.add_node("G0", node)
            best_sn_st.network = tmp_subnet

        best_st.network.replace_with(
            best_res.subnet, best_sn_st.network, best_sn_st.reshape_history
        )
        # print(best_res.subnet)
        # print(best_sn_st.network)
        # print("-"*20)
        best_st.free_indices = best_sn_st.free_indices
        best_st.reshape_history = best_sn_st.reshape_history
        return best_sn_st.unused_delta

    def _to_lower_dim(self, st: HSearchState):
        merge_start = time.time()

        index_sets = self._cluster.cluster(st.network)

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

    @abstractmethod
    def _optimize_node(
        self,
        st: HSearchState,
        subnet: TreeNetwork,
        remaining_delta: float,
    ) -> List[SubnetResult]:
        """Optimize the children nodes in a given network"""
        raise NotImplementedError

    def _split_indices(
        self, st: HSearchState, indices: List[Index], compute_data: bool = True
    ) -> Sequence[IndexSplitResult]:
        if not self.config.topdown.reshape_enabled and compute_data:
            return [IndexSplitResult(st, [])]

        index_splits = self._split_indices_on_budget(st, indices, compute_data)

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
                # tmp_indices = new_st.network.node_tensor(node).indices

                # ndims = len(tmp_indices) + len(split_op.shape) - 1
                # if (
                #     self.config.topdown.reshape_algo == ReshapeOption.ENUMERATE
                #     and ndims > self.config.topdown.group_threshold
                # ):
                #     continue

                new_st = new_st.split_index(split_op, compute_data)
                used_splits.append(split_op)
                refactored = True

            if refactored:
                result_sts.append(IndexSplitResult(new_st, used_splits))

        return result_sts

    def _split_indices_on_budget(
        self, st: HSearchState, indices: Sequence[Index], compute_data: bool = True
    ) -> List[List[IndexSplit]]:
        # distribute the allowed splits between indices

        maxs = []
        for ind in indices:
            if ind in st.free_indices:
                factors = sympy.factorint(ind.size)
                maxs.append(sum(factors.values()) - 1)
            else:
                maxs.append(0)

        all_splits = []
        if self.config.topdown.reshape_algo in (
            ReshapeOption.RANDOM,
            ReshapeOption.ENUMERATE,
        ):
            budget = self.config.topdown.group_threshold - len(indices)
            budget = min(sum(maxs), budget)  # exhaust the budget as possible
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

        if self.config.topdown.reshape_algo == ReshapeOption.CLUSTER:
            splits = []
            for i, ind in enumerate(indices):
                ind_splits = self._get_split_op(st, ind, maxs[i], compute_data)
                if len(ind_splits) != 0:
                    splits.append(ind_splits)

            all_splits.extend(list(itertools.product(*splits)))

        return all_splits

    def _split_scores(
        self, st: HSearchState, index: Index
    ) -> Dict[int, float]:
        # get the svals decay for each split points of the index size
        node = st.network.node_by_free_index(index.name)
        st.network.orthonormalize(node)

        split_scores = {}
        for n in sympy.divisors(index.size):
            if n in (1, index.size):
                continue

            tmp_net = TreeNetwork()
            tmp_net.network = copy.deepcopy(st.network.network)
            # get indices on one side of the node
            nbrs = list(tmp_net.network.neighbors(node))

            if not nbrs:
                linds, rinds = [], tmp_net.free_indices()
            else:
                linds, rinds = index_partition(tmp_net, node, nbrs[0])

            lres = Index(str(index.name) + "_0", n)
            rres = Index(str(index.name) + "_1", index.size // n)
            tmp_net.split_index(
                IndexSplit(
                    index=index,
                    shape=(n, index.size // n),
                    result=[lres, rres],
                )
            )

            target_inds = linds
            if index in linds:
                target_inds = rinds

            target_inds.append(lres)

            max_rank = 2
            s = tmp_net.random_svals(node, target_inds, max_rank=max_rank)
            split_scores[n] = s[0] / s[min(len(s), max_rank) - 1]
            # print(target_inds, n, split_scores[n])

        return split_scores

    def _get_split_op(
        self, st: HSearchState, index: Index, budget: int, compute_data: bool = True
    ) -> Sequence[IndexSplit]:
        if index not in st.free_indices or budget <= 0:
            return []

        res = sympy.factorint(index.size)
        factors = [i for i, n in res.items() for _ in range(n)]
        if len(factors) == 1:
            return []

        if self.config.topdown.reshape_algo == ReshapeOption.RANDOM:
            k = np.random.randint(0, len(factors))
            selected = random.sample(factors, k=k)
            shape = _create_split_target(factors, selected)
            return [IndexSplit(index=index, shape=shape)]

        if compute_data:
            # filter out the top few
            scores = self._split_scores(st, index)
            # we always try our best to decompose the indices to
            # the maximum number and they subsume higher level reshapes
            shape_with_scores = set()
            for shape in select_factors(res, budget):
                # Calculate cumulative product sizes and sum their scores
                cumulative_sizes = itertools.accumulate(shape[1:-1], operator.mul, initial=shape[0])
                # transform the shape such that the low scores are excluded
                cumulative_score = [scores[size] for size in cumulative_sizes]
                pos = 0
                while pos < len(shape) - 1:
                    if cumulative_score[pos] < 2:
                        shape = tuple(shape[:pos]) + (shape[pos] * shape[pos+1],) + tuple(shape[pos+2:])
                        cumulative_score.pop(pos)
                    else:
                        pos += 1

                    # print(shape)

                assert len(cumulative_score) + 1 == len(shape)
                score = sum(cumulative_score)
                shape_with_scores.add((score, shape))

            shape_with_scores = list(shape_with_scores)
            shape_with_scores.sort(reverse=True)
        else:
            shape_with_scores = [(1, shape) for shape in select_factors(res, budget)]

        split_ops = []
        for _, shape in shape_with_scores[:10]:
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


class WhiteBoxTopDownSearch(TopDownSearch):
    """Top down structural search for white box data tensors."""

    def __init__(self, config: SearchConfig, data_tensor: TreeNetwork):
        super().__init__(config)
        self.data_tensor = data_tensor

    def _initialize(self) -> Tuple[TreeNetwork, List[IndexOp]]:
        return self.data_tensor, []

    def _optimize_node(
        self,
        st: HSearchState,
        subnet: TreeNetwork,
        remaining_delta: float,
    ) -> List[SubnetResult]:
        """Optimize the children nodes in a given network"""
        results = []
        subnet_nodes = list(subnet.network.nodes)
        assert len(subnet_nodes) == 1, (
            "only single node should be optimized in white box tensors"
        )
        node = subnet_nodes[0]
        node_indices = st.network.node_tensor(node).indices
        for split_result in self._split_indices(st, node_indices):
            net = split_result.state.network
            net.orthonormalize(node)

            new_sn = TreeNetwork()
            new_sn.add_node(node, copy.deepcopy(net.node_tensor(node)))
            new_st = HSearchState(
                split_result.state.free_indices,
                split_result.state.reshape_history,
                new_sn,
                split_result.state.unused_delta,
            )

            # new_st.network.orthonormalize(node)
            logger.debug("splitted network: %s", new_st.network)

            # exclude actions that split single internal indices
            exclusions = []
            for ind in new_sn.free_indices():
                if ind not in split_result.state.free_indices:
                    exclusions.append(ind)

            sn_st = self._search(
                new_st, remaining_delta, [], exclusions=exclusions
            )
            results.append(SubnetResult(net, new_sn, sn_st))

        return results


class BlackBoxTopDownSearch(TopDownSearch):
    """Top down structural search for black box tensors."""

    def __init__(
        self,
        config: SearchConfig,
        data_tensor: CountingFunc,
        validation_set: np.ndarray,
    ):
        super().__init__(config)
        self.data_tensor = data_tensor
        self._cross_runner = CrossRunner()
        self._validation_set = validation_set

    def set_cross_runner(self, runner: CrossRunner):
        """Modify the cross approximation algorithm."""
        self._cross_runner = runner

    def _initialize(self) -> Tuple[TreeNetwork, List[IndexOp]]:
        # we run index split to split as many as possible
        splits, f = self._gen_split_func(self.data_tensor)
        # run the cross approximation to avoid expensive error querying
        net = self._run_init_cross(self.data_tensor, f, splits)
        tree = TreeNetwork()
        tree.network = net.network
        # revert the index splits by merging the neighbor nodes
        for split_op in splits:
            assert isinstance(split_op, IndexSplit)
            assert split_op.result is not None
            base_index = split_op.result[0]
            base_node = tree.node_by_free_index(base_index.name)
            for ind in split_op.result[1:]:
                tree.merge(base_node, tree.node_by_free_index(ind.name))

            tree.merge_index(IndexMerge(indices=split_op.result, result=split_op.index))

        # TODO: investigate this part
        return net, splits

    def _gen_split_func(
        self, data_tensor: CountingFunc
    ) -> Tuple[List[IndexOp], TensorFunc]:
        if self.config.cross.init_reshape:
            free_indices = data_tensor.indices
            init_st = init_state(data_tensor, self.config.engine.eps)

            splits = []

            # create index splits in the chosen order
            st = HSearchState(free_indices[:], [], init_st.network, 0)
            for n in st.network.network.nodes:
                n_indices = st.network.node_tensor(n).indices
                split_results = self._split_indices(st, n_indices, compute_data=False)
                if len(split_results) == 0:
                    continue

                st = split_results[0].state
                splits.extend(split_results[0].splits)

            split_indices = list(st.network.free_indices())
            split_f = split_func(data_tensor, split_indices, splits)

            if self.config.topdown.cluster_method == "rand_nbr":
                split_f = self._rand_permute_indices(splits, split_f)

            self.init_splits = len(splits)
            new_indices, self._validation_set = unravel_indices(splits, free_indices, self._validation_set)
            perm = [new_indices.index(ind) for ind in split_f.indices]
            self._validation_set = self._validation_set[:, perm]
            return splits, split_f

        return [], data_tensor

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
        self,
        data_tensor: CountingFunc,
        f: TensorFunc,
        splits: Sequence[IndexOp],
    ) -> TreeNetwork:
        cross_res_file = f"{self.config.output.output_dir}/init_net.pkl"
        cross_start = time.time()
        init_eps = self.config.engine.eps * self.config.cross.init_eps

        if isinstance(data_tensor, FuncTensorNetwork) and isinstance(
            data_tensor.net, TensorTrain
        ):
            self.stats.init_cross_size = data_tensor.net.cost()
            # # directly split the nodes into the target by svd
            # f_inds = f.indices
            # net = copy.deepcopy(data_tensor.net)

            # prev_node = None
            # for split_op in splits:
            #     assert isinstance(split_op, IndexSplit)
            #     curr_node = net.split_index(split_op)
            #     assert curr_node is not None
            #     assert split_op.result is not None
            #     # split them into tensor train formats
            #     for ind in split_op.result[:-1]:
            #         curr_inds = net.node_tensor(curr_node).indices
            #         lefts = [curr_inds.index(ind)]
            #         for jj, curr_ind in enumerate(curr_inds):
            #             if (
            #                 prev_node is not None
            #                 and curr_ind
            #                 in net.get_contraction_index(prev_node, curr_node)
            #             ):
            #                 lefts.append(jj)

            #         prev_node, curr_node = net.qr(curr_node, lefts)
            #         tensor = net.node_tensor(prev_node)
            #         if len(tensor.indices) == 3:
            #             net.set_node_tensor(
            #                 prev_node, tensor.permute([1, 0, 2])
            #             )

            #     prev_node = curr_node

            # net = copy.deepcopy(data_tensor.net)
            # print(net)
            # node_map = {n: i for i, n in enumerate(sorted(net.network.nodes))}
            # net.network = nx.relabel_nodes(net.network, node_map)
            # rounded_net = tt_svd_round(net, self.config.engine.eps)
            # print("tt-round before reshape", data_tensor.net.cost() / rounded_net.cost())
            # print(rounded_net)
            # # print(net)
            # print(
            #     "TT before round compression:",
            #     np.prod([ind.size for ind in net.free_indices()])
            #     / rounded_net.cost(),
            # )
            return data_tensor.net
        else:
            # import os
            # if os.path.exists(cross_res_file):
            #     with open(cross_res_file, "rb") as cross_reader:
            #         net = pickle.load(cross_reader)

            #     tmp_net = copy.deepcopy(net)
            #     print(tmp_net)
            #     node_map = {n: i for i, n in enumerate(tmp_net.network.nodes)}
            #     tmp_net.network = nx.relabel_nodes(tmp_net.network, node_map)
            #     for i in tmp_net.network.nodes:
            #         if 0 < i < len(tmp_net.network.nodes) - 1:
            #             tensor = tmp_net.node_tensor(i)
            #             print(tensor.indices)
            #             tmp_net.set_node_tensor(i, tensor.permute([2,0,1]))

            #         if i == len(tmp_net.network.nodes) - 1:
            #             tensor = tmp_net.node_tensor(i)
            #             print(tensor.indices)
            #             tmp_net.set_node_tensor(i, tensor.permute([1,0]))

            #     rounded_net = tt_svd_round(tmp_net, self.config.engine.eps)
            #     print("tt-round before reshape", tmp_net.cost() / rounded_net.cost())
            # else:
            net = self._cross_runner.run(
                f, init_eps, kickrank=self.config.cross.init_kickrank, validation=self._validation_set
            )

        with open(cross_res_file, "wb") as cross_writer:
            pickle.dump(net, cross_writer)

        net.draw()
        plt.savefig(f"{self.config.output.output_dir}/init_net.png", dpi=100)
        plt.close()

        self.stats.cross_time = time.time() - cross_start
        self.stats.init_cross_size = net.cost()
        self.stats.init_cross_evals = data_tensor.num_calls()

        # print(net)
        if isinstance(data_tensor, FuncNeutron):
            self._store_neutron_func(data_tensor)

        return net

    def _store_neutron_func(self, data_tensor: FuncNeutron):
        cache = f"output/neutron_diffusion_{data_tensor.d}.pkl"
        with open(cache, "wb") as cache_file:
            pickle.dump(data_tensor.cache, cache_file)

    def _trigger_merge(self, ind_cnt: int, is_top: bool) -> bool:
        """Determine whether to trigger the index merge operation before search"""
        return (
            (self.config.topdown.reshape_algo == ReshapeOption.CLUSTER)
            and (not is_top or self.config.topdown.merge_mode == "all")
            and ind_cnt > self.config.topdown.group_threshold
        )

    def _optimize_node(
        self,
        st: HSearchState,
        subnet: TreeNetwork,
        remaining_delta: float,
    ) -> List[SubnetResult]:
        """Optimize the children nodes in a given network"""
        results = []
        subnet_nodes = list(subnet.network.nodes)

        node = subnet_nodes[0]
        indices = subnet.free_indices()
        for split_result in self._split_indices(st, indices):
            net = split_result.state.network
            net.orthonormalize(node)

            new_sn = TreeNetwork()
            new_sn.network = nx.subgraph(
                net.network, subnet.network.nodes
            ).copy()
            new_st = HSearchState(
                split_result.state.free_indices,
                split_result.state.reshape_history,
                new_sn,
            )

            # new_st.network.orthonormalize(node)
            logger.debug("splitted network: %s", new_st.network)

            # exclude actions that split single internal indices
            exclusions = []
            for ind in new_sn.free_indices():
                if ind not in split_result.state.free_indices:
                    exclusions.append(ind)

            sn_st = self._search(
                new_st, remaining_delta, [], exclusions=exclusions
            )
            results.append(SubnetResult(net, new_sn, sn_st))

        return results

    @profile
    def _preprocess(self, net: TreeNetwork) -> TreeNetwork:
        # print("preprocess subnet")
        # print(net)

        free_inds = []
        for ind in net.free_indices():
            free_inds.append(ind.with_new_rng(range(ind.size)))

        if len(net.network.nodes) == 1:
            # return TensorTrain.tt_svd(
            #     net.contract().value, free_inds, eps=self.config.engine.eps * 0
            # )
            return net

        func = FuncTensorNetwork(free_inds, net)
        max_rank = max(ind.size for ind in net.inner_indices())
        kickrank = max(2, max_rank // 5)
        return self._cross_runner.run(
            func, eps=self.config.engine.eps * 0.01, kickrank=kickrank
        )

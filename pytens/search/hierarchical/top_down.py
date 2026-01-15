"""Top down reshaping search"""

from collections import defaultdict
import copy
import os
import shutil
import functools
import itertools
import logging
import math
import operator
import pickle
import random
import time
from abc import abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy
from line_profiler import profile

from pytens.algs import Tensor, TensorTrain, TreeNetwork
from pytens.cross.funcs import (
    CountingFunc,
    FuncNeutron,
    FuncTensorNetwork,
    PermuteFunc,
    TensorFunc,
)
from pytens.cross.runner import CrossRunner, TTCrossRunner
from pytens.search.algs.partition import PartitionSearch
from pytens.search.configuration import (
    ClusterMethod,
    ReorderAlgo,
    ReshapeOption,
    SearchAlgo,
    SearchConfig,
)
from pytens.search.hierarchical.error_dist import BaseErrorDist
from pytens.search.hierarchical.index_cluster import (
    CrossIndexCluster,
    RandomIndexCluster,
    SVDIndexCluster,
    SVDNbrIndexCluster,
    eff_rank,
)
from pytens.search.hierarchical.types import (
    HSearchState,
    IndexSplitResult,
    ReplaySweep,
    ReplayTrace,
    SubnetResult,
)
from pytens.search.hierarchical.utils import DisjointSet, split_func
from pytens.search.utils import (
    SearchResult,
    SearchStats,
    index_partition,
    init_state,
    seed_all,
    to_splits,
    unravel_indices,
)
from pytens.types import (
    DimTreeNode,
    Index,
    IndexMerge,
    IndexName,
    IndexOp,
    IndexPermute,
    IndexSplit,
    NodeName,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _permute_unique(nums: List[int]) -> Sequence[Tuple[int, ...]]:
    nums.sort()
    used = [False] * len(nums)

    def backtrack(pat: List[int]) -> List[Tuple[int, ...]]:
        if len(pat) == len(nums):
            return [tuple(pat[:])]

        results = []
        for i, num in enumerate(nums):
            if used[i]:
                continue
            if i > 0 and num == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            pat.append(num)
            results.extend(backtrack(pat))
            used[i] = False
            pat.pop()

        return results

    return backtrack([])


def _split_into_chunks(
    lst: Sequence[int], n: int
) -> Sequence[List[Sequence[int]]]:
    if n == 1:
        # When n is 1, the only chunk is the entire list
        return [[lst]]

    results = []
    for i in range(1, len(lst) - n + 2):  # Ensure at least `n` chunks
        for tail in _split_into_chunks(lst[i:], n - 1):
            results.append([lst[:i]] + tail)

    return results


def _select_factors(
    factors: Dict[int, int], budget: int
) -> List[Sequence[int]]:
    """Select a suitable number of factors for reshaping"""
    # enumerate all possible choices for each factor
    factors_flat = [x for x, c in factors.items() for _ in range(c)]
    # partition the list into splits_allowed groups
    seen = set()
    results = []
    for factors_perm in _permute_unique(factors_flat):
        for chunks in _split_into_chunks(factors_perm, min(budget + 1, len(factors_perm))):
            chunk_factors = tuple([math.prod(chunk) for chunk in chunks])
            if chunk_factors not in seen:
                seen.add(chunk_factors)
                results.append(chunk_factors)

    return results


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


def _merge_ops_from_index_groups(index_sets):
    merge_ops, split_ops = [], []
    for m_indices in index_sets:
        if len(m_indices) < 2:
            continue

        m_shape = [ind.size for ind in m_indices]
        m_ind_size = int(np.prod(m_shape, dtype=int))
        m_ind = Index("_".join(str(ind.name) for ind in m_indices), m_ind_size)

        merge_op = IndexMerge(indices=m_indices, result=m_ind)
        merge_ops.append(merge_op)

        split_op = IndexSplit(index=m_ind, shape=m_shape, result=m_indices)
        split_ops.append(split_op)

    return merge_ops, split_ops


def _index_to_end_distance(net: TreeNetwork, end: NodeName, index: Index):
    """Compute the distance between the end node and the given index"""
    index_node = net.node_by_free_index(index.name)
    return net.distance(end, index_node)


def _merge_op_dist(net: TreeNetwork, end: NodeName, merge_op: IndexMerge):
    """sort the merge ops according to their positions."""

    dist = 0
    for ind in merge_op.indices:
        dist += _index_to_end_distance(net, end, ind)

    return dist / len(merge_op.indices)


def _rename_data_tensor(st: HSearchState, data_tensor: TreeNetwork):
    # we pick the node with the smallest free index as the root
    def split_name(ind: Index):
        segments = str(ind.name).split("_")
        if len(segments) < 2:
            return (ind not in st.free_indices, ind.size, segments[0])

        return (
            ind not in st.free_indices,
            ind.size,
            segments[0],
            int(segments[1]),
        )

    sorted_inds = sorted(data_tensor.free_indices(), key=split_name)
    root_ind = list(sorted_inds)[-1]
    root_node = data_tensor.node_by_free_index(root_ind.name)
    # we label each node by the indices
    tree = data_tensor.dimension_tree(root_node)
    # create the edge remapping by the order of tree traversal
    ind_map, reverse_map = {}, {}

    def tree_traverse(node: DimTreeNode, ind_cnt: int) -> int:
        tensor = data_tensor.node_tensor(node.node)
        node_indices = tensor.indices
        new_indices = []
        perm = []
        for ind in sorted(node_indices, key=split_name):
            if ind in st.free_indices:
                ind_map[ind.name] = ind.name
                reverse_map[ind.name] = ind.name
            elif ind.name not in ind_map:
                ind_map[ind.name] = f"s_{ind_cnt}"
                reverse_map[f"s_{ind_cnt}"] = ind.name
                ind_cnt += 1

            perm.append(node_indices.index(ind))
            new_indices.append(ind.with_new_name(ind_map[ind.name]))

        ordered_tensor = tensor.permute(perm)
        data_tensor.set_node_tensor(
            node.node, Tensor(ordered_tensor.value, new_indices)
        )

        for c in node.down_info.nodes:
            ind_cnt = tree_traverse(c, ind_cnt)

        return ind_cnt

    ind_cnt = tree_traverse(tree, 0)
    assert ind_cnt <= len(data_tensor.all_indices()), (
        f"get mapping for {list(ind_map.values())} but all indices are {data_tensor.all_indices()}"
    )

    return ind_map, reverse_map


def _apply_renaming(
    st: HSearchState, reverse_map: Dict[IndexName, IndexName]
):
    # revert the renaming
    data_tensor = st.network
    for n in data_tensor.network.nodes:
        tensor = data_tensor.node_tensor(n)
        new_indices = []
        for ind in tensor.indices:
            if ind.name in reverse_map:
                new_indices.append(ind.with_new_name(reverse_map[ind.name]))
            else:
                new_indices.append(ind)

        data_tensor.set_node_tensor(n, Tensor(tensor.value, new_indices))


def _split_scores(st: HSearchState, index: Index) -> Dict[int, float]:
    logger.debug(
        "computing split scores for %s in %s with free indices %s",
        index,
        st.network,
        st.free_indices,
    )
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

        max_rank = 10
        s = tmp_net.random_svals(node, target_inds, max_rank=max_rank)
        split_scores[n] = eff_rank(s)  # s[0] / s[min(len(s), 1)]
        logger.debug(
            "target indices %s with size %s has score %s",
            target_inds,
            n,
            split_scores[n],
        )

    return split_scores


class TopDownSearch:
    """Search for reshaped structures from top to bottom"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.error_dist = BaseErrorDist()
        self.stats = SearchStats()
        self.init_splits = 0
        self._cluster = self.set_cluster()

    def set_cluster(self):
        """Set the index clustering method."""
        threshold = self.config.topdown.group_threshold
        cluster_method = {
            ClusterMethod.SVD: SVDIndexCluster(threshold),
            ClusterMethod.RAND: RandomIndexCluster(threshold),
            ClusterMethod.NBR: RandomIndexCluster(threshold, False),
            ClusterMethod.RAND_NBR: RandomIndexCluster(threshold, False),
            ClusterMethod.SVD_NBR: SVDNbrIndexCluster(threshold),
            ClusterMethod.CROSS: CrossIndexCluster(
                threshold, self.config.engine.eps * 0.1
            ),
        }.get(self.config.topdown.cluster_method)

        assert cluster_method is not None, (
            f"unknown cluster method: {self.config.topdown.cluster_method}"
        )
        return cluster_method

    @profile
    def search(
        self,
        delta: Optional[float] = None,
        free_indices: Optional[List[Index]] = None,
        replay_traces: Optional[List[ReplayTrace]] = None,
    ) -> HSearchState:
        """Perform the topdown search starting from the given net"""
        net, splits = self._initialize()

        seed_all(self.config.engine.seed)
        if delta is None:
            delta = net.norm() * self.config.engine.eps

        # print("available delta:", delta)
        if free_indices is None:
            free_indices = net.free_indices()
        init_st = HSearchState(free_indices[:], [], net, 0)

        if self.config.synthesizer.replay_from is not None:
            assert replay_traces is not None
            init_st.replay_traces = replay_traces
        else:
            init_st.replay_traces = [ReplayTrace(0, [], [], [], [])]

        st = self._search(init_st, delta, splits, True)

        # best_st = init_st
        best_net = None
        delta = st.unused_delta
        for n in st.network.network.nodes:
            network = copy.deepcopy(st.network)
            try:
                logger.debug("unused delta: %s", delta)
                network.round(n, delta=math.sqrt(delta))
            except np.linalg.LinAlgError:
                continue

            if best_net is None or network.cost() < best_net.cost():
                best_net = network

        st.network = best_net.compress()
        # print(st.replay_traces)
        return st

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
        tmp_net = TreeNetwork()
        tmp_net.network = bn.network
        for split_op in splits:
            if not isinstance(split_op, IndexSplit):
                continue

            assert split_op.result is not None
            # print(split_op)
            first_ind = split_op.result[0]
            if first_ind not in tmp_net.free_indices():
                continue

            nodes = [
                tmp_net.node_by_free_index(ind.name) for ind in split_op.result
            ]
            unique_nodes = list(set(nodes))
            base_node = unique_nodes[0]
            unique_nodes.sort(
                key=functools.partial(tmp_net.distance, base_node)
            )
            for n in unique_nodes[1:]:
                tmp_net.merge(base_node, n)

            # print(bn)
            # print(split_op)
            tmp_net.merge_index(
                IndexMerge(indices=split_op.result, result=split_op.index)
            )

    def _apply_splits(
        self,
        merges: List[IndexMerge],
        splits: Sequence[IndexOp],
        net: TreeNetwork,
    ):
        for split_op in splits:
            if not isinstance(split_op, IndexSplit):
                continue

            assert split_op.result is not None
            net.split_index(split_op)
            # temporarily modify the merge operations
            inside_merge = False
            for merge_op in merges:
                if split_op.index in merge_op.indices:
                    # print(split_op)
                    pos = merge_op.indices.index(split_op.index)
                    merge_op.indices = list(
                        itertools.chain(
                            merge_op.indices[:pos],
                            split_op.result,
                            merge_op.indices[pos + 1 :],
                        )
                    )
                    # print(merge_op.indices)
                    inside_merge = True

            if not inside_merge:
                merges.append(
                    IndexMerge(indices=split_op.result, result=split_op.index)
                )

    def _cross_to_tt(
        self,
        data_tensor: TreeNetwork,
        merge_ops: List[IndexMerge],
        ind_splits: Sequence[IndexOp],
    ) -> Tuple[bool, TreeNetwork]:
        """Handle index merging during preprocessing for cross results."""
        transform_start = time.time()

        if not isinstance(data_tensor, TensorTrain):
            return True, data_tensor

        # after we know how indices are merged, we create a permuted function
        new_indices = []

        end = data_tensor.end_nodes()[0]
        merge_op_key = partial(_merge_op_dist, data_tensor, end)
        index_key = partial(_index_to_end_distance, data_tensor, end)
        # print(merge_ops, data_tensor)
        for mop in sorted(merge_ops, key=merge_op_key):
            # sort the indices before adding to the list according to positions
            new_indices.extend(sorted(mop.indices, key=index_key))

        # there exists some free indices are not merged
        for ind in data_tensor.free_indices():
            if ind not in new_indices:
                new_indices.append(ind)

        assert len(new_indices) == len(data_tensor.free_indices()), (
            f"get {new_indices} with merges {merge_ops}, but expect {data_tensor.free_indices()}"
        )
        # print("reorder the indices into", new_indices)

        reorder_start = time.time()
        if self.config.preprocess.reorder_algo == ReorderAlgo.SVD:
            ok = True
            tt = data_tensor.reorder_by_svd(
                new_indices,
                self.config.engine.eps
                * self.config.preprocess.reorder_eps
                * 0.01,
            )
        else:
            ok, tt = data_tensor.reorder_by_cross(
                new_indices,
                self.config.engine.eps * self.config.preprocess.reorder_eps,
                kickrank=5,
            )

        logger.debug("reorder time: %s", time.time() - reorder_start)
        logger.debug("after reordering")
        logger.debug(tt)
        logger.debug(merge_ops)

        self.stats.merge_transform_time = time.time() - transform_start
        return ok, tt

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
        logger.debug("original cost %s", st.network.cost())

        # TODO: handle the case when the network is a single node
        if not isinstance(st.network, TensorTrain):
            # tmp_net = TreeNetwork()
            # tmp_net.add_node("G0", st.network.contract())
            # st.network = tmp_net
            logger.debug("before process")
            logger.debug(st.network)
            # turn it into a tensor train
            st.network = self._preprocess(st.network)
            # print("after transforming to tt")
            # print(st.network)

        # decrease the delta budget exponentially
        delta, remaining_delta = self.error_dist.split_delta(remaining_delta)

        ind_map, reverse_map = _rename_data_tensor(st, st.network)
        logger.debug("reverse name mapping is %s", reverse_map)
        before_split = st.network.free_indices()
        # merge_ops, split_ops = [], []

        # apply the splits
        # new_merge_ops = merge_ops[:]
        logger.debug("current free indices: %s", st.free_indices)
        logger.debug("searching better structures for %s", st.network)
        logger.debug(
            "network norm: %s, allowed compression: %s",
            st.network.norm() ** 2,
            delta**2,
        )
        if (
            not self.config.cross.init_cross
            or len(st.network.network.nodes) > 7
        ):
            logger.debug(
                "running sweep with iterations %s", self.config.sweep.max_iters
            )
            config = copy.deepcopy(self.config)
            config.sweep.subnet_size = math.ceil(config.sweep.subnet_size / 2)
            config.sweep.max_iters += 3
            logger.debug("current free indices: %s", st.free_indices)
            sweep = RandomStructureSweep(
                config, copy.deepcopy(st.network), st.free_indices
            )
            if self.config.synthesizer.replay_from is not None:
                sweep.traces = st.replay_traces

            result = sweep.sweep(delta)
            bn = sweep.data_tensor
            self.stats.merge(result.stats)

            best_st = HSearchState(
                list(sweep.free_indices),
                sweep.reshape_history,
                bn,
                result.unused_delta,
            )
            best_st.replay_traces = sweep.traces
            _apply_renaming(best_st, reverse_map)
            # best_st.network.draw()
            # plt.show()
            return best_st

        # best_merge = st
        # best_result = SearchResult()
        logger.debug("after sweeping, get the net %s", st.network)
        tmp_st, merge_ops, split_ops = self._to_lower_dim(st, splits, is_top)
        # new_merge_ops = copy.deepcopy(merge_ops)
        # self._apply_splits(new_merge_ops, splits, st.network)
        if self.config.synthesizer.replay_from is None:
            search_engine = PartitionSearch(self.config, tmp_st.network)
        else:
            logger.debug("popping out the trace %s", tmp_st.replay_traces[0])
            trace = tmp_st.replay_traces.pop(0)
            logger.debug("remaining traces: %s", tmp_st.replay_traces)
            acs = trace.actions
            logger.debug("replaying actions %s", [str(ac) for ac in acs])
            # we need to update the index sizes in actions
            net_indices = tmp_st.network.free_indices()
            for ac in acs:
                new_indices = []
                for ind in ac.indices:
                    for nind in net_indices:
                        if nind.name == ind.name:
                            new_indices.append(ind.with_new_size(nind.size))
                            break

                ac.indices = new_indices

            search_engine = PartitionSearch(self.config, tmp_st.network, acs)

        if exclusions is not None:
            renamed_exclusions = []
            for ind in exclusions:
                renamed_exclusions.append(
                    ind.with_new_name(ind_map.get(ind.name, ind.name))
                )
            exclusions = renamed_exclusions
            logger.debug("exclusions: %s", exclusions)

        result = search_engine.search(
            copy.deepcopy(merge_ops), splits, delta, exclusions
        )
        if self.config.synthesizer.replay_from is not None:
            assert result.best_state is not None

        bn = result.best_state.network

        # self._revert_splits(splits, bn)
        logger.debug("get best structure %s", bn)
        tmp_bn = copy.deepcopy(bn)
        for split_op in reversed(split_ops):
            tmp_bn.split_index(split_op)
        after_split = tmp_bn.free_indices()
        del tmp_bn

        if self.config.synthesizer.replay_from is None:
            # save the splits and actions at the current level
            tmp_st.replay_traces[0].merge_ops = merge_ops
            tmp_st.replay_traces[0].split_ops = split_ops
            acs = to_splits(bn)
            logger.debug("applied actions %s", [str(ac) for ac in acs])
            logger.debug(
                "past actions: %s",
                [str(ac) for ac in result.best_state.past_actions],
            )
            # there might be fewer applied actions
            applied_acs = [
                ac for ac in result.best_state.past_actions if ac in acs
            ]
            tmp_st.replay_traces[0].actions = list(
                sorted(
                    applied_acs,
                    key=result.best_state.past_actions.index,
                )
            )

        best_st = HSearchState(
            tmp_st.free_indices, tmp_st.reshape_history, bn, 0
        )
        _apply_renaming(best_st, reverse_map)
        best_st.level = tmp_st.level
        best_st.replay_traces = tmp_st.replay_traces

        logger.debug("after revert renaming, we get %s with traces %s", bn, best_st.replay_traces)
        logger.debug(
            "used delta: %s, budget: %s, unused: %s",
            tmp_st.network.norm() ** 2 - bn.norm() ** 2,
            delta**2,
            result.unused_delta**2,
        )

        next_nets = self._get_next_nets(bn, best_st.free_indices)
        unused_delta = tmp_st.unused_delta + result.unused_delta**2
        if sorted(before_split) == sorted(after_split) and len(next_nets) == 1:
            best_st.unused_delta = unused_delta + remaining_delta**2
        else:
            # future work: it would be better to dynamically distribute errors
            # distribute delta equally to all subnets
            remaining_delta = remaining_delta / math.sqrt(len(next_nets))
            # enumerate nodes in the order of their scores
            for subnet in next_nets:
                # self._revert_splits(splits, subnet)
                # TODO: check whether we need to connect the unused delta here
                sn_unused = self._search_for_subnet(
                    best_st, remaining_delta, subnet
                )
                unused_delta += sn_unused

            # after we replaced all subnets, we compress the additional edges
            # print(best_st.network.norm() ** 2, sn_unused_delta)
            best_st.unused_delta = unused_delta

        # if (
        #     best_st.network.cost() <= best_merge.network.cost()
        #     or self.config.synthesizer.replay_from is not None
        # ):
        #     best_merge = best_st
        #     best_result = result
        best_st.network = best_st.network.compress()
        best_merge = best_st
        best_result = result

        logger.debug(
            "search result: new cost %s, %s",
            best_merge.network.cost(),
            best_merge.network,
        )

        self.stats.merge(best_result.stats)

        return best_merge

    @profile
    def _preprocess(self, net: TreeNetwork) -> TreeNetwork:
        return net

    def _get_next_nets(self, best_net: TreeNetwork, free_indices: Sequence[Index]) -> List[TreeNetwork]:
        """Get the next level nodes to optimize"""
        subgraph_nodes = DisjointSet()
        for node in best_net.network.nodes:
            subgraph_nodes.union(node, node)

        # for ac in to_splits(best_net):
        #     if ac not in best_st.past_actions and ac.reverse_edge is not None:
        #         subgraph_nodes.union(*ac.reverse_edge)

        subnets = []

        def _group_size(xs):
            # it is safer if we sort by free indices
            free_inds = []
            ind_size = 1
            for x in xs:
                for ind in best_net.node_tensor(x).indices:
                    if ind in free_indices:
                        free_inds.append(ind)
                        ind_size *= ind.size

            logger.debug(free_indices)
            return (ind_size, sorted(free_inds))

        for group in sorted(subgraph_nodes.groups().values(), key=_group_size):
            # print("group:", group)
            tn = TreeNetwork()
            tn.network = nx.subgraph(best_net.network, group).copy()
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
            logger.debug("No split, skip")
            if self.config.synthesizer.replay_from is None:
                best_st.replay_traces.append(
                    ReplayTrace(best_st.level + 1, [], [], [], [])
                )
            else:
                best_st.replay_traces.pop(0)

            return remaining_delta**2

        optimize_res.sort(key=lambda x: x.subnet_state.network.cost())
        # best_res = optimize_res[0]
        # for res in optimize_res[1:]:
        #     res_cost = res.subnet_state.network.cost()
        #     best_cost = best_res.subnet_state.network.cost()
        #     if best_res is None or res_cost < best_cost:
        #         logger.debug("among subnets, select %s", res.subnet_state.network)
        #         best_res = res

        # if "space_1_0" in [ind.name for ind in optimize_res[0].subnet.free_indices()]:
        #     best_res = optimize_res[3]
        # else:
        #     best_res = optimize_res[0]
        best_res = optimize_res[0]
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

        logger.debug("==before replace subnet==")
        logger.debug(best_st.network)
        logger.debug("-" * 20)
        logger.debug(best_res.subnet)
        best_st.network = best_st.network.replace_with(
            best_res.subnet, best_sn_st.network, best_sn_st.reshape_history
        )
        logger.debug("==after replace subnet==")
        logger.debug(best_st.network)
        logger.debug(best_st.replay_traces)
        logger.debug("-" * 20)
        best_st.free_indices = best_sn_st.free_indices
        best_st.reshape_history = best_sn_st.reshape_history
        if self.config.synthesizer.replay_from is None:
            logger.debug("adding subnet replay traces to its parent")
            best_st.replay_traces.extend(best_sn_st.replay_traces)
        else:
            best_st.replay_traces = best_sn_st.replay_traces
        return best_sn_st.unused_delta

    def _to_lower_dim(
        self, st: HSearchState, splits: Sequence[IndexOp], is_top: bool
    ):
        merge_start = time.time()

        if not self._trigger_merge(len(st.network.free_indices()), is_top):
            return (copy.deepcopy(st), [], [])

        if self.config.synthesizer.replay_from is not None:
            logger.debug("extracting the trace %s", st.replay_traces[0])
            trace = st.replay_traces[0]
            merge_ops = trace.merge_ops
            split_ops = trace.split_ops
            return (st, merge_ops, split_ops)

        tts = []
        for index_sets in self._cluster.cluster(st.network, splits):
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

                split_op = IndexSplit(
                    index=m_ind, shape=m_shape, result=m_indices
                )
                split_ops.append(split_op)

            # if len(st.network.network.nodes) == 1:
            #     yield (copy.deepcopy(st), merge_ops, split_ops)
            #     continue

            ok, tt = self._cross_to_tt(
                copy.deepcopy(st.network), merge_ops, splits
            )
            logger.debug("after index merge, we get a tensor train %s", tt)
            if ok:
                tmp_st = copy.deepcopy(st)
                tmp_st.network = tt
                tts.append((tmp_st, merge_ops, split_ops))

        tts.sort(key=lambda x: x[0].network.cost())
        self.stats.merge_time += time.time() - merge_start
        return tts[0]

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

        if self.config.synthesizer.replay_from is not None:
            logger.debug("replaying index splits: %s", st.replay_traces[0].splits)
            index_splits = [st.replay_traces[0].splits]
        else:
            index_splits = self._split_indices_on_budget(
                st, indices, compute_data
            )

        seen = set()
        result_sts = []
        for index_split in index_splits:
            for split_op in index_split:
                if split_op.result is not None:
                    split_op.result = tuple(split_op.result)

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
        self,
        st: HSearchState,
        indices: Sequence[Index],
        compute_data: bool = True,
    ) -> List[List[IndexSplit]]:
        # distribute the allowed splits between indices

        maxs = []
        for ind in indices:
            if ind in st.free_indices and ind.name not in self.config.topdown.disable_reshape:
                factors = sympy.factorint(ind.size)
                maxs.append(sum(factors.values()) - 1)
            else:
                maxs.append(0)

        all_splits = []
        if self.config.topdown.reshape_algo == ReshapeOption.ENUMERATE:
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

        if self.config.topdown.reshape_algo in (ReshapeOption.RANDOM, ReshapeOption.CLUSTER):
            splits = []
            for i, ind in enumerate(indices):
                ind_splits = self._get_split_op(st, ind, maxs[i], compute_data)
                if len(ind_splits) != 0:
                    splits.append(ind_splits)

            all_splits.extend(list(itertools.product(*splits)))

        return all_splits

    def _get_split_op(
        self,
        st: HSearchState,
        index: Index,
        budget: int = 5,
        compute_data: bool = True,
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
            return [IndexSplit(index=index, shape=tuple(shape))]

        if compute_data:
            # filter out the top few
            scores = _split_scores(st, index)
            # we always try our best to decompose the indices to
            # the maximum number and they subsume higher level reshapes
            shape_with_scores = set()
            for shape in _select_factors(res, 3):
                logger.debug("computing scores for shape %s", shape)
                # Calculate cumulative product sizes and sum their scores
                cumulative_sizes = itertools.accumulate(
                    shape[1:-1], operator.mul, initial=shape[0]
                )
                # transform the shape such that the low scores are excluded
                cumulative_score = [scores[size] for size in cumulative_sizes]
                # pos = 0
                # while pos < len(shape) - 1:
                #     if cumulative_score[pos] < 2:
                #         shape = (
                #             tuple(shape[:pos])
                #             + (shape[pos] * shape[pos + 1],)
                #             + tuple(shape[pos + 2 :])
                #         )
                #         cumulative_score.pop(pos)
                #     else:
                #         pos += 1

                # print(shape)

                assert len(cumulative_score) + 1 == len(shape)
                # # compute the score as a tensor train
                # score = 0
                # for i, cs in enumerate(cumulative_score):
                #     if i == 0:
                #         score += shape[i] * cs
                #     elif i == len(cumulative_score) - 1:
                #         score += shape[-1] * cs
                #     else:
                #         score += shape[i] * cs * cumulative_score[i-1]
                score = sum(cumulative_score)

                shape_with_scores.add((score, shape))

            shape_with_scores = list(shape_with_scores)
            shape_with_scores.sort(reverse=False)
        else:
            shape_with_scores = [
                (1, shape) for shape in _select_factors(res, budget)
            ]

        split_ops = []
        for score, shape in shape_with_scores[:10]:
            logger.debug("selecting shape %s with score %s", shape, score)
            # for selected in itertools.combinations(factors, r=k):
            split_ops.append(IndexSplit(index=index, shape=shape))

        return split_ops

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
        logger.debug("before split index, the network is %s", st.network)
        logger.debug("before split index, the norm is %s", st.network.norm())
        logger.debug("before split index, replay traces are %s", st.replay_traces)
        for split_result in self._split_indices(st, node_indices):
            net = split_result.state.network
            logger.debug("split index get %s", net)
            net.orthonormalize(node)
            logger.debug(
                "after normalize %s, the norm is %s", node, net.norm()
            )
            logger.debug(
                "the norm of orthogonality center is %s",
                np.linalg.norm(net.node_tensor(node).value),
            )

            new_sn = TreeNetwork()
            new_sn.add_node(node, copy.deepcopy(net.node_tensor(node)))
            logger.debug("the norm of the subnet is %s", new_sn.norm())
            new_st = HSearchState(
                split_result.state.free_indices,
                split_result.state.reshape_history,
                copy.deepcopy(new_sn),
            )
            new_st.level = st.level + 1
            if self.config.synthesizer.replay_from is None:
                new_st.replay_traces = [
                    ReplayTrace(st.level, split_result.splits, [], [], [])
                ]
            else:
                new_st.replay_traces = st.replay_traces

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
        validation_set: Optional[np.ndarray] = None,
    ):
        super().__init__(config)
        self.data_tensor = data_tensor
        self._cross_runner = CrossRunner()
        self._validation_set = validation_set

    def set_cross_runner(self, runner: CrossRunner):
        """Modify the cross approximation algorithm."""
        self._cross_runner = runner

    @profile
    def _initialize(self) -> Tuple[TreeNetwork, List[IndexOp]]:
        # we run index split to split as many as possible
        splits, f = self._gen_split_func(self.data_tensor)
        # run the cross approximation to avoid expensive error querying
        net = self._run_init_cross(self.data_tensor, f, splits)
        tree = TreeNetwork()
        tree.network = net.network
        # revert the index splits by merging the neighbor nodes
        for split_op in splits:
            if not isinstance(split_op, IndexSplit):
                continue

            assert split_op.result is not None
            base_index = split_op.result[0]
            base_node = tree.node_by_free_index(base_index.name)
            for ind in split_op.result[1:]:
                tree.merge(base_node, tree.node_by_free_index(ind.name))

            tree.merge_index(
                IndexMerge(indices=split_op.result, result=split_op.index)
            )

        # TODO: investigate this part
        return net, splits

    def _gen_split_func(
        self, data_tensor: CountingFunc
    ) -> Tuple[List[IndexOp], TensorFunc]:
        free_indices = data_tensor.indices
        splits = []

        if not self.config.cross.init_reshape:
            return splits, data_tensor

        init_st = init_state(data_tensor, self.config.engine.eps)

        # create index splits in the chosen order
        st = HSearchState(free_indices[:], [], init_st.network, 0)
        for n in st.network.network.nodes:
            n_indices = st.network.node_tensor(n).indices
            split_results = self._split_indices(
                st, n_indices, compute_data=False
            )
            if len(split_results) == 0:
                continue

            st = split_results[0].state
            splits.extend(split_results[0].splits)

        split_indices = list(st.network.free_indices())
        split_f = split_func(data_tensor, split_indices, splits)

        if self.config.topdown.cluster_method == ClusterMethod.RAND_NBR:
            split_f = self._rand_permute_indices(splits, split_f)

        self.init_splits = len(splits)

        assert self._validation_set is not None
        new_indices, self._validation_set = unravel_indices(
            splits, free_indices, self._validation_set
        )
        perm = [new_indices.index(ind) for ind in split_f.indices]
        self._validation_set = self._validation_set[:, perm]
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

    @profile
    def _run_init_cross(
        self,
        data_tensor: CountingFunc,
        f: TensorFunc,
        splits: Sequence[IndexOp],
    ) -> TreeNetwork:
        cross_res_file = f"{self.config.output.output_dir}/init_net.pkl"
        cross_start = time.time()
        init_eps = self.config.engine.eps * self.config.cross.init_eps

        if isinstance(data_tensor, FuncTensorNetwork):
            if (
                isinstance(data_tensor.net, TensorTrain)
                or not self.config.cross.init_cross
            ):
                self.stats.init_cross_size = data_tensor.net.cost()
                assert isinstance(data_tensor.net, TreeNetwork)
                return copy.deepcopy(data_tensor.net)

        if isinstance(data_tensor, FuncNeutron) and os.path.exists(cross_res_file):
            with open(cross_res_file, "rb") as cross_reader:
                net = pickle.load(cross_reader)

        else:
            net = self._cross_runner.run(
                f,
                init_eps,
                kickrank=self.config.cross.init_kickrank,
                validation=self._validation_set,
            )

            with open(cross_res_file, "wb") as cross_writer:
                pickle.dump(net, cross_writer)

        # net.draw()
        # plt.savefig(f"{self.config.output.output_dir}/init_net.png", dpi=100)
        # plt.close()

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
        logger.debug("before splitting, free indices are %s", st.free_indices)
        logger.debug("considering splitting %s in %s", indices, st.network)
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
                copy.deepcopy(new_sn),
            )
            new_st.level = st.level + 1
            if self.config.synthesizer.replay_from is None:
                new_st.replay_traces = [
                    ReplayTrace(st.level, split_result.splits, [], [], [])
                ]
            else:
                new_st.replay_traces = st.replay_traces

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
            logger.debug(
                "used delta: %s, budget: %s, unused: %s, real unused: %s",
                new_st.network.norm() ** 2 - sn_st.network.norm() ** 2,
                remaining_delta**2,
                sn_st.unused_delta,
                remaining_delta**2
                - (new_st.network.norm() ** 2 - sn_st.network.norm() ** 2),
            )
            logger.debug("after optimization, %s", sn_st.network)
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

        # print(net)
        # sort the indices according to the topology of the network
        nodes = [net.node_by_free_index(ind.name) for ind in free_inds]
        # get the two ends where the nodes have only one nbr in nodes
        ends = []
        for n in nodes:
            nbrs = list(net.network.neighbors(n))
            if len(nbrs) == 1 or not all(nbr in nodes for nbr in nbrs):
                ends.append(n)

        node_dists = [net.distance(ends[0], n) for n in nodes]
        free_inds = [free_inds[i] for i in np.argsort(node_dists)]
        func = FuncTensorNetwork(free_inds, net)
        max_rank = max(ind.size for ind in net.inner_indices())
        kickrank = max(2, max_rank // 5)
        return self._cross_runner.run(
            func, eps=self.config.engine.eps * 0.01, kickrank=kickrank
        )
        # core = net.contract()
        # tree = TreeNetwork()
        # tree.add_node("G0", core)
        # return tree


class StructureSweep:
    """Structure refinement algorithm."""

    def __init__(
        self,
        config: SearchConfig,
        data_tensor: TreeNetwork,
        free_indices: Sequence[Index],
    ):
        self.config = config
        self.free_indices = free_indices
        self.reshape_history = []

        self._data_tensor = data_tensor
        self._node_visited = defaultdict(int)

        self.traces = []

    @abstractmethod
    def _get_local_structure(self) -> TreeNetwork:
        raise NotImplementedError

    def sweep(self, delta: float) -> SearchResult:
        """Iteratively extract local structures and refine them."""
        # and replace the original one if a better structure is found
        total_result = SearchResult()
        total_result.stats = SearchStats()
        single_network = (
            len(self._data_tensor.network.nodes)
            < self.config.sweep.subnet_size
        )
        if single_network:
            step_delta = delta
        else:
            step_delta = math.sqrt(delta**2 / self.config.sweep.max_iters)

        for _ in range(self.config.sweep.max_iters):
            if single_network:
                local_struct = copy.deepcopy(self._data_tensor)
            else:
                local_struct = self._get_local_structure()

            logger.debug("get local structure %s", local_struct)
            exclude_inds = []
            for ind in local_struct.free_indices():
                if ind not in self.free_indices:
                    exclude_inds.append(ind)

            config = copy.deepcopy(self.config)
            config.output.collect_stats = False
            config.output.output_dir = (
                f"{config.output.output_dir}/{time.time()}"
            )
            if not os.path.exists(config.output.output_dir):
                os.makedirs(config.output.output_dir)
            config.cross.init_reshape = False
            config.cross.init_cross = (
                len(local_struct.network.nodes)
                <= self.config.topdown.group_threshold
            )

            indices = []
            for ind in local_struct.free_indices():
                indices.append(ind.with_new_rng(range(ind.size)))

            search_engine = BlackBoxTopDownSearch(
                config,
                FuncTensorNetwork(indices, local_struct),
                None,
            )
            search_engine.set_cross_runner(TTCrossRunner())
            search_engine.set_cluster()

            if self.config.synthesizer.replay_from is None:
                traces = None
            else:
                traces = self.traces.pop(0).traces
            result = search_engine.search(
                step_delta, self.free_indices, replay_traces=traces
            )
            result.network = result.network.compress()
            if os.path.exists(config.output.output_dir):
                shutil.rmtree(config.output.output_dir)

            total_result.stats.merge(search_engine.stats)
            # delta = math.sqrt(delta**2 - result.unused_delta)
            if (
                self.config.synthesizer.replay_from is not None
                or result.network.cost() < local_struct.cost()
            ):
                if self.config.synthesizer.replay_from is None:
                    free_inds = []
                    for ind in indices:
                        if ind in self._data_tensor.free_indices():
                            free_inds.append(ind)

                    self.traces.append(
                        ReplaySweep(free_inds, result.replay_traces)
                    )

                logger.debug(
                    "replacing %s with %s inside %s",
                    local_struct,
                    result.network,
                    self._data_tensor,
                )
                self._data_tensor = self._data_tensor.replace_with(
                    local_struct, result.network
                )
                self.free_indices = result.free_indices
                self.reshape_history += result.reshape_history
                logger.debug("optimized into %s", self._data_tensor)
                assert (
                    len(self._data_tensor.network.nodes)
                    == len(self._data_tensor.network.edges) + 1
                ), f"wrong tree network {self._data_tensor}"

                for node in result.network.network.nodes:
                    self._node_visited[node] += 1

            if single_network:
                break

        return total_result

    @property
    def data_tensor(self):
        """Get the modified data tensor."""
        return self._data_tensor


class RandomStructureSweep(StructureSweep):
    """Randomly select a local structure and optimize it"""

    def _get_local_structure(self) -> TreeNetwork:
        """Randomly select a local structure"""
        if self.config.synthesizer.replay_from is not None:
            nodes = []
            for ind in self.traces[0].indices:
                node = self._data_tensor.node_by_free_index(ind.name)
                nodes.append(node)

        else:
            logger.debug("sampling substructures from %s", self._data_tensor)
            # randomly sample one node from the network and randomly expand
            net_nodes = list(sorted(list(self._data_tensor.network.nodes)))
            weights = [9999] * len(net_nodes)
            for ni, node in enumerate(net_nodes):
                weights[ni] = self._node_visited.get(node, 9999)

            root_node = random.choices(net_nodes, weights=weights, k=1)[0]
            logger.debug("select root node %s", root_node)
            self._data_tensor.orthonormalize(root_node)
            if (
                len(self._data_tensor.network.nodes)
                <= self.config.sweep.subnet_size
            ):
                return copy.deepcopy(self._data_tensor)

            nodes = [root_node]
            while len(nodes) < self.config.sweep.subnet_size:
                expand_node = random.choice(nodes)
                logger.debug("expansion node %s", expand_node)
                nbrs = list(self._data_tensor.network.neighbors(expand_node))
                logger.debug("neighbors %s", nbrs)
                logger.debug("network %s", self._data_tensor)
                nbr_node = None
                while nbr_node is None or nbr_node in nodes:
                    nbr_node = random.choice(nbrs)

                    if len(nbrs) == 0 or all(nbr in nodes for nbr in nbrs):
                        nbr_node = None
                        break

                if nbr_node is not None:
                    nodes.append(nbr_node)
                    nodes = list(sorted(nodes))

        local_tree = TreeNetwork()
        for n in nodes:
            local_tree.add_node(n, self._data_tensor.node_tensor(n))

        for n in nodes:
            for m in nodes:
                if (n, m) in self._data_tensor.network:
                    local_tree.add_edge(n, m)

        return local_tree


class TraversalStructureSweep(StructureSweep):
    """Traverse the entire tree and do the structure refinement."""

    def __init__(
        self,
        config: SearchConfig,
        data_tensor: TreeNetwork,
        free_indices: Sequence[Index],
    ):
        super().__init__(config, data_tensor, free_indices)

        # record visited index pairs
        self._visited_edges = set()

        icount = data_tensor.all_indices()
        inner_indices = [i for i, v in icount.items() if v == 2]

        if inner_indices:
            self._next_edge = inner_indices[0]
        else:
            self._next_edge = None

    def _get_local_structure(self) -> TreeNetwork:
        # traverse the network from a given point
        if self._next_edge is None:
            return self._data_tensor

        # select at least one node from each end of the next edge and update
        # the next edge to be one of the free edges that haven't visited
        next_nodes = []
        for n in self._data_tensor.network.nodes:
            n_indices = self._data_tensor.node_tensor(n).indices
            if self._next_edge in n_indices:
                next_nodes.append(n)

        assert len(next_nodes) == 2

        u, v = next_nodes[0], next_nodes[1]
        u_inds, v_inds = index_partition(self._data_tensor, u, v)
        self._visited_edges.add((tuple(u_inds), tuple(v_inds)))

        u_nbrs = list(self._data_tensor.network.neighbors(u))
        u_nbrs.remove(v)
        v_nbrs = list(self._data_tensor.network.neighbors(v))
        v_nbrs.remove(u)

        selected_nodes = [u, v]
        selected_nodes.extend(random.choices(u_nbrs, k=min(len(u_nbrs), 2)))
        selected_nodes.extend(random.choices(v_nbrs, k=min(len(v_nbrs), 2)))
        threshold = self.config.topdown.group_threshold
        if len(selected_nodes) > threshold:
            selected_nodes = selected_nodes[:threshold]

        tree = TreeNetwork()
        tree.network = nx.subgraph(
            self._data_tensor.network, selected_nodes
        ).copy()

        # from the free indices of tree, we pick the next edge
        candidates = tree.free_indices()
        random.shuffle(candidates)
        for ind in candidates:
            if ind in self.free_indices:
                continue

            nodes = self._data_tensor.nodes_by_contraction_index(ind)
            assert len(nodes) == 2
            l_inds, r_inds = index_partition(self._data_tensor, *nodes)
            inds_parts = (tuple(l_inds), tuple(r_inds))
            if inds_parts not in self._visited_edges:
                self._next_edge = ind
                break

        return tree

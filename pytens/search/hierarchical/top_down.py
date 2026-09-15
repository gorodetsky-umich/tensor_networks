"""Top down reshaping search"""

from collections import defaultdict
import copy
import os
import shutil
import itertools
import logging
import math
import operator
import pickle
import random
import time
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Set

import networkx as nx
import numpy as np
import sympy

from pytens.algs import Tensor, TensorNetwork
from pytens.cross.func_interface import CachedFunc, TensorFunc
from pytens.cross.func_impl import FuncTensorNetwork, PermuteFunc
from pytens.cross.runner import CrossRunner, TTCrossRunner
from pytens.search.algs.partition import PartitionSearch
from pytens.search.configuration import (
    ClusterMethod,
    ReshapeOption,
    SearchConfig,
)
from pytens.search.hierarchical.error_dist import BaseErrorDist
from pytens.search.hierarchical.index_cluster import (
    CrossIndexCluster,
    IndexCluster,
    RandomIndexCluster,
    SVDIndexCluster,
    eff_rank,
)
from pytens.search.hierarchical.types import (
    HSearchState,
    IndexSplitResult,
    PartitionSearchInput,
    Replay,
    ReplaySweep,
    ReplayTrace,
    SubnetResult,
)
from pytens.search.hierarchical.utils import (
    DisjointSet,
    build_bipartite_sample,
    split_func,
)
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
    SValsParams,
)
from pytens.search.state import SearchState
from pytens.search.types import SearchContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _permute_unique(nums: Sequence[int]) -> Sequence[Tuple[int, ...]]:
    """All distinct orderings of a multiset of integers."""
    nums = sorted(nums)
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
    results: List[Sequence[int]] = []
    for factors_perm in _permute_unique(factors_flat):
        for chunks in _split_into_chunks(
            factors_perm, min(budget + 1, len(factors_perm))
        ):
            chunk_factors = tuple(math.prod(chunk) for chunk in chunks)
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


def _rename_data_tensor(
    st: HSearchState, data_tensor: TensorNetwork
) -> Dict[IndexName, IndexName]:
    """Give the internal indices canonical names ``s_0, s_1, ...``.

    Names are assigned in the traversal order of the dimension tree rooted
    at the node holding the "largest" free index, so that structurally
    equal networks get equal names. Free indices keep their names. Returns
    the map from new names back to the original ones.
    """
    free = frozenset(st.free_indices)

    def split_name(ind: Index) -> Tuple[bool, int, str, int]:
        segments = str(ind.name).split("_")
        # names such as "x_nu" have no numeric suffix
        has_suffix = len(segments) > 1 and segments[1].isdigit()
        suffix = int(segments[1]) if has_suffix else 0
        return (ind not in free, ind.size, segments[0], suffix)

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
            if ind in free:
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
        f"get mapping for {list(ind_map.values())} but all indices are "
        f"{data_tensor.all_indices()}"
    )

    return reverse_map


def _apply_renaming(
    st: HSearchState, reverse_map: Dict[IndexName, IndexName]
) -> None:
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

    # The cut is scored between the far side of the node's first bond and
    # everything else, with the leading factor of the split index moved to
    # the far side. That far side does not depend on the divisor.
    net = st.network
    nbrs = list(net.network.neighbors(node))
    far_side: List[Index] = []
    if nbrs:
        near_side, far_side = index_partition(net, node, nbrs[0])
        if index not in near_side:
            far_side = near_side

    split_scores = {}
    for n in sympy.divisors(index.size):
        if n in (1, index.size):
            continue

        lres = Index(str(index.name) + "_0", n)
        rres = Index(str(index.name) + "_1", index.size // n)
        net.split_index(
            IndexSplit(
                index=index,
                shape=(n, index.size // n),
                result=[lres, rres],
            )
        )

        target_inds = far_side + [lres]
        s = net.random_svals(node, target_inds, SValsParams(max_rank=100))
        split_scores[n] = eff_rank(s)  # s[0] / s[min(len(s), 1)]
        logger.debug(
            "target indices %s with size %s has score %s",
            target_inds,
            n,
            split_scores[n],
        )

        net.merge_index(IndexMerge(indices=[lres, rres], result=index))

    return split_scores


def _shape_score(
    shape: Sequence[int], scores: Dict[int, float], node_size: float
) -> float:
    """Estimate the cost of reshaping an index into `shape`.

    Reshaping turns one core into a small tensor train over the factors;
    `scores[k]` is the effective rank of the cut after the first `k`
    entries of the index (from ``_split_scores``), and `node_size` is the
    size of the rest of the core. The score adds up the sizes of the
    resulting train's cores.
    """
    cuts = itertools.accumulate(shape[1:-1], operator.mul, initial=shape[0])
    ranks = [scores[size] for size in cuts]
    assert len(ranks) + 1 == len(shape)

    score = shape[0] * ranks[0] * node_size
    for i in range(1, len(ranks)):
        score += shape[i] * ranks[i] * ranks[i - 1]
    score += shape[-1] * ranks[-1]
    return score


class TopDownSearch:
    """Search for reshaped structures from top to bottom"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.error_dist = BaseErrorDist()
        self.stats = SearchStats()
        self.init_splits = 0
        self._cluster = self.set_cluster()

    def set_cluster(self) -> IndexCluster:
        """Set the index clustering method."""
        threshold = self.config.topdown.group_threshold
        cluster_method = {
            ClusterMethod.SVD: SVDIndexCluster(threshold),
            ClusterMethod.RAND: RandomIndexCluster(threshold),
            ClusterMethod.NBR: RandomIndexCluster(threshold, False),
            ClusterMethod.RAND_NBR: RandomIndexCluster(threshold, False),
            ClusterMethod.CROSS: CrossIndexCluster(
                threshold, self.config.engine.eps * 0.1
            ),
        }.get(self.config.topdown.cluster_method)

        assert cluster_method is not None, (
            f"unknown cluster method: {self.config.topdown.cluster_method}"
        )
        return cluster_method

    def search(
        self,
        delta: Optional[float] = None,
        free_indices: Optional[List[Index]] = None,
        replay_traces: Optional[List[Replay]] = None,
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

        st: HSearchState = self._search(
            init_st,
            SearchContext(remaining_delta=delta, splits=splits, is_top=True),
        )

        # best_st = init_st
        best_net = None
        delta = st.unused_delta
        for n in st.network.network.nodes:
            network = copy.deepcopy(st.network)
            try:
                logger.debug("unused delta: %s", delta)
                network.round(n, atol=math.sqrt(delta))
            except np.linalg.LinAlgError:
                continue

            if best_net is None or network.cost() < best_net.cost():
                best_net = network

        logger.debug("obtained rounded network %s", best_net)
        assert best_net is not None
        st.network = best_net.compress()
        logger.debug("=======")
        return st

    @abstractmethod
    def _initialize(self) -> Tuple[TensorNetwork, List[IndexOp]]:
        raise NotImplementedError

    def _trigger_merge(self, ind_cnt: int, is_top: bool) -> bool:
        """Determine whether to trigger the index merge operation before
        search."""
        return (
            (self.config.topdown.reshape_algo == ReshapeOption.CLUSTER)
            and (not is_top or self.config.topdown.merge_mode == "all")
            and ind_cnt > self.config.topdown.group_threshold
        )

    def _merge_score(
        self, data_tensor: TensorNetwork, merge_ops: List[IndexMerge]
    ) -> float:
        """Estimate how well a set of index merges suits `data_tensor`.

        The free indices are reordered so that merged indices sit next to
        each other, ordered by their position along the network; the score
        is the effective rank of a random sample of the matrix that splits
        this order in half. Lower is better.
        """
        transform_start = time.time()
        free_inds = data_tensor.free_indices()

        # position of every free index along the network: its distance
        # from one end node
        end = data_tensor.end_nodes()[0]
        node_dist = nx.single_source_shortest_path_length(
            data_tensor.network, end
        )
        index_dist: Dict[Index, int] = {}
        for node in data_tensor.network.nodes:
            for ind in data_tensor.node_tensor(node).indices:
                if ind in free_inds:
                    index_dist[ind] = int(node_dist[node])

        def merge_op_dist(mop: IndexMerge) -> float:
            total = sum(index_dist[ind] for ind in mop.indices)
            return total / len(mop.indices)

        # merged indices sit next to each other, ordered by position, and
        # the remaining free indices follow
        new_indices = []
        for mop in sorted(merge_ops, key=merge_op_dist):
            new_indices.extend(sorted(mop.indices, key=index_dist.__getitem__))
        for ind in free_inds:
            if ind not in new_indices:
                new_indices.append(ind)

        assert len(new_indices) == len(free_inds), (
            f"get {new_indices} with merges {merge_ops}, but expect "
            f"{free_inds}"
        )

        # random sample points and evaluate the middle effective ranks
        selected_inds = []
        sample_size = 100
        for ind in free_inds:
            selected_inds.append(
                np.random.randint(0, ind.size, size=(sample_size,))
            )

        # reorganize the values according to reordered indices
        left_inds = new_indices[: len(new_indices) // 2]
        right_inds = new_indices[len(new_indices) // 2 :]
        full_indices, eval_inds = build_bipartite_sample(
            list(left_inds), list(right_inds), free_inds, selected_inds
        )

        vals = data_tensor.evaluate(eval_inds, full_indices)
        s = np.linalg.svdvals(vals.reshape(sample_size, sample_size))
        score = eff_rank(s)
        logger.debug("score for %s is %s", merge_ops, score)

        self.stats.merge_transform_time = time.time() - transform_start
        return score

    def _search(
        self,
        st: HSearchState,
        ctx: SearchContext = SearchContext(),
    ) -> HSearchState:
        """Search one level: optimize `st.network`, then recurse into the
        resulting subnets with the leftover error budget."""
        assert ctx.remaining_delta is not None, "delta must be specified"
        logger.debug("original cost %s", st.network.cost())

        if not st.network.is_tensor_train():
            logger.debug("turning %s into a tensor train", st.network)
            st.network = self._preprocess(st.network)

        # this level gets `level_delta`, the levels below `sub_delta`
        level_delta, sub_delta = self.error_dist.split_delta(
            ctx.remaining_delta
        )
        assert math.isclose(
            level_delta**2 + sub_delta**2, ctx.remaining_delta**2, rel_tol=1e-9
        ), f"budget split {level_delta}, {sub_delta} != {ctx.remaining_delta}"

        reverse_map = _rename_data_tensor(st, st.network)
        logger.debug("reverse name mapping is %s", reverse_map)
        before_split = st.network.free_indices()
        logger.debug("searching better structures for %s", st.network)
        logger.debug(
            "network norm: %s, allowed compression: %s",
            st.network.norm() ** 2,
            level_delta**2,
        )
        if len(st.network.network.nodes) > self.config.sweep.subnet_size:
            return self._search_via_sweep(st, level_delta, reverse_map)

        merge_ops, split_ops = self._to_lower_dim(st, ctx.splits, ctx.is_top)
        logger.debug("select merge operations: %s", merge_ops)
        search_engine = self._build_search_engine(st)
        result = search_engine.search(
            SearchContext(
                splits=ctx.splits,
                merge_ops=copy.deepcopy(merge_ops),
                remaining_delta=level_delta,
                exclusions=ctx.exclusions,
            ),
        )
        if self.config.synthesizer.replay_from is not None:
            assert result.best_state is not None

        partition_input = PartitionSearchInput(
            result=result,
            merge_ops=merge_ops,
            split_ops=split_ops,
            before_split=before_split,
            remaining_delta=sub_delta,
            reverse_map=reverse_map,
        )
        best_st = self._finalize_partition_result(st, partition_input)
        self.stats.merge(result.stats)
        return best_st

    def _search_via_sweep(
        self,
        st: HSearchState,
        delta: float,
        reverse_map: dict,
    ) -> HSearchState:
        """Run sweep-based search for networks exceeding the subnet size."""
        logger.debug(
            "running sweep with iterations %s", self.config.sweep.max_iters
        )
        config = copy.deepcopy(self.config)
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
        return best_st

    def _build_search_engine(self, st: HSearchState) -> "PartitionSearch":
        """Build a PartitionSearch engine, replaying actions if needed."""
        if self.config.synthesizer.replay_from is None:
            return PartitionSearch(self.config, st.network)

        logger.debug("popping out the trace %s", st.replay_traces[0])
        trace = st.replay_traces.pop(0)
        assert isinstance(trace, ReplayTrace)
        logger.debug("remaining traces: %s", st.replay_traces)
        acs = trace.actions
        logger.debug("replaying actions %s", [str(ac) for ac in acs])

        # recorded actions carry the index sizes of the recording run;
        # refresh them from the current network
        size_by_name = {
            ind.name: ind.size for ind in st.network.free_indices()
        }
        for ac in acs:
            ac.indices = [
                ind.with_new_size(size_by_name[ind.name])
                for ind in ac.indices
                if ind.name in size_by_name
            ]

        return PartitionSearch(self.config, st.network, acs)

    def _finalize_partition_result(
        self,
        tmp_st: HSearchState,
        inp: PartitionSearchInput,
    ) -> HSearchState:
        """Build best state after partition search and recurse into subnets."""
        result = inp.result
        assert result.best_state is not None
        bn = result.best_state.network
        logger.debug("get best structure %s", bn)

        # compare in the renamed index space, before names are reverted
        after_split = self._free_after_split(inp)

        if self.config.synthesizer.replay_from is None:
            self._record_trace(tmp_st, result.best_state, inp)

        best_st = HSearchState(
            tmp_st.free_indices, tmp_st.reshape_history, bn, 0
        )
        _apply_renaming(best_st, inp.reverse_map)
        best_st.level = tmp_st.level
        best_st.replay_traces = tmp_st.replay_traces

        # the clusters were formed on the renamed network, which has just
        # been renamed back, so translate them before grouping nodes
        clusters = []
        for merge_op in inp.merge_ops:
            clusters.append(
                [
                    ind.with_new_name(inp.reverse_map.get(ind.name, ind.name))
                    for ind in merge_op.indices
                ]
            )
        next_nets = self._get_next_nets(bn, best_st.free_indices, clusters)

        # Nothing to recurse into when the search neither split the node
        # nor reshaped its indices; the whole sub-budget is then unused.
        # `unused` accumulates squared deltas.
        unused = tmp_st.unused_delta + result.unused_delta**2
        same_indices = sorted(inp.before_split) == sorted(after_split)
        unchanged = same_indices and len(next_nets) == 1
        if unchanged:
            unused += inp.remaining_delta**2
        else:
            sub_delta = inp.remaining_delta / math.sqrt(len(next_nets))
            for subnet in next_nets:
                unused += self._search_for_subnet(best_st, sub_delta, subnet)

        best_st.unused_delta = unused
        best_st.network = best_st.network.compress()
        return best_st

    @staticmethod
    def _free_after_split(inp: PartitionSearchInput) -> List[Index]:
        """Free indices of the best network once the merged indices are
        split back into their original ones."""
        assert inp.result.best_state is not None
        net = copy.deepcopy(inp.result.best_state.network)
        for split_op in reversed(inp.split_ops):
            net.split_index(split_op)
        return net.free_indices()

    @staticmethod
    def _record_trace(
        st: HSearchState, best: SearchState, inp: PartitionSearchInput
    ) -> None:
        """Store what this level did in its replay trace: the index merges
        and splits, and the actions that survived into the best network."""
        trace = st.replay_traces[0]
        assert isinstance(trace, ReplayTrace)
        trace.merge_ops = inp.merge_ops
        trace.split_ops = inp.split_ops
        kept = to_splits(best.network)
        applied = [ac for ac in best.past_actions if ac in kept]
        trace.actions = sorted(applied, key=best.past_actions.index)

    def _preprocess(self, net: TensorNetwork) -> TensorNetwork:
        return net

    def _get_next_nets(
        self,
        best_net: TensorNetwork,
        free_indices: Sequence[Index],
        clusters: Sequence[Sequence[Index]] = (),
    ) -> List[TensorNetwork]:
        """Split the network into the subnets optimized at the next level.

        Indices that were clustered together are only ever moved as a
        whole by the search, so the nodes carrying one cluster may still be
        several cores of the input. Those nodes, together with the nodes on
        the paths between them, form one subnet; every other node is a
        subnet of its own. Subnets are ordered by their free size.
        """
        groups = DisjointSet()
        for node in best_net.network.nodes:
            groups.union(node, node)

        for cluster in clusters:
            nodes = {best_net.node_by_free_index(ind.name) for ind in cluster}
            anchor = next(iter(nodes))
            for node in nodes:
                # union along the path so the subnet stays a connected tree
                for on_path in nx.shortest_path(
                    best_net.network, anchor, node
                ):
                    groups.union(anchor, on_path)

        def _group_size(xs: Sequence[NodeName]) -> Tuple[int, List[Index]]:
            # it is safer if we sort by free indices
            free_inds = []
            ind_size = 1
            for x in xs:
                for ind in best_net.node_tensor(x).indices:
                    if ind in free_indices:
                        free_inds.append(ind)
                        ind_size *= ind.size

            return (ind_size, sorted(free_inds))

        subnets = []
        for group in sorted(groups.groups().values(), key=_group_size):
            tn = TensorNetwork()
            tn.network = nx.subgraph(best_net.network, group).copy()
            subnets.append(tn)

        return subnets

    def _search_for_subnet(
        self,
        best_st: HSearchState,
        remaining_delta: float,
        subnet: TensorNetwork,
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

        best_res = optimize_res[0]
        best_st.network = best_res.network
        best_sn_st = best_res.subnet_state

        # if nothing happened in the subnet, we contract the entire subnet
        contraction_size = np.prod(
            [ind.size for ind in best_sn_st.network.free_indices()]
        )
        if contraction_size < best_sn_st.network.cost():
            node = best_sn_st.network.contract()
            tmp_subnet = TensorNetwork()
            tmp_subnet.add_node("G0", node)
            best_sn_st.network = tmp_subnet

        logger.debug(
            "replacing %s inside %s", best_res.subnet, best_st.network
        )
        best_st.network = best_st.network.replace_with(
            best_res.subnet, best_sn_st.network, best_sn_st.reshape_history
        )
        logger.debug("after replacing the subnet: %s", best_st.network)
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
    ) -> Tuple[Sequence[IndexMerge], Sequence[IndexSplit]]:
        merge_start = time.time()

        if not self._trigger_merge(len(st.network.free_indices()), is_top):
            return ([], [])

        if self.config.synthesizer.replay_from is not None:
            logger.debug("extracting the trace %s", st.replay_traces[0])
            trace = st.replay_traces[0]
            assert isinstance(trace, ReplayTrace)
            merge_ops = trace.merge_ops
            split_ops = trace.split_ops
            return (merge_ops, split_ops)

        candidates = []
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

            score = self._merge_score(st.network, merge_ops)
            candidates.append((score, merge_ops, split_ops))

        _, merge_ops, split_ops = min(candidates, key=lambda x: x[0])
        self.stats.merge_time += time.time() - merge_start
        return merge_ops, split_ops

    def _optimize_node(
        self,
        st: HSearchState,
        subnet: TensorNetwork,
        remaining_delta: float,
    ) -> List[SubnetResult]:
        """Search every reshape of `subnet` one level down.

        For each candidate reshape of the subnet's indices the full network
        is orthonormalized at the subnet, the subnet is cut out and searched
        on its own with the given budget.
        """
        node = list(subnet.network.nodes)[0]
        indices = self._subnet_indices(st, subnet)
        logger.debug("considering splitting %s in %s", indices, st.network)

        results = []
        for split_result in self._split_indices(st, indices):
            net = split_result.state.network
            net.orthonormalize(node)
            new_sn = self._extract_subnet(net, subnet)

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

            # bonds cut by extracting the subnet are free indices of it
            # now, but must not be split on their own
            exclusions = [
                ind
                for ind in new_sn.free_indices()
                if ind not in split_result.state.free_indices
            ]
            sn_st = self._search(
                new_st, SearchContext(remaining_delta, exclusions=exclusions)
            )
            logger.debug("after optimization, %s", sn_st.network)
            results.append(SubnetResult(net, new_sn, sn_st))

        return results

    @abstractmethod
    def _subnet_indices(
        self, st: HSearchState, subnet: TensorNetwork
    ) -> List[Index]:
        """The indices whose reshapes are tried for `subnet`."""
        raise NotImplementedError

    @abstractmethod
    def _extract_subnet(
        self, net: TensorNetwork, subnet: TensorNetwork
    ) -> TensorNetwork:
        """Cut the nodes of `subnet` out of the orthonormalized `net`."""
        raise NotImplementedError

    def _split_indices(
        self, st: HSearchState, indices: List[Index], compute_data: bool = True
    ) -> Sequence[IndexSplitResult]:
        if not self.config.topdown.reshape_enabled and compute_data:
            return [IndexSplitResult(st, [])]

        if self.config.synthesizer.replay_from is not None:
            assert isinstance(st.replay_traces[0], ReplayTrace)
            logger.debug(
                "replaying index splits: %s", st.replay_traces[0].splits
            )
            index_splits = [list(st.replay_traces[0].splits)]
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

            if not index_split or tuple(index_split) in seen:
                continue

            seen.add(tuple(index_split))

            new_st = copy.deepcopy(st)
            used_splits = []
            for split_op in index_split:
                split_op = copy.deepcopy(split_op)
                new_st = new_st.split_index(split_op, compute_data)
                used_splits.append(split_op)

            result_sts.append(IndexSplitResult(new_st, used_splits))

        return result_sts

    def _split_indices_on_budget(
        self,
        st: HSearchState,
        indices: Sequence[Index],
        compute_data: bool = True,
    ) -> List[List[IndexSplit]]:
        """Enumerate combinations of index splits, one per index.

        Each index may be split at most as many times as it has prime
        factors. With ``ENUMERATE`` the total number of splits is fixed by
        the group threshold and every distribution of that budget over the
        indices is tried; otherwise every index gets its full budget.
        """
        # the most splits each index allows
        maxs = []
        for ind in indices:
            restricted = ind.name in self.config.topdown.reshape_restriction
            if ind in st.free_indices and not restricted:
                factors = sympy.factorint(ind.size)
                maxs.append(sum(factors.values()) - 1)
            else:
                maxs.append(0)

        if self.config.topdown.reshape_algo == ReshapeOption.ENUMERATE:
            budget = self.config.topdown.group_threshold - len(indices)
            budget = min(sum(maxs), budget)  # exhaust the budget as possible
            budgets = [
                b
                for b in itertools.product(*[range(x + 1) for x in maxs])
                if sum(b) == budget
            ]
        elif self.config.topdown.reshape_algo in (
            ReshapeOption.RANDOM,
            ReshapeOption.CLUSTER,
        ):
            budgets = [tuple(maxs)]
        else:
            budgets = []

        all_splits: List[List[IndexSplit]] = []
        for ind_budget in budgets:
            per_index = []
            for ind, b in zip(indices, ind_budget):
                ind_splits = self._get_split_op(st, ind, b, compute_data)
                if ind_splits:
                    per_index.append(ind_splits)

            all_splits.extend(
                list(combo) for combo in itertools.product(*per_index)
            )

        return all_splits

    def _get_split_op(
        self,
        st: HSearchState,
        index: Index,
        budget: int = 5,
        compute_data: bool = True,
    ) -> Sequence[IndexSplit]:
        """Candidate reshapes of one free index into a tuple of factors."""
        if index not in st.free_indices or budget <= 0:
            return []

        res: Dict[int, int] = sympy.factorint(index.size)
        factors = [i for i, n in res.items() for _ in range(n)]
        if len(factors) == 1:
            return []

        if self.config.topdown.reshape_algo == ReshapeOption.RANDOM:
            k = np.random.randint(0, len(factors))
            selected = random.sample(factors, k=k)
            shape: Sequence[int] = _create_split_target(factors, selected)
            return [IndexSplit(index=index, shape=tuple(shape))]

        if not compute_data:
            shapes = _select_factors(res, budget)
        else:
            # score every 3-way factorization by the cost of the tensor
            # train it induces and keep the best few
            scores = _split_scores(st, index)
            node = st.network.node_by_free_index(index.name)
            node_size = st.network.node_size(node) / index.size
            scored = set()
            for shape in _select_factors(res, 3):
                score = _shape_score(shape, scores, node_size)
                logger.debug("get score for shape %s: %s", shape, score)
                scored.add((score, shape))

            top = sorted(scored)[: self.config.topdown.reshape_opts]
            shapes = [shape for _, shape in top]

        return [IndexSplit(index=index, shape=shape) for shape in shapes]


class WhiteBoxTopDownSearch(TopDownSearch):
    """Top down structural search for white box data tensors."""

    def __init__(self, config: SearchConfig, data_tensor: TensorNetwork):
        super().__init__(config)
        self.data_tensor = data_tensor

    def _initialize(self) -> Tuple[TensorNetwork, List[IndexOp]]:
        return self.data_tensor, []

    def _subnet_indices(
        self, st: HSearchState, subnet: TensorNetwork
    ) -> List[Index]:
        nodes = list(subnet.network.nodes)
        assert len(nodes) == 1, (
            "only single node should be optimized in white box tensors"
        )
        return list(st.network.node_tensor(nodes[0]).indices)

    def _extract_subnet(
        self, net: TensorNetwork, subnet: TensorNetwork
    ) -> TensorNetwork:
        node = list(subnet.network.nodes)[0]
        new_sn = TensorNetwork()
        new_sn.add_node(node, copy.deepcopy(net.node_tensor(node)))
        return new_sn


class BlackBoxTopDownSearch(TopDownSearch):
    """Top down structural search for black box tensors."""

    def __init__(
        self,
        config: SearchConfig,
        data_tensor: CachedFunc,
        validation_set: Optional[np.ndarray] = None,
    ):
        super().__init__(config)
        self.data_tensor = data_tensor
        self._cross_runner: CrossRunner
        self._validation_set = validation_set

    def set_cross_runner(self, runner: CrossRunner) -> None:
        """Modify the cross approximation algorithm."""
        self._cross_runner = runner

    def _initialize(self) -> Tuple[TensorNetwork, List[IndexOp]]:
        # we run index split to split as many as possible
        splits, f = self._gen_split_func(self.data_tensor)
        # run the cross approximation to avoid expensive error querying
        net = self._run_init_cross(self.data_tensor, f)
        tree = TensorNetwork()
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

        return net, splits

    def _gen_split_func(
        self, data_tensor: CachedFunc
    ) -> Tuple[List[IndexOp], TensorFunc]:
        free_indices = data_tensor.indices
        splits: List[IndexOp] = []

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

        perm_f: TensorFunc
        if self.config.topdown.cluster_method == ClusterMethod.RAND_NBR:
            perm_f = self._rand_permute_indices(splits, split_f)
        else:
            perm_f = split_f

        self.init_splits = len(splits)

        assert self._validation_set is not None
        new_indices, self._validation_set = unravel_indices(
            splits, free_indices, self._validation_set
        )
        perm = [new_indices.index(ind) for ind in perm_f.indices]
        self._validation_set = self._validation_set[:, perm]
        return splits, perm_f

    def _rand_permute_indices(
        self, splits: List[IndexOp], split_f: TensorFunc
    ) -> PermuteFunc:
        # randomly permute the indices of f
        perm = list(range(len(split_f.indices)))
        random.shuffle(perm)
        perm_indices = [split_f.indices[i] for i in perm]
        unperm = np.argsort(perm).tolist()
        splits.append(IndexPermute(perm=tuple(perm), unperm=tuple(unperm)))
        return PermuteFunc(perm_indices, split_f, unperm)

    def _run_init_cross(
        self, data_tensor: CachedFunc, f: TensorFunc
    ) -> TensorNetwork:
        cross_res_file = f"{self.config.output.output_dir}/init_net.pkl"
        cross_start = time.time()
        init_eps = self.config.engine.eps * self.config.cross.init_eps

        if isinstance(data_tensor, FuncTensorNetwork):
            if (
                data_tensor.net.is_tensor_train()
                or not self.config.cross.init_cross
            ):
                self.stats.init_cross_size = data_tensor.net.cost()
                assert isinstance(data_tensor.net, TensorNetwork)
                return copy.deepcopy(data_tensor.net)

        net: TensorNetwork
        if self.config.cross.use_input_net and os.path.exists(cross_res_file):
            with open(cross_res_file, "rb") as cross_reader:
                net = pickle.load(cross_reader)

        else:
            net = self._cross_runner.run(
                f,
                init_eps,
                kickrank=self.config.cross.init_kickrank,
                validation=self._validation_set,
            )

            if os.path.exists(cross_res_file):
                with open(cross_res_file, "wb") as cross_writer:
                    pickle.dump(net, cross_writer)

        self.stats.cross_time = time.time() - cross_start
        self.stats.init_cross_size = net.cost()
        self.stats.init_cross_evals = data_tensor.num_calls()

        return net

    def _subnet_indices(
        self, st: HSearchState, subnet: TensorNetwork
    ) -> List[Index]:
        return list(subnet.free_indices())

    def _extract_subnet(
        self, net: TensorNetwork, subnet: TensorNetwork
    ) -> TensorNetwork:
        new_sn = TensorNetwork()
        new_sn.network = nx.subgraph(net.network, subnet.network.nodes).copy()
        return new_sn


class StructureSweep:
    """Structure refinement algorithm."""

    def __init__(
        self,
        config: SearchConfig,
        data_tensor: TensorNetwork,
        free_indices: List[Index],
    ):
        self.config = config
        self.free_indices = free_indices
        self.reshape_history: List[IndexOp] = []

        self._data_tensor = data_tensor
        self._node_visited: Dict[NodeName, int] = defaultdict(int)

        self.traces: List[Replay] = []

    @abstractmethod
    def _get_local_structure(self) -> TensorNetwork:
        raise NotImplementedError

    def sweep(self, delta: float) -> SearchResult:
        """Iteratively extract local structures and refine them."""
        # and replace the original one if a better structure is found
        total_result = SearchResult()
        total_result.stats = SearchStats()
        single_network = (
            len(self._data_tensor.network.nodes)
            <= self.config.sweep.subnet_size
        )
        if single_network:
            step_delta = delta
        else:
            step_delta = math.sqrt(delta**2 / self.config.sweep.max_iters)

        for _ in range(self.config.sweep.max_iters):
            added_delta = self._run_sweep_iteration(
                single_network, step_delta, total_result
            )
            total_result.unused_delta += added_delta
            if single_network:
                break

        total_result.unused_delta = total_result.unused_delta**0.5
        return total_result

    def _run_sweep_iteration(
        self,
        single_network: bool,
        step_delta: float,
        total_result: SearchResult,
    ) -> float:
        """
        Run one sweep iteration;
        return the squared unused delta contribution.
        """
        if single_network:
            local_struct = copy.deepcopy(self._data_tensor)
        else:
            local_struct = self._get_local_structure()

        logger.debug("get local structure %s", local_struct)
        config = copy.deepcopy(self.config)
        config.output.collect_stats = False
        config.output.output_dir = f"{config.output.output_dir}/{time.time()}"
        if not os.path.exists(config.output.output_dir):
            os.makedirs(config.output.output_dir)
        config.cross.init_reshape = False
        config.cross.init_cross = False

        indices = [
            ind.with_new_rng(range(ind.size))
            for ind in local_struct.free_indices()
        ]
        search_engine = BlackBoxTopDownSearch(
            config,
            FuncTensorNetwork(indices, local_struct),
            None,
        )
        search_engine.set_cross_runner(TTCrossRunner())
        search_engine.set_cluster()

        traces: Optional[List[Replay]] = None
        if self.config.synthesizer.replay_from is not None:
            sweep_traces = self.traces.pop(0)
            assert isinstance(sweep_traces, ReplaySweep)
            traces = list(sweep_traces.traces)

        result = search_engine.search(
            step_delta, self.free_indices, replay_traces=traces
        )
        result.network = result.network.compress()
        if os.path.exists(config.output.output_dir):
            shutil.rmtree(config.output.output_dir)

        total_result.stats.merge(search_engine.stats)
        if (
            self.config.synthesizer.replay_from is not None
            or result.network.cost() < local_struct.cost()
        ):
            return self._apply_sweep_result(indices, local_struct, result)

        return step_delta**2

    def _apply_sweep_result(
        self,
        indices: Sequence[Index],
        local_struct: TensorNetwork,
        result: HSearchState,
    ) -> float:
        """
        Apply an improved sweep result to the data tensor;
        return squared unused delta.
        """
        if self.config.synthesizer.replay_from is None:
            free_inds = [
                ind
                for ind in indices
                if ind in self._data_tensor.free_indices()
            ]
            self.traces.append(ReplaySweep(free_inds, result.replay_traces))

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
        return result.unused_delta**2

    @property
    def data_tensor(self) -> TensorNetwork:
        """Get the modified data tensor."""
        return self._data_tensor


class RandomStructureSweep(StructureSweep):
    """Randomly select a local structure and optimize it"""

    def _get_local_structure(self) -> TensorNetwork:
        """Randomly select a local structure"""
        if self.config.synthesizer.replay_from is not None:
            nodes = []
            assert isinstance(self.traces[0], ReplaySweep)
            for ind in self.traces[0].indices:
                node = self._data_tensor.node_by_free_index(ind.name)
                nodes.append(node)
        else:
            mb_nodes = self._sample_subnet_nodes()
            if mb_nodes is None:
                return copy.deepcopy(self._data_tensor)

            nodes = list(mb_nodes)

        local_tree = TensorNetwork()
        local_tree.network = nx.subgraph(
            self._data_tensor.network, nodes
        ).copy()
        return local_tree

    def _sample_subnet_nodes(self) -> Optional[Sequence[NodeName]]:
        """
        Sample a connected subnet;
        return None when the whole network fits.
        """
        logger.debug("sampling substructures from %s", self._data_tensor)
        net_nodes = sorted(self._data_tensor.network.nodes, key=str)
        weights = [self._node_visited.get(node, 9999) for node in net_nodes]
        root_node = random.choices(net_nodes, weights=weights, k=1)[0]
        logger.debug("select root node %s", root_node)
        self._data_tensor.orthonormalize(root_node)
        if (
            len(self._data_tensor.network.nodes)
            <= self.config.sweep.subnet_size
        ):
            return None

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
                nodes = sorted(nodes, key=str)
        return nodes


class TraversalStructureSweep(StructureSweep):
    """Traverse the entire tree and do the structure refinement."""

    def __init__(
        self,
        config: SearchConfig,
        data_tensor: TensorNetwork,
        free_indices: List[Index],
    ):
        super().__init__(config, data_tensor, free_indices)

        # record visited index pairs
        self._visited_es: Set[Tuple[Sequence[Index], Sequence[Index]]] = set()

        icount = data_tensor.all_indices()
        inner_indices = [i for i, v in icount.items() if v == 2]

        self._next_edge: Optional[Index]
        if inner_indices:
            self._next_edge = inner_indices[0]
        else:
            self._next_edge = None

    def _get_local_structure(self) -> TensorNetwork:
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
        self._visited_es.add((tuple(u_inds), tuple(v_inds)))

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

        tree = TensorNetwork()
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
            if inds_parts not in self._visited_es:
                self._next_edge = ind
                break

        return tree

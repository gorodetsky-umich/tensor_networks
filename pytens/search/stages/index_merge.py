"""Search algorithm with index merge"""

import copy
import logging
import time
from functools import partial
from typing import List, Tuple

import numpy as np

from pytens.algs import TensorTrain, TreeNetwork
from pytens.search.configuration import (
    ClusterMethod,
    ReorderAlgo,
    ReshapeOption,
    SearchConfig,
)
from pytens.search.hierarchical.index_cluster import (
    CrossIndexCluster,
    IndexCluster,
    RandomIndexCluster,
    SVDIndexCluster,
    SVDNbrIndexCluster,
)
from pytens.search.hierarchical.types import HSearchState
from pytens.search.stages.base import SearchStage, StageRunParams
from pytens.search.stages.stage_runner import StageRunner
from pytens.search.utils import SearchResult
from pytens.types import Index, IndexMerge, IndexSplit, NodeName

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class IndexMergeStage(SearchStage):
    """Decorate the core search algorithm with index merging"""

    def __init__(self, config: SearchConfig):
        super().__init__(config)

        self._cluster = self._set_cluster()

    def run(self, runner: StageRunner, params: StageRunParams) -> SearchResult:
        """Search the optimized network structure with index merging."""
        st = params.state
        free_inds = st.network.free_indices()

        merge_ops = []
        if self._trigger_merge(len(free_inds), params.is_top):
            merge_ops, _ = self._to_lower_dim(st)

        params.merge_ops = merge_ops
        result = runner.run(params)

        # assert result.best_state is not None
        # for split_op in reversed(split_ops):
        #     result.best_state.network.split_index(split_op)

        return result

    def _trigger_merge(self, ind_cnt: int, is_top: bool) -> bool:
        """Whether to trigger the index merge operation before search"""
        return (
            (self._config.topdown.reshape_algo == ReshapeOption.CLUSTER)
            and (not is_top or self._config.topdown.merge_mode == "all")
            and ind_cnt > self._config.topdown.group_threshold
        )

    def _to_lower_dim(self, st: HSearchState):
        merge_start = time.time()
        list_of_index_sets = self._cluster.cluster(st.network, [])
        self._add_merge_time(time.time() - merge_start)

        best_net, best_ops = None, ([], [])
        for index_sets in list_of_index_sets:
            merge_ops, split_ops = _merge_ops_from_index_groups(index_sets)

            if len(st.network.network.nodes) == 1:
                best_ops = merge_ops, split_ops
                best_net = st.network
                break

            ok, tt = self._cross_to_tt(copy.deepcopy(st.network), merge_ops)
            logger.debug("after index merge, we get a tensor train %s", tt)
            if ok and (best_net is None or tt.cost() < best_net.cost()):
                best_net = tt
                best_ops = merge_ops, split_ops

        assert best_net is not None
        st.network = best_net
        return best_ops

    def _cross_to_tt(
        self, data_tensor: TreeNetwork, merge_ops: List[IndexMerge]
    ) -> Tuple[bool, TensorTrain]:
        """Handle index merging during preprocessing for cross results."""
        transform_start = time.time()

        # after we know how indices are merged, we create a permuted function
        new_indices = []

        end = data_tensor.end_nodes()[0]
        merge_op_key = partial(_merge_op_dist, net=data_tensor, end=end)
        index_key = partial(_index_to_end_distance, net=data_tensor, end=end)
        for mop in sorted(merge_ops, key=merge_op_key):
            # sort the indices before adding to the list according to positions
            new_indices.extend(sorted(mop.indices, key=index_key))

        # there exists some free indices are not merged
        for ind in data_tensor.free_indices():
            if ind not in new_indices:
                new_indices.append(ind)

        assert len(new_indices) == len(data_tensor.free_indices())
        # print("reorder the indices into", new_indices)

        reorder_start = time.time()
        if self._config.preprocess.reorder_algo == ReorderAlgo.SVD:
            ok = True
            eps = (
                self._config.engine.eps
                * self._config.preprocess.reorder_eps
                * 0.01
            )
            tt = data_tensor.reorder_by_svd(new_indices, eps)
        else:
            ok, tt = data_tensor.reorder_by_cross(
                new_indices,
                self._config.engine.eps * self._config.preprocess.reorder_eps,
                kickrank=5,
            )

        logger.debug("reorder time: %s", time.time() - reorder_start)
        logger.debug("after reordering")
        logger.debug(tt)
        logger.debug(merge_ops)

        self._add_merge_transform_time(time.time() - transform_start)
        return ok, tt

    def _set_cluster(self) -> IndexCluster:
        threshold = self._config.topdown.group_threshold
        cluster_method = {
            ClusterMethod.SVD: SVDIndexCluster(threshold),
            ClusterMethod.RAND: RandomIndexCluster(threshold),
            ClusterMethod.NBR: RandomIndexCluster(threshold, False),
            ClusterMethod.RAND_NBR: RandomIndexCluster(threshold, False),
            ClusterMethod.SVD_NBR: SVDNbrIndexCluster(threshold),
            ClusterMethod.CROSS: CrossIndexCluster(
                threshold, self._config.engine.eps * 0.1
            ),
        }.get(self._config.topdown.cluster_method)

        assert cluster_method is not None, (
            f"unknown cluster method: {self._config.topdown.cluster_method}"
        )
        return cluster_method

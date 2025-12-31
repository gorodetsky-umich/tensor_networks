import copy
from typing import List
import logging

import networkx as nx
import numpy as np

from pytens.algs import TreeNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.hierarchical.types import HSearchState, SubnetResult
from pytens.search.hierarchical.utils import DisjointSet
from pytens.search.stages.base import SearchStage, StageContext, StageRunParams
from pytens.search.stages.stage_runner import StageRunner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SubnetRefinementStage(SearchStage):
    def __init__(self, config: SearchConfig, subnet_runner: StageRunner):
        super().__init__(config)

        self._subnet_runner = subnet_runner

    def run(self, runner: StageRunner, params: StageRunParams):
        assert params.ctx is not None
        next_nets = self._get_next_nets(params.ctx)

        unused_delta = st.unused_delta + result.unused_delta**2
        best_st = HSearchState(st.free_indices, st.reshape_history, bn, 0)
        if len(next_nets) == 1 and not params.is_modified:
            best_st.unused_delta = unused_delta + remaining_delta**2
        else:
            # future work: it would be better to dynamically distribute errors
            # distribute delta equally to all subnets
            remaining_delta = remaining_delta / math.sqrt(len(next_nets))
            # enumerate nodes in the order of their scores
            for subnet in next_nets:
                self._search_for_subnet(best_st, remaining_delta, subnet)

            best_st.unused_delta = unused_delta

        return SearchResult(self._stats, best_st)

    def _get_next_nets(self, best_net: TreeNetwork) -> List[TreeNetwork]:
        """Get the next level nodes to optimize"""
        subgraph_nodes = DisjointSet()
        for node in best_net.network.nodes:
            subgraph_nodes.union(node, node)

        # for ac in to_splits(best_net):
        #     if ac not in best_st.past_actions and ac.reverse_edge is not None:
        #         subgraph_nodes.union(*ac.reverse_edge)

        subnets = []
        for group in sorted(subgraph_nodes.groups().values()):
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

        subnet_nodes = list(subnet.network.nodes)
        best_st.network.orthonormalize(subnet_nodes[0])
        ctx = StageContext(
            nodes=subnet_nodes,
            indices=subnet.free_indices(),
            is_modified=True,
        )
        optimize_res = self._subnet_runner.run(
            StageRunParams(best_st, remaining_delta, [], ctx)
        )
        if not optimize_res:
            return remaining_delta**2

        best_st.network = optimize_res.network
        best_sn_st = optimize_res.subnet_state

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
        best_st.network.replace_with(
            optimize_res.subnet, best_sn_st.network, best_sn_st.reshape_history
        )
        # print(best_st.network)
        # print(best_sn_st.network)
        # print("-"*20)
        best_st.free_indices = best_sn_st.free_indices
        best_st.reshape_history = best_sn_st.reshape_history
        return best_sn_st.unused_delta

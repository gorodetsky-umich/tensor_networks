"""Search algorithsm for tensor networks."""

import time
import itertools

import numpy as np

from pytens.algs import TensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.exhaustive import DFSSearch, BFSSearch
from pytens.search.partition import PartitionSearch
from pytens.search.hierarchical.top_down import TopDownSearch
from pytens.search.hierarchical.error_dist import BaseErrorDist, AlphaErrorDist
from pytens.search.utils import approx_error, SearchResult, reshape_indices


class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, config: SearchConfig):
        self.config = config

    def partition_search(self, net: TensorNetwork):
        """Perform an search with output-directed splits + constraint solve."""

        engine = PartitionSearch(self.config)
        result = engine.search(net)
        free_indices = net.free_indices()
        result.stats.cr_core = (
            float(np.prod([i.size for i in free_indices]))
            / result.best_network.cost()
        )
        result.stats.cr_start = net.cost() / result.best_network.cost()
        result.stats.re = float(
            np.linalg.norm(
                result.best_network.contract().value
                - net.contract().value
            )
            / np.linalg.norm(net.contract().value)
        )
        return result

    def dfs(
        self,
        net: TensorNetwork,
    ):
        """Perform an exhaustive enumeration with the DFS algorithm."""

        dfs_runner = DFSSearch(self.config)
        result = dfs_runner.run(net)
        end = time.time()

        result.stats.time = end - dfs_runner.start - dfs_runner.logging_time
        # result.best_network = dfs_runner.best_network
        result.stats.cr_core = (
            np.prod([i.size for i in net.free_indices()])
            / result.best_network.cost()
        )
        result.stats.cr_start = net.cost() / dfs_runner.best_network.cost()
        err = approx_error(dfs_runner.target_tensor, dfs_runner.best_network)
        result.stats.re = err

        return result

    def bfs(self, net: TensorNetwork):
        """Perform an exhaustive enumeration with the BFS algorithm."""

        bfs_runner = BFSSearch(self.config)
        result = bfs_runner.run(net)

        best_network = result.best_network
        # search_stats["best_network"] = best_network
        result.stats.cr_core = (
            np.prod([i.size for i in net.free_indices()]) / best_network.cost()
        )
        result.stats.cr_start = net.cost() / best_network.cost()
        err = approx_error(bfs_runner.target_tensor, best_network)
        result.stats.re = err

        return result

    def top_down(self, net: TensorNetwork):
        """Start point of a top down hierarchical search."""

        top_down_runner = TopDownSearch(self.config)
        start = time.time()
        best_network, best_st = top_down_runner.search(net, AlphaErrorDist(alpha=2.5))
        end = time.time()

        result = SearchResult()
        result.best_network = best_network
        result.stats.time = end - start
        result.stats.cr_start = net.cost() / best_network.cost()
        result.stats.cr_core = np.prod([i.size for i in net.free_indices()]) / best_network.cost()
        approx_tensor = best_network.contract()
        data_val = net.contract().value
        # print(best_st.reshape_history)
        reordered_indices, data_val = reshape_indices(best_st.reshape_history, net.free_indices(), data_val)
        # print(top_down_runner.reshape_info)
        # free_indices_name = [ind.name for ind in net.free_indices()]
        # approx_net = undo_reshape(top_down_runner.reshape_info, best_network)

        # print(reordered_indices)
        reordered_indices = itertools.chain(*reordered_indices)
        # print(reordered_indices)
        approx_tensor = approx_tensor.permute_by_name([ind.name for ind in reordered_indices])
        approx_val = approx_tensor.value
        # for reshape_op in top_down_runner.reshape_info:
        #     if isinstance(reshape_op, IndexSplit):
        #         net.reshape(index_split=[reshape_op])
        #     elif isinstance(reshape_op, IndexMerge):
        #         net.reshape(index_merge=[reshape_op])
        # print(best_network.free_indices())
        # print(net.free_indices())
        
        result.stats.re = float(np.linalg.norm(approx_val - data_val) / np.linalg.norm(data_val))
        return result
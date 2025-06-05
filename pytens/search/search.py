"""Search algorithsm for tensor networks."""

from typing import Optional
import time
import itertools

import numpy as np

from pytens.algs import TreeNetwork
from pytens.cross.cross import TensorFunc
from pytens.search.configuration import SearchConfig
from pytens.search.exhaustive import DFSSearch, BFSSearch
from pytens.search.partition import PartitionSearch
from pytens.search.hierarchical.top_down import TopDownSearch
from pytens.search.hierarchical.error_dist import AlphaErrorDist
from pytens.search.utils import approx_error, SearchResult, reshape_indices


class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, config: SearchConfig):
        self.config = config

    def partition_search(
        self, net: TreeNetwork, tensor_func: Optional[TensorFunc] = None
    ):
        """Perform an search with output-directed splits + constraint solve."""

        engine = PartitionSearch(self.config)
        for result in engine.search(net, tensor_func=tensor_func):
            assert result.best_network is not None

            free_indices = net.free_indices()
            unopt_size = float(np.prod([i.size for i in free_indices]))
            best_size = result.best_network.cost()
            best_val = result.best_network.contract().value
            net_val = net.contract().value
            net_norm = np.linalg.norm(net_val)
            result.stats.cr_core = unopt_size / best_size
            result.stats.cr_start = net.cost() / best_size
            if net_norm != 0:
                result.stats.re = float(np.linalg.norm(best_val) / net_norm)
            return result

    def dfs(
        self,
        net: TreeNetwork,
    ):
        """Perform an exhaustive enumeration with the DFS algorithm."""

        dfs_runner = DFSSearch(self.config)
        result = dfs_runner.run(net)
        assert result.best_network is not None
        end = time.time()

        result.stats.time = end - dfs_runner.start - dfs_runner.logging_time
        # result.best_network = dfs_runner.best_network
        unopt_size = float(np.prod([i.size for i in net.free_indices()]))
        result.stats.cr_core = unopt_size / result.best_network.cost()
        result.stats.cr_start = net.cost() / dfs_runner.best_network.cost()
        err = approx_error(dfs_runner.target_tensor, dfs_runner.best_network)
        result.stats.re = err

        return result

    def bfs(self, net: TreeNetwork):
        """Perform an exhaustive enumeration with the BFS algorithm."""

        bfs_runner = BFSSearch(self.config)
        result = bfs_runner.run(net)
        best_network = result.best_network
        assert best_network is not None

        # search_stats["best_network"] = best_network
        unopt_size = np.prod([i.size for i in net.free_indices()])
        result.stats.cr_core = float(unopt_size) / best_network.cost()
        result.stats.cr_start = net.cost() / best_network.cost()
        err = approx_error(bfs_runner.target_tensor, best_network)
        result.stats.re = err

        return result

    def top_down(self, net: TreeNetwork):
        """Start point of a top down hierarchical search."""

        top_down_runner = TopDownSearch(self.config)
        start = time.time()
        if self.config.topdown.random_algorithm == "random":
            error_dist = AlphaErrorDist(alpha=self.config.topdown.alpha)
            best_network, best_st = top_down_runner.search(net, error_dist)
        else:
            raise ValueError("Random search algorithm not implemented yet.")
        end = time.time()

        best_network.compress()
        result = SearchResult()
        result.best_network = best_network
        result.stats.time = end - start
        result.stats.cr_start = net.cost() / best_network.cost()
        unopt_size = float(np.prod([i.size for i in net.free_indices()]))
        result.stats.cr_core = unopt_size / best_network.cost()
        approx_tensor = best_network.contract()
        data_val = net.contract().value
        # print(best_st.network)
        # print(best_st.reshape_history)
        reordered_indices, data_val = reshape_indices(
            best_st.reshape_history, net.free_indices(), data_val
        )
        # print(best_st.reshape_history)
        # free_indices_name = [ind.name for ind in net.free_indices()]
        # approx_net = undo_reshape(top_down_runner.reshape_info, best_network)

        # print(reordered_indices)
        reordered_indices = list(itertools.chain(*reordered_indices))
        # print(list(reordered_indices))
        # print(approx_tensor.indices)
        # print(approx_tensor.value.shape)
        approx_tensor = approx_tensor.permute_by_name(
            [ind.name for ind in reordered_indices]
        )
        approx_val = approx_tensor.value
        # for reshape_op in top_down_runner.reshape_info:
        #     if isinstance(reshape_op, IndexSplit):
        #         net.reshape(index_split=[reshape_op])
        #     elif isinstance(reshape_op, IndexMerge):
        #         net.reshape(index_merge=[reshape_op])
        # print(best_network.free_indices())
        # print(net.free_indices())

        result.stats.re = float(
            np.linalg.norm(approx_val - data_val) / np.linalg.norm(data_val)
        )
        return result

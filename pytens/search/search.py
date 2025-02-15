"""Search algorithsm for tensor networks."""

import time

import numpy as np

from pytens.algs import TensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.exhaustive import DFSSearch, BFSSearch
from pytens.search.partition import PartitionSearch
from pytens.search.utils import approx_error


class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, config: SearchConfig):
        self.config = config

    def partition_search(self, net: TensorNetwork):
        """Perform an search with output-directed splits + constraint solve."""

        engine = PartitionSearch(self.config)
        stats = engine.search(net)

        # clear up the temporary storage
        return stats

    def dfs(
        self,
        net: TensorNetwork,
    ):
        """Perform an exhaustive enumeration with the DFS algorithm."""

        dfs_runner = DFSSearch(self.config)
        search_stats = dfs_runner.run(net)
        end = time.time()

        search_stats["time"] = end - dfs_runner.start - dfs_runner.logging_time
        search_stats["best_network"] = dfs_runner.best_network
        search_stats["cr_core"] = (
            np.prod([i.size for i in net.free_indices()])
            / dfs_runner.best_network.cost()
        )
        search_stats["cr_start"] = net.cost() / dfs_runner.best_network.cost()
        err = approx_error(dfs_runner.target_tensor, dfs_runner.best_network)
        search_stats["reconstruction_error"] = err

        return search_stats

    def bfs(self, net: TensorNetwork):
        """Perform an exhaustive enumeration with the BFS algorithm."""

        bfs_runner = BFSSearch(self.config)
        search_stats = bfs_runner.run(net)

        best_network = bfs_runner.best_network
        search_stats["best_network"] = best_network
        search_stats["cr_core"] = (
            np.prod([i.size for i in net.free_indices()]) / best_network.cost()
        )
        search_stats["cr_start"] = net.cost() / best_network.cost()
        err = approx_error(bfs_runner.target_tensor, best_network)
        search_stats["reconstruction_error"] = err

        return search_stats

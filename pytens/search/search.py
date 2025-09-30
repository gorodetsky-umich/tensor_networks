"""Search algorithsm for tensor networks."""

import itertools
import time

import numpy as np

from pytens.algs import TreeNetwork
from pytens.cross.cross import TensorFunc
from pytens.search.configuration import SearchConfig
from pytens.search.exhaustive import BFSSearch, DFSSearch
from pytens.search.hierarchical.error_dist import AlphaErrorDist
from pytens.search.hierarchical.top_down import TopDownSearch
from pytens.search.partition import PartitionSearch
from pytens.search.state import SearchState
from pytens.search.utils import (
    DataTensor,
    SearchResult,
    approx_error,
    reshape_indices,
    rtol,
    unravel_indices,
    reshape_func,
)
from pytens.search.hierarchical.types import TopDownSearchResult


class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, config: SearchConfig):
        self.config = config

    def partition_search(self, data_tensor: DataTensor):
        """Perform an search with output-directed splits + constraint solve."""

        engine = PartitionSearch(self.config)
        result = engine.search(data_tensor)
        assert result.best_state is not None

        free_indices = data_tensor.free_indices()
        unopt_size = float(np.prod([i.size for i in free_indices]))
        best_size = result.best_state.network.cost()

        if isinstance(data_tensor, TreeNetwork):
            best_val = result.best_state.network.contract().value
            net_val = data_tensor.contract().value
            start_cost = data_tensor.cost()
        elif isinstance(data_tensor, TensorFunc):
            sizes = [ind.size for ind in data_tensor.indices]
            val_size = 10000
            validation = [np.random.randint(i, size=val_size) for i in sizes]
            validation = np.stack(validation, axis=-1)
            net_val = data_tensor(validation)
            best_val = result.best_state.network.evaluate(result.best_state.network.free_indices(), validation)
            start_cost = np.prod(sizes)
        else:
            raise TypeError("unknown data tensor type")

        result.stats.re_f = rtol(net_val, best_val, "F")
        result.stats.re_max = rtol(net_val, best_val, "M")
        result.stats.cr_core = unopt_size / best_size
        result.stats.cr_start = float(start_cost / best_size)
        return result

    def dfs(
        self,
        net: TreeNetwork,
    ):
        """Perform an exhaustive enumeration with the DFS algorithm."""

        dfs_runner = DFSSearch(self.config)
        result = dfs_runner.run(net)
        assert result.best_state is not None
        end = time.time()

        result.stats.search_start = dfs_runner.start
        result.stats.search_end = end - dfs_runner.logging_time
        # result.best_network = dfs_runner.best_network
        unopt_size = float(np.prod([i.size for i in net.free_indices()]))
        best_network = result.best_state.network
        best_cost = best_network.cost()
        result.stats.cr_core = unopt_size / best_cost
        result.stats.cr_start = net.cost() / best_cost
        err = approx_error(dfs_runner.target_tensor, best_network)
        result.stats.re_f = err

        return result

    def bfs(self, net: TreeNetwork):
        """Perform an exhaustive enumeration with the BFS algorithm."""

        bfs_runner = BFSSearch(self.config)
        result = bfs_runner.run(net)
        assert result.best_state is not None
        best_network = result.best_state.network
        assert best_network is not None

        # search_stats["best_network"] = best_network
        unopt_size = np.prod([i.size for i in net.free_indices()])
        result.stats.cr_core = float(unopt_size) / best_network.cost()
        result.stats.cr_start = net.cost() / best_network.cost()
        err = approx_error(bfs_runner.target_tensor, best_network)
        result.stats.re_f = err

        return result

    def top_down(self, data_tensor: DataTensor):
        """Start point of a top down hierarchical search."""

        top_down_runner = TopDownSearch(self.config)
        top_down_runner.error_dist = AlphaErrorDist(alpha=self.config.topdown.alpha)
        start = time.time()
        if self.config.topdown.random_algorithm == "random":
            best_st = top_down_runner.search(data_tensor)
        else:
            raise ValueError("Random search algorithm not implemented yet.")
        end = time.time()

        assert best_st is not None
        best_st.network.compress()
        best_network = best_st.network
        result = TopDownSearchResult()
        result.best_state = SearchState(best_network, 0)
        result.stats = top_down_runner.stats
        result.stats.search_start = start
        result.stats.search_end = end

        if isinstance(data_tensor, TreeNetwork):
            free_indices = data_tensor.free_indices()
            unopt_size = float(np.prod([i.size for i in free_indices]))
            init_size = data_tensor.cost()
            data_val = data_tensor.contract().value
            approx_val = best_network.contract().value
            reordered_indices, data_val = reshape_indices(
                best_st.reshape_history, free_indices, data_val
            )
            reordered_indices = list(itertools.chain(*reordered_indices))
            approx_val = approx_val.transpose(
                [free_indices.index(ind) for ind in reordered_indices]
            )
        elif isinstance(data_tensor, TensorFunc):
            sample_size = 10000
            free_indices = data_tensor.indices
            unopt_size = float(np.prod([i.size for i in free_indices]))
            init_size = unopt_size

            valid = []
            for ind in best_network.free_indices():
                valid.append(np.random.randint(0, ind.size, size=sample_size))
            # valid = np.stack(np.unravel_index(np.arange(unopt_size).astype(int), [int(i.size) for i in best_network.free_indices()]), axis=-1)
            valid = np.stack(valid, axis=-1)
            # indices, new_valid = unravel_indices(best_st.reshape_history, free_indices, valid)
            # approx_val = best_network.evaluate(indices, new_valid)
            # print(21**3 / best_network.cost())
            # raise Exception("end")
            approx_val = best_network.evaluate(best_network.free_indices(), valid)
            # print(np.allclose(approx_val, best_network.contract().value.reshape(-1)))
            # print(np.where(approx_val != best_network.contract().value.reshape(-1)))
            # approx_val = best_network.contract().value.reshape(-1)
            # data_val = data_tensor(valid)
            best_indices = best_network.free_indices()
            reshaped_func = reshape_func(best_st.reshape_history, free_indices, data_tensor)
            perm = [best_indices.index(ind) for ind in reshaped_func.indices]
            result.valid_set = valid
            result.valid_indices = best_indices
            result.reshape_history = best_st.reshape_history
            result.init_splits = top_down_runner.init_splits
            data_val = reshaped_func(valid[:, perm])
        else:
            raise TypeError("unsupported data tensor type")

        result.stats.cr_start = init_size / best_network.cost()
        result.stats.cr_core = unopt_size / best_network.cost()
        result.stats.re_f = float(
            np.linalg.norm(approx_val - data_val) / np.linalg.norm(data_val)
        )
        return result

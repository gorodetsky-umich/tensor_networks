"""Search algorithsm for tensor networks."""

import time
from abc import abstractmethod

import numpy as np

from pytens.algs import TreeNetwork
from pytens.cross.funcs import CountingFunc, TensorFunc
from pytens.cross.runner import (
    CrossRunner,
    FTTCrossRunner,
    HTCrossRunner,
    TnTorchCrossRunner,
    TTCrossRunner,
)
from pytens.search.configuration import (
    ClusterMethod,
    InitStructType,
    SearchConfig,
)
from pytens.search.exhaustive import BFSSearch, DFSSearch
from pytens.search.hierarchical.error_dist import AlphaErrorDist
from pytens.search.hierarchical.index_cluster import (
    RandomIndexCluster,
    SVDIndexCluster,
)
from pytens.search.hierarchical.top_down import (
    BlackBoxTopDownSearch,
    TopDownSearch,
    WhiteBoxTopDownSearch,
)
from pytens.search.hierarchical.types import HSearchState, TopDownSearchResult
from pytens.search.partition import PartitionSearch
from pytens.search.state import SearchState
from pytens.search.utils import (
    DataTensor,
    approx_error,
    reshape_func,
    reshape_indices,
    rtol,
)


class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, config: SearchConfig):
        self.config = config

    def partition_search(self, data_tensor: DataTensor):
        """Perform an search with output-directed splits + constraint solve."""

        engine = PartitionSearch(self.config)
        result = engine.search(data_tensor, [])
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
            best_val = result.best_state.network.evaluate(
                result.best_state.network.free_indices(), validation
            )
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


class TopDownSearchEngine(SearchEngine):
    """The search engine with the top down search strategy."""

    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self._top_down_runner = TopDownSearch(config)

    def top_down(self) -> TopDownSearchResult:
        """Start point of a top down hierarchical search."""
        self._initialize()
        self._set_cluster()
        self._top_down_runner.error_dist = AlphaErrorDist(
            alpha=self.config.topdown.alpha
        )

        start = time.time()
        best_st = self._top_down_runner.search()
        end = time.time()

        assert best_st is not None
        best_st.network.compress()
        best_network = best_st.network
        result = TopDownSearchResult()
        result.best_state = SearchState(best_network, 0)
        result.stats = self._top_down_runner.stats
        result.stats.search_start = start
        result.stats.search_end = end

        self._collect_stats(result, best_st)
        return result

    def _set_cluster(self):
        threshold = self.config.topdown.group_threshold
        cluster_method = {
            ClusterMethod.SVD: SVDIndexCluster(threshold),
            ClusterMethod.RAND: RandomIndexCluster(threshold),
            ClusterMethod.NBR: RandomIndexCluster(threshold, False),
        }.get(self.config.topdown.cluster_method, ClusterMethod(threshold))
        self._top_down_runner.set_cluster(cluster_method)

    @abstractmethod
    def _initialize(self):
        raise NotImplementedError

    @abstractmethod
    def _collect_stats(
        self, result: TopDownSearchResult, best_st: HSearchState
    ):
        raise NotImplementedError


class WhiteBoxTopDownSearchEngine(TopDownSearchEngine):
    """Search engine for the white box tensors."""

    def __init__(self, config, data_tensor: TreeNetwork):
        super().__init__(config)
        self._data_tensor = data_tensor

    def _initialize(self):
        self._top_down_runner = WhiteBoxTopDownSearch(
            self.config, self._data_tensor
        )

    def _collect_stats(
        self, result: TopDownSearchResult, best_st: HSearchState
    ):
        free_indices = self._data_tensor.free_indices()
        best_network = best_st.network
        unopt_size = float(np.prod([i.size for i in free_indices]))
        init_size = self._data_tensor.cost()
        data_val = self._data_tensor.contract().value
        approx_val = best_network.contract().value
        reordered_indices, data_val = reshape_indices(
            best_st.reshape_history, free_indices, data_val
        )
        approx_val = approx_val.transpose(
            [free_indices.index(ind) for ind in reordered_indices]
        )
        result.stats.cr_start = init_size / best_network.cost()
        result.stats.cr_core = unopt_size / best_network.cost()
        result.stats.re_f = float(
            np.linalg.norm(approx_val - data_val) / np.linalg.norm(data_val)
        )


class BlackBoxTopDownSearchEngine(TopDownSearchEngine):
    """Search engine for the black box functions."""

    def __init__(self, config, data_tensor: CountingFunc):
        super().__init__(config)
        self._data_tensor = data_tensor

    def _initialize(self):
        top_down_runner = BlackBoxTopDownSearch(self.config, self._data_tensor)

        cross_runner = {
            InitStructType.TT: TTCrossRunner(),
            InitStructType.TT_CROSS: TnTorchCrossRunner(),
            InitStructType.HT: HTCrossRunner(),
            InitStructType.FTT: FTTCrossRunner(),
        }.get(self.config.cross.init_struct, CrossRunner())

        top_down_runner.set_cross_runner(cross_runner)
        self._top_down_runner = top_down_runner

    def _collect_stats(
        self, result: TopDownSearchResult, best_st: HSearchState
    ):
        sample_size = 10000
        best_network = best_st.network
        free_indices = self._data_tensor.indices
        unopt_size = float(np.prod([i.size for i in free_indices]))
        init_size = unopt_size

        valid = []
        for ind in best_network.free_indices():
            valid.append(np.random.randint(0, ind.size, size=sample_size))
        valid = np.stack(valid, axis=-1)

        approx_val = best_network.evaluate(best_network.free_indices(), valid)
        best_indices = best_network.free_indices()
        reshaped_func = reshape_func(
            best_st.reshape_history, self._data_tensor
        )
        perm = [best_indices.index(ind) for ind in reshaped_func.indices]
        result.valid_set = valid
        result.valid_indices = best_indices
        result.reshape_history = best_st.reshape_history
        result.init_splits = self._top_down_runner.init_splits
        data_val = reshaped_func(valid[:, perm])

        result.stats.cr_start = init_size / best_network.cost()
        result.stats.cr_core = unopt_size / best_network.cost()
        result.stats.re_f = float(
            np.linalg.norm(approx_val - data_val) / np.linalg.norm(data_val)
        )

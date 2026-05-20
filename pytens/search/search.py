"""Search algorithsm for tensor networks."""

import copy
import logging
import time
from abc import abstractmethod
from typing import List, Optional

import numpy as np

from pytens.algs import TreeNetwork
from pytens.cross.func_interface import CachedFunc, TensorFunc
from pytens.cross.runner import (
    HTCrossRunner,
    TTCrossRunner,
    TuckerCrossRunner,
)
from pytens.search.algs.exhaustive import BFSSearch, DFSSearch
from pytens.search.algs.partition import PartitionSearch
from pytens.search.configuration import (
    InitStructType,
    SearchConfig,
)
from pytens.search.hierarchical.error_dist import AlphaErrorDist
from pytens.search.hierarchical.top_down import (
    BlackBoxTopDownSearch,
    TopDownSearch,
    WhiteBoxTopDownSearch,
)
from pytens.search.hierarchical.types import (
    HSearchState,
    Replay,
    TopDownSearchResult,
)
from pytens.search.state import SearchState
from pytens.search.types import SearchContext
from pytens.search.utils import (
    SearchResult,
    approx_error,
    reshape_indices,
    rtol,
    unravel_indices,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    def partition_search(self, data_tensor: TreeNetwork) -> SearchResult:
        """Perform an search with output-directed splits + constraint solve."""

        engine = PartitionSearch(self.config, data_tensor)
        result: SearchResult = engine.search(SearchContext())
        assert result.best_state is not None

        free_indices = data_tensor.free_indices()
        unopt_size = float(np.prod([i.size for i in free_indices]))
        best_size = result.best_state.network.cost()

        if isinstance(data_tensor, TreeNetwork):
            best_tensor = result.best_state.network.contract()
            best_val = best_tensor.value
            perm = [
                best_tensor.indices.index(ind)
                for ind in data_tensor.free_indices()
            ]
            best_val = best_val.transpose(perm)
            net_val = data_tensor.contract().value

            start_cost = data_tensor.cost()
        elif isinstance(data_tensor, TensorFunc):
            sizes = [ind.size for ind in data_tensor.indices]
            net_val = np.array(0)
            best_val = np.array(0)
            start_cost = np.prod(sizes)
        else:
            raise TypeError("unknown data tensor type")

        result.stats.re_f = rtol(net_val, best_val, "F")
        result.stats.re_max = rtol(net_val, best_val, "M")
        result.stats.cr_core = unopt_size / best_size
        result.stats.cr_start = float(start_cost / best_size)
        return result

    def dfs(self, net: TreeNetwork) -> SearchResult:
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

        assert dfs_runner.target_tensor is not None
        err = approx_error(dfs_runner.target_tensor, best_network)
        result.stats.re_f = err

        return result

    def bfs(self, net: TreeNetwork) -> SearchResult:
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

        assert bfs_runner.target_tensor is not None
        err = approx_error(bfs_runner.target_tensor, best_network)
        result.stats.re_f = err

        return result


class TopDownSearchEngine(SearchEngine):
    """The search engine with the top down search strategy."""

    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self._top_down_runner: TopDownSearch

    def top_down(
        self, replay_traces: Optional[List[Replay]] = None
    ) -> TopDownSearchResult:
        """Start point of a top down hierarchical search."""
        self._initialize()
        self._top_down_runner.error_dist = AlphaErrorDist(
            alpha=self.config.topdown.alpha
        )

        start = time.time()
        best_st = self._top_down_runner.search(replay_traces=replay_traces)
        end = time.time()

        assert best_st is not None
        best_st.network.compress()
        best_network = best_st.network

        result = TopDownSearchResult()
        result.best_state = SearchState(best_network, 0)
        result.stats = self._top_down_runner.stats
        result.stats.search_start = start
        result.stats.search_end = end
        result.replay_traces = best_st.replay_traces

        if self.config.output.collect_stats:
            self._collect_stats(result, best_st)
        return result

    @abstractmethod
    def _initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _collect_stats(
        self, result: TopDownSearchResult, best_st: HSearchState
    ) -> None:
        raise NotImplementedError


class WhiteBoxTopDownSearchEngine(TopDownSearchEngine):
    """Search engine for the white box tensors."""

    def __init__(self, config: SearchConfig, data_tensor: TreeNetwork) -> None:
        super().__init__(config)
        self._data_tensor = data_tensor

    def _initialize(self) -> None:
        self._top_down_runner = WhiteBoxTopDownSearch(
            self.config, copy.deepcopy(self._data_tensor)
        )

    def _collect_stats(
        self, result: TopDownSearchResult, best_st: HSearchState
    ) -> None:
        free_indices = self._data_tensor.free_indices()
        best_network = best_st.network
        unopt_size = float(np.prod([i.size for i in free_indices]))
        init_size = self._data_tensor.cost()
        data_val = self._data_tensor.contract().value
        approx_val = best_network.contract().value
        best_indices = best_network.free_indices()
        reordered_indices, data_val = reshape_indices(
            best_st.reshape_history, free_indices, data_val
        )
        approx_val = approx_val.transpose(
            [best_indices.index(ind) for ind in reordered_indices]
        )
        result.stats.cr_start = init_size / best_network.cost()
        result.stats.cr_core = unopt_size / best_network.cost()
        result.stats.re_f = float(
            np.linalg.norm(approx_val - data_val) / np.linalg.norm(data_val)
        )


class BlackBoxTopDownSearchEngine(TopDownSearchEngine):
    """Search engine for the black box functions."""

    def __init__(
        self,
        config: SearchConfig,
        data_tensor: CachedFunc,
        validation_set: np.ndarray,
    ) -> None:
        super().__init__(config)
        self._data_tensor = data_tensor
        self._validation_set = validation_set

    def _initialize(self) -> None:
        top_down_runner = BlackBoxTopDownSearch(
            self.config, copy.deepcopy(self._data_tensor), self._validation_set
        )

        cross_runner = {
            InitStructType.TT: TTCrossRunner(),
            InitStructType.HT: HTCrossRunner(),
            InitStructType.TUCKER: TuckerCrossRunner(),
        }.get(self.config.cross.init_struct)
        assert cross_runner is not None, (
            f"Unknown init struct: {self.config.cross.init_struct}"
        )

        top_down_runner.set_cross_runner(cross_runner)
        self._top_down_runner = top_down_runner

    def _collect_stats(
        self, result: TopDownSearchResult, best_st: HSearchState
    ) -> None:
        best_network = best_st.network
        logger.debug("result best network %s", best_network)
        free_indices = self._data_tensor.indices
        unopt_size = float(np.prod([i.size for i in free_indices]))
        init_size = unopt_size

        data_val = self._data_tensor(self._validation_set)

        logger.debug("reshape history: %s", best_st.reshape_history)
        new_inds, new_valid = unravel_indices(
            best_st.reshape_history, free_indices, self._validation_set
        )
        best_indices = best_network.free_indices()
        perm = [new_inds.index(ind) for ind in best_indices]
        approx_val = best_network.evaluate(best_indices, new_valid[:, perm])

        result.valid_set = self._validation_set
        result.valid_indices = best_indices
        result.reshape_history = best_st.reshape_history
        result.init_splits = self._top_down_runner.init_splits
        # data_val = reshaped_func(valid[:, perm])

        result.stats.cr_start = init_size / best_network.cost()
        result.stats.cr_core = unopt_size / best_network.cost()
        result.stats.re_f = float(
            np.linalg.norm(approx_val - data_val) / np.linalg.norm(data_val)
        )

        # data = best_network.contract()
        # # print(data.indices)
        # np.save(f"{self.config.output.output_dir}/result.npy", data.value)

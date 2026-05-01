"""Exhaustive search of tensor network structures."""

from abc import abstractmethod
from typing import List, Optional, Set
import time
import copy

from pytens.algs import TreeNetwork, Tensor
from pytens.search.configuration import SearchConfig
from pytens.search.types import SearchContext
from pytens.search.hierarchical.types import HSearchState
from pytens.search.state import SearchState
from pytens.search.utils import log_stats, SearchResult, SearchStats


class ExhaustiveSearch:
    """Base class for exhaustive search"""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.stats = SearchStats()

        self.delta = 0.0
        self.target_tensor: Tensor
        self.best_network: TreeNetwork

        self.start = 0.0
        self.logging_time = 0.0

    @abstractmethod
    def search(self, st: HSearchState, context: SearchContext) -> SearchResult:
        raise NotImplementedError


class BFSSearch(ExhaustiveSearch):
    """Implementation of BFS search."""

    def _add_wodup(
        self,
        best_network: Optional[TreeNetwork],
        new_st: SearchState,
        worked: set,
        worklist: List[SearchState],
    ) -> TreeNetwork:
        """Add a network to a worked set to remove duplicates."""
        if best_network is None or best_network.cost() > new_st.network.cost():
            best_network = new_st.network

        h = new_st.network.canonical_structure(
            consider_ranks=self.config.heuristics.prune_by_ranks
        )
        if self.config.heuristics.prune_duplicates:
            if h in worked:
                return best_network

            worked.add(h)

        if len(new_st.past_actions) < self.config.engine.max_ops:
            worklist.append(new_st)

        return best_network

    def run(self, net: TreeNetwork) -> SearchResult:
        """Execute the BFS search algorithm on the given tensor network"""

        self.target_tensor = net.contract()
        logging_time = 0.0
        start = time.time()

        network = copy.deepcopy(net)
        delta = self.config.engine.eps * net.norm()

        worked = set()
        worklist = [SearchState(network, delta)]
        worked.add(network.canonical_structure())
        best_network = None
        count = 0

        while len(worklist) != 0:
            st = worklist.pop(0)

            if (
                self.config.engine.timeout is not None
                and time.time() - start >= self.config.engine.timeout
            ):
                break

            for ac in st.get_legal_actions(
                index_actions=self.config.synthesizer.action_type == "osplit"
            ):
                new_st = st.take_action(ac)
                if new_st is None:
                    continue

                if self.config.heuristics.prune_full_rank:
                    continue

                ts = time.time() - start - logging_time
                best_network = self._add_wodup(
                    best_network,
                    new_st,
                    worked,
                    worklist,
                )
                count += 1

                verbose_start = time.time()
                if self.config.engine.verbose:
                    log_stats(
                        self.stats,
                        self.target_tensor,
                        ts,
                        new_st,
                        best_network,
                    )
                verbose_end = time.time()
                logging_time += verbose_end - verbose_start

        end = time.time()

        self.stats.search_start = start
        self.stats.search_end = end - logging_time
        self.stats.count = count

        result = SearchResult()
        if best_network is not None:
            result.best_state = SearchState(best_network, 0)
        result.stats = self.stats
        return result

    def search(self, st: HSearchState, context: SearchContext) -> SearchResult:
        return self.run(st.network)


class DFSSearch(ExhaustiveSearch):
    """Implementation of DFS search."""

    def log(self, new_st: SearchState) -> None:
        """Log statistics during search."""
        ts = time.time() - self.start - self.logging_time
        verbose_start = time.time()
        if self.config.engine.verbose:
            log_stats(
                self.stats,
                self.target_tensor,
                ts,
                new_st,
                self.best_network,
            )
        verbose_end = time.time()
        self.logging_time += verbose_end - verbose_start

    def dfs(self, worked: Set[int], curr_st: SearchState) -> None:
        """Implementation of the DFS recursion."""
        self.stats.count += 1
        used_ops = len(curr_st.past_actions)
        if used_ops >= self.config.engine.max_ops:
            return

        if (
            self.config.engine.timeout is not None
            and time.time() - self.start > self.config.engine.timeout
        ):
            return

        for ac in curr_st.get_legal_actions(
            index_actions=self.config.synthesizer.action_type == "osplit"
        ):
            if used_ops + 1 >= self.config.engine.max_ops:
                split_errors = 1
            else:
                split_errors = self.config.rank_search.error_split_stepsize

            config = copy.deepcopy(self.config)
            config.rank_search.error_split_stepsize = split_errors

            new_st = curr_st.take_action(ac)
            if new_st is None:
                continue

            if self.config.heuristics.prune_full_rank:
                continue

            if new_st.network.cost() < self.best_network.cost():
                self.best_network = new_st.network

            self.log(new_st)

            if self.config.heuristics.prune_duplicates:
                h = new_st.network.canonical_structure(
                    consider_ranks=self.config.heuristics.prune_by_ranks
                )
                # print(h)
                if h in worked:
                    return

                worked.add(h)

            if used_ops + 1 >= self.config.engine.max_ops:
                # print("max op")
                return

            self.dfs(worked, new_st)

    def run(self, net: TreeNetwork) -> SearchResult:
        """Run a DFS search from the given tensor network."""

        self.target_tensor = net.contract()
        self.delta = self.config.engine.eps * net.norm()
        self.best_network = net

        self.logging_time = 0.0
        self.start = time.time()

        # network = copy.deepcopy(net)
        worked: Set[int] = set()
        self.dfs(worked, SearchState(net, self.delta))

        result = SearchResult()
        result.stats = self.stats
        result.best_state = SearchState(self.best_network, 0)
        return result

    def search(self, st: HSearchState, context: SearchContext) -> SearchResult:
        return self.run(st.network)

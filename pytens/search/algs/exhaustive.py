"""Exhaustive search of tensor network structures."""

from typing import List, Optional
import time
import copy

from pytens.algs import TreeNetwork
from pytens.search.algs.base import SearchAlgo
from pytens.search.configuration import SearchConfig
from pytens.search.state import SearchState
from pytens.search.utils import log_stats, SearchStats, SearchResult


class ExhaustiveSearch(SearchAlgo):
    """Base class for exhaustive search"""

    def __init__(self, config: SearchConfig):
        self.config = config

        self.delta = 0
        self.target_tensor = None
        self.best_network = None

        self.start = 0
        self.logging_time = 0
        self.search_stats = SearchStats()


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
        # new_net.draw()
        # plt.show()
        # new_net_hash = hash(new_net)
        # if new_net_hash not in worked:
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
        logging_time = 0
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
                # plt.subplot(2,1,1)
                # st.network.draw()
                new_st = st.take_action(ac)
                if new_st is None:
                    continue
                # plt.subplot(2,1,2)
                # new_st.network.draw()
                # plt.show()
                if self.config.heuristics.prune_full_rank and new_st.is_noop:
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
                        self.search_stats,
                        self.target_tensor,
                        ts,
                        new_st,
                        best_network,
                    )
                verbose_end = time.time()
                logging_time += verbose_end - verbose_start

        end = time.time()

        self.search_stats.search_start = start
        self.search_stats.search_end = end - logging_time
        self.search_stats.count = count

        result = SearchResult()
        if best_network is not None:
            result.best_state = SearchState(best_network, 0)
        result.stats = self.search_stats
        return result


class DFSSearch(ExhaustiveSearch):
    """Implementation of DFS search."""

    def log(self, new_st: SearchState):
        """Log statistics during search."""
        ts = time.time() - self.start - self.logging_time
        verbose_start = time.time()
        if self.config.engine.verbose:
            log_stats(
                self.search_stats,
                self.target_tensor,
                ts,
                new_st,
                self.best_network,
            )
        verbose_end = time.time()
        self.logging_time += verbose_end - verbose_start

    def dfs(self, worked: set, curr_st: SearchState):
        """Implementation of the DFS recursion."""
        self.search_stats.count += 1
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
            # greedy = False

            if self.config.heuristics.prune_full_rank and new_st.is_noop:
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

            # best_before = best_network.cost()
            self.dfs(worked, new_st)
            # best_after = best_network.cost()
            # if best_before == best_after:
            #     # greedy = True
            #     break

    def run(self, net: TreeNetwork) -> SearchResult:
        """Run a DFS search from the given tensor network."""

        self.target_tensor = net.contract()
        self.delta = self.config.engine.eps * net.norm()
        self.best_network = net

        self.logging_time = 0
        self.start = time.time()

        # network = copy.deepcopy(net)
        worked = set()
        self.dfs(worked, SearchState(net, self.delta))

        result = SearchResult()
        result.stats = self.search_stats
        result.best_state = SearchState(self.best_network, 0)
        return result

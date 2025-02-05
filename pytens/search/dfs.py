"""DFS search of tensor network structures."""

import time
import copy
from pytens.algs import TensorNetwork
from pytens.search.state import SearchState
from pytens.search.utils import log_stats, EMPTY_SEARCH_STATS


class DFSSearch:
    """Implementation of DFS search."""

    def __init__(self, params):
        self.params = params

        self.delta = 0
        self.target_tensor = None
        self.best_network = None

        self.start = 0
        self.logging_time = 0
        self.search_stats = EMPTY_SEARCH_STATS

    def log(self, new_st: SearchState):
        """Log statistics during search."""
        ts = time.time() - self.start - self.logging_time
        verbose_start = time.time()
        if self.params["verbose"]:
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
        self.search_stats["count"] += 1
        used_ops = len(curr_st.past_actions)
        if used_ops >= self.params["max_ops"]:
            # print("max op")
            return

        if (
            self.params["timeout"] is not None
            and time.time() - self.start > self.params["timeout"]
        ):
            return

        for ac in curr_st.get_legal_actions(
            index_actions=self.params["partition"]
        ):
            # print(ac)
            if used_ops + 1 >= self.params["max_ops"]:
                split_errors = 0
            else:
                split_errors = self.params["split_errors"]

            params = copy.deepcopy(self.params)
            params["split_errors"] = split_errors

            gen = curr_st.take_action(ac, params=params)
            # greedy = False
            for new_st in gen:
                if not self.params["no_heuristic"] and new_st.is_noop:
                    # print("noop")
                    continue

                if new_st.network.cost() < self.best_network.cost():
                    self.best_network = new_st.network

                self.log(new_st)

                if self.params["prune"]:
                    h = new_st.network.canonical_structure(
                        consider_ranks=self.params["consider_ranks"]
                    )
                    # print(h)
                    if h in worked:
                        return

                    worked.add(h)

                if used_ops + 1 >= self.params["max_ops"]:
                    # print("max op")
                    return

                # best_before = best_network.cost()
                self.dfs(worked, new_st)
                # best_after = best_network.cost()
                # if best_before == best_after:
                #     # greedy = True
                #     break

    def run(self, net: TensorNetwork):
        """Run a DFS search from the given tensor network."""

        self.target_tensor = net.contract()
        self.delta = self.params["eps"] * net.norm()
        self.best_network = net

        self.logging_time = 0
        self.start = time.time()

        # network = copy.deepcopy(net)
        worked = set()
        self.dfs(worked, SearchState(net, self.delta))
        return self.search_stats

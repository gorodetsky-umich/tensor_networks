"""Search algorithsm for tensor networks."""

import time
import copy
import argparse
from typing import Dict, List
from pydantic.dataclasses import dataclass

import numpy as np

from pytens.algs import TensorNetwork, Index, Tensor
from pytens.search.state import SearchState
from pytens.search.beam import BeamSearch
from pytens.search.mcts import MCTS
from pytens.search.partition import PartitionSearch
from pytens.search.utils import log_stats, approx_error, EMPTY_SEARCH_STATS
from pytens.search.dfs import DFSSearch


parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str)
parser.add_argument("--log", type=str)


class MyConfig:
    """Configuring data classes"""

    arbitrary_types_allowed = True


@dataclass(config=MyConfig)
class EnumState:
    """Enumeration state."""

    network: TensorNetwork
    ops: int
    delta: float

    def __lt__(self, other):
        if self.network != other.network:
            return self.network < other.network

        if self.ops != other.ops:
            return self.ops < other.ops

        return self.delta < other.delta


class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, params: Dict):
        self.params = params

    def add_wodup(
        self,
        best_network: TensorNetwork,
        new_st: SearchState,
        worked: set,
        worklist: List[SearchState],
    ) -> TensorNetwork:
        """Add a network to a worked set to remove duplicates."""
        # new_net.draw()
        # plt.show()
        # new_net_hash = hash(new_net)
        # if new_net_hash not in worked:
        if best_network is None or best_network.cost() > new_st.network.cost():
            best_network = new_st.network

        h = new_st.network.canonical_structure(
            consider_ranks=self.params["consider_ranks"]
        )
        if self.params["prune"]:
            if h in worked:
                return best_network

            worked.add(h)

        if len(new_st.past_actions) < self.params["max_ops"]:
            worklist.append(new_st)

        return best_network

    def mcts(self, net: TensorNetwork, budget: int = 10000):
        """Run the MCTS as a search engine."""
        engine = MCTS()
        delta = self.params["eps"] * net.norm()
        initial_state = SearchState(net, delta, max_ops=8, threshold=0.2)

        start = time.time()
        engine.search(initial_state, simulations=budget)
        end = time.time()

        best = engine.best_network

        stats = {}
        stats["time"] = end - start
        target_tensor = net.contract().value
        stats["cr_core"] = np.prod(target_tensor.shape) / best.cost()
        stats["cr_start"] = net.cost() / best.cost()
        stats["reconstruction_error"] = np.linalg.norm(
            best.contract().value - target_tensor
        ) / np.linalg.norm(target_tensor)
        stats["best_network"] = best

        # best.draw()
        # plt.show()
        return stats

    def beam(self, net: TensorNetwork, target_tensor: np.ndarray):
        """Run the beam search as a search engine."""
        engine = BeamSearch(self.params)
        if target_tensor is None:
            target_tensor = net.contract().value

        delta = np.sqrt(
            (self.params["eps"] * np.linalg.norm(target_tensor)) ** 2
            - np.linalg.norm(net.contract().value.squeeze() - target_tensor)
            ** 2
        )
        # print(delta)
        initial_state = SearchState(net, delta)

        start = time.time()
        engine.search(initial_state, guided=self.params["guided"])
        end = time.time()

        best = engine.best_network

        stats = engine.stats
        stats["time"] = end - start
        stats["cr_core"] = np.prod(target_tensor.shape) / best.cost()
        stats["cr_start"] = net.cost() / best.cost()
        stats["reconstruction_error"] = np.linalg.norm(
            best.contract().value.squeeze() - target_tensor
        ) / np.linalg.norm(target_tensor)
        stats["best_network"] = best

        return stats

    def partition_search(self, net: TensorNetwork):
        """Perform an search with output-directed splits + constraint solve."""
        engine = PartitionSearch(self.params)
        return engine.search(net)

    def dfs(
        self,
        net: TensorNetwork,
    ):
        """Perform an exhaustive enumeration with the DFS algorithm."""
        dfs_runner = DFSSearch(self.params)
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
        target_tensor = net.contract()
        search_stats = EMPTY_SEARCH_STATS
        logging_time = 0
        start = time.time()

        network = copy.deepcopy(net)
        delta = self.params["eps"] * net.norm()

        worked = set()
        worklist = [SearchState(network, delta)]
        worked.add(network.canonical_structure())
        best_network = None
        count = 0

        while len(worklist) != 0:
            st = worklist.pop(0)

            if time.time() - start >= self.params["timeout"]:
                break

            for ac in st.get_legal_actions():
                # plt.subplot(2,1,1)
                # st.network.draw()
                for new_st in st.take_action(ac, params=self.params):
                    # plt.subplot(2,1,2)
                    # new_st.network.draw()
                    # plt.show()
                    if not self.params["no_heuristic"] and new_st.is_noop:
                        continue

                    if self.params["optimize"]:
                        new_st.optimize()

                    ts = time.time() - start - logging_time
                    best_network = self.add_wodup(
                        best_network,
                        new_st,
                        worked,
                        worklist,
                    )
                    count += 1

                    verbose_start = time.time()
                    if self.params["verbose"]:
                        log_stats(
                            search_stats,
                            target_tensor,
                            ts,
                            new_st,
                            best_network,
                        )
                    verbose_end = time.time()
                    logging_time += verbose_end - verbose_start

        end = time.time()

        search_stats["time"] = end - start - logging_time
        search_stats["best_network"] = best_network
        search_stats["cr_core"] = (
            np.prod([i.size for i in net.free_indices()]) / best_network.cost()
        )
        search_stats["cr_start"] = net.cost() / best_network.cost()
        err = approx_error(target_tensor, best_network)
        search_stats["reconstruction_error"] = err
        search_stats["count"] = count

        return search_stats


def test_case_3():
    """Test exhaustive search.

    Target size: 14 x 16 x 18 x 20 x22
    Ranks:
    R12 = 3
    R23 = 4
    R34 = 3
    R45 = 2
    """
    target_net = TensorNetwork()

    g1 = np.random.randn(14, 3)
    g1_indices = [Index("I1", 14), Index("r12", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(3, 16, 4)
    g2_indices = [Index("r12", 3), Index("I2", 16), Index("r23", 4)]
    target_net.add_node("G2", Tensor(g2, g2_indices))

    g3 = np.random.randn(4, 18, 3)
    g3_indices = [Index("r23", 4), Index("I3", 18), Index("r34", 3)]
    target_net.add_node("G3", Tensor(g3, g3_indices))

    g4 = np.random.randn(3, 20, 2)
    g4_indices = [Index("r34", 3), Index("I4", 20), Index("r45", 2)]
    target_net.add_node("G4", Tensor(g4, g4_indices))

    g5 = np.random.randn(2, 22)
    g5_indices = [Index("r45", 2), Index("I5", 22)]
    target_net.add_node("G5", Tensor(g5, g5_indices))

    target_net.add_edge("G1", "G2")
    target_net.add_edge("G2", "G3")
    target_net.add_edge("G3", "G4")
    target_net.add_edge("G4", "G5")

    return target_net


def test_case_4():
    """Test exhaustive search.

    Target size: 40 x 60 x 3 x 9 x 9
    Ranks:
    R12 = 3
    R13 = 3
    R34 = 3
    R35 = 3
    """
    target_net = TensorNetwork()

    g1 = np.random.randn(40, 3, 3)
    g1_indices = [Index("I1", 40), Index("r12", 3), Index("r13", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(3, 60)
    g2_indices = [Index("r12", 3), Index("I2", 60)]
    target_net.add_node("G2", Tensor(g2, g2_indices))

    g3 = np.random.randn(3, 3, 3, 3)
    g3_indices = [
        Index("r13", 3),
        Index("I3", 3),
        Index("r34", 3),
        Index("r35", 3),
    ]
    target_net.add_node("G3", Tensor(g3, g3_indices))

    g4 = np.random.randn(3, 9)
    g4_indices = [Index("r34", 3), Index("I4", 9)]
    target_net.add_node("G4", Tensor(g4, g4_indices))

    g5 = np.random.randn(3, 9)
    g5_indices = [Index("r35", 3), Index("I5", 9)]
    target_net.add_node("G5", Tensor(g5, g5_indices))

    target_net.add_edge("G1", "G2")
    target_net.add_edge("G1", "G3")
    target_net.add_edge("G3", "G4")
    target_net.add_edge("G3", "G5")

    return target_net


def test_case_5():
    """Test exhaustive search.

    Target size: 14 x 16 x 18 x 20 x22
    Ranks:
    R12 = 3
    R1i1 = 3
    R1i1 = 3
    i1i2 = 2
    i1R4 = 4
    i2R3 = 3
    i2R5 = 3
    """
    target_net = TensorNetwork()

    g1 = np.random.randn(14, 4, 3)
    g1_indices = [Index("I1", 14), Index("r12", 4), Index("r1i1", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(4, 16)
    g2_indices = [Index("r12", 4), Index("I2", 16)]
    target_net.add_node("G2", Tensor(g2, g2_indices))

    i1 = np.random.randn(3, 2, 4)
    i1_indices = [Index("r1i1", 3), Index("i1i2", 2), Index("i1r4", 4)]
    target_net.add_node("i1", Tensor(i1, i1_indices))

    i2 = np.random.randn(2, 3, 3)
    i2_indices = [Index("i1i2", 2), Index("i2r3", 3), Index("i2r5", 3)]
    target_net.add_node("i2", Tensor(i2, i2_indices))

    g3 = np.random.randn(3, 18)
    g3_indices = [Index("i2r3", 3), Index("I3", 18)]
    target_net.add_node("G3", Tensor(g3, g3_indices))

    g4 = np.random.randn(20, 4)
    g4_indices = [Index("I4", 20), Index("i1r4", 4)]
    target_net.add_node("G4", Tensor(g4, g4_indices))

    g5 = np.random.randn(22, 3)
    g5_indices = [Index("I5", 22), Index("i2r5", 3)]
    target_net.add_node("G5", Tensor(g5, g5_indices))

    target_net.add_edge("G1", "G2")
    target_net.add_edge("G1", "i1")
    target_net.add_edge("i1", "i2")
    target_net.add_edge("i1", "G4")
    target_net.add_edge("i2", "G3")
    target_net.add_edge("i2", "G5")

    return target_net


if __name__ == "__main__":
    a = np.random.randn(14, 16, 18, 20, 22)
    initial_net = TensorNetwork()
    initial_net.add_node(
        "G",
        Tensor(
            a,
            [
                Index("a", 14),
                Index("I0", 16),
                Index("I1", 18),
                Index("I2", 20),
                Index("I3", 22),
            ],
        ),
    )

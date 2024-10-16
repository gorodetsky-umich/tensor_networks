"""Search algorithsm for tensor networks."""

import itertools
import time
import copy
import heapq
import argparse
from typing import Sequence, Dict, Set, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from pytens import NodeName, TensorNetwork, Index, Tensor


parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str)
parser.add_argument("--log", type=str)


class Split:
    """Split action."""

    def __init__(
        self,
        delta: float,
        node: NodeName,
        left_orders: Sequence[int],
        right_orders: Sequence[int],
    ):
        self.delta = delta
        self.node = node
        self.left_orders = left_orders
        self.right_orders = right_orders

    def __str__(self) -> str:
        return f"Split({self.node}, {self.left_orders}, {self.right_orders})"

    def __hash__(self) -> int:
        return hash(self.__str__())

    def execute(self, network: TensorNetwork) -> TensorNetwork:
        """Execute a split action."""
        network.split(self.node, self.left_orders, self.right_orders, delta=self.delta)
        return network


class Merge:
    """Merge action."""

    def __init__(self, node1: NodeName, node2: NodeName):
        self.node1 = node1
        self.node2 = node2

    def __str__(self) -> str:
        return f"Merge({self.node1}, {self.node2})"

    def __hash__(self) -> int:
        return hash(self.__str__())

    def execute(self, network: TensorNetwork) -> TensorNetwork:
        """Execute a merge action."""
        network.merge(self.node1, self.node2)
        return network


def approx_error(tensor: Tensor, net: TensorNetwork):
    """Compute the reconstruction error.

    Given a tensor network TN and the target tensor X,
    it returns ||X - TN|| / ||X||.
    """
    target_free_indices = tensor.indices
    net_free_indices = net.free_indices()
    net_value = net.contract().value
    perm = [net_free_indices.index(i) for i in target_free_indices]
    net_value = net_value.transpose(perm)
    error = np.linalg.norm(net_value - tensor.value) / np.linalg.norm(tensor.value)
    return error


class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, params: Dict):
        self.params = params

    def add_wodup(self,
        best_network: TensorNetwork,
        new_net: TensorNetwork,
        worked: Set[Tuple],
        worklist: List[TensorNetwork],
        # used for stats collections
        search_stats: Dict[str, Any],
        ts: float,
        ops: int,
    ) -> TensorNetwork:
        """Add a network to a worked set to remove duplicates."""
        # new_net.draw()
        # plt.show()
        canonical_new_net = new_net.canonicalize()
        if canonical_new_net not in worked:
            if best_network is None or best_network.cost() > new_net.cost():
                best_network = new_net

            heapq.heappush(worklist, (new_net, ops))
            worked.add(canonical_new_net)

        if self.params["verbose"]:
            search_stats["networks"].append((ts, new_net, ops))
            search_stats["best_networks"].append(best_network)

        return best_network

    def exhaustive(self, net: TensorNetwork, budget: int = 10000):
        """Perform an exhaustive enumeration."""
        target_tensor = net.contract()

        search_stats = {
            "networks": [],
            "best_networks": [],
            "best_cost": [],
            "costs": [],
            "errors": [],
            "ops": [],
        }
        start = time.time()

        network = TensorNetwork()
        network.add_node("tt", target_tensor)

        worked = set()
        worklist = []
        worked.add(network.canonicalize())
        heapq.heappush(worklist, (network, 0))
        best_network = None

        while len(worklist) != 0:
            net, ops = heapq.heappop(worklist)

            # print(net.canonicalize())
            # net.draw()
            # plt.show()

            # can we perform split?
            for n in net.network.nodes:
                indices = net.network.nodes[n]["tensor"].indices
                indices = range(len(indices))
                # get all partitions of indices
                for sz in range(1, len(indices) // 2 + 1):
                    combs = list(itertools.combinations(indices, sz))
                    if len(indices) % 2 == 0 and sz == len(indices) // 2:
                        combs = combs[: len(combs) // 2]

                    for comb in combs:
                        new_net = copy.deepcopy(net)
                        new_net.split(
                            n,
                            comb,
                            tuple(j for j in indices if j not in comb),
                            delta=1e-5
                        )
                        best_network = self.add_wodup(
                            best_network,
                            new_net,
                            worked,
                            worklist,
                            search_stats,
                            time.time() - start,
                            ops + 1,
                        )
                        budget -= 1
                        new_net.split(n, comb, tuple([j for j in indices if j not in comb]))
                        canonical_new_net = new_net.canonicalize()
                        if canonical_new_net not in worked:
                            heapq.heappush(worklist, new_net)
                            worked.add(canonical_new_net)

            # can we perform merge?
            for n in net.network.nodes:
                for m in net.network.neighbors(n):
                    if n < m:
                        new_net = copy.deepcopy(net)
                        new_net.merge(n, m)
                        best_network = self.add_wodup(
                            best_network,
                            new_net,
                            worked,
                            worklist,
                            search_stats,
                            time.time() - start,
                            ops + 1,
                        )
                        budget -= 1

            if budget < 0:
                break

        end = time.time()

        if self.params["verbose"]:
            for ((ts, n, ops), bn) in zip(search_stats["networks"], search_stats["best_networks"]):
                search_stats["ops"].append((ts, ops))
                search_stats["costs"].append((ts, n.cost()))
                err = approx_error(target_tensor, n)
                search_stats["errors"].append((ts, err))
                search_stats["best_cost"].append(
                    (ts, bn.cost())
                )

        search_stats["time"] = end - start
        search_stats["best_network"] = best_network
        err = approx_error(target_tensor, best_network)
        search_stats["reconstruction_error"] = err

        # remove temporary stats
        search_stats.pop("networks")
        search_stats.pop("best_networks")

        return search_stats


def test_case_1():
    """Test exhaustive search.

    Target size: 16 x 18 x 20 x22
    Ranks:
    R12 = 3
    R23 = 4
    R24 = 2
    """
    target_net = TensorNetwork()

    g1 = np.random.randn(16, 3)
    g1_indices = [Index("I1", 16), Index("r12", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(18, 3, 4, 2)
    g2_indices = [
        Index("I2", 18),
        Index("r12", 3),
        Index("r23", 4),
        Index("r24", 2),
    ]
    target_net.add_node("G2", Tensor(g2, g2_indices))

    g3 = np.random.randn(20, 4)
    g3_indices = [Index("I3", 20), Index("r23", 4)]
    target_net.add_node("G3", Tensor(g3, g3_indices))

    g4 = np.random.randn(22, 2)
    g4_indices = [Index("I4", 22), Index("r24", 2)]
    target_net.add_node("G4", Tensor(g4, g4_indices))

    target_net.add_edge("G1", "G2")
    target_net.add_edge("G2", "G3")
    target_net.add_edge("G2", "G4")

    return target_net


def test_case_2():
    """Test exhaustive search.

    Target size: 16 x 18 x 20 x22
    Ranks:
    R12 = 3
    R23 = 4
    R34 = 4
    """
    target_net = TensorNetwork()

    g1 = np.random.randn(16, 3)
    g1_indices = [Index("I1", 16), Index("r12", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(18, 3, 4)
    g2_indices = [Index("I2", 18), Index("r12", 3), Index("r23", 4)]
    target_net.add_node("G2", Tensor(g2, g2_indices))

    g3 = np.random.randn(20, 4, 4)
    g3_indices = [Index("I3", 20), Index("r23", 4), Index("r34", 4)]
    target_net.add_node("G3", Tensor(g3, g3_indices))

    g4 = np.random.randn(22, 4)
    g4_indices = [Index("I4", 22), Index("r34", 4)]
    target_net.add_node("G4", Tensor(g4, g4_indices))

    target_net.add_edge("G1", "G2")
    target_net.add_edge("G2", "G3")
    target_net.add_edge("G3", "G4")

    return target_net


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
    args = parser.parse_args()

    target = test_case_5()
    target.draw()
    plt.savefig(f"{args.tag}_gt.png")
    plt.close()

    with open(args.log, "w", encoding="utf-8") as f:
        for i in range(10):
            engine = SearchEngine({"tag": args.tag})
            stats = engine.exhaustive(target)
            f.write(f"{i},{stats['time']},{stats['reconstruction_error']}\n")
            stats["best_network"].draw()
            plt.savefig(f"{args.tag}_result.png")
            plt.close()

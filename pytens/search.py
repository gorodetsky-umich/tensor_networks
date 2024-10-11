from pytens import *

import numpy as np
import matplotlib.pyplot as plt
import copy
import heapq
import itertools
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str)
parser.add_argument("--log", type=str)

class Split:
    def __init__(self, eps: float, node: NodeName, left_orders: Sequence[int], right_orders: Sequence[int]):
        self.eps = eps
        self.node = node
        self.left_orders = left_orders
        self.right_orders = right_orders

    def __str__(self):
        return f"Split({self.node}, {self.left_orders}, {self.right_orders})"
    
    def __hash__(self):
        return hash(self.__str__())
    
    def execute(self, network: TensorNetwork) -> TensorNetwork:
        network.pull_through(self.node)
        network.split(self.node, self.left_orders, self.right_orders, eps=self.eps)
        return network

class Merge:
    def __init__(self, node1: NodeName, node2: NodeName):
        self.node1 = node1
        self.node2 = node2

    def __str__(self):
        return f"Merge({self.node1}, {self.node2})"
    
    def __hash__(self):
        return hash(self.__str__())
    
    def execute(self, network: TensorNetwork) -> TensorNetwork:
        network.merge(self.node1, self.node2)

class SearchEngine:
    def __init__(self, params, target):
        self.params = params
        self.target = target

    def exhaustive(self, budget = 10000):
        network = TensorNetwork()
        network.add_node("tt", self.target)

        worked = set()
        worklist = []
        worked.add(network.canonicalize())
        heapq.heappush(worklist, network)
        best_network = None

        while len(worklist) != 0:
            net = heapq.heappop(worklist)
            
            # print(net.canonicalize())
            # net.draw()
            # plt.show()

            if best_network is None or best_network.cost() > net.cost():
                best_network = net

            # can we perform split?
            for n in net.network.nodes:
                indices = net.network.nodes[n]["tensor"].indices
                indices = range(len(indices))
                # get all partitions of indices
                for sz in range(1, len(indices) // 2 + 1):
                    combs = list(itertools.combinations(indices, sz))
                    if len(indices) % 2 == 0 and sz == len(indices) // 2:
                        combs = combs[:len(combs)//2]

                    for comb in combs:
                        new_net = copy.deepcopy(net)
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
                        canonical_new_net = new_net.canonicalize()
                        if canonical_new_net not in worked:
                            heapq.heappush(worklist, new_net)
                            worked.add(canonical_new_net)
                            budget -= 1

            if budget < 0:
                break

        plt.figure(2)
        best_network.draw()
        # best_result = best_network.contract()
        # best_value = best_result.value.transpose(2,1,3,4,0)
        # error = np.linalg.norm(best_value - self.target.value) / np.linalg.norm(self.target.value)
        # print("Reconstruction error", error)
        plt.savefig(f"{self.params['tag']}_result.png")

def test_case_1():
    target_net = TensorNetwork()

    g1 = np.random.randn(16, 3)
    g1_indices = [Index("I1", 16), Index("r12", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(18, 3, 4, 2)
    g2_indices = [Index("I2", 18), Index("r12", 3), Index("r23", 4), Index("r24", 2)]
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
    # 14 x 16 x 18 x 20 x22
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
    # 40 x 60 x 3 x 9 x 9
    target_net = TensorNetwork()

    g1 = np.random.randn(40, 3, 3)
    g1_indices = [Index("I1", 40), Index("r12", 3), Index("r13", 3)]
    target_net.add_node("G1", Tensor(g1, g1_indices))

    g2 = np.random.randn(3, 60)
    g2_indices = [Index("r12", 3), Index("I2", 60)]
    target_net.add_node("G2", Tensor(g2, g2_indices))

    g3 = np.random.randn(3, 3, 3, 3)
    g3_indices = [Index("r13", 3), Index("I3", 3), Index("r34", 3), Index("r35", 3)]
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
    # 14 x 16 x 18 x 20 x22
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

    target_net = test_case_2()
    target = target_net.contract()
    plt.figure(1)
    target_net.draw()
    plt.savefig(f"{args.tag}_gt.png")

    print(target.value.shape, target.indices)
    with open(args.log, "w") as f:
        for i in range(10):
            engine = SearchEngine({"tag": args.tag}, target)
            start = time.time()
            engine.exhaustive()
            end = time.time()
            f.write(f"{i},{end-start}\n")
        
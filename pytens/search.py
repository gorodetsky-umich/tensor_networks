"""Search algorithsm for tensor networks."""

import itertools
import time
import copy
import heapq
import math
import random
import argparse
from typing import Sequence, Dict, Set, List, Tuple, Any
from pydantic.dataclasses import dataclass

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from pytens.algs import NodeName, TensorNetwork, Index, Tensor


parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str)
parser.add_argument("--log", type=str)


class Split:
    """Split action."""

    def __init__(
        self,
        node: NodeName,
        left_indices: Sequence[int],
        right_indices: Sequence[int],
    ):
        self.node = node
        self.left_indices = left_indices
        self.right_indices = right_indices

    def __str__(self) -> str:
        return f"Split({self.node}, {self.left_indices}, {self.right_indices})"

    def __hash__(self) -> int:
        return hash(self.__str__())

    def execute(self, network: TensorNetwork, delta: float) -> Tuple[bool, float]:
        """Execute a split action."""
        node_indices = network.network.nodes[self.node]["tensor"].indices
        left_dims = [node_indices[i].size for i in self.left_indices]
        right_dims = [node_indices[i].size for i in self.right_indices]
        [u, v], new_delta = network.split(self.node, self.left_indices, self.right_indices, delta=delta)
        index_sz = network.get_contraction_index(u, v)[0].size
        index_max = min(np.prod(left_dims), np.prod(right_dims))
        return index_sz == index_max, new_delta

class Merge:
    """Merge action."""

    def __init__(self, node1: NodeName, node2: NodeName):
        self.node1 = node1
        self.node2 = node2

    def __str__(self) -> str:
        return f"Merge({self.node1}, {self.node2})"

    def __hash__(self) -> int:
        return hash(self.__str__())

    def execute(self, network: TensorNetwork):
        """Execute a merge action."""
        network.merge(self.node1, self.node2)
        return network

class SearchState:
    def __init__(self, net, delta, threshold=0.1, max_ops = 5):
        self.network = net
        self.curr_delta = delta
        self.last_action = None      # Last action taken to reach this state
        self.max_ops = max_ops
        self.threshold = threshold
        self.is_noop = False

    def get_legal_actions(self):
        """Return a list of all legal actions in this state."""
        actions = []
        for n in self.network.network.nodes:
            indices = self.network.network.nodes[n]["tensor"].indices
            indices = range(len(indices))
            # get all partitions of indices
            for sz in range(1, len(indices) // 2 + 1):
                combs = list(itertools.combinations(indices, sz))
                if len(indices) % 2 == 0 and sz == len(indices) // 2:
                    combs = combs[: len(combs) // 2]

                for comb in combs:
                    left_indices = comb
                    right_indices = tuple(j for j in indices if j not in comb)
                    ac = Split(n, left_indices, right_indices)
                    actions.append(ac)

            for m in self.network.network.neighbors(n):
                if n < m:
                    n_indices = self.network.network.nodes[n]["tensor"].indices
                    m_indices = self.network.network.nodes[m]["tensor"].indices
                    if len(set(n_indices).union(set(m_indices))) < 5:
                        ac = Merge(n, m)
                        actions.append(ac)

        return actions

    def take_action(self, action):
        """Return a new GameState after taking the specified action."""
        if isinstance(action, Split):
            new_net = copy.deepcopy(self.network)
            try:
                is_noop, new_delta = action.execute(new_net, self.curr_delta)
            except np.linalg.LinAlgError:
                is_noop, new_delta = True, self.curr_delta
        elif isinstance(action, Merge):
            new_net = copy.deepcopy(self.network)
            action.execute(new_net)
            new_delta = self.curr_delta
            is_noop = False
        else:
            raise TypeError("Unrecognized action type")
        
        # new_net.draw()
        # plt.show()
        new_state = SearchState(new_net, new_delta, max_ops=self.max_ops, threshold=self.threshold)
        new_state.last_action = action
        new_state.is_noop = isinstance(action, Split) and is_noop
        return new_state

    def is_terminal(self):
        return len(self.network.network.nodes) >= self.max_ops or self.is_noop
    
    def get_result(self, total_cost):
        if self.is_noop:
            return 0
        
        return self.network.cost() <= self.threshold * total_cost

class Node:
    def __init__(self, state, parent=None):
        self.state = state                  # Game state for this node
        self.parent = parent                # Parent node
        self.children = []                  # List of child nodes
        self.visits = 0                     # Number of times this node was visited
        self.wins = 0                       # Number of wins after visiting this node

    def is_fully_expanded(self):
        """Check if all possible moves have been expanded."""
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight=1.41):
        """Use UCB1 to select the best child node."""
        choices_weights = []
        for child in self.children:
            if child.state.is_noop:
                weight = 0
            elif child.state.is_terminal() and child.wins == 0:
                weight = 0
            else:
                weight = (child.wins / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)

            choices_weights.append(weight)

        max_weight = max(choices_weights)
        # candidates = [self.children[i] for i, w in enumerate(choices_weights) if w == max_weight]
        return self.children[choices_weights.index(max_weight)]

    def expand(self):
        """Expand by creating a new child node for a random untried action."""
        legal_actions = self.state.get_legal_actions()
        tried_actions = [child.state.last_action for child in self.children]
        untried_actions = [action for action in legal_actions if action not in tried_actions]

        action = random.choice(untried_actions)
        start = time.time()
        next_state = self.state.take_action(action)
        if isinstance(action, Split):
            print("completing the action", action, self.state.network.network.nodes[action.node]["tensor"].indices, "takes", time.time() - start)
        else:
            print("completing the action", action, "takes", time.time() - start)
        child_node = Node(state=next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        """Backpropagate the result of the simulation up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, exploration_weight=1.41):
        self.exploration_weight = exploration_weight
        self.initial_cost = 0
        self.best_network = None

    def search(self, initial_state, simulations=1000):
        root = Node(initial_state)
        self.initial_cost = initial_state.network.cost()
        self.best_network = initial_state.network
        
        for _ in range(simulations):
            start = time.time()
            node = self.select(root)
            if not node.state.is_terminal():
                node = node.expand()
            result = self.simulate(node)
            node.backpropagate(result)
            print("one simulation time", time.time() - start)

    def select(self, node):
        """Select a leaf node."""
        while not node.state.is_terminal():
            if node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            else:
                return node
        return node

    def simulate(self, node):
        """Run a random simulation from the given node to a terminal state."""
        current_state = node.state
        start = time.time()
        step = 0
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.take_action(action)
            step += 1

        print("complete", step, "steps in", time.time() - start)

        if current_state.network.cost() < self.best_network.cost():
            self.best_network = current_state.network

        return current_state.get_result(self.initial_cost)


def approx_error(tensor: Tensor, net: TensorNetwork) -> float:
    """Compute the reconstruction error.

    Given a tensor network TN and the target tensor X,
    it returns ||X - TN|| / ||X||.
    """
    target_free_indices = tensor.indices
    net_free_indices = net.free_indices()
    net_value = net.contract().value
    perm = [net_free_indices.index(i) for i in target_free_indices]
    net_value = net_value.transpose(perm)
    error = float(np.linalg.norm(net_value - tensor.value) / np.linalg.norm(tensor.value))
    return error

def log_stats(search_stats, target_tensor, ts, n, ops, bn):
    search_stats["ops"].append((ts, ops))
    search_stats["costs"].append((ts, n.cost()))
    err = approx_error(target_tensor, n)
    search_stats["errors"].append((ts, err))
    search_stats["best_cost"].append((ts, bn.cost()))

class MyConfig:
    arbitrary_types_allowed=True

@dataclass(config=MyConfig)
class EnumState:
    """Enumeration state."""
    network: TensorNetwork
    ops: int
    delta: float

    def __lt__(self, other):
        if self.network != other.network:
            return self.network < other.network
        elif self.ops != other.ops:
            return self.ops < other.ops
        else:
            return self.delta < other.delta

class StructureFactory:
    def __init__(self):
        self.space = nx.DiGraph()

    def initialize(self, network: TensorNetwork, max_ops: int = 10):
        if max_ops == 0:
            return
        
        curr_node = network.canonical_structure()
        for n in network.network.nodes:
            curr_indices = network.network.nodes[n]["tensor"].indices
            indices = range(len(curr_indices))
            # get all partitions of indices
            for sz in range(1, len(indices) // 2 + 1):
                combs = list(itertools.combinations(indices, sz))
                if len(indices) % 2 == 0 and sz == len(indices) // 2:
                    combs = combs[: len(combs) // 2]

                for comb in combs:
                    new_net = copy.deepcopy(network)
                    new_net.split(
                        n,
                        comb,
                        tuple(j for j in indices if j not in comb),
                        preview=True
                    )

                    new_node = new_net.canonical_structure()
                    # print("checking", new_node)
                    if new_node not in self.space.nodes:
                        self.space.add_node(new_node, network=new_net)
                    self.space.add_edge(curr_node, new_node, action=Split(n, comb, tuple(j for j in indices if j not in comb)))
                    self.initialize(new_net, max_ops-1)

            # can we perform merge?
            for m in network.network.neighbors(n):
                if n < m:
                    new_net = copy.deepcopy(network)
                    new_net.merge(n, m, preview=True)
                    new_node = new_net.canonical_structure()
                    # print("checking", new_node)
                    if new_node not in self.space.nodes:
                        self.space.add_node(new_node, network=new_net)
                    self.space.add_edge(curr_node, new_node, action=Merge(n, m))
                    self.initialize(new_net, max_ops-1)

class SearchEngine:
    """Tensor network topology search engine."""

    def __init__(self, params: Dict):
        self.params = params

    def add_wodup(self,
        best_network: TensorNetwork,
        new_net: TensorNetwork,
        delta: float,
        worked: Set[Tuple],
        worklist: List[TensorNetwork],
        ops: int,
    ) -> TensorNetwork:
        """Add a network to a worked set to remove duplicates."""
        # new_net.draw()
        # plt.show()
        # new_net_hash = hash(new_net)
        # if new_net_hash not in worked:
        if best_network is None or best_network.cost() > new_net.cost():
            best_network = new_net

        # if new_net.cost() < 2 * best_network.cost():
        if ops < 3:
            worklist.append(EnumState(new_net, ops, delta))
        # worked.add(new_net_hash)

        return best_network

    def a_star(self, max_ops: int = 5, timeout: float = 3600):
        """Perform the A-star search with a priority queue"""

    def mcts(self, net: TensorNetwork, budget: int = 10000):
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
        stats["reconstruction_error"] = np.linalg.norm(best.contract().value - target_tensor) / np.linalg.norm(target_tensor)
        stats["best_network"] = best
        
        best.draw()
        plt.show()
        return stats


    def dfs(self, net: TensorNetwork, max_ops: int = 4, timeout: float = 3600 * 10, budget: int = 5000):
        """Perform an exhaustive enumeration with the DFS algorithm."""
        target_tensor = net.contract()

        search_stats = {
            "networks": [],
            "best_networks": [],
            "best_cost": [],
            "costs": [],
            "errors": [],
            "ops": [],
        }
        logging_time = 0
        start = time.time()

        network = copy.deepcopy(net)
        delta = self.params["eps"] * net.norm()
        best_network = net
        worked = set()
        count = 0

        def helper(curr_net: TensorNetwork, curr_ops: int, curr_delta: float):
            # plt.figure(curr_net.canonical_structure())
            nonlocal best_network
            nonlocal logging_time
            nonlocal search_stats
            nonlocal start
            nonlocal count

            count += 1

            # canon = curr_net.canonical_structure()
            # if canon == 1666526642026772368:
            #     curr_net.draw()
            #     plt.show()

            frees = curr_net.free_indices()
            # if len(curr_net.network.nodes) == 4 and all(len(d["tensor"].indices) <= 3 and len(set(d["tensor"].indices).intersection(set(frees))) == 1 for _, d in curr_net.network.nodes(data=True)):
            #     curr_net.draw()
            #     plt.show()
            # print(len(curr_net.network.nodes))
            # for n, d in curr_net.network.nodes(data=True):
            #     print(n, len(d["tensor"].indices))
            #     print(n, len(set(d["tensor"].indices).intersection(set(frees))))
            # curr_net.draw()
            # plt.show()

            # if (curr_ops, canon) in worked:
            #     print("prune")
            #     return
            
            # # print("keep")            
            # worked.add((curr_ops, canon))

            if curr_ops >= max_ops:
                return
            
            if time.time() - start > timeout:
                return

            # can we perform split?
            for n in curr_net.network.nodes:
                curr_indices = curr_net.network.nodes[n]["tensor"].indices
                indices = range(len(curr_indices))
                # get all partitions of indices
                for sz in range(1, len(indices) // 2 + 1):
                    combs = list(itertools.combinations(indices, sz))
                    if len(indices) % 2 == 0 and sz == len(indices) // 2:
                        combs = combs[: len(combs) // 2]

                    for comb in combs:
                        new_net = copy.deepcopy(curr_net)
                        try:
                            # print("split", n, [curr_indices[i] for i in comb])
                            #TODO: look ahead and check whether this is a duplicate
                            [u, v], new_delta = new_net.split(
                                n,
                                comb,
                                tuple(j for j in indices if j not in comb),
                                delta=curr_delta
                            )
                            if new_net.canonical_structure() == -4013424694000593114:
                                print(curr_net.canonical_structure())
                                curr_net.draw()
                                plt.show()
                            # index_uv = new_net.get_contraction_index(u, v)[0]
                            # after split, we already orthonormalized the env
                            # print("before optimize")
                            # _, new_delta = new_net.optimize(v, new_delta)
                            # if len(new_net.network.nodes) == 5 and all(len(d["tensor"].indices) <= 3 for _, d in new_net.network.nodes(data=True)):
                            #     plt.subplot(211)
                            #     curr_net.draw()
                            #     plt.subplot(212)
                            #     new_net.draw()
                            #     plt.show()
                            # after update, we need to orthonormalize the env before next optimization
                            # print("before normalize")
                            # new_net.draw()
                            # plt.show()
                            # u = new_net.orthonormalize(u)
                            # print("after optimize")
                            # index_uv = new_net.get_contraction_index(u, v)[0]
                            # _, new_delta = new_net.optimize(u, new_delta, set([index_uv]))
                        except np.linalg.LinAlgError as e:
                            print("exception", e)
                            continue

                        if new_net.cost() < best_network.cost():
                            best_network = new_net

                        ts = time.time() - start - logging_time
                        verbose_start = time.time()
                        if self.params["verbose"]:
                            # print(time.time() - start)
                            log_stats(search_stats, target_tensor, ts, new_net, curr_ops+1, best_network)
                        verbose_end = time.time()
                        logging_time += verbose_end - verbose_start

                        helper(new_net, curr_ops+1, new_delta)

                # can we perform merge?
                for m in curr_net.network.neighbors(n):
                    if n < m:
                        new_net = copy.deepcopy(curr_net)
                        # print("merge", n, m)
                        new_net.merge(n, m)
                        # plt.subplot(211)
                        # curr_net.draw()
                        # plt.subplot(212)
                        # new_net.draw()
                        # plt.show()
                        if new_net.cost() < best_network.cost():
                            best_network = new_net
                        
                        ts = time.time() - start - logging_time

                        verbose_start = time.time()
                        if self.params["verbose"]:
                            log_stats(search_stats, target_tensor, ts, new_net, curr_ops+1, best_network)
                        verbose_end = time.time()
                        logging_time += verbose_end - verbose_start

                        helper(new_net, curr_ops+1, curr_delta)

                # plt.close(curr_net.canonical_structure())

        helper(network, 0, delta)
        end = time.time()

        print("unique structures", len(worked))
        print("best hash", best_network.canonical_structure())
        search_stats["time"] = end - start - logging_time
        search_stats["best_network"] = best_network
        search_stats["cr_core"] = np.prod([i.size for i in net.free_indices()]) / best_network.cost()
        search_stats["cr_start"] = net.cost() / best_network.cost()
        err = approx_error(target_tensor, best_network)
        search_stats["reconstruction_error"] = err
        search_stats["count"] = count

        return search_stats

    def bfs(self, net: TensorNetwork, max_ops = 3, timeout = 3600):
        """Perform an exhaustive enumeration with the BFS algorithm."""
        target_tensor = net.contract()

        search_stats = {
            "networks": [],
            "best_networks": [],
            "best_cost": [],
            "costs": [],
            "errors": [],
            "ops": [],
        }
        logging_time = 0
        start = time.time()

        network = copy.deepcopy(net)
        delta = self.params["eps"] * net.norm()

        worked = set()
        worklist = [EnumState(network, 0, delta)]
        best_network = None
        count = 0

        while len(worklist) != 0:
            st = worklist.pop(0)
            curr_net = st.network
            ops = st.ops
            delta = st.delta

            # print(net.canonicalize())
            # net.draw()
            # plt.show()
            if time.time() - start >= timeout:
                break

            # can we perform split?
            for n in curr_net.network.nodes:
                curr_indices = curr_net.network.nodes[n]["tensor"].indices
                indices = range(len(curr_indices))
                # get all partitions of indices
                for sz in range(1, len(indices) // 2 + 1):
                    combs = list(itertools.combinations(indices, sz))
                    if len(indices) % 2 == 0 and sz == len(indices) // 2:
                        combs = combs[: len(combs) // 2]

                    for comb in combs:
                        print("split", n, [curr_indices[i] for i in comb])
                        new_net = copy.deepcopy(curr_net)
                        _, new_delta = new_net.split(
                            n,
                            comb,
                            tuple(j for j in indices if j not in comb),
                            delta=delta
                        )
                        ts = time.time() - start - logging_time
                        best_network = self.add_wodup(
                            best_network,
                            new_net,
                            new_delta,
                            worked,
                            worklist,
                            ops + 1,
                        )
                        count += 1

                        verbose_start = time.time()
                        if self.params["verbose"]:
                            log_stats(search_stats, target_tensor, ts, new_net, ops+1, best_network)
                        verbose_end = time.time()
                        logging_time += verbose_end - verbose_start

                for m in curr_net.network.neighbors(n):
                    if n < m:
                        new_net = copy.deepcopy(curr_net)
                        new_net.merge(n, m)
                        ts = time.time() - start - logging_time
                        best_network = self.add_wodup(
                            best_network,
                            new_net,
                            delta,
                            worked,
                            worklist,
                            ops + 1,
                        )
                        count += 1

                        verbose_start = time.time()
                        if self.params["verbose"]:
                            log_stats(search_stats, target_tensor, ts, new_net, ops+1, best_network)
                        verbose_end = time.time()
                        logging_time += verbose_end - verbose_start

                if count > 10000:
                    break

        end = time.time()

        search_stats["time"] = end - start - logging_time
        search_stats["best_network"] = best_network
        search_stats["cr_core"] = best_network.cost() / np.prod([i.size for i in net.free_indices()])
        search_stats["cr_start"] = best_network.cost() / net.cost()
        err = approx_error(target_tensor, best_network)
        search_stats["reconstruction_error"] = err

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
    factory = StructureFactory()
    a = np.random.randn(14,16,18,20,22)
    initial_net = TensorNetwork()
    initial_net.add_node("G", Tensor(a, [Index("a", 14), Index("I0", 16), Index("I1", 18), Index("I2", 20), Index("I3", 22)]))
    factory.initialize(initial_net, max_ops=5)
    print(len(factory.space.nodes))
    print(len(factory.space.edges))

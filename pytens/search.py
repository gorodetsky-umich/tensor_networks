"""Search algorithsm for tensor networks."""

import itertools
import time
import copy
import heapq
import math
import random
import argparse
from typing import Sequence, Dict, List, Tuple, Self
from pydantic.dataclasses import dataclass

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from pytens.algs import NodeName, TensorNetwork, Index, Tensor


parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str)
parser.add_argument("--log", type=str)


class Action:
    """Base action."""

class Split(Action):
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
        # print("node indices", node_indices)
        left_dims = [node_indices[i].size for i in self.left_indices]
        right_dims = [node_indices[i].size for i in self.right_indices]
        [u, v], new_delta = network.split(
            self.node, self.left_indices, self.right_indices, delta=delta
        )
        index_sz = network.get_contraction_index(u, v)[0].size
        index_max = min(np.prod(left_dims), np.prod(right_dims))
        return index_sz == index_max, new_delta


class Merge(Action):
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
    """Class for representation of intermediate search states."""

    def __init__(
        self, net: TensorNetwork, delta: float, threshold: float = 0.1, max_ops: int = 5
    ):
        self.network = net
        self.curr_delta = delta
        self.last_action = None  # Last action taken to reach this state
        self.max_ops = max_ops
        self.threshold = threshold
        self.is_noop = False
        self.used_ops = 0

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

    def take_action(self, action: Action):
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
        new_state = SearchState(
            new_net, new_delta, max_ops=self.max_ops, threshold=self.threshold
        )
        new_state.last_action = action
        new_state.used_ops = self.used_ops + 1
        new_state.is_noop = isinstance(action, Split) and is_noop
        return new_state

    def optimize(self):
        """Optimize the current structure."""
        free_indices = self.network.free_indices()
        root = None
        for n, t in self.network.network.nodes(data=True):
            if free_indices[0] in t["tensor"].indices:
                root = n
                break

        root = self.network.orthonormalize(root)
        _, self.curr_delta = self.network.optimize(root, self.curr_delta)

    def is_terminal(self) -> bool:
        """Whether the current state is a terminal state."""
        return self.is_noop or len(self.network.network.nodes) >= self.max_ops

    def get_result(self, total_cost: float) -> float:
        """Whether the current state succeeds or not."""
        if self.is_noop:
            return 0

        return float(self.network.cost() <= self.threshold * total_cost)

    def __lt__(self, other: Self) -> bool:
        return self.network.cost() > other.network.cost()


class Node:
    """Representation of one node in MCTS."""

    def __init__(self, state: SearchState, parent: Self = None):
        self.state = state  # Game state for this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node was visited
        self.wins = 0  # Number of wins after visiting this node

    def is_fully_expanded(self):
        """Check if all possible moves have been expanded."""
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight: float = 1.41) -> Self:
        """Use UCB1 to select the best child node."""
        choices_weights = []
        for child in self.children:
            if child.state.is_noop:
                weight = 0
            elif child.state.is_terminal() and child.wins == 0:
                weight = 0
            else:
                weight = (child.wins / child.visits) + exploration_weight * math.sqrt(
                    math.log(self.visits) / child.visits
                )

            choices_weights.append(weight)

        max_weight = max(choices_weights)
        # candidates = [self.children[i] for i, w in enumerate(choices_weights) if w == max_weight]
        return self.children[choices_weights.index(max_weight)]

    def expand(self):
        """Expand by creating a new child node for a random untried action."""
        legal_actions = self.state.get_legal_actions()
        tried_actions = [child.state.last_action for child in self.children]
        untried_actions = [
            action for action in legal_actions if action not in tried_actions
        ]

        action = random.choice(untried_actions)
        start = time.time()
        next_state = self.state.take_action(action)
        if isinstance(action, Split):
            print(
                "completing the action",
                action,
                self.state.network.network.nodes[action.node]["tensor"].indices,
                "takes",
                time.time() - start,
            )
        else:
            print("completing the action", action, "takes", time.time() - start)
        child_node = Node(state=next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result: int):
        """Backpropagate the result of the simulation up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    """The MCTS search engine."""

    def __init__(self, exploration_weight: float = 1.41):
        self.exploration_weight = exploration_weight
        self.initial_cost = 0
        self.best_network = None

    def search(self, initial_state: SearchState, simulations: int = 1000):
        """Perform the mcts search."""
        root = Node(initial_state)
        self.initial_cost = initial_state.network.cost()
        self.best_network = initial_state.network

        for _ in range(simulations):
            node = self.select(root)
            if not node.state.is_terminal():
                node = node.expand()
            result = self.simulate(node)
            node.backpropagate(result)
            # print("one simulation time", time.time() - start)

    def select(self, node: Node) -> Node:
        """Select a leaf node."""
        while not node.state.is_terminal():
            if node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            else:
                return node
        return node

    def simulate(self, node: Node) -> float:
        """Run a random simulation from the given node to a terminal state."""
        curr_state = node.state
        prev_state = node.state
        step = 0
        while not curr_state.is_terminal():
            prev_state = curr_state
            action = random.choice(curr_state.get_legal_actions())
            curr_state = curr_state.take_action(action)
            step += 1

        # print("complete", step, "steps in", time.time() - start)
        best_candidate = curr_state
        if curr_state.is_noop:
            best_candidate = prev_state

        if best_candidate.network.cost() < self.best_network.cost():
            self.best_network = best_candidate.network

        return curr_state.get_result(self.initial_cost)


class BeamSearch:
    """Beam search with a given beam size."""

    def __init__(self, params):
        self.params = params
        self.heap = None
        self.initial_cost = 0
        self.best_network = None

    def search(self, initial_state):
        """Perform the beam search from the given initial state."""
        self.initial_cost = initial_state.network.cost()
        self.best_network = initial_state.network
        self.heap = [initial_state]

        for _ in range(self.params["max_ops"]):
            start = time.time()
            # maintain a set of networks of at most k
            self.step()
            print("one step time", time.time() - start)

    def step(self):
        """Make a step in a beam search."""
        next_level = []
        while len(self.heap) > 0:
            state = heapq.heappop(self.heap)
            for ac in state.get_legal_actions():
                new_state = state.take_action(ac)
                if new_state.is_noop:
                    continue

                if len(next_level) < self.params["beam_size"]:
                    heapq.heappush(next_level, new_state)
                elif next_level[0] < new_state:
                    heapq.heappushpop(next_level, new_state)

                if new_state.network.cost() < self.best_network.cost():
                    self.best_network = new_state.network

        self.heap = next_level


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
    error = float(
        np.linalg.norm(net_value - tensor.value) / np.linalg.norm(tensor.value)
    )
    return error


def log_stats(
    search_stats: dict,
    target_tensor: np.ndarray,
    ts: float,
    st: SearchState,
    bn: TensorNetwork,
):
    """Log statistics of a given state."""
    search_stats["ops"].append((ts, st.used_ops))
    search_stats["costs"].append((ts, st.network.cost()))
    err = approx_error(target_tensor, st.network)
    search_stats["errors"].append((ts, err))
    search_stats["best_cost"].append((ts, bn.cost()))


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
        elif self.ops != other.ops:
            return self.ops < other.ops
        else:
            return self.delta < other.delta


class StructureFactory:
    """Pure structure generation."""

    def __init__(self):
        self.space = nx.DiGraph()

    def initialize(self, network: TensorNetwork, max_ops: int = 6):
        """Initial the factory with all possible structures 
        up to a given maximum number of operations.
        """
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
                        preview=True,
                    )

                    new_node = new_net.canonical_structure()
                    # print("checking", new_node)
                    if new_node not in self.space.nodes:
                        self.space.add_node(new_node, network=new_net)
                    self.space.add_edge(
                        curr_node,
                        new_node,
                        action=Split(
                            n, comb, tuple(j for j in indices if j not in comb)
                        ),
                    )
                    self.initialize(new_net, max_ops - 1)

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
                    self.initialize(new_net, max_ops - 1)


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
            else:
                worked.add(h)

        if new_st.used_ops < self.params["max_ops"]:
            worklist.append(new_st)

        return best_network

    def a_star(self, max_ops: int = 5, timeout: float = 3600):
        """Perform the A-star search with a priority queue"""

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

        best.draw()
        plt.show()
        return stats

    def beam(self, net: TensorNetwork):
        """Run the beam search as a search engine."""
        engine = BeamSearch(self.params)
        delta = self.params["eps"] * net.norm()
        initial_state = SearchState(net, delta)

        start = time.time()
        engine.search(initial_state)
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

        best.draw()
        plt.show()
        return stats

    def dfs(
        self,
        net: TensorNetwork,
    ):
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
        worked = set([network.canonical_structure()])
        count = 0

        def helper(curr_st: SearchState):
            # plt.figure(curr_net.canonical_structure())
            nonlocal best_network
            nonlocal logging_time
            nonlocal search_stats
            nonlocal start
            nonlocal count

            count += 1

            if self.params["prune"]:
                h = curr_st.network.canonical_structure(
                    consider_ranks=self.params["consider_ranks"]
                )
                if h in worked:
                    return
                else:
                    worked.add(h)

            if curr_st.used_ops >= self.params["max_ops"]:
                return

            if time.time() - start > self.params["timeout"]:
                return

            for ac in curr_st.get_legal_actions():
                new_st = curr_st.take_action(ac)
                # new_st.network.draw()
                # plt.show()
                if not self.params["no_heuristic"] and new_st.is_noop:
                    continue

                if new_st.network.cost() < best_network.cost():
                    best_network = new_st.network

                ts = time.time() - start - logging_time
                verbose_start = time.time()
                if self.params["verbose"]:
                    log_stats(search_stats, target_tensor, ts, new_st, best_network)
                verbose_end = time.time()
                logging_time += verbose_end - verbose_start

                helper(new_st)

                # plt.close(curr_net.canonical_structure())

        helper(SearchState(network, delta))
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

    def bfs(self, net: TensorNetwork):
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
                new_st = st.take_action(ac)
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
                    log_stats(search_stats, target_tensor, ts, new_st, best_network)
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
    factory = StructureFactory()
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
    factory.initialize(initial_net, max_ops=5)
    print(len(factory.space.nodes))
    print(len(factory.space.edges))

"""Class for Monte Carlo Tree Search."""

import math
import random
import time
from typing import Self

from pytens.search.state import SearchState, Split


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
                weight = (
                    child.wins / child.visits
                ) + exploration_weight * math.sqrt(
                    math.log(self.visits) / child.visits
                )

            choices_weights.append(weight)

        max_weight = max(choices_weights)
        return self.children[choices_weights.index(max_weight)]

    def expand(self):
        """Expand by creating a new child node for a random untried action."""
        legal_actions = self.state.get_legal_actions()
        tried_actions = [
            child.state.past_actions[-1] for child in self.children
        ]
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
                self.state.network.network.nodes[action.node][
                    "tensor"
                ].indices,
                "takes",
                time.time() - start,
            )
        else:
            print(
                "completing the action", action, "takes", time.time() - start
            )
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

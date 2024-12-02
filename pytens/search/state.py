"""Classes for search states."""
from typing import Sequence, Tuple, Self, Generator
import itertools
import copy

import numpy as np

from pytens.algs import NodeName, TensorNetwork

class Action:
    """Base action."""
    def __lt__(self, other) -> bool:
        return str(self) < str(other)

    def __hash__(self) -> int:
        return hash(self.__str__())

class Split(Action):
    """Split action."""

    def __init__(
        self,
        node: NodeName,
        left_indices: Sequence[int],
    ):
        self.node = node
        self.left_indices = left_indices

    def __str__(self) -> str:
        return f"Split({self.node}, {self.left_indices})"

    def execute(self, network: TensorNetwork) -> Tuple[NodeName, NodeName, NodeName]:
        """Execute a split action."""
        node_indices = network.network.nodes[self.node]["tensor"].indices
        # print("node indices", node_indices)
        # left_dims = [node_indices[i].size for i in self.left_indices]
        # right_dims = [node_indices[i].size for i in self.right_indices]
        right_indices = [i for i in range(len(node_indices)) if i not in self.left_indices]
        return network.split(
            self.node, self.left_indices, right_indices
        )


class Merge(Action):
    """Merge action."""

    def __init__(self, node1: NodeName, node2: NodeName):
        self.node1 = node1
        self.node2 = node2

    def __str__(self) -> str:
        return f"Merge({self.node1}, {self.node2})"

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
        self.past_actions = []  # How we reach this state
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
                    ac = Split(n, left_indices)
                    actions.append(ac)

        for n in self.network.network.nodes:
            for m in self.network.network.neighbors(n):
                if str(n) < str(m):
                    n_indices = self.network.network.nodes[n]["tensor"].indices
                    m_indices = self.network.network.nodes[m]["tensor"].indices
                    if len(set(n_indices).union(set(m_indices))) <= 5:
                        ac = Merge(n, m)
                        actions.append(ac)

        return actions

    def take_action(self, action: Action, split_errors: int = 0, no_heuristic: bool = False) -> Generator["SearchState", None, None]:
        """Return a new GameState after taking the specified action."""
        if isinstance(action, Split) and split_errors != 0:
            new_net = copy.deepcopy(self.network)
            try:
                indices = new_net.network.nodes[action.node]["tensor"].indices
                # print(indices, action.left_indices)
                left_sz = np.prod([indices[i].size for i in action.left_indices])
                right_sz = np.prod([indices[i].size for i in range(len(indices)) if i not in action.left_indices])
                max_sz = min(left_sz, right_sz)
                u, s, v = action.execute(new_net)
                # print(u, new_net.network.nodes[u]["tensor"].indices)
                # print(s, new_net.network.nodes[s]["tensor"].indices)
                # print(v, new_net.network.nodes[v]["tensor"].indices)
                # new_net.draw()
                # plt.show()
                u_val = new_net.network.nodes[u]["tensor"].value
                v_val = new_net.network.nodes[v]["tensor"].value
                # This should produce a lot of new states
                s_val = np.diag(new_net.network.nodes[s]["tensor"].value)
                
                slist = list(s_val * s_val)
                slist.reverse()
                truncpost = []
                for elem in np.cumsum(slist):
                    if elem <= self.curr_delta ** 2:
                        truncpost.append(elem)
                    else:
                        break

                if not no_heuristic and (len(truncpost) == 0 and max_sz == len(s_val)):
                    return

                split_num = min(split_errors, len(truncpost))
                # print("split_num", split_num)
                if split_num == 0:
                    tmp_net = copy.deepcopy(new_net)
                    tmp_net.merge(v, s)
                    new_state = SearchState(
                        tmp_net, self.curr_delta, max_ops=self.max_ops, threshold=self.threshold
                    )
                    new_state.past_actions = self.past_actions + [action]
                    new_state.used_ops = self.used_ops + 1
                    return new_state

                for idx, elem in enumerate(reversed(truncpost[-split_num:])):
                    truncation_rank = max(max_sz - len(truncpost) + idx, 1)
                    used_delta = elem

                    # it is possible to do the truncation at this point
                    tmp_net = copy.deepcopy(new_net)
                    # truncate u, s, v according to idx
                    
                    tmp_net.network.nodes[u]["tensor"].update_val_size(u_val[..., :truncation_rank])
                    tmp_net.network.nodes[s]["tensor"].update_val_size(np.diag(s_val[:truncation_rank]))
                    tmp_net.network.nodes[v]["tensor"].update_val_size(v_val[:truncation_rank, ...])
                    # tmp_net.draw()
                    # plt.show()
                    # print(tmp_net.network.nodes[u]["tensor"].indices)
                    # print(tmp_net.network.nodes[s]["tensor"].indices)
                    # print(tmp_net.network.nodes[v]["tensor"].indices)
                    tmp_net.merge(v, s)
                    # print("merging", v, s)
                    # tmp_net.draw()
                    # plt.show()

                    # print(idx, self.curr_delta ** 2, cum_slist[idx-1])
                    remaining_delta = float(np.sqrt(self.curr_delta**2 - used_delta))
                    # we cannot afford to put this into a list, so generator
                    new_state = SearchState(
                        tmp_net, remaining_delta, max_ops=self.max_ops, threshold=self.threshold
                    )
                    new_state.past_actions = self.past_actions + [action]
                    new_state.used_ops = self.used_ops + 1
                    # if new_state.network.cost() > self.network.cost():
                    #     continue

                    yield new_state
            except np.linalg.LinAlgError:
                pass

        elif isinstance(action, Split) and split_errors == 0:
            new_net = copy.deepcopy(self.network)
            indices = new_net.network.nodes[action.node]["tensor"].indices
            left_sz = np.prod([indices[i].size for i in action.left_indices])
            right_indices = [i for i in range(len(indices)) if i not in action.left_indices]
            right_sz = np.prod([indices[i].size for i in right_indices])
            max_sz = min(left_sz, right_sz)
            (u, v), new_delta = new_net.delta_split(action.node, action.left_indices, right_indices, delta=self.curr_delta)
            new_state = SearchState(
                new_net, new_delta, max_ops=self.max_ops, threshold=self.threshold
            )
            new_state.past_actions = self.past_actions + [action]
            new_state.used_ops = self.used_ops + 1
            index_sz = new_net.get_contraction_index(u, v)[0].size
            if max_sz == index_sz: # or new_state.network.cost() > self.network.cost():
                return

            yield new_state

        elif isinstance(action, Merge):
            new_net = copy.deepcopy(self.network)
            action.execute(new_net)
            # new_net.draw()
            # plt.show()
            new_state = SearchState(
                new_net, self.curr_delta, max_ops=self.max_ops, threshold=self.threshold
            )
            new_state.past_actions = self.past_actions + [action]
            new_state.used_ops = self.used_ops + 1
            # new_state.is_noop = len(new_net.network.nodes) == 1

            yield new_state

        else:
            raise TypeError("Unrecognized action type")

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

"""Classes for search states."""
from typing import Sequence, Tuple, Self, Generator, Dict
import itertools
import copy

import numpy as np
import matplotlib.pyplot as plt

from pytens.algs import NodeName, TensorNetwork, Index

class Action:
    """Base action."""
    def __lt__(self, other) -> bool:
        return str(self) < str(other)

    def __hash__(self) -> int:
        return hash(self.__str__())


class SplitIndex(Action):
    def __init__(self, indices: Sequence[Index]):
        self.indices = indices

    def __str__(self) -> str:
        return f"Split({[i.name for i in self.indices]})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SplitIndex):
            return False
        
        if len(self.indices) != len(other.indices):
            return False
        
        for i, j in zip(self.indices, other.indices):
            if i.name != j.name:
                return False
            
        return True
    
    def __hash__(self) -> int:
        return hash(self.__str__())
    
    def __lt__(self, other: Self) -> bool:
        if len(self.indices) != len(other.indices):
            return len(self.indices) < len(other.indices)
        
        return sorted(self.indices) < sorted(other.indices)

    def is_valid(self, past_actions) -> bool:
        """Check whether this action is valid given its execution history."""
        if self in past_actions:
            return False
        
        for ac in past_actions:
            if not isinstance(ac, SplitIndex):
                continue

            if len(ac.indices) > 1 and any([i in ac.indices for i in self.indices]):
                # print(f"{self} is invalid with past actions {[str(ac) for ac in past_actions]}")
                return False
            
        return True

    def to_split(self, net: TensorNetwork):
        """Convert a split index operation into split"""
        lca_node = None
        lca_indices = []
        # we should find a node where the expected indices and the unexpected indices are on different indices
        def postorder(visited, node):
            visited.add(node)
            results = []
            for m in net.network.neighbors(node):
                if m not in visited:
                    ok, finds = postorder(visited, m)
                    if not ok:
                        return False, []
                    
                    # print("get", finds, "for", m, "with parent", node)
                    inds = []
                    for x in finds:
                        inds.extend(list(x[1]))

                    # if finds include both desired and undesired, skip
                    desired = set(self.indices).intersection(set(inds))
                    undesired = set(inds).difference(set(self.indices))
                    if len(desired) > 0 and len(undesired) > 0:
                        return False, []

                    results.append((net.get_contraction_index(m, node)[0], inds))
            
            free_indices = net.free_indices()
            node_indices = net.network.nodes[node]["tensor"].indices
            for i in node_indices:
                if i in free_indices:
                    results.append((i, [i]))

            return True, results
        
        for n in net.network.nodes:
            # postorder traversal from each node and 
            # if we find each index 
            visited = set()
            # print("postordering", n)
            ok, results = postorder(visited, n)
            if ok:
                lca_node = n
                for i in self.indices:
                    for e, inds in results:
                        if i in inds:
                            lca_indices.append(e)
                            break

                break

        if lca_node is None:
            raise ValueError("Cannot find the lca for indices", self.indices)
        # net.draw()
        # plt.show()
        # Once we find the node and indices, we perform the split
        node_indices = net.network.nodes[lca_node]["tensor"].indices
        # print(path_views)
        # print(lca_node, self.indices, node_indices)
        # net.draw()
        # plt.show()
        left_indices = [node_indices.index(i) for i in lca_indices]
        
        return Split(lca_node, left_indices)

    def execute(self, net: TensorNetwork):
        """Execute the split index action on the given tensor network"""
        # find the nodes that include @indices@,
        # if there are multiple such nodes, go to the common ancestor
        ac = self.to_split(net)
        lca_node = ac.node
        node_indices = net.network.nodes[ac.node]["tensor"].indices
        left_indices = ac.left_indices
        right_indices = [i for i in range(len(node_indices)) if i not in left_indices]

        left_sz = np.prod([node_indices[i].size for i in left_indices])
        right_sz = np.prod([node_indices[i].size for i in right_indices])
        max_sz = min(left_sz, right_sz)
        u, s, v = net.split(lca_node, left_indices, right_indices, with_orthonormalize=True)
        return (u, s, v), max_sz

class SplitIndexAround(SplitIndex):
    """Split a given index set and truncate it to the desired rank size."""

    def __init__(self, indices, target_size):
        super().__init__(indices)
        self.target_size = target_size

    def __str__(self) -> str:
        return f"SplitIndexAround({self.indices, self.target_size})"

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
        self.ac_to_link = {}

    def get_legal_actions(self, index_actions=False):
        """Return a list of all legal actions in this state."""
        if index_actions:
            return self.get_legal_index_actions()
        
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

        # for n in self.network.network.nodes:
        #     for m in self.network.network.neighbors(n):
        #         if str(n) < str(m):
        #             n_indices = self.network.network.nodes[n]["tensor"].indices
        #             m_indices = self.network.network.nodes[m]["tensor"].indices
        #             if len(set(n_indices).union(set(m_indices))) <= 5:
        #                 ac = Merge(n, m)
        #                 actions.append(ac)

        return actions
    
    def get_legal_index_actions(self):
        """Produce a list of legal index splitting actions over the current network."""
        actions = []
        free_indices = self.network.free_indices()
        for k in range(1, len(free_indices) // 2 + 1):
            combs = list(itertools.combinations(free_indices, k))
            if len(free_indices) % 2 == 0 and k == len(free_indices) // 2:
                combs = combs[:len(combs) // 2]

            for comb in combs:
                ac = SplitIndex(comb)
                if ac not in self.past_actions and ac.is_valid(self.past_actions):
                    actions.append(SplitIndex(comb))

        return actions

    def take_action(self, action: Action, split_errors: int = 0, no_heuristic: bool = False) -> Generator["SearchState", None, None]:
        """Return a new GameState after taking the specified action."""
        if (isinstance(action, Split) and split_errors != 0) or isinstance(action, SplitIndex):
            # try the error splitting from large to small
            new_net = copy.deepcopy(self.network)
            try:
                if isinstance(action, Split):
                    indices = new_net.network.nodes[action.node]["tensor"].indices
                    # print(indices, action.left_indices)
                    left_sz = np.prod([indices[i].size for i in action.left_indices])
                    right_sz = np.prod([indices[i].size for i in range(len(indices)) if i not in action.left_indices])
                    max_sz = min(left_sz, right_sz)
                    u, s, v = action.execute(new_net)
                elif isinstance(action, SplitIndex):
                    if not action.is_valid(self.past_actions):
                        return
                    
                    (u, s, v), max_sz = action.execute(new_net)
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

                # print(slist[:10], truncpost, self.curr_delta ** 2)

                if not no_heuristic and (len(truncpost) == 0 and max_sz == len(s_val)):
                    # print("heuristic prune")
                    return
                
                # print("original truncpost", len(truncpost), "target_size", action.target_size, "max size", max_sz)
                if isinstance(action, SplitIndexAround):
                    target_trunc = max(len(s_val) - action.target_size + split_errors // 2, 0)
                    truncpost = truncpost[:target_trunc]

                # print("remaining truncpost", len(truncpost))

                split_num = min(split_errors, len(truncpost))
                # print("split_num", split_num)
                if split_num == 0:
                    tmp_net = copy.deepcopy(new_net)
                    truncation_rank = max(len(s_val) - len(truncpost), 1)
                    used_delta = truncpost[-1] if len(truncpost) > 0 else 0
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

                    remaining_delta = float(np.sqrt(self.curr_delta**2 - used_delta))
                    new_state = SearchState(
                        tmp_net, remaining_delta, max_ops=self.max_ops, threshold=self.threshold
                    )
                    new_state.past_actions = self.past_actions + [action]
                    new_state.used_ops = self.used_ops + 1
                    new_state.ac_to_link[action] = tmp_net.get_contraction_index(u, v)[0].name
                    yield new_state
                    return

                for idx, elem in enumerate(truncpost[-split_num:]):
                    truncation_rank = max(max_sz - len(truncpost) + split_num - idx - 1, 1)
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
                    new_state.ac_to_link[action] = tmp_net.get_contraction_index(u, v)[0].name
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
            new_index = new_net.get_contraction_index(u, v)[0]
            new_state.ac_to_link[action] = new_index.name
            index_sz = new_index.size
            if not no_heuristic and max_sz == index_sz: # or new_state.network.cost() > self.network.cost():
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
        return (self.curr_delta ** 2 / self.network.cost()) < (other.curr_delta ** 2 / other.network.cost())
        # return self.network.cost() > other.network.cost()
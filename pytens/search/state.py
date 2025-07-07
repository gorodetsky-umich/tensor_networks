"""Classes for search states."""

from typing import Sequence, Tuple, Self, Generator, Optional, Dict, List
import itertools
import copy

import numpy as np
import networkx as nx

from pytens.algs import NodeName, TreeNetwork, Index, SVDConfig, IndexName
from pytens.cross.cross import TensorFunc


class Action:
    """Base action."""

    def __init__(self):
        self.delta = None
        self.target_size = None
        self.indices = None

    def __lt__(self, other) -> bool:
        return str(self) < str(other)

    def __hash__(self) -> int:
        return hash(self.__str__())

    def is_valid(self, past_actions: Sequence["Action"]) -> bool:
        """Check whether the current action is valid against the history."""
        return True


class OSplit(Action):
    """Class for output-directed splits."""

    def __init__(
        self,
        indices: Sequence[Index],
        target_size: Optional[int] = None,
        delta: Optional[float] = None,
    ):
        super().__init__()
        self.indices = sorted(indices)
        self.target_size = target_size
        self.delta = delta

    def __str__(self) -> str:
        return f"OSplit({[i.name for i in self.indices]})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OSplit):
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

    def is_valid(self, past_actions: Sequence[Action]) -> bool:
        """Check whether this action is valid given its execution history."""
        if self in past_actions:
            return False

        for ac in past_actions:
            if not isinstance(ac, OSplit):
                continue

            if len(ac.indices) > 1 and any(
                i in ac.indices for i in self.indices
            ):
                return False

        return True

    # TODO: Reorganize the code in this function
    def to_isplit(self, net: TreeNetwork):
        """Convert an output-directed split to an input-directed one."""
        lca_node = None
        lca_indices = []

        # we should find a node where the expected indices and
        # the unexpected indices are on different indices
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
                    # print(desired, undesired)
                    if len(desired) > 0 and len(undesired) > 0:
                        return False, []

                    results.append(
                        (net.get_contraction_index(m, node)[0], inds)
                    )

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

        return ISplit(
            lca_node,
            left_indices,
            target_size=self.target_size,
            delta=self.delta,
        )

    def cross(
        self, net: TreeNetwork
    ) -> Tuple[NodeName, NodeName]:
        """Execute the split index action with cross approximation"""
        ac = self.to_isplit(net)
        return ac.cross(net)

    def svd(
        self,
        net: TreeNetwork,
        svd: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        compute_data: bool = True,
        compute_uv: bool = True,
    ):
        """Execute the split index action on the given tensor network"""
        # find the nodes that include @indices@,
        # if there are multiple such nodes, go to the common ancestor
        ac = self.to_isplit(net)
        return ac.svd(
            net, svd, compute_data=compute_data, compute_uv=compute_uv
        )


class ISplit(Action):
    """Class for input-directed splits."""

    def __init__(
        self,
        node: NodeName,
        left_indices: Sequence[int],
        target_size: Optional[int] = None,
        delta: Optional[float] = None,
    ):
        super().__init__()
        self.node = node
        self.left_indices = sorted(left_indices)
        self.target_size = target_size
        self.delta = delta

    def __str__(self) -> str:
        return f"ISplit({self.node}, {self.left_indices})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ISplit):
            return False

        if self.node != other.node:
            return False

        if len(self.left_indices) != len(other.left_indices):
            return False

        for i, j in zip(self.left_indices, other.left_indices):
            if i != j:
                return False

        return True

    def cross(
        self, net: TreeNetwork
    ) -> Tuple[NodeName, NodeName]:
        """Execute the split action with cross approximation."""
        (u, s, v), _ = net.svd(
            self.node,
            self.left_indices,
            SVDConfig(compute_data=False),
        )
        net.merge(v, s, compute_data=False)
        if self.target_size is not None:
            net.get_contraction_index(u, v)[0].with_new_size(self.target_size)

        return u, v

    def svd(
        self,
        net: TreeNetwork,
        svd: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        compute_data: bool = True,
        compute_uv: bool = True,
    ) -> Tuple[Tuple[NodeName, NodeName, NodeName], float]:
        """Execute a split action."""
        node_indices = net.node_tensor(self.node).indices
        linds = self.left_indices
        rinds = [i for i in range(len(node_indices)) if i not in linds]
        lszs = [node_indices[i].size for i in linds]
        rszs = [node_indices[i].size for i in rinds]
        max_sz = min(int(np.prod(lszs)), int(np.prod(rszs)))

        if svd is None:
            if compute_data:
                net.orthonormalize(self.node)
            (u, s, v), _ = net.svd(
                self.node,
                linds,
                SVDConfig(compute_data=compute_data, compute_uv=compute_uv),
            )
        else:
            # print("read preprocessing result")
            (u, s, v), _ = net.svd(
                self.node,
                linds,
                SVDConfig(compute_data=False, compute_uv=compute_uv),
            )
            net.node_tensor(u).update_val_size(svd[0].reshape(*lszs, -1))
            net.node_tensor(s).update_val_size(np.diag(svd[1]))
            net.node_tensor(v).update_val_size(svd[2].reshape(-1, *rszs))

        # truncate the network to the target ranks
        s_val = np.diag(net.node_tensor(s).value)
        trunc_error = np.cumsum(np.flip(s_val**2))
        if self.target_size is not None:
            r = self.target_size
            err = trunc_error[max_sz - r - 1]
        elif self.delta is not None:
            # find the first index where the truncation error is less than delta
            r_discard = np.searchsorted(trunc_error, self.delta**2)
            r = max_sz - r_discard
            err = trunc_error[r_discard - 1] if r_discard > 0 else 0.0
        else:
            r = max_sz
            err = 0.0

        if compute_data:
            net.node_tensor(s).update_val_size(net.value(s)[:r, :r])
            if compute_uv:
                net.node_tensor(u).update_val_size(net.value(u)[..., :r])
                net.node_tensor(v).update_val_size(net.value(v)[:r])

        return (u, s, v), err

    # TODO: reorganize the code in this funciton
    def to_osplit(self, st, idx):
        """Convert a split action to OSplit."""
        connect_nodes = []
        for n, d in st.network.network.nodes(data=True):
            for ind in d["tensor"].indices:
                if ind.name == st.links[idx]:
                    connect_nodes.append(n)
                    break

        if len(connect_nodes) != 2:
            print("Unusual edge label found in nodes:", connect_nodes)

        all_free_indices = st.network.free_indices()
        tmp_net = copy.deepcopy(st.network.network)
        tmp_net.remove_edge(connect_nodes[0], connect_nodes[1])
        curr_indices = None
        for subgraph in nx.connected_components(tmp_net):
            tn = TreeNetwork()
            tn.network = st.network.network.subgraph(subgraph)
            indices = [
                ind for ind in tn.free_indices() if ind in all_free_indices
            ]
            if (
                curr_indices is None
                or len(indices) < len(curr_indices)
                or (
                    len(indices) == len(curr_indices)
                    and indices < curr_indices
                )
            ):
                curr_indices = indices

        assert curr_indices is not None
        return OSplit(curr_indices)


class Merge(Action):
    """Merge action."""

    def __init__(self, node1: NodeName, node2: NodeName):
        self.node1 = node1
        self.node2 = node2

    def __str__(self) -> str:
        return f"Merge({self.node1}, {self.node2})"

    def execute(self, network: TreeNetwork):
        """Execute a merge action."""
        network.merge(self.node1, self.node2)
        return network


class SearchState:
    """Class for representation of intermediate search states."""

    def __init__(
        self,
        net: TreeNetwork,
        delta: float,
        max_ops: int = 5,
    ):
        self.network = net
        self.curr_delta = delta
        self.past_actions = []  # How we reach this state
        self.max_ops = max_ops
        self.links: List[IndexName] = []

    def count_actions_of_size(self, k: int = 2):
        """Count the number of actions of the given size in the history."""
        cnt = 0
        for ac in self.past_actions:
            if len(ac.indices) >= k:
                cnt += 1

        return cnt

    # TODO: check how to implement isplit osplit in a better way
    def get_legal_actions(self, index_actions=False):
        """Return a list of all legal actions in this state."""
        if index_actions:
            return self.get_legal_index_actions()

        actions = []
        for n in self.network.network.nodes:
            indices = self.network.network.nodes[n]["tensor"].indices
            indices = range(len(indices))
            half_size = len(indices) // 2
            # get all partitions of indices
            for sz in range(1, half_size + 1):
                combs = list(itertools.combinations(indices, sz))
                if len(indices) % 2 == 0 and sz == half_size:
                    combs = combs[: len(combs) // 2]

                for comb in combs:
                    left_indices = comb
                    ac = ISplit(n, left_indices)
                    actions.append(ac)

        return actions

    @staticmethod
    def all_index_combs(
        free_indices: Sequence[Index], k: Optional[int] = None
    ) -> Generator[Sequence[Index], None, None]:
        """Compute all index partitions for the given index set."""
        half_size = len(free_indices) // 2
        if k is not None:
            upper = min(k, half_size + 1)
        else:
            upper = half_size + 1

        for i in range(1, upper):
            combs = list(itertools.combinations(free_indices, i))
            if len(free_indices) % 2 == 0 and i == half_size:
                combs = combs[: len(combs) // 2]

            yield from combs

    def get_legal_index_actions(self):
        """
        Produce a list of legal index splitting actions
        over the current network.
        """
        actions = []
        free_indices = self.network.free_indices()
        for comb in SearchState.all_index_combs(free_indices):
            ac = OSplit(comb)
            if not self.past_actions or (
                self.past_actions[-1] < ac and ac.is_valid(self.past_actions)
            ):
                actions.append(ac)

        return actions

    def take_action(
        self,
        action: Action,
        svd: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        tensor_func: Optional[TensorFunc] = None,
    ) -> Optional["SearchState"]:
        """Return a new SearchState after taking the specified action."""
        if isinstance(action, (ISplit, OSplit)):
            # try the error splitting from large to small
            new_net = copy.deepcopy(self.network)

            if not action.is_valid(self.past_actions):
                return None

            if action.delta is None and action.target_size is None:
                action.delta = self.curr_delta

            new_state = SearchState(
                new_net,
                self.curr_delta,
                max_ops=self.max_ops,
            )

            if tensor_func is not None:
                # print("running cross for", ac)
                u, v = action.cross(new_net)
                # new_err = cross_st.ranks_and_errors[-1][1]
                # new_state.curr_delta =
                # np.sqrt(self.curr_delta ** 2 - new_err ** 2)
            else:
                # we allow specify the node values
                (u, s, v), used_delta = action.svd(new_net, svd)
                new_net.merge(v, s)
                remaining_delta = np.sqrt(self.curr_delta**2 - used_delta)
                new_state.curr_delta = remaining_delta

            new_ind = new_net.get_contraction_index(u, v)[0].name
            new_state.links.append(new_ind)
            new_state.past_actions = self.past_actions + [action]
            return new_state

        elif isinstance(action, Merge):
            new_net = copy.deepcopy(self.network)
            action.execute(new_net)
            # new_net.draw()
            # plt.show()
            new_state = SearchState(
                new_net,
                self.curr_delta,
                max_ops=self.max_ops,
            )
            new_state.past_actions = self.past_actions + [action]
            return new_state

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

        assert root is not None
        root = self.network.orthonormalize(root)
        _, self.curr_delta = self.network.round(root, self.curr_delta)

    def __lt__(self, other: Self) -> bool:
        # return (self.curr_delta**2 / self.network.cost()) < (
        #     other.curr_delta**2 / other.network.cost()
        # )
        return self.network.cost() < other.network.cost()

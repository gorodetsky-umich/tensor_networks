"""Classes for search states."""

from typing import Sequence, Tuple, Self, Generator, Optional
import itertools
import copy

import numpy as np
import networkx as nx

from pytens.algs import NodeName, TensorNetwork, Index, SVDConfig
from pytens.search.configuration import SearchConfig


class Action:
    """Base action."""

    def __lt__(self, other) -> bool:
        return str(self) < str(other)

    def __hash__(self) -> int:
        return hash(self.__str__())

    def is_valid(self, _: Sequence["Action"]) -> bool:
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

    def is_valid(self, past_actions) -> bool:
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

    def to_isplit(self, net: TensorNetwork):
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

        return ISplit(lca_node, left_indices)

    def execute(self, net: TensorNetwork, svd: Tuple[np.ndarray] = None):
        """Execute the split index action on the given tensor network"""
        # find the nodes that include @indices@,
        # if there are multiple such nodes, go to the common ancestor
        ac = self.to_isplit(net)
        return ac.execute(net, svd)


class ISplit(Action):
    """Class for input-directed splits."""

    def __init__(
        self,
        node: NodeName,
        left_indices: Sequence[int],
        target_size: Optional[int] = None,
        delta: Optional[float] = None,
    ):
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

    def execute(
        self, net: TensorNetwork, svd: Tuple[np.ndarray] = None
    ) -> Tuple[Tuple[NodeName, NodeName, NodeName], int]:
        """Execute a split action."""
        node_indices = net.network.nodes[self.node]["tensor"].indices
        l_indices = self.left_indices
        r_indices = [i for i in range(len(node_indices)) if i not in l_indices]

        left_szs = [node_indices[i].size for i in l_indices]
        left_sz = np.prod(left_szs)
        right_szs = [node_indices[i].size for i in r_indices]
        right_sz = np.prod(right_szs)
        max_sz = min(left_sz, right_sz)

        if svd is None:
            (u, s, v), _ = net.svd(
                self.node, l_indices, SVDConfig(with_orthonormal=True)
            )
        else:
            # print("read preprocessing result")
            (u, s, v), _ = net.svd(
                self.node, l_indices, SVDConfig(compute_data=False)
            )
            net.network.nodes[u]["tensor"].update_val_size(
                svd[0].reshape(*left_szs, -1)
            )
            net.network.nodes[s]["tensor"].update_val_size(np.diag(svd[1]))
            net.network.nodes[v]["tensor"].update_val_size(
                svd[2].reshape(-1, *right_szs)
            )

        return (u, s, v), max_sz

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
            tn = TensorNetwork()
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

        return OSplit(curr_indices)


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
        self,
        net: TensorNetwork,
        delta: float,
        threshold: float = 0.1,
        max_ops: int = 5,
    ):
        self.network = net
        self.curr_delta = delta
        self.past_actions = []  # How we reach this state
        self.max_ops = max_ops
        self.threshold = threshold
        self.is_noop = False
        self.links = []

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
                    ac = ISplit(n, left_indices)
                    actions.append(ac)

        return actions

    @staticmethod
    def all_index_combs(
        free_indices: Sequence[Index],
    ) -> Generator[Sequence[Index], None, None]:
        """Compute all index partitions for the given index set."""
        for k in range(1, len(free_indices) // 2 + 1):
            combs = list(itertools.combinations(free_indices, k))
            if len(free_indices) % 2 == 0 and k == len(free_indices) // 2:
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

    def truncate(
        self,
        new_net: TensorNetwork,
        usv: Tuple[Tuple[NodeName, NodeName, NodeName], int],
        config: SearchConfig,
        target_size: int = None,
    ) -> Generator["SearchState", None, None]:
        """Truncate the node u, s, v in the specified tensor network."""
        [u, s, v], max_sz = usv
        u_val = new_net.network.nodes[u]["tensor"].value
        v_val = new_net.network.nodes[v]["tensor"].value
        s_val = np.diag(new_net.network.nodes[s]["tensor"].value)

        slist = list(s_val * s_val)
        slist.reverse()
        truncpost = []
        for elem in np.cumsum(slist):
            if elem <= self.curr_delta**2:
                truncpost.append(elem)
            else:
                break

        if len(truncpost) == 0:
            if config.heuristics.prune_full_rank and max_sz == len(s_val):
                return

            tmp_net = copy.deepcopy(new_net)
            tmp_net.merge(v, s)

            remaining_delta = self.curr_delta
            new_state = SearchState(
                tmp_net,
                remaining_delta,
                max_ops=self.max_ops,
                threshold=self.threshold,
            )
            new_state.links.append(tmp_net.get_contraction_index(u, v)[0].name)

            yield new_state
            return

        split_errors = config.rank_search.error_split_stepsize
        if target_size is not None:
            target_trunc = max(len(s_val) - target_size + split_errors // 2, 0)
            truncpost = truncpost[:target_trunc]

        # print("remaining truncpost", len(truncpost))

        if split_errors == 0:
            split_num = 1
        else:
            split_num = min(split_errors, len(truncpost))

        for idx, elem in enumerate(truncpost[-split_num:]):
            truncation_rank = max(
                len(s_val) - len(truncpost) + split_num - idx - 1, 1
            )
            used_delta = truncpost[-1] if len(truncpost) > 0 else 0

            # it is possible to do the truncation at this point
            tmp_net = copy.deepcopy(new_net)
            # truncate u, s, v according to idx

            tmp_net.network.nodes[u]["tensor"].update_val_size(
                u_val[..., :truncation_rank]
            )
            tmp_net.network.nodes[s]["tensor"].update_val_size(
                np.diag(s_val[:truncation_rank])
            )
            tmp_net.network.nodes[v]["tensor"].update_val_size(
                v_val[:truncation_rank, ...]
            )
            tmp_net.merge(v, s)

            remaining_delta = float(np.sqrt(self.curr_delta**2 - used_delta))
            new_state = SearchState(
                tmp_net,
                remaining_delta,
                max_ops=self.max_ops,
                threshold=self.threshold,
            )
            new_state.links.append(tmp_net.get_contraction_index(u, v)[0].name)

            yield new_state

    def take_action(
        self,
        action: Action,
        config: SearchConfig,
        svd: Tuple[np.ndarray] = None,
    ) -> Generator["SearchState", None, None]:
        """Return a new GameState after taking the specified action."""
        if isinstance(action, (ISplit, OSplit)):
            # try the error splitting from large to small
            new_net = copy.deepcopy(self.network)

            if not action.is_valid(self.past_actions):
                return

            if action.delta is not None:
                self.curr_delta = action.delta

            try:
                # we allow specify the node values
                exec_result = action.execute(new_net, svd)
                for new_state in self.truncate(
                    new_net,
                    exec_result,
                    config=config,
                    target_size=action.target_size,
                ):
                    new_state.past_actions = self.past_actions + [action]
                    yield new_state
            except np.linalg.LinAlgError:
                pass

        elif isinstance(action, Merge):
            new_net = copy.deepcopy(self.network)
            action.execute(new_net)
            # new_net.draw()
            # plt.show()
            new_state = SearchState(
                new_net,
                self.curr_delta,
                max_ops=self.max_ops,
                threshold=self.threshold,
            )
            new_state.past_actions = self.past_actions + [action]
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
        return (self.curr_delta**2 / self.network.cost()) < (
            other.curr_delta**2 / other.network.cost()
        )
        # return self.network.cost() > other.network.cost()

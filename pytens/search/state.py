"""Classes for search states."""

import copy
import itertools
from typing import Dict, Generator, List, Optional, Self, Sequence, Tuple
import logging

import networkx as nx
import numpy as np

from pytens.algs import (
    FoldedTensorTrain,
    HierarchicalTucker,
    Index,
    IndexName,
    NodeName,
    SVDConfig,
    TensorTrain,
    TreeNetwork,
)
from pytens.cross.cross import TensorFunc
from pytens.types import IndexMerge, PartitionStatus, SVDAlgorithm
from pytens.search.types import Action

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OSplit(Action):
    """Class for output-directed splits."""

    def __init__(
        self,
        indices: Sequence[Index],
        target_size: Optional[int] = None,
        delta: Optional[float] = None,
        reversible: bool = False,
        reverse_edge: Optional[Tuple[NodeName, NodeName]] = None,
    ):
        super().__init__()
        self.indices = sorted(indices)
        self.target_size = target_size
        self.delta = delta
        self.reversible = reversible
        self.reverse_edge = reverse_edge

    def __str__(self) -> str:
        return f"OSplit({[i.name for i in self.indices]})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OSplit):
            return False

        if len(self.indices) != len(other.indices):
            return False

        self_names = set(i.name for i in self.indices)
        other_names = set(i.name for i in other.indices)
        return self_names == other_names

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
        res = net.partition_node(self.indices)
        if res.code == PartitionStatus.EXIST:
            return None

        while res.code != PartitionStatus.OK:
            print(
                "Cannot find the lca for indices",
                self.indices,
                "try swap indices",
            )
            print("before swap", net)
            # swap indices until they are in the same subtree
            ind_nodes = [net.node_by_free_index(i.name) for i in self.indices]
            net.swap(ind_nodes)
            if len(ind_nodes) < 2:
                raise ValueError(
                    "Cannot find the common ancestor for the given indices"
                )

            for n in ind_nodes[1:]:
                net.merge(ind_nodes[0], n)

            res = net.partition_node(self.indices)

        # net.draw()
        # plt.show()
        # Once we find the node and indices, we perform the split
        node_indices = net.node_tensor(res.lca_node).indices
        # print(path_views)
        # print(lca_node, self.indices, node_indices)
        # net.draw()
        # plt.show()

        left_indices = [node_indices.index(i) for i in res.lca_indices]
        left_indices = list(set(left_indices))

        return ISplit(
            res.lca_node,
            left_indices,
            target_size=self.target_size,
            delta=self.delta,
        )

    def cross(self, net: TreeNetwork) -> Tuple[NodeName, NodeName]:
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
        if ac is None:
            return None

        # print("OSplit svd", ac)
        # print(net)
        return ac.svd(
            net, svd, compute_data=compute_data, compute_uv=compute_uv
        )

    def svals(
        self,
        net: TreeNetwork,
        svd: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        algo: SVDAlgorithm = SVDAlgorithm.SVD,
        max_rank=100,
        orthonormal=None,
        eps=0,
        rand: bool = True,
    ) -> np.ndarray:
        """Compute the singular values of the split action."""
        logger.debug("performing actions: %s", self)

        if algo == SVDAlgorithm.CROSS:
            return net.svals_by_cross(self.indices, max_rank=max_rank, eps=eps)
        
        if isinstance(net, TensorTrain):
            logger.debug("computing singular values for a tensor train: %s", net)
            # if algo == SVDAlgorithm.FOLD:
            #     return net.svals_by_fold(self.indices, max_rank=max_rank)

            return net.svals_by_merge(self.indices, max_rank=max_rank, rand=rand)

            # return net.svals(self.indices, max_rank=max_rank, delta=delta)

        if isinstance(net, HierarchicalTucker):
            return net.svals(
                self.indices, max_rank=max_rank, orthonormal=orthonormal
            )

        if isinstance(net, FoldedTensorTrain):
            logger.debug("computing singular values for a folded tensor train")
            return net.svals(self.indices, max_rank=max_rank, rand=rand)

        ac = self.to_isplit(net)
        return ac.svals(net, svd)


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

    def cross(self, net: TreeNetwork) -> Tuple[NodeName, NodeName]:
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
        linds = self.left_indices

        # TODO: clean up this if block
        if svd is None:
            if compute_data:
                net.orthonormalize(self.node)

            node_indices = net.node_tensor(self.node).indices
            rinds = [i for i in range(len(node_indices)) if i not in linds]
            lszs = [node_indices[i].size for i in linds]
            rszs = [node_indices[i].size for i in rinds]
            max_sz = min(int(np.prod(lszs)), int(np.prod(rszs)))

            (u, s, v), _ = net.svd(
                self.node,
                linds,
                SVDConfig(
                    delta=0, compute_data=compute_data, compute_uv=compute_uv
                ),
            )
        else:
            node_indices = net.node_tensor(self.node).indices
            rinds = [i for i in range(len(node_indices)) if i not in linds]
            lszs = [node_indices[i].size for i in linds]
            rszs = [node_indices[i].size for i in rinds]
            max_sz = min(int(np.prod(lszs)), int(np.prod(rszs)))

            # print("read preprocessing result")
            (u, s, v), _ = net.svd(
                self.node,
                linds,
                SVDConfig(delta=0, compute_data=False, compute_uv=compute_uv),
            )
            net.node_tensor(u).update_val_size(svd[0].reshape(*lszs, -1))
            net.node_tensor(s).update_val_size(np.diag(svd[1]))
            net.node_tensor(v).update_val_size(svd[2].reshape(-1, *rszs))

        # truncate the network to the target ranks
        s_val = np.diag(net.node_tensor(s).value)
        trunc_error = np.cumsum(np.flip(s_val**2))
        if self.target_size is not None:
            max_sz = min(max_sz, len(trunc_error))
            r = min(max_sz, self.target_size)
            # r = self.target_size
            if r < max_sz:
                err = trunc_error[max_sz - r - 1]
            else:
                err = 0.0
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

    def svals(
        self,
        net: TreeNetwork,
        svd: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute the singular values for the current split action."""
        (_, s, _), _ = self.svd(net, svd, compute_data=True, compute_uv=False)
        return np.diag(net.value(s))

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
        actions = []
        for subgraph in nx.connected_components(tmp_net):
            tn = TreeNetwork()
            tn.network = st.network.network.subgraph(subgraph)
            indices = [
                ind for ind in tn.free_indices() if ind in all_free_indices
            ]
            actions.append(OSplit(indices))

        return min(actions)


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
    def get_legal_actions(self, index_actions=False, merge_ops=None):
        """Return a list of all legal actions in this state."""
        if index_actions:
            return self.get_legal_index_actions(merge_ops)

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
        free_indices = sorted(free_indices)
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

    def get_legal_index_actions(self, merge_ops=None):
        """
        Produce a list of legal index splitting actions
        over the current network.
        """
        actions = []
        if merge_ops is None:
            free_indices = [[ind] for ind in self.network.free_indices()]
        else:
            free_indices = []
            for merge_op in merge_ops:
                free_indices.append(merge_op.indices)

            for ind in self.network.free_indices():
                found = False
                for merge_op in merge_ops:
                    if ind in merge_op.indices:
                        found = True
                        break

                if not found:
                    free_indices.append([ind])

        sorted_free_indices = list(sorted(free_indices))
        for comb in SearchState.all_index_combs(sorted_free_indices):
            ac = OSplit([ind for ind_group in comb for ind in ind_group])

            # consider the complement index set
            ac_comp = OSplit([ind for ind in self.network.free_indices() if ind not in ac.indices])
            ac = min(ac, ac_comp)

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

            # if not action.is_valid(self.past_actions):
            #     return None

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
                res = action.svd(new_net, svd)
                if res is None:  # no-op
                    return self

                (u, s, v), used_delta = res
                new_net.merge(v, s)
                logger.debug("current delta: %s, used delta: %s", self.curr_delta, used_delta)
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

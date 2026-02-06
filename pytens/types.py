"""Type definitions."""

import logging
import itertools
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Self, Sequence, Union, Tuple

import numpy as np

logger = logging.getLogger(__name__)

IntOrStr = Union[str, int]
IndexName = IntOrStr
IndexChain = Union[List[int], Tuple[int]]
NodeName = IntOrStr


@dataclass(frozen=True, eq=True)
class Index:
    """Class for denoting an index."""

    name: Union[str, int]
    size: int
    value_choices: Sequence[float] = tuple([])

    def with_new_size(self, new_size: int) -> "Index":
        """Create a new index with same name but new size"""
        return Index(self.name, new_size)

    def with_new_name(self, name: IntOrStr) -> "Index":
        """Create a new index with same size but new name"""
        return Index(name, self.size)

    def with_new_rng(self, rng: Sequence[float]) -> "Index":
        """Create a new index with different value choices"""
        return Index(self.name, self.size, rng)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return False
        return self.name == other.name and self.size == other.size

    def __lt__(self, other: Self) -> bool:
        return str(self.name) < str(other.name)

    def __hash__(self) -> int:
        return hash((self.name, self.size))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data_dict: dict) -> "Index":
        """Reconstruct from dictionary."""
        return cls(**data_dict)


@dataclass
class SVDConfig:
    """Configuration fields for SVD in tensor networks."""

    delta: float = 1e-5
    with_orthonormal: bool = True
    compute_data: bool = True


class NodeInfo:
    """Information at each dim tree node."""

    def __init__(
        self,
        nodes: List["DimTreeNode"],
        indices: List[Index],
        vals: np.ndarray,
    ):
        self.nodes = nodes
        self.indices = indices
        self.vals = vals
        self.rank = 0


class DimTreeNode:
    """Class for a dimension tree node"""

    def __init__(  # pylint: disable=R0913,R0917
        self,
        node: NodeName,
        indices: List[Index],
        free_indices: List[Index],
        up_info: NodeInfo,
        down_info: NodeInfo,
    ):
        self.node = node
        self.indices = indices
        self.free_indices = free_indices
        self.up_info = up_info
        self.down_info = down_info
        self.perm = list(
            range(
                len(free_indices) + len(down_info.nodes) + len(up_info.nodes)
            )
        )

    def __lt__(self, other: Self) -> bool:
        return sorted(self.indices) < sorted(other.indices)

    def preorder(self) -> Sequence["DimTreeNode"]:
        """Get the list of tree nodes in the pre-order traversal."""
        results = [self]
        for c in self.down_info.nodes:
            results = list(itertools.chain(results, c.preorder()))

        return list(results)

    def increment_ranks(
        self, kickrank: int = 1, max_rank: Optional[int] = None
    ) -> None:
        """Increment the ranks without value modification"""
        self.up_info.rank += kickrank
        if max_rank is not None:
            self.up_info.rank = min(max_rank, self.up_info.rank)

        for c in self.down_info.nodes:
            c.increment_ranks(kickrank, max_rank)

    def ranks(self) -> List[int]:
        """Get all up ranks in the dimension tree."""
        res = [self.up_info.rank]
        for c in self.down_info.nodes:
            res.extend(c.ranks())

        return res

    def bound_ranks(self) -> None:
        """Adjust the ranks according to the ranks of neighbor edges"""
        # if we move leaves to root
        rank_up = 1
        for c in self.down_info.nodes:
            if c.up_info.rank != 0:
                rank_up *= c.up_info.rank

        for ind in self.free_indices:
            rank_up *= ind.size

        # if we move root to leaves
        rank_down = self.up_info.rank
        for p in self.up_info.nodes:
            rank_down = 1
            if p.up_info.rank != 0:
                rank_down *= p.up_info.rank

            for s in p.down_info.nodes:
                if s.node != self.node and s.up_info.rank != 0:
                    rank_down *= s.up_info.rank

            for ind in p.free_indices:
                rank_down *= ind.size

        # rank_up = max(1, rank_up)
        # rank_down = max(1, rank_down)
        logger.debug(
            "node: %s, indices: %s, rank_up: %i, rank_down: %i, curr_rankL %i",
            self.node,
            self.free_indices,
            rank_up,
            rank_down,
            self.up_info.rank,
        )
        self.up_info.rank = min([rank_up, rank_down, self.up_info.rank])

        for c in self.down_info.nodes:
            c.bound_ranks()

    def add_values(self, up_vals: np.ndarray) -> None:
        """Initialize the up and down values for the given dimension tree."""
        # if len(self.up_info.nodes) == 0:
        #     self.up_info.rank = 1
        # else:
        #     self.up_info.rank += len(up_vals)

        for c in self.down_info.nodes:
            cvals = up_vals[:, [self.indices.index(ind) for ind in c.indices]]
            c.up_info.vals = np.append(c.up_info.vals, cvals, axis=0)
            c.up_info.vals = c.up_info.vals[: c.up_info.rank]
            c.add_values(cvals)

    def locate(self, node: NodeName) -> Optional["DimTreeNode"]:
        """Locate a node by its name."""
        if node == self.node:
            return self

        for c in self.down_info.nodes:
            res = c.locate(node)
            if res is not None:
                return res

        return None

    def leaves(self) -> Sequence["DimTreeNode"]:
        """Get the leaf nodes in the current tree."""
        results = []
        if len(self.up_info.nodes) == 0:
            results.append(self)
            return results

        for c in self.up_info.nodes:
            results.extend(c.leaves())

        return results

    def height(self) -> int:
        """Get the height of the tree."""
        max_c = 0
        for c in self.up_info.nodes:
            max_c = max(max_c, c.height())

        return max_c + 1

    def path(self, node1: NodeName, node2: NodeName) -> List["DimTreeNode"]:
        """Get the list of nodes between the source and destination."""

        n1 = self.locate(node1)
        assert n1 is not None
        n2 = self.locate(node2)
        assert n2 is not None
        # find the common ancestor that subsumes both n1 and n2

        res = [n1]
        p = n1
        while p is not None:
            if all(ind in p.indices for ind in n1.indices + n2.indices):
                break

            p = p.up_info.nodes[0]
            res.append(p)

        if p is None:
            raise RuntimeError("not a valid tree")

        p2 = n2
        res2 = [p2]
        while p2 is not None and p2 != p:
            p2 = p2.up_info.nodes[0]
            res2.append(p2)

        if p2 is None:
            raise RuntimeError("not a valid tree")

        return res + list(reversed(res2[:-1]))

    def distance(self, node1: NodeName, node2: NodeName) -> int:
        """Get the distance between the two indices."""
        return len(self.path(node1, node2))

    def entries(self) -> np.ndarray:
        """Extract the up entries"""

        if len(self.up_info.vals) != 0:
            vals = self.up_info.vals
        else:
            vals = np.empty((0, len(self.up_info.indices)))

        return vals

    def known_entries(self) -> np.ndarray:
        """Extract the up and down entries"""

        vals = np.empty((0, len(self.indices)))
        if len(self.up_info.vals) != 0:
            vals = np.concat([self.down_info.vals, self.up_info.vals], axis=-1)

        self_inds = self.down_info.indices + self.up_info.indices
        for c in self.down_info.nodes:
            cvals = c.known_entries()
            # cvals follows the order of
            # c.down_info.indices + c.up_info.indices
            # reorder the values to match self
            cindices = c.down_info.indices + c.up_info.indices
            perm = [self_inds.index(ind) for ind in cindices]
            vals = np.concat([vals, cvals[:, perm]], axis=0)

        return vals

    def highest_frontier(
        self, indices: Sequence[Index]
    ) -> List["DimTreeNode"]:
        """Find the frontier of nodes that contain the given indices."""
        inds = self.indices
        if len(inds) > 0 and all(ind in indices for ind in inds):
            return [self]

        res = []
        for c in self.down_info.nodes:
            res.extend(c.highest_frontier(indices))

        return res

    def sibling(self, node: "DimTreeNode") -> "DimTreeNode":
        """Get one of the sibling node of the given node"""
        if len(node.up_info.nodes) != 1:
            raise ValueError("root node does not have a sibling")

        p = node.up_info.nodes[0]
        for c in p.down_info.nodes:
            if c.node == node.node:
                continue

            return c

        raise ValueError("No sibling for the given node")

    def is_ancestor(self, other: "DimTreeNode") -> bool:
        """Return true if the current node is an ancestor of other"""
        while len(other.up_info.nodes) > 0:
            other = other.up_info.nodes[0]
            if other.node == self.node:
                return True

        return False

"""Type definitions."""

import copy
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Self, Sequence, Union, Tuple

import numpy as np
import pydantic

IntOrStr = Union[str, int]
NodeName = IntOrStr
IndexName = IntOrStr


@dataclass(frozen=True, eq=True)
class Index:
    """Class for denoting an index."""

    name: Union[str, int]
    size: int
    value_choices: Sequence[float] = tuple([])

    def with_new_size(self, new_size: int) -> "Index":
        """Create a new index with same name but new size"""
        return Index(self.name, new_size)

    def with_new_name(self, name: IndexName) -> "Index":
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

    def __gt__(self, other: Self) -> bool:
        return str(self.name) > str(other.name)

    def __hash__(self) -> int:
        return hash((self.name, self.size))


@dataclass
class SVDConfig:
    """Configuration fields for SVD in tensor networks."""

    delta: float = 1e-6
    compute_data: bool = True
    compute_uv: bool = True


class IndexMerge(pydantic.BaseModel):
    """An index merge request and response."""

    indices: Sequence[Index]
    result: Optional[Index] = None

    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.__dict__.values()))


class IndexSplit(pydantic.BaseModel):
    """An index split request and response."""

    index: Index
    shape: Sequence[int]
    result: Optional[Sequence[Index]] = None

    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.__dict__.values()))


class IndexPermute(pydantic.BaseModel):
    """Permute all indices in a function"""

    perm: Sequence[int]
    unperm: Sequence[int]

    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.__dict__.values()))


def split_index(
    ind: Index, indices: List[Index], vals: np.ndarray, split_op: IndexSplit
) -> Tuple[List[Index], np.ndarray]:
    assert split_op.result is not None

    pos = indices.index(ind)
    split_sizes = [i.size for i in split_op.result]
    new_vals = np.empty(
        (vals.shape[0], vals.shape[1] - 1 + len(split_sizes)), dtype=int
    )
    new_vals[:, : vals.shape[1] - 1] = np.hstack(
        [vals[:, :pos], vals[:, pos + 1 :]]
    )
    new_vals[:, -len(split_sizes) :] = np.vstack(
        np.unravel_index(vals[:, pos], split_sizes)
    ).T
    indices.remove(ind)
    indices.extend(split_op.result)
    return indices, new_vals

class NodeInfo:
    """Information at each dim tree node."""

    def __init__(self, nodes: List["DimTreeNode"], indices: List[Index], vals: np.ndarray):
        self.nodes = nodes
        self.indices = indices
        self.vals = vals
        self.rank = 0

class DimTreeNode:
    """Class for a dimension tree node"""

    def __init__(
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

    def __lt__(self, other: Self) -> bool:
        return sorted(self.indices) < sorted(other.indices)

    def preorder(self) -> Sequence["DimTreeNode"]:
        """Get the list of tree nodes in the pre-order traversal."""
        results = [self]
        for c in self.down_info.nodes:
            results = itertools.chain(results, c.preorder())

        return list(results)

    def increment_ranks(self, kickrank: int = 1) -> None:
        """Increment the ranks without value modification"""
        self.up_info.rank += kickrank
        self.down_info.rank += kickrank
        for c in self.down_info.nodes:
            c.increment_ranks(kickrank)

    def ranks(self) -> List[int]:
        res = [(self.up_info.rank, self.down_info.rank)]
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
        rank_down = self.down_info.rank
        for p in self.up_info.nodes:
            rank_down = 1
            if p.down_info.rank != 0:
                rank_down *= p.down_info.rank

            for s in p.down_info.nodes:
                if s.node != self.node and s.up_info.rank != 0:
                    rank_down *= s.up_info.rank

            for ind in p.free_indices:
                rank_down *= ind.size

        # rank_up = max(1, rank_up)
        # rank_down = max(1, rank_down)
        self.up_info.rank = min([rank_up, rank_down, self.up_info.rank])
        self.down_info.rank = min([rank_up, rank_down, self.down_info.rank])

        for c in self.down_info.nodes:
            c.bound_ranks()

    def add_values(self, up_vals: np.ndarray) -> None:
        """Initialize the up and down values for the given dimension tree."""
        if len(self.up_info.nodes) == 0:
            self.up_info.rank = 1
        else:
            self.up_info.rank += len(up_vals)

        for c in self.down_info.nodes:
            cvals = up_vals[:, [self.indices.index(ind) for ind in c.indices]]
            c.up_info.vals = np.append(c.up_info.vals, cvals, axis=0)
            c.up_info.vals = c.up_info.vals[:c.up_info.rank]
            c.add_values(cvals)

    def locate(self, node: NodeName) -> Optional["DimTreeNode"]:
        """Locate a node by its name."""
        if node == self.node:
            return self

        for c in self.up_info.nodes:
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
    
    def distance(self, node1: NodeName, node2: NodeName) -> int:
        """Get the distance between the two indices."""
        n1 = self.locate(node1)
        assert n1 is not None
        n2 = self.locate(node2)
        assert n2 is not None
        # find the common ancestor that subsumes both n1 and n2
        p = n1
        dist1 = 0
        while p is not None:
            if all(ind in p.indices for ind in n1.indices + n2.indices):
                break

            p = p.down_info.nodes[0]
            dist1 += 1

        if p is None:
            raise RuntimeError("not a valid tree")
        
        p2 = n2
        dist2 = 0
        while p2 is not None and p2 != p:
            p2 = p2.down_info.nodes[0]
            dist2 += 1

        if p2 is None:
            raise RuntimeError("not a valid tree")
        
        return dist1 + dist2

    def entries(self) -> np.ndarray:
        if len(self.up_info.vals) != 0:
            vals = self.up_info.vals
        else:
            vals = np.empty((0, len(self.up_info.indices)))

        # for c in self.down_info.nodes:
        #     cvals = c.entries()
        #     print(c.indices)
        #     if len(vals) == 0:
        #         vals = cvals
        #     else:
        #         size = min(len(vals), len(cvals))
        #         vals = np.concat([vals[:size], cvals[:size]], axis=-1)

        return vals

    def known_entries(self) -> np.ndarray:
        if len(self.up_info.vals) != 0:
            vals = np.concat([self.down_info.vals, self.up_info.vals], axis=-1)
        else:
            vals = np.empty((0, len(self.indices)))

        self_inds = self.down_info.indices + self.up_info.indices
        for c in self.down_info.nodes:
            cvals = c.known_entries()
            # cvals follows the order of c.down_info.indices + c.up_info.indices
            # reorder the values to match self
            cindices = c.down_info.indices + c.up_info.indices
            perm = [self_inds.index(ind) for ind in cindices]
            vals = np.concat([vals, cvals[:, perm]], axis=0)

        return vals

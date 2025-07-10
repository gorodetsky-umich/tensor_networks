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


@dataclass
class NodeInfo:
    """Node name and the corresponding indices"""

    node: NodeName
    level: int
    indices: List[Index]
    free_indices: List[Index]
    up_indices: List[Index]
    down_indices: List[Index]


class Connection:
    """Node connections including children and parent"""

    def __init__(
        self,
        children: List["DimTreeNode"],
        parent: Optional["DimTreeNode"] = None,
    ):
        self.children = children
        self.parent = parent


class CrossVals:
    """Values for cross approximation"""

    def __init__(self, up_vals: np.ndarray, down_vals: np.ndarray):
        self.up_vals = up_vals
        self.down_vals = down_vals


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


class DimTreeNode:
    """Class for a dimension tree node"""

    def __init__(
        self,
        node: NodeName,
        level: int,
        indices: List[Index],
        free_indices: List[Index],
        up_indices: List[Index],
        down_indices: List[Index],
        children: List["DimTreeNode"],
        down_vals: np.ndarray,
        up_vals: np.ndarray,
        parent: Optional["DimTreeNode"] = None,
    ):
        self.info = NodeInfo(
            node, level, indices, free_indices, up_indices, down_indices
        )
        self.conn = Connection(children, parent)
        self.values = CrossVals(up_vals, down_vals)

    def __lt__(self, other: Self) -> bool:
        return sorted(self.info.indices) < sorted(other.info.indices)

    def preorder(self) -> Sequence["DimTreeNode"]:
        """Get the list of tree nodes in the pre-order traversal."""
        results = [self]
        for c in self.conn.children:
            results = itertools.chain(results, c.preorder())

        return list(results)

    def locate(self, node: NodeName) -> Optional["DimTreeNode"]:
        """Locate a node by its name."""
        if node == self.info.node:
            return self

        for c in self.conn.children:
            res = c.locate(node)
            if res is not None:
                return res

        return None

    def leaves(self) -> Sequence["DimTreeNode"]:
        """Get the leaf nodes in the current tree."""
        results = []
        if len(self.conn.children) == 0:
            results.append(self)
            return results

        for c in self.conn.children:
            results.extend(c.leaves())

        return results
    
    def height(self) -> int:
        """Get the height of the tree."""
        max_c = 0
        for c in self.conn.children:
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
            if all(ind in p.info.indices for ind in n1.info.indices + n2.info.indices):
                break

            p = p.conn.parent
            dist1 += 1

        if p is None:
            raise RuntimeError("not a valid tree")
        
        p2 = n2
        dist2 = 0
        while p2 is not None and p2 != p:
            p2 = p2.conn.parent
            dist2 += 1

        if p2 is None:
            raise RuntimeError("not a valid tree")
        
        return dist1 + dist2

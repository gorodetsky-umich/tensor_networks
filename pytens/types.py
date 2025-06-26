"""Type definitions."""

from typing import Union, Sequence, Self, Optional, Tuple, List
from dataclasses import dataclass

import pydantic
import numpy as np
import pytens.algs as algs

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
    indices: List[Index]
    free_indices: List[Index]
    up_indices: List[Index]
    down_indices: List[Index]

class Connection:
    """Node connections including children and parent"""
    def __init__(self, children: List["DimTreeNode"], parent: Optional["DimTreeNode"] = None):
        self.children = children
        self.parent = parent

class CrossVals:
    """Values for cross approximation"""
    def __init__(self, up_vals: List[List[int]], down_vals: List[List[int]]):
        self.up_vals = up_vals
        self.down_vals = down_vals

class DimTreeNode:
    """Class for a dimension tree node"""

    def __init__(
        self,
        node: NodeName,
        indices: List[Index],
        free_indices: List[Index],
        up_indices: List[Index],
        down_indices: List[Index],
        children: List["DimTreeNode"],
        down_vals: List[List[int]],
        up_vals: List[List[int]],
        parent: Optional["DimTreeNode"] = None,
    ):
        self.info = NodeInfo(node, indices, free_indices, up_indices, down_indices)
        self.conn = Connection(children, parent)
        self.values = CrossVals(up_vals, down_vals)

    def __lt__(self, other: Self) -> bool:
        return sorted(self.info.indices) < sorted(other.info.indices)

    def preorder(self) -> List["DimTreeNode"]:
        """Get the list of tree nodes in the pre-order traversal."""
        results = [self]
        for c in self.conn.children:
            results.extend(c.preorder())

        return results
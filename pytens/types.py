"""Type definitions."""

from typing import Union, Sequence, Self, Optional, Any
from dataclasses import dataclass

import pydantic

IntOrStr = Union[str, int]
NodeName = IntOrStr
IndexName = IntOrStr


@dataclass(frozen=True, eq=True)
class Index:
    """Class for denoting an index."""

    name: Union[str, int]
    size: Any

    def with_new_size(self, new_size: int) -> "Index":
        """Create a new index with same name but new size"""
        return Index(self.name, new_size)

    def with_new_name(self, name: IntOrStr) -> "Index":
        """Create a new index with same size but new name"""
        return Index(name, self.size)

    def __lt__(self, other: Self) -> bool:
        return str(self.name) < str(other.name)


@dataclass
class SVDConfig:
    """Configuration fields for SVD in tensor networks."""

    delta: float = 1e-5
    with_orthonormal: bool = True
    compute_data: bool = True


class IndexMerge(pydantic.BaseModel):
    """An index merge request and response."""

    merging_indices: Sequence[Index]
    merging_positions: Optional[Sequence[int]] = None
    merge_result: Optional[Index] = None

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))


class IndexSplit(pydantic.BaseModel):
    """An index split request and response."""

    splitting_index: Index
    split_target: Sequence[int]
    split_result: Optional[Sequence[Index]] = None

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

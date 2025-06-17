"""Type definitions."""

from typing import Union, Sequence, Self, Optional, Tuple, List
from dataclasses import dataclass

import pydantic
import numpy as np

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

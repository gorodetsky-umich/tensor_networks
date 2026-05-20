"""Type definitions for search related concepts."""

from dataclasses import dataclass, field

from typing import Sequence, Optional
from pytens.types import Index, IndexOp, IndexMerge


class Action:
    """Base action."""

    def __init__(self) -> None:
        self.delta: Optional[float] = None
        self.target_size: Optional[int] = None
        self.indices: Sequence[Index] = []

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Action):
            raise NotImplementedError

        return str(self) < str(other)

    def __hash__(self) -> int:
        return hash(self.__str__())

    def is_valid(self, _past_actions: Sequence["Action"]) -> bool:
        """Check whether the current action is valid against the history."""
        return True


@dataclass
class SearchContext:
    """Context passed into a search call: delta budget, splits, and flags."""

    remaining_delta: Optional[float] = None
    splits: Sequence[IndexOp] = field(default_factory=list)
    merge_ops: Sequence[IndexMerge] = field(default_factory=list)
    is_top: bool = False
    exclusions: Optional[Sequence[Index]] = None

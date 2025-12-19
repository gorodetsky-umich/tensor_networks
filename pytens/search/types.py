"""Type definitions for search related concepts."""

from typing import Sequence, Optional

class Action:
    """Base action."""

    def __init__(self):
        self.delta: Optional[float] = None
        self.target_size: Optional[int] = None
        self.indices = []

    def __lt__(self, other) -> bool:
        return str(self) < str(other)

    def __hash__(self) -> int:
        return hash(self.__str__())

    def is_valid(self, past_actions: Sequence["Action"]) -> bool:
        """Check whether the current action is valid against the history."""
        return True

"""Type definitions for search related concepts."""

from typing import Sequence

class Action:
    """Base action."""

    def __init__(self):
        self.delta = None
        self.target_size = None
        self.indices = []

    def __lt__(self, other) -> bool:
        return str(self) < str(other)

    def __hash__(self) -> int:
        return hash(self.__str__())

    def is_valid(self, past_actions: Sequence["Action"]) -> bool:
        """Check whether the current action is valid against the history."""
        return True

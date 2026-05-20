"""Different error distribution functions"""

import math
from typing import Tuple


class BaseErrorDist:
    """Base class for error distribution methods."""

    def split_delta(self, delta: float) -> Tuple[float, float]:
        """Split the remaining delta between the current level and sub-levels.

        Returns ``(node_delta, remaining_delta)`` satisfying the Pythagorean
        constraint ``node_delta**2 + remaining_delta**2 == delta**2`` exactly
        (up to float precision).  The equal split allocates half the squared
        budget to each side.
        """
        node_delta = delta / math.sqrt(2)
        remaining_delta = math.sqrt(max(0.0, delta**2 - node_delta**2))
        return node_delta, remaining_delta


class AlphaErrorDist(BaseErrorDist):
    """Divide the errors by a constant factor alpha.

    The node receives ``alpha`` times as much budget as the subtree, while
    the Pythagorean constraint is preserved exactly.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def split_delta(self, delta: float) -> Tuple[float, float]:
        """Split delta with ratio alpha:1 between node and subtree."""
        denom = math.sqrt(self.alpha**2 + 1)
        node_delta = delta * self.alpha / denom
        remaining_delta = math.sqrt(max(0.0, delta**2 - node_delta**2))
        return node_delta, remaining_delta

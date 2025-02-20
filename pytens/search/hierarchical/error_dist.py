"""Different error distribution functions"""

from typing import Tuple


def exponential_decrease(_: int, delta: float) -> Tuple[float, float]:
    """Split the remaining delta exponentially between levels"""
    return delta * 0.5, delta * 0.5

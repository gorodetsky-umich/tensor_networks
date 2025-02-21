"""Different error distribution functions"""

from typing import Tuple

class BaseErrorDist:
    def get_delta(self, _: int, delta: float) -> Tuple[float, float]:
        """Split the remaining delta exponentially between levels"""
        return delta / (2 ** 0.5), delta / (2 ** 0.5)

class AlphaErrorDist(BaseErrorDist):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def get_delta(self, _: int, delta: float) -> Tuple[float, float]:
        denom = (self.alpha ** 2 + 1) ** 0.5
        return delta * self.alpha / denom, delta / denom

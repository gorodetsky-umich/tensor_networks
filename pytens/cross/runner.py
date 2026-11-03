"""Runners for various cross approximation algorithms."""

from abc import abstractmethod
from typing import Optional
import warnings

import numpy as np

from pytens.algs import TensorNetwork, rand_ht, rand_tt, rand_tucker
from pytens.cross.cross import CrossApproximation, CrossConfig, CrossResult
from pytens.cross.func_interface import TensorFunc
from pytens.types import NodeName


class CrossRunner:
    """Base class for running cross approximation."""

    def __init__(self, config: Optional[CrossConfig] = None):
        self.config = config if config is not None else CrossConfig()
        self.last_result: Optional[CrossResult] = None

    def _cross(
        self,
        f: TensorFunc,
        net: TensorNetwork,
        root: NodeName,
        eps: float,
        kickrank: Optional[int],
        validation: Optional[np.ndarray],
    ) -> TensorNetwork:
        config = self.config.model_copy()
        if kickrank is not None:
            # Revalidate overrides, including negative rank increments.
            config = CrossConfig(
                **{**config.model_dump(), "kickrank": kickrank}
            )
        self.last_result = CrossApproximation(f, config).cross(
            net, root, validation, eps=eps
        )
        if not self.last_result.converged:
            warnings.warn(
                f"Cross approximation stopped: {self.last_result.stop_reason}; "
                "validation tolerance was not met",
                RuntimeWarning,
                stacklevel=3,
            )
        return net

    @abstractmethod
    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: Optional[int] = None,
        validation: Optional[np.ndarray] = None,
    ) -> TensorNetwork:
        """Run the cross approximation on the given function
        with the specified error.
        """
        raise NotImplementedError


class TTCrossRunner(CrossRunner):
    """Runner for our tt-cross implementation."""

    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: Optional[int] = None,
        validation: Optional[np.ndarray] = None,
    ) -> TensorNetwork:
        indices = f.indices[:]
        net = rand_tt(indices, [1] * (len(indices) - 1))
        return self._cross(
            f, net, list(net.network.nodes)[0], eps, kickrank, validation
        )


class HTCrossRunner(CrossRunner):
    """Runner for hierarchical tucker cross implementation."""

    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: Optional[int] = None,
        validation: Optional[np.ndarray] = None,
    ) -> TensorNetwork:
        net = rand_ht(f.indices, 1)
        return self._cross(
            f, net, list(net.network.nodes)[0], eps, kickrank, validation
        )


class TuckerCrossRunner(CrossRunner):
    """Runner for tucker cross implementation."""

    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: Optional[int] = None,
        validation: Optional[np.ndarray] = None,
    ) -> TensorNetwork:
        tucker = rand_tucker(f.indices)
        return self._cross(f, tucker, "root", eps, kickrank, validation)

"""Runners for various cross approximation algorithms."""

from abc import abstractmethod
from typing import Optional

import numpy as np

from pytens.algs import TensorNetwork
from pytens.cross.cross import CrossApproximation, CrossConfig
from pytens.cross.funcs import TensorFunc


class CrossRunner:
    """Base class for running cross approximation."""

    @abstractmethod
    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: int = 2,
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
        kickrank: int = 2,
        validation: Optional[np.ndarray] = None,
    ) -> TensorNetwork:
        indices = f.indices[:]
        net = TensorNetwork.rand_tt(indices, [1] * len(indices))
        cross_config = CrossConfig(kickrank=kickrank)
        cross_engine = CrossApproximation(f, cross_config)
        cross_engine.cross(
            net, list(net.network.nodes)[0], validation, eps=eps
        )
        return net


class HTCrossRunner(CrossRunner):
    """Runner for hierarchical tucker cross implementation."""

    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: int = 2,
        validation: Optional[np.ndarray] = None,
    ) -> TensorNetwork:
        net = TensorNetwork.rand_ht(f.indices, 1)
        cross_config = CrossConfig(kickrank=kickrank)
        cross_engine = CrossApproximation(f, cross_config)
        cross_engine.cross(
            net, list(net.network.nodes)[0], validation, eps=eps
        )
        return net


class TuckerCrossRunner(CrossRunner):
    """Runner for tucker cross implementation."""

    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: int = 2,
        validation: Optional[np.ndarray] = None,
    ) -> TensorNetwork:
        tucker = TensorNetwork.rand_tucker(f.indices)
        cross_config = CrossConfig(kickrank=kickrank)
        cross_engine = CrossApproximation(f, cross_config)
        cross_engine.cross(tucker, "root", validation, eps=eps)
        return tucker

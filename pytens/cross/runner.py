"""Runners for various cross approximation algorithms."""

from abc import abstractmethod
from typing import Optional

import tntorch
import torch
import numpy as np

from pytens.algs import (
    FoldedTensorTrain,
    HierarchicalTucker,
    Tensor,
    TensorTrain,
    TreeNetwork,
)
from pytens.cross.cross import cross
from pytens.cross.funcs import TensorFunc
from pytens.search.hierarchical.utils import tntorch_to_tt, tntorch_wrapper
from pytens.types import Index


class CrossRunner:
    """Base class for running cross approximation."""

    @abstractmethod
    def run(self, f: TensorFunc, eps: float, kickrank: int = 2, validation: Optional[np.ndarray] = None) -> TreeNetwork:
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
    ) -> TensorTrain:
        indices = f.indices[:]
        net = TensorTrain.rand_tt(indices)
        cross(f, net, net.end_nodes()[0], validation, eps=eps, kickrank=kickrank)
        return net


class TnTorchCrossRunner(CrossRunner):
    """Runner for tntorch cross implementation."""

    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: int = 2,
        validation: Optional[np.ndarray] = None,
    ) -> TensorTrain:
        domains = [torch.arange(ind.size) for ind in f.indices]
        res = tntorch.cross(
            tntorch_wrapper(f),
            domains,
            eps=eps,
            kickrank=kickrank,
            max_iter=100,
            val_size=2500,
            rmax=1000,
            verbose=True,
        )
        net = tntorch_to_tt(res, f.indices)
        return net


class HTCrossRunner(CrossRunner):
    """Runner for hierarchical tucker cross implementation."""

    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: int = 2,
        validation: Optional[np.ndarray] = None,
    ) -> HierarchicalTucker:
        net = HierarchicalTucker.rand_ht(f.indices, 1)
        cross(f, net, net.root(), validation, eps=eps, kickrank=kickrank)
        return net

class TuckerCrossRunner(CrossRunner):
    """Runner for tucker cross implementation."""

    def run(self, f: TensorFunc, eps: float, kickrank: int = 2, validation: Optional[np.ndarray] = None) -> TreeNetwork:
        tucker = TreeNetwork()
        root_val = np.random.random([1] * f.d)
        root_inds = [Index(f"s_{i}", 1) for i in range(f.d)]
        tucker.add_node("root", Tensor(root_val, root_inds))
        for i, ind in enumerate(f.indices):
            tensor_val = np.random.random((ind.size, 1))
            tensor_inds = [ind, Index(f"s_{i}", 1)]
            tucker.add_node(f"G{i}", Tensor(tensor_val, tensor_inds))
            tucker.add_edge(f"G{i}", "root")

        cross(f, tucker, "root", validation, eps=eps, kickrank=kickrank)
        return tucker

class FTTCrossRunner(CrossRunner):
    """Runner for hierarchical tucker cross implementation."""

    def run(
        self,
        f: TensorFunc,
        eps: float,
        kickrank: int = 2,
        validation: Optional[np.ndarray] = None,
    ) -> FoldedTensorTrain:
        inds = f.indices
        grouped_inds = []
        group_size = len(inds) // 4
        i = 0
        while i < len(inds):
            grouped_inds.append(inds[i:i+group_size])
            i += group_size
        net = FoldedTensorTrain.rand_ftt(grouped_inds)
        cross(
            f,
            net,
            net.backbone_nodes[0],
            validation,
            eps=eps,
            kickrank=kickrank,
        )
        return net

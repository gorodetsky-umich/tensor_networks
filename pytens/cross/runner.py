"""Runners for various cross approximation algorithms."""

from abc import abstractmethod

import tntorch
import torch

from pytens.algs import (
    FoldedTensorTrain,
    HierarchicalTucker,
    TensorTrain,
    TreeNetwork,
)
from pytens.cross.cross import cross
from pytens.cross.funcs import TensorFunc
from pytens.search.hierarchical.utils import tntorch_to_tt, tntorch_wrapper


class CrossRunner:
    """Base class for running cross approximation."""

    @abstractmethod
    def run(self, f: TensorFunc, eps: float) -> TreeNetwork:
        """Run the cross approximation on the given function
        with the specified error.
        """
        raise NotImplementedError


class TTCrossRunner(CrossRunner):
    """Runner for our tt-cross implementation."""

    def run(self, f: TensorFunc, eps: float) -> TensorTrain:
        net = TensorTrain.rand_tt(f.indices)
        cross(f, net, net.end_nodes()[0], eps=eps)
        return net


class TnTorchCrossRunner(CrossRunner):
    """Runner for tntorch cross implementation."""

    def run(self, f: TensorFunc, eps: float) -> TensorTrain:
        domains = [torch.arange(ind.size) for ind in f.indices]
        res = tntorch.cross(
            tntorch_wrapper(f),
            domains,
            eps=eps,
            kickrank=10,
            max_iter=100,
            val_size=2500,
            rmax=1000,
            verbose=False,
        )
        net = tntorch_to_tt(res, f.indices)
        return net


class HTCrossRunner(CrossRunner):
    """Runner for hierarchical tucker cross implementation."""

    def run(self, f: TensorFunc, eps: float) -> HierarchicalTucker:
        net = HierarchicalTucker.rand_ht(f.indices, 1)
        cross(f, net, net.root(), eps=eps)
        return net


class FTTCrossRunner(CrossRunner):
    """Runner for hierarchical tucker cross implementation."""

    def run(self, f: TensorFunc, eps: float) -> FoldedTensorTrain:
        inds = f.indices
        grouped_inds = []
        group_size = len(inds) // 4
        i = 0
        while i < len(inds):
            grouped_inds.append(inds[i:i+group_size])
            i += group_size
        net = FoldedTensorTrain.rand_ftt(grouped_inds)
        cross(f, net, net.backbone_nodes[0], eps=eps)
        return net

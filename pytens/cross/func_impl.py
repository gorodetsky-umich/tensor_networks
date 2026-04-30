"""Test functions for cross approximation"""

from typing import TYPE_CHECKING, List

import numpy as np

from pytens.types import Index
from pytens.cross.func_interface import CachedFunc, TensorFunc

if TYPE_CHECKING:
    import pytens.algs as pt


class FuncData(CachedFunc):
    """Class for data tensors as cross approximation targets."""

    def __init__(self, indices: List[Index], data: np.ndarray):
        super().__init__(indices)
        self.data = data

    def _run(self, args: np.ndarray) -> np.ndarray:
        return self.data[*args.astype(int).T]


class FuncTensorNetwork(CachedFunc):
    """Class for data tensors as cross approximation targets."""

    def __init__(self, indices: List[Index], net: "pt.TensorNetwork"):
        super().__init__(indices)
        self.net = net

    def _run(self, args: np.ndarray) -> np.ndarray:
        return self.net.evaluate(self.indices, args.astype(int))

    def cost(self) -> int:
        """Return the evaluation cost of the underlying tensor network."""
        return self.net.cost()


class PermuteFunc(TensorFunc):
    """Tensor functions for index permutation."""

    def __init__(self, indices, old_func, ind_unperm):
        super().__init__(indices)
        self.old_func = old_func
        self.ind_unperm = ind_unperm

    def index_to_args(self, indices: np.ndarray):
        # permute the indices back into the order before the permutation
        return self.old_func.index_to_args(indices[:, self.ind_unperm])

    def run(self, args: np.ndarray):
        return self.old_func.run(args)


class SplitFunc(TensorFunc):
    """Reduce the tensor function after split into the old one."""

    def __init__(self, indices, old_func, ind_mapping):
        super().__init__(indices)
        self.old_func = old_func
        self.ind_mapping = ind_mapping

    def index_to_args(self, indices: np.ndarray) -> np.ndarray:
        indices = indices.astype(int)
        old_free = self.old_func.indices
        old_indices = np.empty((len(indices), len(old_free)), dtype=int)
        for i, ind in enumerate(old_free):
            if i not in self.ind_mapping:
                assert ind in self.indices, "index not found"
                j = self.indices.index(ind)
                old_indices[:, i] = indices[:, j]
            else:
                split_inds, split_sizes = self.ind_mapping[i]
                old_indices[:, i] = np.ravel_multi_index(
                    tuple(indices[:, split_inds].T), split_sizes
                )

        # turn indices into arguments
        return self.old_func.index_to_args(old_indices)

    def run(self, args: np.ndarray):
        return self.old_func.run(args)

"""Test functions for cross approximation"""

from abc import abstractmethod
from typing import List

import numpy as np

from pytens.types import Index


class TensorFunc:
    """An abstract base class for tensor functions.

    The derived classes should implement the ``run`` method,
    which evalutes the function at vectorized arguments.

    Attributes:
        d: Number of dimensions (equal to ``len(indices)``).
        indices: One ``Index`` per dimension, each carrying the discrete grid
            points via ``space`` and the grid size via ``size``.
        name: Human-readable identifier used for logging and file names.
            Defaults to ``"_func_"``; subclasses should override it.
        calls: Array of shape ``(n_calls, d)`` recording every set of integer
            grid indices the function has been evaluated at, accumulated
            across all calls to ``__call__``.
    """

    def __init__(self, indices: List[Index]):
        self.d = len(indices)
        self.indices = indices
        self.name = "_func_"
        self.calls = np.empty((0, self.d))

    def index_to_args(self, indices: np.ndarray) -> np.ndarray:
        """Convert vectorized integer indices to vectorized function arguments.

        This maps each discrete index (i_k) to its associated argument value
        using ``self.indices[k].space``.

        Parameters
        ----------
        indices:
            Array of shape ``(n, d)`` of integer-like indices.

        Returns
        -------
        np.ndarray
            Array of shape ``(n, d)`` containing the corresponding argument
            values (dtype float).
        """
        indices = indices.astype(int)
        args = np.empty_like(indices, dtype=float)
        for i, ind in enumerate(self.indices):
            args[:, i] = np.array(ind.space)[indices[:, i]]

        return args

    def size(self) -> int:
        """Get the size of the tensor function."""

        res = 1
        for ind in self.indices:
            res *= ind.size

        return res

    @property
    def shape(self) -> List[int]:
        """Get the shape of the tensor function."""

        result = [0] * len(self.indices)
        for i, ind in enumerate(self.indices):
            if isinstance(ind.size, int):
                result[i] = ind.size
            elif isinstance(ind.size, tuple):
                result[i] = ind.size[-1]
            else:
                raise TypeError("Unsupported index size type")

        return result

    def cost(self) -> int:
        """Return the cost proxy for evaluating/storing the full tensor.

        By default, this is the total number of entries, i.e. ``prod(shape)``.
        """

        return int(np.prod(self.shape))

    def free_indices(self) -> List[Index]:
        """Return the free indices of the function.

        Returns
        -------
        list[Index]
            The indices that define the domain of this function. For simple
            functions this is just ``self.indices``; composite functions may
            override this.
        """

        return self.indices

    @abstractmethod
    def run(self, args: np.ndarray) -> np.ndarray:
        """Evaluate the function for a batch of vectorized arguments.

        Implementations should accept a 2D array of shape ``(n, d)`` and return
        a 1D array of length ``n`` (or a compatible vectorized output).
        """
        raise NotImplementedError

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        # print("recording", indices.shape[0])
        args = self.index_to_args(indices)
        return self.run(args)


class CachedFunc(TensorFunc):
    """An abstract class for tensor function with cache.

    Subclasses should implement the ``_run`` method to
    evaluate the function values using given arguments.
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.calls = np.empty((0, self.d))

    def num_calls(self) -> int:
        """Return the number of unique calls observed so far."""
        return len(np.unique(self.calls, axis=0))

    @abstractmethod
    def _run(self, args: np.ndarray) -> np.ndarray:
        """Subclass hook for evaluating the underlying function.

        The input is an array of vectorized arguments with shape ``(n, d)`` and
        the output is an array of vectorized function vals with shape ``(n,)``.
        """
        raise NotImplementedError

    def run(self, args: np.ndarray) -> np.ndarray:
        self.calls = np.concatenate([args, self.calls])
        return self._run(args)

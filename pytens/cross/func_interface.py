"""Test functions for cross approximation"""

from abc import abstractmethod
from typing import List, Set

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
    """

    def __init__(self, indices: List[Index]):
        self.d = len(indices)
        self.indices = indices
        self.name = "_func_"

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
        # 64-bit hashes of the unique rows evaluated so far
        self.calls: Set[int] = set()
        rng = np.random.default_rng(0)
        self._row_hash_mults = rng.integers(
            1, np.iinfo(np.uint64).max, size=self.d, dtype=np.uint64
        ) | np.uint64(1)

    def _hash_rows(self, args: np.ndarray) -> np.ndarray:
        """One 64-bit hash per row of `args` (splitmix64 mixing per entry,
        then a random linear combination across the columns)."""
        x = np.ascontiguousarray(args, dtype=np.float64).view(np.uint64)
        with np.errstate(over="ignore"):
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
            return np.asarray(
                (x * self._row_hash_mults).sum(axis=1, dtype=np.uint64)
            )

    def num_calls(self) -> int:
        """Return the number of unique calls observed so far."""
        return len(self.calls)

    @abstractmethod
    def _run(self, args: np.ndarray) -> np.ndarray:
        """Subclass hook for evaluating the underlying function.

        The input is an array of vectorized arguments with shape ``(n, d)`` and
        the output is an array of vectorized function vals with shape ``(n,)``.
        """
        raise NotImplementedError

    def run(self, args: np.ndarray) -> np.ndarray:
        self.calls.update(self._hash_rows(args).tolist())
        return self._run(args)

"""Bounded, run-local evaluation of discrete cross-approximation samples."""

from dataclasses import dataclass
from math import prod
from time import perf_counter

import numpy as np

from pytens.cross.func_interface import TensorFunc


@dataclass
class EvaluationStats:
    """Point counts, including validation; hits include within-batch reuse."""

    requested: int = 0
    evaluated: int = 0
    cache_hits: int = 0
    evaluation_seconds: float = 0.0


class CrossEvaluator:
    """Batch evaluations with an optional exact, bounded LRU value cache.

    Cache lookup and updates operate on NumPy arrays instead of Python tuples
    and dictionaries. Discrete coordinates are encoded without hash collisions.
    The evaluator and its cache are recreated for every cross run.
    """

    def __init__(self, function: TensorFunc, batch_size: int, cache_size: int):
        self.function = function
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.stats = EvaluationStats()
        sizes = [ind.size for ind in function.indices]
        self._strides: np.ndarray | None
        key_dtype: np.dtype
        if not cache_size or prod(sizes) <= 2**64:
            # Mixed-radix keys are exact even when the domain has 2**64 points.
            strides = [0] * len(sizes)
            stride = 1
            if cache_size:
                for i in reversed(range(len(sizes))):
                    # A singleton coordinate contributes zero. Its stride can
                    # otherwise equal 2**64 at the representable boundary.
                    strides[i] = stride if sizes[i] != 1 else 0
                    stride *= sizes[i]
            self._strides = np.asarray(strides, dtype=np.uint64)
            self._coordinate_dtype = None
            key_dtype = np.dtype("uint64")
        else:
            # Fixed-width coordinate bytes avoid overflow in larger domains.
            self._strides = None
            self._coordinate_dtype = np.min_scalar_type(max(sizes) - 1)
            key_dtype = np.dtype(
                ("V", len(sizes) * self._coordinate_dtype.itemsize)
            )
        self.cache = np.empty(0, dtype=key_dtype)
        self._values = np.empty(0)
        self._ages = np.empty(0, dtype=np.int64)
        self._clock = 0

    def _keys(self, points: np.ndarray) -> np.ndarray:
        if self._strides is not None:
            return np.asarray(
                np.asarray(points, dtype=np.uint64) @ self._strides
            )
        return (
            np.ascontiguousarray(points, dtype=self._coordinate_dtype)
            .view(self.cache.dtype)
            .reshape(-1)
        )

    def _evaluate(self, points: np.ndarray) -> np.ndarray:
        start = perf_counter()
        values = np.asarray(self.function(points))
        self.stats.evaluation_seconds += perf_counter() - start
        self.stats.evaluated += len(points)
        if values.shape == (len(points), 1):
            values = values[:, 0]
        if values.shape != (len(points),):
            raise ValueError(
                "Tensor function must return shape (n,) or (n, 1)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Tensor function returned nonfinite values")
        return values

    def _batch(self, points: np.ndarray) -> np.ndarray:
        self.stats.requested += len(points)
        if not self.cache_size:
            return self._evaluate(points)

        keys = self._keys(points)
        positions = np.asarray(np.searchsorted(self.cache, keys))
        hits = positions < len(self.cache)
        hits[hits] = self.cache[positions[hits]] == keys[hits]
        ages = self._clock + np.arange(len(points), dtype=np.int64)
        self._clock += len(points)
        # maximum.at handles repeated points and preserves true access order.
        np.maximum.at(self._ages, positions[hits], ages[hits])
        if np.all(hits):
            self.stats.cache_hits += len(points)
            return self._values[positions]

        missing_positions = np.flatnonzero(~hits)
        new_keys, first, inverse = np.unique(
            keys[missing_positions], return_index=True, return_inverse=True
        )
        new_values = self._evaluate(points[missing_positions[first]])
        self.stats.cache_hits += len(points) - len(new_keys)
        dtype = (
            np.result_type(self._values.dtype, new_values.dtype)
            if len(self.cache)
            else new_values.dtype
        )
        values = np.empty(len(points), dtype=dtype)
        values[hits] = self._values[positions[hits]]
        values[missing_positions] = new_values[inverse]
        new_ages = np.zeros(len(new_keys), dtype=np.int64)
        np.maximum.at(new_ages, inverse, ages[missing_positions])

        cache = np.concatenate((self.cache, new_keys))
        cached_values = np.concatenate(
            (self._values.astype(dtype, copy=False), new_values)
        )
        cached_ages = np.concatenate((self._ages, new_ages))
        if len(cache) > self.cache_size:
            keep = np.argpartition(cached_ages, len(cache) - self.cache_size)[
                -self.cache_size :
            ]
            cache = cache[keep]
            cached_values = cached_values[keep]
            cached_ages = cached_ages[keep]
        order = np.argsort(cache, kind="stable")
        self.cache = cache[order]
        self._values = cached_values[order]
        self._ages = cached_ages[order]
        return values

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= self.batch_size:
            return self._batch(points) if len(points) else np.empty(0)
        return np.concatenate(
            [
                self._batch(points[start : start + self.batch_size])
                for start in range(0, len(points), self.batch_size)
            ]
        )

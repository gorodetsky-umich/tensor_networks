"""Utility functions for hierarchical search"""

import math
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np

from pytens.search.configuration import SearchConfig
from pytens.types import IndexSplit, Index
from pytens.cross.funcs import TensorFunc, SplitFunc


def trigger_merge(config: SearchConfig, is_top: bool) -> bool:
    """Determine whether to trigger the index merge operation before search"""
    return (config.topdown.search_algo == "correlation") and (
        not is_top or config.topdown.merge_mode == "all"
    )


def corr(
    corr_data: np.ndarray,
    agg: Literal["mean", "det", "norm", "sval"],
    sample_size: int = 50000,
) -> float:
    """Compute the correlation over the random samples of the given data."""
    samples = np.random.choice(
        corr_data.shape[0],
        size=min(sample_size, corr_data.shape[0]),
        replace=False,
    )
    sample_data = corr_data[samples]
    # TODO: modify this part to support correlation for cross approximation
    corr_res = np.corrcoef(
        sample_data + np.random.random(sample_data.shape) * 1e-13
    )
    if agg == "mean":
        return -np.mean(np.abs(corr_res))

    if agg == "det":
        return np.linalg.det(corr_res)

    if agg == "norm":
        return float(np.linalg.norm(corr_res))

    if agg == "sval":
        return np.linalg.svdvals(corr_res)[0]

    raise ValueError("unknown aggregation method")


def permute_unique(nums: List[int]) -> Sequence[Tuple[int, ...]]:
    nums.sort()
    used = [False] * len(nums)

    def backtrack(pat: List[int]) -> List[Tuple[int, ...]]:
        if len(pat) == len(nums):
            return [tuple(pat[:])]

        results = []
        for i, num in enumerate(nums):
            if used[i]:
                continue
            if i > 0 and num == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            pat.append(num)
            results.extend(backtrack(pat))
            used[i] = False
            pat.pop()

        return results

    return backtrack([])


def split_into_chunks(
    lst: Sequence[int], n: int
) -> Sequence[List[Sequence[int]]]:
    if n == 1:
        # When n is 1, the only chunk is the entire list
        return [[lst]]
    else:
        results = []
        for i in range(1, len(lst) - n + 2):  # Ensure at least `n` chunks
            for tail in split_into_chunks(lst[i:], n - 1):
                results.append([lst[:i]] + tail)

        return results


def select_factors(
    factors: Dict[int, int], budget: int
) -> List[Sequence[int]]:
    """Select a suitable number of factors for reshaping"""
    # enumerate all possible choices for each factor
    factors_flat = [x for x, c in factors.items() for _ in range(c)]
    # partition the list into splits_allowed groups
    seen = set()
    results = []
    for factors_perm in permute_unique(factors_flat):
        for chunks in split_into_chunks(factors_perm, budget + 1):
            chunk_factors = tuple([math.prod(chunk) for chunk in chunks])
            if chunk_factors not in seen:
                seen.add(chunk_factors)
                results.append(chunk_factors)

    return results


def split_func(
    old_func: TensorFunc,
    free_indices: Sequence[Index],
    split_ops: Sequence[IndexSplit],
) -> SplitFunc:
    """Get the tensor function for the sequence of split operations."""
    old_free = old_func.indices
    var_mapping = {}
    for split_op in split_ops:
        split_out = split_op.result
        if split_out is None:
            continue

        split_inds, split_sizes = [], []
        for ind in split_out:
            split_inds.append(free_indices.index(ind))
            split_sizes.append(int(ind.size))

        before_split = old_free.index(split_op.index)
        var_mapping[before_split] = (split_inds, split_sizes)

    return SplitFunc(free_indices, old_func, var_mapping)

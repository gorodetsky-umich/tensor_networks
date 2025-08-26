"""Utility functions for hierarchical search"""

import math
from typing import Dict, List, Literal, Sequence, Tuple
from collections import defaultdict

import numpy as np
import torch

from pytens.algs import TensorTrain, Tensor
from pytens.search.configuration import SearchConfig
from pytens.types import IndexSplit, Index
from pytens.cross.funcs import TensorFunc, SplitFunc

class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.elems = set()

    def find(self, i):
        if i not in self.parent:
            return i

        self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i, j):
        self.elems.add(i)
        self.elems.add(j)
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j  # Union by setting one root's parent to the other
            return True
        return False  # Already in the same set
    
    def groups(self):
        groups = defaultdict(list)

        for x in self.elems:
            root = self.find(x)
            groups[root].append(x)

        return groups

def trigger_merge(config: SearchConfig, is_top: bool) -> bool:
    """Determine whether to trigger the index merge operation before search"""
    return (config.topdown.search_algo == "correlation") and (
        not is_top or config.topdown.merge_mode == "all"
    )


def corr(
    corr_res: np.ndarray,
    agg: Literal["mean", "det", "norm", "sval"],
) -> float:
    """Compute the correlation over the random samples of the given data."""
    # samples = np.random.choice(
    #     corr_data.shape[0],
    #     size=min(sample_size, corr_data.shape[0]),
    #     replace=False,
    # )
    # sample_data = corr_data[samples]
    # # TODO: modify this part to support correlation for cross approximation
    # corr_res = np.corrcoef(
    #     sample_data + np.random.random(sample_data.shape) * 1e-13
    # )
    if agg == "mean":
        return float(-np.mean(np.abs(corr_res)))

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

def tntorch_wrapper(f):
    def g(*args):
        if len(args[0].shape) == 1:
            inds = np.stack([a.numpy() for a in args], axis=-1)
        else:
            inds = np.concat([a.numpy() for a in args], axis=-1)
        return torch.from_numpy(f(inds.astype(int)))
    
    return g

def tntorch_to_tt(res, split_indices):
    net = TensorTrain()
    for ni, n in enumerate(res.cores):
        n = n.squeeze([0, -1])
        if ni == 0:
            n_indices = [split_indices[ni], Index(f"s{ni}", n.shape[1])]
        elif ni == len(res.cores) - 1:
            n_indices = [Index(f"s{ni-1}", n.shape[0]), split_indices[ni]]
        else:
            n_indices = [Index(f"s{ni-1}", n.shape[0]), split_indices[ni], Index(f"s{ni}", n.shape[2])]
        net.add_node(str(ni), Tensor(n.numpy(), n_indices))

    for i in range(len(res.cores) - 1):
        net.add_edge(str(i), str(i + 1))

    return net
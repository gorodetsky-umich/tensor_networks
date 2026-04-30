"""Utility functions for hierarchical search"""

from typing import List, Literal, Sequence, Tuple
from collections import defaultdict

import numpy as np

from pytens.types import IndexSplit, Index
from pytens.cross.func_interface import TensorFunc
from pytens.cross.func_impl import SplitFunc


class DisjointSet:
    """A disjoint-set (union-find) data structure."""

    def __init__(self):
        self.parent = {}
        self.elems = set()

    def find(self, i):
        """Find the root representative of element i with path compression."""
        if i not in self.parent:
            return i

        self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i, j):
        """
        Union the sets containing i and j;
        return True if they were disjoint.
        """
        self.elems.add(i)
        self.elems.add(j)
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = (
                root_j  # Union by setting one root's parent to the other
            )
            return True
        return False  # Already in the same set

    def groups(self):
        """Return the mapping from each root to its members."""
        groups = defaultdict(list)

        for x in self.elems:
            root = self.find(x)
            groups[root].append(x)

        for k in groups:
            groups[k] = sorted(groups[k])

        return groups


def corr(
    corr_res: np.ndarray,
    agg: Literal["mean", "det", "norm", "sval"],
) -> float:
    """Compute the correlation over the random samples of the given data."""
    if agg == "mean":
        return float(-np.mean(np.abs(corr_res)))

    if agg == "det":
        return np.linalg.det(corr_res)

    if agg == "norm":
        return float(np.linalg.norm(corr_res))

    if agg == "sval":
        return np.linalg.svdvals(corr_res)[0]

    raise ValueError("unknown aggregation method")


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


def build_bipartite_sample(
    left_inds: List[Index],
    right_inds: List[Index],
    free_inds: List[Index],
    selected_inds: List[np.ndarray],
) -> Tuple[np.ndarray, List[Index]]:
    """Build a Cartesian-product sample matrix for a bipartite index split.

    Returns the stacked index matrix (num_left * num_right, total_dims) and
    the ordering of indices used (left_inds + right_inds).
    """
    left_ind_vals = []
    for ind in left_inds:
        left_ind_vals.append(selected_inds[free_inds.index(ind)])

    right_ind_vals = []
    for ind in right_inds:
        right_ind_vals.append(selected_inds[free_inds.index(ind)])

    left_stacked = np.stack(left_ind_vals, axis=-1)
    right_stacked = np.stack(right_ind_vals, axis=-1)

    # Cartesian product: repeat left N times, tile right M times
    left = np.repeat(left_stacked, len(right_stacked), axis=0)
    right = np.tile(right_stacked, (len(left_stacked), 1))
    full_indices = np.hstack((left, right))
    return full_indices, left_inds + right_inds

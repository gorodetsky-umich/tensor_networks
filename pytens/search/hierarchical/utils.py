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
 
        for k in groups:
            groups[k] = sorted(groups[k])

        return groups


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
        if ni == 0:
            n = n.squeeze([0])
            n_indices = [split_indices[ni], Index(f"s{ni}", n.shape[1])]
        elif ni == len(res.cores) - 1:
            n = n.squeeze([-1])
            n_indices = [Index(f"s{ni-1}", n.shape[0]), split_indices[ni]]
        else:
            assert len(n.shape) == 3
            n_indices = [Index(f"s{ni-1}", n.shape[0]), split_indices[ni], Index(f"s{ni}", n.shape[2])]
        net.add_node(str(ni), Tensor(n.numpy(), n_indices))

    for i in range(len(res.cores) - 1):
        net.add_edge(str(i), str(i + 1))

    return net

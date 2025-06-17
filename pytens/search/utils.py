"""Utility function for structure search."""

import os
from typing import Tuple, List, Dict, Optional, Self, Union, Literal

import numpy as np
import pydantic

from pytens.search.state import SearchState
from pytens.algs import TreeNetwork, Tensor
from pytens.cross.funcs import TensorFunc
from pytens.types import IndexSplit, IndexMerge


class SearchStats(pydantic.BaseModel):
    """Statistics collected during the search process"""

    best_cost: List[Tuple[float, float]] = []
    costs: List[Tuple[float, float]] = []
    errors: List[Tuple[float, float]] = []
    ops: List[Tuple[float, int]] = []
    unique: Dict[int, int] = {}

    # results
    count: int = 0
    preprocess_time: float = 0.0
    search_start: float = 0.0
    search_end: float = 0.0
    cr_core: float = 0.0
    cr_start: float = 0.0
    re_f: float = 0.0
    re_max: float = 0.0

    def incr_unique(self, key: int):
        """Increment the unique counter."""
        self.unique[key] = self.unique.get(key, 0) + 1


class SearchResult:
    """Result returned by the search process"""

    stats: SearchStats
    unused_delta: float = 0.0
    best_state: Optional[SearchState] = None
    best_solver_cost: int = -1

    def __init__(
        self,
        stats=SearchStats(),
        best_state=None,
        unused_delta=0.0,
    ):
        self.stats = stats
        self.best_state = best_state
        self.unused_delta = unused_delta

    def __lt__(self, other: Self) -> bool:
        if self.best_state is None:
            return False

        if other.best_state is None:
            return True

        if self.best_state < other.best_state:
            return True

        if self.best_state == other.best_state:
            return self.unused_delta > other.unused_delta

        return False

    def update_best_state(self, other: Self) -> Self:
        """Update the field of best_state if other is better"""
        assert other.best_state is not None
        if other < self:
            self.best_state = other.best_state
            self.unused_delta = other.unused_delta

        return self


def approx_error(tensor: Tensor, net: TreeNetwork) -> float:
    """Compute the reconstruction error.

    Given a tensor network TN and the target tensor X,
    it returns ||X - TN|| / ||X||.
    """
    target_free_indices = tensor.indices
    net_free_indices = net.free_indices()
    net_value = net.contract().value
    perm = [net_free_indices.index(i) for i in target_free_indices]
    net_value = net_value.transpose(perm)
    error = float(
        np.linalg.norm(net_value - tensor.value) / np.linalg.norm(tensor.value)
    )
    return error


def log_stats(
    search_stats: SearchStats,
    target_tensor: Tensor,
    ts: float,
    st: SearchState,
    bn: TreeNetwork,
):
    """Log statistics of a given state."""
    search_stats.ops.append((ts, len(st.past_actions)))
    search_stats.costs.append((ts, st.network.cost()))
    err = approx_error(target_tensor, st.network)
    search_stats.errors.append((ts, err))
    search_stats.best_cost.append((ts, bn.cost()))
    ukey = st.network.canonical_structure()
    search_stats.unique[ukey] = search_stats.unique.get(ukey, 0) + 1


def rtol(
    base_tensor: np.ndarray,
    approx_tensor: np.ndarray,
    norm: Literal["F", "M"] = "F",
) -> float:
    """
    Compute the relative error between two given tensors
    on the specified norms.
    """
    if norm == "F":
        return float(
            np.linalg.norm(base_tensor - approx_tensor)
            / np.linalg.norm(base_tensor)
        )

    if norm == "M":
        return np.max(abs(base_tensor - approx_tensor)) / np.max(
            abs(base_tensor)
        )

    raise ValueError("unsupported norm type")


def remove_temp_dir(temp_dir, temp_files):
    """Remove temporary npz files"""
    try:
        for temp_file in temp_files:
            os.remove(temp_file)

        if len(os.listdir(temp_dir)) == 0:
            os.rmdir(temp_dir)

    except FileNotFoundError:
        pass


def reshape_indices(reshape_ops, indices, data):
    """Get corresponding indices after splitting"""
    indices = [[ind] for ind in indices]
    for reshape_op in reshape_ops:
        new_indices = []
        new_sizes = []
        if isinstance(reshape_op, IndexSplit):
            for ind_group in indices:
                new_ind_group = []
                for ind in ind_group:
                    if ind == reshape_op.index:
                        assert reshape_op.result is not None
                        new_ind_group.extend(reshape_op.result)
                    else:
                        new_ind_group.append(ind)

                new_indices.append(new_ind_group)
                new_sizes.extend([ind.size for ind in new_ind_group])

        elif isinstance(reshape_op, IndexMerge):
            for group_idx, ind_group in enumerate(indices):
                new_ind_group = []
                for ind in ind_group:
                    if ind in reshape_op.indices:
                        unchanged = [
                            ind
                            for ind in ind_group
                            if ind not in reshape_op.indices
                        ]
                        assert reshape_op.result is not None
                        new_ind_group = [reshape_op.result] + unchanged
                        # we want to permute these indices before comparison
                        cnt_before = sum(len(g) for g in indices[:group_idx])
                        cnt_after = sum(
                            len(g) for g in indices[group_idx + 1 :]
                        )
                        curr_perm = [
                            ind_group.index(ind) + cnt_before
                            for ind in reshape_op.indices
                        ] + [
                            ind_group.index(ind) + cnt_before
                            for ind in unchanged
                        ]
                        prev_perm = list(range(cnt_before))
                        next_perm = [
                            cnt_before + len(curr_perm) + i
                            for i in range(cnt_after)
                        ]
                        data = data.transpose(
                            *(prev_perm + curr_perm + next_perm)
                        )
                        break
                    else:
                        new_ind_group.append(ind)

                new_sizes.extend([ind.size for ind in new_ind_group])
                new_indices.append(new_ind_group)

        data = data.reshape(*new_sizes)
        indices = new_indices

    return indices, data


def init_state(
    data_tensor: Union[TreeNetwork, TensorFunc], delta
) -> SearchState:
    if isinstance(data_tensor, TreeNetwork):
        return SearchState(data_tensor, delta)

    if isinstance(data_tensor, TensorFunc):
        net = TreeNetwork()
        net.add_node("G0", Tensor(np.empty(0), data_tensor.indices))
        return SearchState(net, delta)

    raise TypeError(
        f"Expect data tensors to have types TreeNetwork or TensorFunc, but get {type(data_tensor)}"
    )

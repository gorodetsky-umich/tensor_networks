"""Cross Approximation."""

from __future__ import annotations

import copy
import logging
import math
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import pydantic
from scipy.linalg import qr as sp_qr

import pytens.algs as pt
from pytens.cross.func_interface import TensorFunc
from pytens.cross.evaluation import CrossEvaluator, EvaluationStats
from pytens.types import DimTreeNode, Index

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CrossAlgo(Enum):
    """Enumeration of cross algorithms."""

    MAXVOL = auto()
    DEIM = auto()


class ConvergenceCheck(Enum):
    """Enumeration of convergence check methods."""

    # Also compute the full iterate-change norm (a stagnation diagnostic).
    # Accuracy is always checked against function values on validation points.
    NORM = auto()
    # use the error on a validation set as the criteria for convergence
    VALID_ERROR = auto()


class CrossConfig(pydantic.BaseModel):
    """Configuration fields for the cross procedure."""

    cross_algo: CrossAlgo = pydantic.Field(
        default=CrossAlgo.MAXVOL,
        description="Configure the algorithm for index selection",
    )
    kickrank: int = pydantic.Field(
        default=2,
        ge=0,
        description="Configure the rank increment between iterations",
    )
    max_rank: Optional[int] = pydantic.Field(
        default=None,
        ge=1,
        description="Configure the maximum rank that is used in cross",
    )
    max_iters: int = pydantic.Field(
        default=25,
        ge=1,
        description="Limit the maximum number of sweeps over the entire tree",
    )
    validation_size: int = pydantic.Field(
        default=1000,
        ge=1,
        description="Configure the number of validation points",
    )
    convergence: ConvergenceCheck = pydantic.Field(
        default=ConvergenceCheck.VALID_ERROR,
        description="Validation stopping, optionally with full norm diagnostics",
    )
    atol: float = pydantic.Field(default=0.0, ge=0, allow_inf_nan=False)
    batch_size: int = pydantic.Field(default=8192, ge=1)
    cache_size: int = pydantic.Field(
        default=0,
        ge=0,
        description="LRU entries; opt in only for deterministic functions",
    )
    stagnation_patience: int = pydantic.Field(default=3, ge=1)


class CrossSweep(pydantic.BaseModel):
    """One complete sweep; ranks follow preorder, excluding the root."""

    sweep: int
    ranks: List[int]
    validation_error: float
    validation_rmse: float
    norm_change: Optional[float] = None
    evaluated: int


class CrossResult(pydantic.BaseModel):
    """Class to record cross approximation results."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # The resulting tensor network
    net: "pt.TensorNetwork"
    # The used rows and columns are stored in the dimension tree
    dim_tree: DimTreeNode
    # The ranks and corresponding errors info during the cross algo
    ranks_and_errors: Sequence[Tuple[int, float]]
    converged: bool = False
    stop_reason: str = "unknown"
    history: List[CrossSweep] = pydantic.Field(default_factory=list)
    evaluations: EvaluationStats = pydantic.Field(
        default_factory=EvaluationStats
    )


def _py_maxvol(
    arr: np.ndarray,
    tol: float = 1.05,
    max_iters: int = 100,
    top_k_index: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find a rxr submatrix of maximum volume in an nxr matrix.

    Implements the 1-maxvol algorithm: given an nxr tall matrix ``arr``
    (n > r), finds r row indices whose rxr submatrix has (approximately)
    maximum absolute determinant.  The search is restricted to the first
    ``top_k_index`` rows; pass -1 (default) to search all rows.

    The algorithm initialises with a QR-pivoted basis, then greedily swaps
    rows using a Sherman-Morrison rank-1 update until no swap
    improves the volume by more than ``tol``.

    Args:
        arr: Input matrix of shape (n, r) with n > r.
        tol: Stop when the best swap improves volume by less than this factor
            (must be ≥ 1; values < 1 are clamped to 1.0).
        max_iters: Maximum number of row swaps before terminating.
        top_k_index: Restrict the row search to the first this-many rows.
            -1 means no restriction (all n rows are candidates).

    Returns:
        index: Integer array of shape (r,) containing the selected row
            indices into ``arr``.
        c: Coefficient matrix of shape (n, r) such that
            ``arr ≈ c @ arr[index]``.

    This implementation is adapted from the
    tntorch library (https://github.com/rballester/tntorch).
    """
    # some work on parameters
    if tol < 1:
        tol = 1.0
    n, r = arr.shape
    if n <= r:
        return np.arange(n, dtype=np.int32), np.eye(n, dtype=arr.dtype)
    if top_k_index == -1 or top_k_index > n:
        top_k_index = n

    top_k_index = max(top_k_index, r)

    # Select r initial pivot rows via QR with
    # column pivoting on arr[:top_k_index].T,
    # then compute the coefficient matrix C = inv(arr[index[:r]]).T @ arr.T.
    _, _, piv = sp_qr(arr[:top_k_index].T, pivoting=True)
    index = np.arange(n, dtype=np.int32)
    index[:r] = piv[:r]
    # C has shape (r, N): C = arr_sub^{-T} @ arr.T so that arr ≈ C.T @ arr_sub
    c: np.ndarray = np.linalg.solve(arr[index[:r]].T, arr.T)
    # find max value in C
    i, j = divmod(int(abs(c[:, :top_k_index]).argmax()), top_k_index)
    # set number of iters to 0
    iters = 0
    # check if need to swap rows
    while abs(c[i, j]) > tol and iters < max_iters:
        # add j to index and recompute C by rank-1 (Sherman-Morrison) update
        index[i] = j
        tmp_row = c[i].copy()
        tmp_column = c[:, j].copy()
        tmp_column[i] -= 1.0
        alpha = -1.0 / c[i, j]
        c += alpha * np.outer(tmp_column, tmp_row)
        iters += 1
        i, j = divmod(int(abs(c[:, :top_k_index]).argmax()), top_k_index)
    return index[:r].copy(), c.T


def _select_indices_maxvol(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select proper indices by maxvol algorithm.

    This method takes the value matrix as input.
    """
    # q, r, _ = scipy.linalg.qr(v, pivoting=True, mode="economic")
    # real_rank = (np.abs(np.diag(r) / r[0, 0]) > 1e-10).sum()
    # # if real_rank < r.shape[0]:
    # #     print("Warning: cutting internal ranks")
    # q = q[:, :real_rank]
    # # print(v)
    q: np.ndarray = v
    q, _ = np.linalg.qr(q)
    selected_inds: np.ndarray
    b: np.ndarray
    selected_inds, b = _py_maxvol(q)
    return selected_inds, b


def _deim(u: np.ndarray) -> np.ndarray:
    """
    Select indices by Discrete Empirical Interpolation Method (DEIM)
    """
    r = u.shape[1]
    indices = np.empty(r, dtype=int)
    indices[0] = np.argmax(np.abs(u[:, 0]))

    for j in range(1, r):
        uselect = u[indices[:j], :j]
        target = u[indices[:j], j]
        try:
            alpha, *_ = np.linalg.lstsq(uselect, target)
        except np.linalg.LinAlgError:
            alpha = np.linalg.pinv(uselect) @ target

        # compute the residual
        rvec = u[:, j] - (u[:, :j] @ alpha)
        # find the maximum from the residual
        indices[j] = np.argmax(np.abs(rvec))

    # print(indices)
    return indices


def _select_indices_deim(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the cross for a single point"""
    u, _, _ = np.linalg.svd(v, full_matrices=False)
    i = _deim(u)
    g = u @ np.linalg.pinv(u[i])
    return g, i


def _select_indices_greedy(
    v: np.ndarray, u: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select indices by maximum the difference between real and approximation.
    """
    diff = np.abs(v - u)
    np.argmax(diff, axis=1)
    return (np.empty(0), np.empty(0))


def _cartesian_product_arrays(*arrays: np.ndarray) -> np.ndarray:
    """
    Compute the Cartesian product of multiple arrays of shape (ni, di),
    resulting in shape (n1*n2*...*nk, d1 + d2 + ... + dk).
    """
    if len(arrays) == 0:
        return np.empty((1, 0), dtype=int)

    shapes = [arr.shape for arr in arrays]
    ns = [s[0] for s in shapes]
    ds = [s[1] for s in shapes]
    total_n = math.prod(ns)

    reshaped = []
    for i, arr in enumerate(arrays):
        # Create shape like (1, ..., ni, ..., 1, di) for broadcasting
        shape = [1] * len(arrays) + [ds[i]]
        shape[i] = arr.shape[0]
        reshaped_arr = arr.reshape(shape)
        reshaped.append(np.broadcast_to(reshaped_arr, ns + [ds[i]]))

    # Concatenate along last axis and reshape
    stacked = np.concatenate(reshaped, axis=-1)
    res: np.ndarray = stacked.reshape(total_n, sum(ds))
    return res


class CrossApproximation:
    """The engine for cross approximation run"""

    def __init__(
        self, tensor_func: TensorFunc, config: Optional[CrossConfig] = None
    ):
        self._config = config if config is not None else CrossConfig()
        self._tensor_func = tensor_func
        self._evaluator = CrossEvaluator(
            tensor_func, self._config.batch_size, self._config.cache_size
        )

    def _construct_matrix(
        self,
        rows: Tuple[List[Index], np.ndarray],
        cols: Tuple[List[Index], np.ndarray],
    ) -> np.ndarray:
        """
        Constructs a matrix from the tensor function by
        evaluating it on the provided row and column indices.
        """
        row_idx, row_vals = rows
        col_idx, col_vals = cols
        nrows, ncols = len(row_vals), len(col_vals)
        positions = {ind: i for i, ind in enumerate(self._tensor_func.indices)}
        row_positions = [positions[ind] for ind in row_idx]
        col_positions = [positions[ind] for ind in col_idx]
        result = None
        # Only coordinate batches are materialized, not the full (n*r, d)
        # Cartesian product and its permuted copy.
        for start in range(0, nrows * ncols, self._config.batch_size):
            flat = np.arange(
                start, min(start + self._config.batch_size, nrows * ncols)
            )
            args = np.empty((len(flat), len(positions)), dtype=int)
            args[:, row_positions] = row_vals[flat % nrows]
            args[:, col_positions] = col_vals[flat // nrows]
            values = self._evaluator(args)
            if result is None:
                result = np.empty(nrows * ncols, dtype=values.dtype)
            result[start : start + len(flat)] = values
        if result is None:
            raise ValueError("Cross sample sets must be nonempty")
        return result.reshape(ncols, nrows)

    def _select_indices(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._config.cross_algo == CrossAlgo.MAXVOL:
            ind, b = _select_indices_maxvol(v)
        elif self._config.cross_algo == CrossAlgo.DEIM:
            b, ind = _select_indices_deim(v)
        else:
            raise ValueError(f"unsupported algo {self._config.cross_algo}")

        return ind, b

    @staticmethod
    def _update_tensor(
        net: "pt.TensorNetwork", node: DimTreeNode, value: np.ndarray
    ) -> None:
        tensor = net.node_tensor(node.node)
        old_indices = tensor.indices[:]
        for axis, ind in enumerate(old_indices):
            if ind in node.free_indices and value.shape[axis] != ind.size:
                raise ValueError("Cross update changed a physical dimension")
        tensor.update_val_size(value)
        # update_val_size replaces Index objects. Preserve physical grids.
        tensor.indices = [
            old if old in node.free_indices else new
            for old, new in zip(old_indices, tensor.indices)
        ]

    def _root_to_leaves(self, node: DimTreeNode) -> None:
        """Update the indices by propagating info from root to leaves."""
        down_ranges = []

        # the indices in the DimTreeNode are up indices
        # when traversing from root to leaves, we need to consider
        # the down indices of the root and the up indices of the siblings
        if len(node.up_info.nodes) > 0:
            p = node.up_info.nodes[0]
            for ind in node.down_info.indices:
                if ind in p.free_indices:
                    down_ranges.append(np.arange(ind.size)[:, None])

            if len(p.up_info.nodes) > 0:
                down_ranges.append(p.down_info.vals)

            for c in p.down_info.nodes:
                if c.node != node.node:
                    down_ranges.append(c.up_info.vals)

            down_vals = _cartesian_product_arrays(*down_ranges)
            v = self._construct_matrix(
                (node.up_info.indices, node.up_info.vals),
                (node.down_info.indices, down_vals),
            )

            selected_inds, _ = self._select_indices(v)
            node.down_info.vals = down_vals[selected_inds, :]
            node.down_info.rank = len(selected_inds)

    def _leaves_to_root(
        self, node: DimTreeNode, net: "pt.TensorNetwork"
    ) -> None:
        """Update the down index values by sweeping from leaves to the root."""
        up_ranges, up_sizes = [], []

        for ind in node.up_info.indices:
            if ind in node.free_indices:
                up_sizes.append(ind.size)
                up_ranges.append(np.arange(ind.size)[:, None])

        for c in sorted(node.down_info.nodes):
            up_sizes.append(len(c.up_info.vals))
            up_ranges.append(c.up_info.vals)

        up_vals = _cartesian_product_arrays(*up_ranges)
        v = self._construct_matrix(
            (node.down_info.indices, node.down_info.vals),
            (node.up_info.indices, up_vals),
        )
        selected_inds, b = self._select_indices(v)
        # print(ind)
        node.up_info.vals = up_vals[selected_inds, :]
        node.up_info.rank = len(selected_inds)
        # print("====>", node.values.up_vals)
        self._update_tensor(
            net,
            node,
            b.reshape(*up_sizes, -1).transpose(np.argsort(node.perm)),
        )

    def _incr_ranks(
        self, tree: DimTreeNode, known: Optional[np.ndarray] = None
    ) -> None:
        """Increment the ranks for all edges"""
        # compute the target size of ranks
        tree.increment_ranks(self._config.kickrank, self._config.max_rank)
        logger.debug("after increment %s", tree.ranks())
        new_ranks = tree.ranks()
        old_ranks = None
        while new_ranks != old_ranks:
            tree.bound_ranks()
            logger.debug("after bounding %s", tree.ranks())
            old_ranks = new_ranks
            new_ranks = tree.ranks()

        if known is None:
            up_val_elems = [
                np.random.randint(0, ind.size, [self._config.kickrank, 1])
                for ind in tree.indices
            ]
            up_vals = np.concatenate(up_val_elems, axis=-1)
        else:
            up_vals = known[
                np.random.randint(
                    0,
                    len(known),
                    [
                        self._config.kickrank,
                    ],
                )
            ]
        tree.add_values(up_vals)

    def _create_validation_set(self) -> np.ndarray:
        if not self._tensor_func.indices:
            return np.empty((self._config.validation_size, 0), dtype=int)
        valid_list = []
        for ind in self._tensor_func.indices:
            valid_list.append(
                np.random.randint(
                    0, ind.size, size=self._config.validation_size
                )
            )
        return np.stack(valid_list, axis=-1)

    def _iterate_tree_nodes(
        self, net: pt.TensorNetwork, tree_nodes: Sequence[DimTreeNode]
    ) -> None:
        for n in tree_nodes:
            if len(n.up_info.nodes) == 0:
                continue

            logger.debug(
                "root to leaves: %s, up indices: %s, down indices: %s",
                n.node,
                [ind.name for ind in n.up_info.indices],
                [ind.name for ind in n.down_info.indices],
            )
            self._root_to_leaves(n)

        for n in reversed(tree_nodes[1:]):
            logger.debug(
                "leaves to root: %s, up indices: %s, down indices: %s",
                n.node,
                [ind.name for ind in n.up_info.indices],
                [ind.name for ind in n.down_info.indices],
            )
            self._leaves_to_root(n, net)

    def _get_root_value(
        self, tree: DimTreeNode, f_sizes: Sequence[int], f_vals: np.ndarray
    ) -> np.ndarray:
        ordered_down_nodes = sorted(tree.down_info.nodes)
        c_indices = [
            ind for c in ordered_down_nodes for ind in c.up_info.indices
        ]
        c_vals = [c.up_info.vals for c in ordered_down_nodes]
        up_vals = _cartesian_product_arrays(*c_vals)
        c_sizes = [len(v) for v in c_vals]
        root_matrix = self._construct_matrix(
            (tree.free_indices, f_vals),
            (c_indices, up_vals),
        )
        return root_matrix.T.reshape(
            tuple(f_sizes) + tuple(c_sizes)
        ).transpose(np.argsort(tree.perm))

    def _check_points(self, points: np.ndarray, name: str) -> np.ndarray:
        points = np.asarray(points)
        sizes = np.array([ind.size for ind in self._tensor_func.indices])
        if (
            points.ndim != 2
            or points.shape[1] != len(sizes)
            or len(points) == 0
        ):
            raise ValueError(f"{name} must have nonempty shape (n, d)")
        if (
            not np.all(np.isfinite(points))
            or np.any(points < 0)
            or np.any(points >= sizes)
            or np.any(points != np.floor(points))
        ):
            raise ValueError(f"{name} contains invalid discrete indices")
        return points.astype(int, copy=False)

    @staticmethod
    def _rms(values: np.ndarray) -> float:
        """Scale before squaring to avoid under/overflow in norm checks."""
        scale = float(np.max(np.abs(values)))
        if scale == 0:
            return 0.0
        return scale * float(np.sqrt(np.mean(np.abs(values / scale) ** 2)))

    def cross(  # pylint: disable=R0913,R0917
        self,
        net: "pt.TensorNetwork",
        root: Optional["pt.NodeName"] = None,
        validation: Optional[np.ndarray] = None,
        eps: float = 0.1,
        initialization: Optional[np.ndarray] = None,
        known: Optional[np.ndarray] = None,
    ) -> CrossResult:
        """Approximate a tree in place, returning measured convergence status.

        All input point arrays use ``tensor_func.indices`` column order.
        Accuracy uses validation RMSE <= atol + eps * target RMS, including
        in NORM mode (which adds full iterate-change diagnostics). Validation
        is sampled evidence, not a bound on unseen entries. ``max_iters`` is
        a positive, exact sweep budget. Caching is opt-in and run-local.
        """
        if not np.isfinite(eps) or eps < 0:
            raise ValueError("eps must be finite and nonnegative")
        if len(net.network) == 0 or not nx.is_tree(net.network):
            raise ValueError("Cross approximation requires a connected tree")
        function_indices = self._tensor_func.indices
        free = net.free_indices()
        if len(set(function_indices)) != len(function_indices) or set(
            free
        ) != set(function_indices):
            raise ValueError(
                "Function indices must match network free indices"
            )
        for name in net.network:
            tensor = net.node_tensor(name)
            if len(set(tensor.indices)) != len(
                tensor.indices
            ) or tensor.value.shape != tuple(i.size for i in tensor.indices):
                raise ValueError("Invalid tensor indices or dimensions")
        for u, v in net.network.edges:
            if len(net.get_contraction_index(u, v)) != 1:
                raise ValueError("Each tree edge must have exactly one bond")
        if root is None:
            root = list(net.network.nodes)[0]
        if root not in net.network:
            raise ValueError("Root is not a network node")
        if validation is None:
            validation = self._create_validation_set()
        validation = self._check_points(validation, "validation")
        if initialization is not None:
            initialization = self._check_points(
                initialization, "initialization"
            )
        if known is not None:
            known = self._check_points(known, "known")
        self._evaluator = CrossEvaluator(
            self._tensor_func, self._config.batch_size, self._config.cache_size
        )
        real = self._evaluator(validation)
        target_rms = self._rms(real)
        tree = net.dimension_tree(root)
        tree_order = [function_indices.index(ind) for ind in tree.indices]
        if known is not None:
            known = known[:, tree_order]
        if initialization is None:
            tree.increment_ranks(1, self._config.max_rank)
            up_vals = [np.random.randint(0, ind.size) for ind in tree.indices]
            tree.add_values(np.asarray([up_vals]))
        else:
            tree.increment_ranks(len(initialization), self._config.max_rank)
            tree.add_values(initialization[:, tree_order])

        f_sizes = [ind.size for ind in tree.free_indices]
        f_vals = _cartesian_product_arrays(
            *[np.arange(sz)[:, None] for sz in f_sizes]
        )

        tree_nodes = tree.preorder()
        history = []
        previous_estimate = None
        previous_ranks = None
        stalled = 0
        converged = False
        stop_reason = "max_iters"
        for trial in range(self._config.max_iters):
            old_net = (
                copy.deepcopy(net)
                if self._config.convergence == ConvergenceCheck.NORM
                else None
            )
            self._iterate_tree_nodes(net, tree_nodes)

            # get the value for the root node
            root_val = self._get_root_value(tree, f_sizes, f_vals)
            self._update_tensor(net, tree, root_val)
            # Bound validation contractions as well as oracle batches.
            estimate = np.concatenate(
                [
                    net.evaluate(
                        function_indices,
                        validation[start : start + self._config.batch_size],
                    ).reshape(-1)
                    for start in range(
                        0, len(validation), self._config.batch_size
                    )
                ]
            )
            if estimate.shape != real.shape or not np.all(
                np.isfinite(estimate)
            ):
                raise ValueError("Network evaluation is invalid")
            rmse = self._rms(real - estimate)
            err = rmse / target_rms if target_rms else rmse
            norm_change = None
            if old_net is not None:
                diff_net = net - old_net
                delta, norm = diff_net.norm(), net.norm()
                norm_change = delta / norm if norm else delta
            ranks = [n.up_info.rank for n in tree_nodes[1:]]
            history.append(
                CrossSweep(
                    sweep=trial + 1,
                    ranks=ranks,
                    validation_error=err,
                    validation_rmse=rmse,
                    norm_change=norm_change,
                    evaluated=self._evaluator.stats.evaluated,
                )
            )
            logger.debug("step: %s, error: %s", trial, err)
            if rmse <= self._config.atol + eps * target_rms:
                converged = True
                stop_reason = "validation"
                break
            if previous_estimate is not None and ranks == previous_ranks:
                change = self._rms(estimate - previous_estimate)
                stationary = change <= self._config.atol + 1e-12 * max(
                    target_rms, self._rms(estimate)
                )
                stalled = stalled + 1 if stationary else 0
            else:
                stalled = 0
            if stalled >= self._config.stagnation_patience:
                stop_reason = (
                    "rank_limit"
                    if self._config.max_rank is not None
                    and any(r >= self._config.max_rank for r in ranks)
                    else "stagnated"
                )
                break
            previous_estimate, previous_ranks = estimate, ranks
            if trial + 1 < self._config.max_iters:
                self._incr_ranks(tree, known=known)

        return CrossResult(
            net=net,
            dim_tree=tree,
            ranks_and_errors=[
                (max(h.ranks, default=1), h.validation_error) for h in history
            ],
            converged=converged,
            stop_reason=stop_reason,
            history=history,
            evaluations=self._evaluator.stats,
        )

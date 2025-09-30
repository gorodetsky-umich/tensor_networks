"""Cross Approximation."""

import random
import copy
from typing import Optional, Sequence, Tuple

import numpy as np
from line_profiler import profile
from tntorch.maxvol import py_maxvol
import scipy

import pytens.algs as pt
from pytens.cross.funcs import TensorFunc
from pytens.types import DimTreeNode


def cartesian_product_arrays(*arrays):
    """
    Compute the Cartesian product of multiple arrays of shape (ni, di),
    resulting in shape (n1*n2*...*nk, d1 + d2 + ... + dk).
    """
    if len(arrays) == 0:
        return np.array([[]])

    shapes = [arr.shape for arr in arrays]
    ns = [s[0] for s in shapes]
    ds = [s[1] for s in shapes]
    total_n = np.prod(ns)

    reshaped = []
    for i, arr in enumerate(arrays):
        # Create shape like (1, ..., ni, ..., 1, di) for broadcasting
        shape = [1] * len(arrays) + [ds[i]]
        shape[i] = arr.shape[0]
        reshaped_arr = arr.reshape(shape)
        reshaped.append(np.broadcast_to(reshaped_arr, ns + [ds[i]]))

    # Concatenate along last axis and reshape
    stacked = np.concatenate(reshaped, axis=-1)
    return stacked.reshape(total_n, sum(ds))


@profile
def construct_matrix(tensor_func: TensorFunc, rows, cols) -> np.ndarray:
    """
    Constructs a matrix from the tensor function by
    evaluating it on the provided row and column indices.
    """
    row_idx, row_vals = rows
    col_idx, col_vals = cols
    args = cartesian_product_arrays(col_vals, row_vals).astype(int, copy=False)
    indices = col_idx + row_idx
    perm = [indices.index(ind) for ind in tensor_func.indices]
    args = args[:, perm]
    # print("constructing", len(args))
    return tensor_func(args).reshape(len(col_vals), len(row_vals))


@profile
def select_indices(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select proper indices by maxvol algorithm.

    This method takes the value matrix as input.
    """
    q, r, _ = scipy.linalg.qr(v, pivoting=True, mode="economic")
    real_rank = (np.abs(np.diag(r) / r[0, 0]) > 1e-14).sum()
    # q, _ = np.linalg.qr(v)
    q = q[:, :real_rank]
    return py_maxvol(q)


def select_indices_greedy(
    v: np.ndarray, u: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select indices by maximum the difference between real and approximation.
    """
    diff = np.abs(v - u)
    np.argmax(diff, axis=1)
    return (np.empty(0), np.empty(0))


@profile
def root_to_leaves(tensor_func: TensorFunc, node: DimTreeNode) -> None:
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

        down_vals = cartesian_product_arrays(*down_ranges)
        v = construct_matrix(
            tensor_func,
            (node.up_info.indices, node.up_info.vals),
            (node.down_info.indices, down_vals),
        )
        ind, _ = select_indices(v)
        node.down_info.vals = down_vals[ind, :]
        node.down_info.rank = len(ind)


@profile
def leaves_to_root(
    tensor_func: TensorFunc, node: DimTreeNode, net: "pt.TreeNetwork"
) -> None:
    """Update the down index values by sweeping from leaves to the root."""
    up_ranges, up_sizes = [], []

    for ind in node.up_info.indices:
        if ind in node.free_indices:
            up_sizes.append(ind.size)
            up_ranges.append(np.arange(ind.size)[:, None])

    for c in node.down_info.nodes:
        up_sizes.append(len(c.up_info.vals))
        up_ranges.append(c.up_info.vals)

    up_vals = cartesian_product_arrays(*up_ranges)
    v = construct_matrix(
        tensor_func,
        (node.down_info.indices, node.down_info.vals),
        (node.up_info.indices, up_vals),
    )
    ind, b = select_indices(v)
    node.up_info.vals = up_vals[ind, :]
    node.up_info.rank = len(ind)
    # print("====>", node.values.up_vals)
    net.node_tensor(node.node).update_val_size(b.reshape(*up_sizes, -1))

def incr_ranks(tree: DimTreeNode, kickrank: int = 2, known: Optional[np.ndarray] = None) -> None:
    """Increment the ranks for all edges"""
    # compute the target size of ranks
    tree.increment_ranks(kickrank)
    tree.bound_ranks()
    tree.bound_ranks()
    print(tree.ranks())

    if known is None:
        up_vals = [np.random.randint(0, ind.size, [kickrank, 1]) for ind in tree.indices]
        up_vals = np.concatenate(up_vals, axis=-1)
    else:
        up_vals = known[np.random.randint(0, len(known), [kickrank,])]
    tree.add_values(up_vals)

class CrossResult:
    """Class to record cross approximation results."""

    def __init__(
        self,
        dim_tree: DimTreeNode,
        ranks_and_errors: Sequence[Tuple[int, float]],
    ):
        self.dim_tree = dim_tree
        self.ranks_and_errors = ranks_and_errors


@profile
def cross(
    f: TensorFunc,
    net: "pt.TreeNetwork",
    root: "pt.NodeName",
    eps: float = 0.1,
    val_size: int = 1000,
    max_size: Optional[int] = None,
    initialization: Optional[np.ndarray] = None,
    known: Optional[np.ndarray] = None,
) -> CrossResult:
    """Cross approximation for the given network structure."""
    # print("root is", root)
    # print(net)
    tree = net.dimension_tree(root)
    if initialization is None:
        tree.increment_ranks(1)
        up_vals = [np.random.randint(0, ind.size) for ind in tree.indices]
        tree.add_values(np.asarray([up_vals]))
    else:
        tree.increment_ranks(len(initialization))
        tree.add_values(initialization)

    converged = False

    validation = []
    for ind in f.indices:
        validation.append(np.random.randint(0, ind.size, size=val_size))
    validation = np.stack(validation, axis=-1)
    real = f(validation)
    f_sizes = [ind.size for ind in tree.free_indices]
    f_vals = cartesian_product_arrays(
        *[np.arange(sz)[:, None] for sz in f_sizes]
    )

    tree_nodes = tree.preorder()
    ranks_and_errs = {}
    trial = 0
    while not converged:
        for n in tree_nodes:
            if len(n.up_info.nodes) == 0:
                continue

            root_to_leaves(f, n)

        for n in reversed(tree_nodes[1:]):
            leaves_to_root(f, n, net)

        # get the value for the root node
        c_indices = [
            ind for c in tree.down_info.nodes for ind in c.up_info.indices
        ]
        c_vals = [c.up_info.vals for c in tree.down_info.nodes]
        up_vals = cartesian_product_arrays(*c_vals)
        c_sizes = [len(v) for v in c_vals]
        root_matrix = construct_matrix(
            f,
            (tree.free_indices, f_vals),
            (c_indices, up_vals),
        )
        # print(c_indices, up_vals)
        root_val = root_matrix.T.reshape(*f_sizes, *c_sizes)
        net.node_tensor(tree.node).update_val_size(root_val)

        estimate = net.evaluate(net.free_indices(), validation).reshape(-1)
        err = np.linalg.norm(real - estimate) / np.linalg.norm(real)
        ranks_and_errs[len(up_vals)] = err
        print("rank:", trial, "error:", err)
        # print(net)
        if err <= eps or (max_size is not None and len(up_vals) >= max_size):
            break

        trial += 1
        incr_ranks(tree, known=known)

    # print(net)
    ranks_and_errs = sorted(list(ranks_and_errs.items()))
    return CrossResult(tree, ranks_and_errs)

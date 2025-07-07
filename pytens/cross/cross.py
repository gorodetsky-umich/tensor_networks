"""Cross Approximation."""

import random
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
    return tensor_func(args).reshape(len(col_vals), len(row_vals))

@profile
def select_indices(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select proper indices by maxvol algorithm.

    This method takes the value matrix as input.
    """
    q, r, _ = scipy.linalg.qr(v, pivoting=True, mode="economic")
    real_rank = (np.abs(np.diag(r) / r[0, 0]) > 1e-14).sum()

    q = q[:, :real_rank]
    return py_maxvol(
        q, max_iters=3
    )

def select_indices_greedy(v: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    if node.conn.parent is not None:
        for ind in node.info.down_indices:
            if ind in node.conn.parent.info.free_indices:
                down_ranges.append(np.arange(ind.size)[:, None])

        if node.conn.parent.conn.parent is not None:
            down_ranges.append(node.conn.parent.values.down_vals)

        for c in node.conn.parent.conn.children:
            if c.info.node != node.info.node:
                down_ranges.append(c.values.up_vals)

        down_vals = cartesian_product_arrays(*down_ranges)
        v = construct_matrix(
            tensor_func,
            (node.info.up_indices, node.values.up_vals),
            (node.info.down_indices, down_vals),
        )
        ind, _ = select_indices(v)
        node.values.down_vals = down_vals[ind, :]

@profile
def leaves_to_root(
    tensor_func: TensorFunc, node: DimTreeNode, net: "pt.TreeNetwork"
) -> None:
    """Update the down index values by sweeping from leaves to the root."""
    up_ranges, up_sizes = [], []

    for ind in node.info.up_indices:
        if ind in node.info.free_indices:
            up_sizes.append(ind.size)
            up_ranges.append(np.arange(ind.size)[:, None])

    for c in node.conn.children:
        up_sizes.append(len(c.values.up_vals))
        up_ranges.append(c.values.up_vals)

    up_vals = cartesian_product_arrays(*up_ranges)
    v = construct_matrix(
        tensor_func,
        (node.info.down_indices, node.values.down_vals),
        (node.info.up_indices, up_vals),
    )
    ind, b = select_indices(v)
    node.values.up_vals = up_vals[ind, :]
    # print("====>", node.values.up_vals)
    net.node_tensor(node.info.node).update_val_size(b.reshape(*up_sizes, -1))


def incr_ranks(tree: DimTreeNode, net):
    """Increment the ranks for all edges"""
    if tree.conn.parent is not None:
        new_up = [random.randint(0, ind.size - 1) for ind in tree.info.up_indices]
        new_up = np.asarray(new_up)[None,:]
        tree.values.up_vals = np.append(tree.values.up_vals, new_up, axis=0)

        new_down = [random.randint(0, ind.size - 1) for ind in tree.info.down_indices]
        new_down = np.asarray(new_down)[None,:]
        tree.values.down_vals = np.append(tree.values.down_vals, new_down, axis=0)

    for c in tree.conn.children:
        incr_ranks(c, net)


def init_values(net: "pt.TreeNetwork", tree: DimTreeNode) -> None:
    """Initialize the up and down values for the given dimension tree."""
    for c in tree.conn.children:
        rank = net.get_contraction_index(tree.info.node, c.info.node)[0].size

        up_vals = []
        for ind in c.info.up_indices:
            up_vals.append(np.random.randint(0, ind.size - 1, rank))
        c.values.up_vals = np.stack(up_vals, axis=-1)
        if len(c.values.up_vals.shape) == 1:
            c.values.up_vals = c.values.up_vals[:, None]

        down_vals = []
        for ind in c.info.down_indices:
            down_vals.append(np.random.randint(0, ind.size - 1, rank))
        c.values.down_vals = np.stack(down_vals, axis=-1)
        if len(c.values.down_vals.shape) == 1:
            c.values.down_vals = c.values.down_vals[:, None]

        init_values(net, c)

class CrossResult:
    """Class to record cross approximation results."""
    def __init__(self, dim_tree: DimTreeNode, ranks_and_errors: Sequence[Tuple[int, float]]):
        self.dim_tree = dim_tree
        self.ranks_and_errors = ranks_and_errors

@profile
def cross(
    f: TensorFunc,
    net: "pt.TreeNetwork",
    root: "pt.NodeName",
    eps: float = 0.1,
    val_size: int = 10000,
    max_size: Optional[int] = None,
) -> CrossResult:
    """Cross approximation for the given network structure."""
    # print("root is", root)
    # print(net)
    tree = net.dimension_tree(root)
    init_values(net, tree)
    converged = False

    validation = []
    for ind in f.indices:
        validation.append(np.random.randint(0, ind.size - 1, size=val_size))
    validation = np.stack(validation, axis=-1)
    real = f(validation)
    f_sizes = [ind.size for ind in tree.info.free_indices]
    f_vals = cartesian_product_arrays(*[np.arange(sz)[:,None] for sz in f_sizes])

    tree_nodes = tree.preorder()
    ranks_and_errs = {}
    while not converged:
        for n in tree_nodes:
            if n.conn.parent is None:
                continue

            root_to_leaves(f, n)

        for n in reversed(tree_nodes[1:]):
            leaves_to_root(f, n, net)

        # get the value for the root node
        c_indices = [ind for c in tree.conn.children for ind in c.info.up_indices]
        c_vals = [c.values.up_vals for c in tree.conn.children]
        up_vals = cartesian_product_arrays(*c_vals)
        c_sizes = [len(v) for v in c_vals]
        root_matrix = construct_matrix(
            f,
            (tree.info.free_indices, f_vals),
            (c_indices, up_vals),
        )
        # print(c_indices, up_vals)
        root_val = root_matrix.T.reshape(*f_sizes, *c_sizes)
        net.node_tensor(tree.info.node).update_val_size(root_val)

        estimate = net.evaluate(validation).reshape(-1)
        err = np.linalg.norm(real - estimate) / np.linalg.norm(real)
        ranks_and_errs[len(up_vals)] = err
        # print("Error:", err, eps)
        if err <= eps or (max_size is not None and len(up_vals) >= max_size):
            break

        incr_ranks(tree, net)

    # print(net)
    ranks_and_errs = sorted(list(ranks_and_errs.items()))
    return CrossResult(tree, ranks_and_errs)

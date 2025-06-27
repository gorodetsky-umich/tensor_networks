import random
from itertools import product
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy
import teneva
from line_profiler import profile
from tntorch.maxvol import py_maxvol, py_rect_maxvol

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
    # construct the arguments
    # print("row_vals", row_vals.shape, "col_vals", col_vals.shape)
    # row_args = row_vals[None, :, :]
    # col_args = col_vals[:, None, :]

    # # Broadcasted addition: (m, n, a + b)
    # args = np.concatenate([np.broadcast_to(col_args, (len(col_vals), len(row_vals), len(col_idx))),
    #                        np.broadcast_to(row_args, (len(col_vals), len(row_vals), len(row_idx)))], axis=2)

    # # Reshape to (m * n, a + b)
    # args = args.reshape(-1, len(col_idx + row_idx)).astype(int, copy=False)
    args = cartesian_product_arrays(col_vals, row_vals).astype(int, copy=False)
    # args = [list(xs) + list(ys) for xs, ys in product(col_vals, row_vals)]
    # args = np.asarray(args, dtype=int)
    args = args[:, np.argsort(col_idx + row_idx)]
    return tensor_func(args).reshape(len(col_vals), len(row_vals))

@profile
def select_indices(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select proper indices by maxvol algorithm.

    This method takes the value matrix as input.
    """
    # prev_rank = v.shape[1]
    # q, r, _ = scipy.linalg.qr(v, pivoting=True, mode="economic")
    # real_rank = (np.abs(np.diag(r) / r[0, 0]) > 1e-14).sum()
    # dr_min = dr_max = 0
    # if real_rank == prev_rank:
    #     dr_max = 1
    q, _ = np.linalg.qr(v)

    # q = q[:, :real_rank]
    return py_maxvol(
        q, max_iters=3
    )


@profile
def root_to_leaves(tensor_func: TensorFunc, node: DimTreeNode, net: "pt.TreeNetwork") -> None:
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
        down_sizes = [len(r) for r in down_ranges]
        v = construct_matrix(
            tensor_func,
            (node.info.up_indices, node.values.up_vals),
            (node.info.down_indices, down_vals),
        )
        ind, b = select_indices(v)
        node.values.down_vals = down_vals[ind, :]
        # net.node_tensor(node.conn.parent.info.node).update_val_size(b.reshape(*down_sizes, -1))
        # print("====>", node.values.down_vals.shape)

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
    # print("====>", node.values.up_vals.shape)
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


@profile
def cross(
    f: TensorFunc,
    net: "pt.TreeNetwork",
    root: "pt.NodeName",
    eps: float = 0.1,
    val_size: int = 10000,
    max_size: Optional[int] = None,
) -> Sequence[Tuple[int, float]]:
    """Cross approximation for the given network structure."""
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
    ranks_and_errs = []
    while not converged:
        for n in tree_nodes:
            if n.conn.parent is None:
                continue

            root_to_leaves(f, n, net)

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
        root_val = root_matrix.T.reshape(*f_sizes, *c_sizes)
        net.node_tensor(tree.info.node).update_val_size(root_val)

        estimate = net.evaluate(validation).reshape(-1)
        err = np.linalg.norm(real - estimate) / np.linalg.norm(real)
        ranks_and_errs.append((len(up_vals), err))
        # print("Error:", err)
        if err <= eps or (max_size is not None and len(up_vals) >= max_size):
            break

        incr_ranks(tree, net)

    return ranks_and_errs

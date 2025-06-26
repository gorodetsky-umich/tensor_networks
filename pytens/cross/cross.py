from typing import Optional, Sequence, Tuple, List, Literal, Dict
import time
from itertools import product
import random

import numpy as np
import scipy
import teneva

from pytens.cross.funcs import TensorFunc, FuncHilbert
import pytens.algs as pt
from pytens.types import Index, DimTreeNode

from line_profiler import profile


def construct_matrix(tensor_func: TensorFunc, rows, cols) -> np.ndarray:
    """
    Constructs a matrix from the tensor function by
    evaluating it on the provided row and column indices.
    """
    row_idx, row_vals = rows
    col_idx, col_vals = cols
    # construct the arguments
    args = [list(xs) + list(ys) for xs, ys in product(col_vals, row_vals)]
    args = np.asarray(args, dtype=int)
    args = args[:, np.argsort(col_idx + row_idx)]
    return tensor_func(args).reshape(len(col_vals), len(row_vals))


def select_indices(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select proper indices by maxvol algorithm.

    This method takes the value matrix as input.
    """
    # try to grow the index value sets
    prev_rank = v.shape[1]
    q, r, _ = scipy.linalg.qr(v, pivoting=True, mode="economic")
    real_rank = (np.abs(np.diag(r) / r[0, 0]) > 1e-14).sum()
    dr_min = dr_max = 0
    if real_rank == prev_rank:
        dr_max = 1

    q = q[:, :real_rank]
    return teneva._maxvol(
        q, tau=1.01, dr_min=dr_min, dr_max=dr_max, tau0=1.01, k0=100
    )


@profile
def root_to_leaves(tensor_func: TensorFunc, node: DimTreeNode) -> None:
    """Update the indices by propagating info from root to leaves."""
    down_ranges = []

    # the indices in the DimTreeNode are up indices
    # when traversing from root to leaves, we need to consider the down indices of the root and the up indices of the siblings
    if node.conn.parent is not None:
        for ind in node.info.down_indices:
            if ind in node.conn.parent.info.free_indices:
                down_ranges.append([[i] for i in range(ind.size)])

        if node.conn.parent.conn.parent is not None:
            down_ranges.append(node.conn.parent.values.down_vals)

        for c in node.conn.parent.conn.children:
            if c.info.node != node.info.node:
                down_ranges.append(c.values.up_vals)

        down_vals = [sum(list(g), []) for g in product(*down_ranges)]
        v = construct_matrix(
            tensor_func,
            (node.info.up_indices, node.values.up_vals),
            (node.info.down_indices, down_vals),
        )
        ind, _ = select_indices(v)
        node.values.down_vals = [down_vals[i] for i in ind]


@profile
def leaves_to_root(
    tensor_func: TensorFunc, node: DimTreeNode, net: "pt.TreeNetwork"
) -> None:
    """Update the down index values by sweeping from leaves to the root."""
    up_ranges, up_sizes = [], []

    for ind in node.info.up_indices:
        if ind in node.info.free_indices:
            up_sizes.append(ind.size)
            up_ranges.append(np.arange(ind.size)[:, None].tolist())

    for c in node.conn.children:
        up_sizes.append(len(c.values.up_vals))
        up_ranges.append(c.values.up_vals)

    up_vals = [sum(list(g), []) for g in product(*up_ranges)]
    v = construct_matrix(
        tensor_func,
        (node.info.down_indices, node.values.down_vals),
        (node.info.up_indices, up_vals),
    )
    ind, b = select_indices(v)
    node.values.up_vals = [up_vals[i] for i in ind]
    net.node_tensor(node.info.node).update_val_size(b.reshape(*up_sizes, -1))


def incr_ranks(tree: DimTreeNode, net):
    """Increment the ranks for all edges"""
    if tree.conn.parent is not None:
        new_up = [random.randint(0, ind.size - 1) for ind in tree.info.indices]
        tree.values.up_vals.append(new_up)
        new_down = []
        for ind in tree.conn.parent.info.indices:
            if ind not in tree.info.indices:
                new_down.append(random.randint(0, ind.size - 1))
        tree.values.down_vals.append(new_down)

    for c in tree.conn.children:
        incr_ranks(c, net)


def init_values(net: "pt.TreeNetwork", tree: DimTreeNode) -> None:
    """Initialize the up and down values for the given dimension tree."""
    for c in tree.conn.children:
        rank = net.get_contraction_index(tree.info.node, c.info.node)[0].size

        up_vals = []
        for ind in c.info.up_indices:
            up_vals.append(np.random.randint(0, ind.size - 1, rank))
        c.values.up_vals = np.stack(up_vals, axis=-1).tolist()

        down_vals = []
        for ind in c.info.down_indices:
            down_vals.append(np.random.randint(0, ind.size - 1, rank))
        c.values.down_vals = np.stack(down_vals, axis=-1).tolist()

        init_values(net, c)


@profile
def cross(
    f: TensorFunc,
    net: "pt.TreeNetwork",
    root: "pt.NodeName",
    eps: float = 0.1,
    val_size: int = 10000,
):
    """Cross approximation for the given network structure."""
    tree = net.dimension_tree(root)
    init_values(net, tree)
    converged = False

    validation = []
    for ind in f.indices:
        validation.append(np.random.randint(0, ind.size - 1, size=val_size))
    validation = np.stack(validation, axis=-1)

    tree_nodes = tree.preorder()
    while not converged:
        for n in tree_nodes:
            if n.conn.parent is None:
                continue

            root_to_leaves(f, n)

        for n in reversed(tree_nodes[1:]):
            leaves_to_root(f, n, net)

        # get the value for the root node
        f_sizes = [ind.size for ind in tree.info.free_indices]
        f_vals = [range(sz) for sz in f_sizes]
        c_indices = [ind for c in tree.conn.children for ind in c.info.indices]
        c_vals = [c.values.up_vals for c in tree.conn.children]
        up_vals = [sum(list(g), []) for g in product(*c_vals)]
        c_sizes = [len(v) for v in c_vals]
        root_matrix = construct_matrix(
            f,
            (tree.info.free_indices, list(product(*f_vals))),
            (c_indices, up_vals),
        )
        root_val = root_matrix.T.reshape(*f_sizes, *c_sizes)
        net.node_tensor(tree.info.node).update_val_size(root_val)

        estimate = net.evaluate(validation)
        estimate = estimate.reshape(-1)
        real = f(validation)
        err = np.linalg.norm(real - estimate) / np.linalg.norm(real)
        # print("Error:", err)
        if err <= eps:
            return
        
        incr_ranks(tree, net)


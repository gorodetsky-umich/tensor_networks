"""Cross Approximation."""

import random
import copy
from typing import Optional, Sequence, Tuple
import time
import logging

import numpy as np
from line_profiler import profile
from tntorch.maxvol import py_maxvol
import scipy

import pytens.algs as pt
from pytens.cross.funcs import TensorFunc, PermuteFunc
from pytens.types import DimTreeNode
from pytens.logger import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ALGO = "maxvol"

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
    # q, r, _ = scipy.linalg.qr(v, pivoting=True, mode="economic")
    # real_rank = (np.abs(np.diag(r) / r[0, 0]) > 1e-10).sum()
    # # if real_rank < r.shape[0]:
    # #     print("Warning: cutting internal ranks")
    # q = q[:, :real_rank]
    # # print(v)
    q = v
    q, _ = np.linalg.qr(q)
    return py_maxvol(q)

def deim(u: np.ndarray):
    """
    Select indices by Descrete Empirical Interpolation Method (DEIM)
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

def select_indices_deim(v: np.ndarray):
    """Compute the cross for a single point"""
    u, _, _ = np.linalg.svd(v, full_matrices=False)
    i = deim(u)
    g = u @ np.linalg.pinv(u[i])
    return g, i

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
        # print(
        #     (node.up_info.indices, node.up_info.vals),
        #     (node.down_info.indices, down_vals),
        # )
        v = construct_matrix(
            tensor_func,
            (node.up_info.indices, node.up_info.vals),
            (node.down_info.indices, down_vals),
        )
        if ALGO == "maxvol":
            ind, _ = select_indices(v)
        elif ALGO == "deim":
            _, ind = select_indices_deim(v)
        else:
            raise Exception("unsupported algo")
        # print(ind)
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

    for c in sorted(node.down_info.nodes):
        up_sizes.append(len(c.up_info.vals))
        up_ranges.append(c.up_info.vals)

    up_vals = cartesian_product_arrays(*up_ranges)
    v = construct_matrix(
        tensor_func,
        (node.down_info.indices, node.down_info.vals),
        (node.up_info.indices, up_vals),
    )
    if ALGO == "maxvol":
        ind, b = select_indices(v)
    elif ALGO == "deim":
        b, ind = select_indices_deim(v)
    else:
        raise Exception("unsupported algo")
    # print(ind)
    node.up_info.vals = up_vals[ind, :]
    node.up_info.rank = len(ind)
    # print("====>", node.values.up_vals)
    net.node_tensor(node.node).update_val_size(b.reshape(*up_sizes, -1))

def incr_ranks(tree: DimTreeNode, kickrank: int = 2, max_rank: Optional[int] = None, known: Optional[np.ndarray] = None) -> None:
    """Increment the ranks for all edges"""
    # compute the target size of ranks
    tree.increment_ranks(kickrank, max_rank)
    logger.trace("after increment %s", tree.ranks())
    new_ranks = tree.ranks()
    old_ranks = None
    while new_ranks != old_ranks:
        tree.bound_ranks()
        logger.trace("after bounding %s", tree.ranks())
        old_ranks = new_ranks
        new_ranks = tree.ranks()
        
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
    validation: Optional[np.ndarray] = None,
    eps: float = 0.1,
    val_size: int = 1000,
    max_iters: Optional[int] = None,
    max_rank: Optional[int] = None,
    initialization: Optional[np.ndarray] = None,
    known: Optional[np.ndarray] = None,
    kickrank: int = 2,
) -> CrossResult:
    """Cross approximation for the given network structure."""
    # print("root is", root)
    # print(net)
    tree = net.dimension_tree(root)
    if initialization is None:
        tree.increment_ranks(1, max_rank)
        up_vals = [np.random.randint(0, ind.size) for ind in tree.indices]
        tree.add_values(np.asarray([up_vals]))
    else:
        tree.increment_ranks(len(initialization), max_rank)
        tree.add_values(initialization)

    converged = False

    if validation is None:
        valid_list = []
        for ind in f.indices:
            valid_list.append(np.random.randint(0, ind.size, size=val_size))
        validation = np.stack(valid_list, axis=-1)

    real = f(validation)
    f_sizes = [ind.size for ind in tree.free_indices]
    f_vals = cartesian_product_arrays(
        *[np.arange(sz)[:, None] for sz in f_sizes]
    )

    tree_nodes = tree.preorder()
    ranks_and_errs = {}
    trial = 0
    while not converged:
        # print(net)
        for n in tree_nodes:
            if len(n.up_info.nodes) == 0:
                continue

            logger.trace("root to leaves: %s, up indices: %s, down indices: %s", n.node, [ind.name for ind in n.up_info.indices], [ind.name for ind in n.down_info.indices])
            root_to_leaves(f, n)

        for n in reversed(tree_nodes[1:]):
            logger.trace("leaves to root: %s, up indices: %s, down indices: %s", n.node, [ind.name for ind in n.up_info.indices], [ind.name for ind in n.down_info.indices])
            leaves_to_root(f, n, net)

        # get the value for the root node
        ordered_down_nodes = sorted(tree.down_info.nodes)
        c_indices = [
            ind for c in ordered_down_nodes for ind in c.up_info.indices
        ]
        c_vals = [c.up_info.vals for c in ordered_down_nodes]
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

        # eval_start = time.time()
        # estimate_tensor = net.contract()
        # ind_perm = [estimate_tensor.indices.index(ind) for ind in f.indices]
        # estimate = estimate_tensor.value.transpose(ind_perm)[*validation]
        # if isinstance(net, pt.TensorTrain):
        #     estimate = net.evaluate(f.indices, validation).reshape(-1)
        # else:
        #     estimate = net.evaluate_cross(f.indices, validation).reshape(-1)
        estimate = net.evaluate(f.indices, validation).reshape(-1)

        # print("evaluate time:", time.time() - eval_start)
        # logger.debug("%s", net)
        # print(estimate.shape, real.shape)
        err = np.linalg.norm(real - estimate) / np.linalg.norm(real)
        ranks_and_errs[len(up_vals)] = err
        logger.debug("step: %s, error: %s", trial, err)
        # import sys
        # sys.stdout.flush()
        # print(net)
        if err <= eps or (max_iters is not None and trial >= max_iters):
            break

        trial += 1
        incr_ranks(tree, kickrank=kickrank+trial, max_rank=max_rank, known=known)

    # print(net)
    ranks_and_errs = sorted(list(ranks_and_errs.items()))
    # print(ranks_and_errs)
    return CrossResult(tree, ranks_and_errs)

# def tt_cur_deim(f: TensorFunc, js: Sequence[np.ndarray]):
#     d = len(f.indices)
#     cores = [np.empty(0) for _ in range(d)]
#     ranks = [1 for _ in range(d+1)]
#     indices = [np.empty(0) for _ in range(d-1)]

#     # print(np.arange(f.indices[0].size, dtype=int)[:, None], js[0])
#     cores[0], indices[0] = cur_deim(f, np.arange(f.indices[0].size, dtype=int)[:, None], js[0])
#     indices[0] = indices[0][:, None]
#     # r = f(cartesian_product_arrays(indices[0], *[np.arange(ind.size, dtype=int)[:, None] for ind in f.indices[1:]]))
#     ranks[1] = indices[0].shape[0]
#     for z in range(1, d - 1):
#         # print(z)
#         # print(indices[z - 1])
#         left_indices = cartesian_product_arrays(indices[z-1], np.arange(f.indices[z].size, dtype=int)[:, None])
#         # print(left_indices)
#         cores[z], group_indices = cur_deim(f, left_indices, js[z])
#         ranks[z+1] = len(group_indices)
#         # print(group_indices, tuple(ranks[1:z] + [f.indices[z].size]))
#         local_indices = np.unravel_index(group_indices.astype(int), tuple(ranks[1:z+1] + [f.indices[z].size]))
#         indices[z] = np.stack([indices[z-1][:,i][inds] for i, inds in enumerate(local_indices[:-1])] + [local_indices[-1]], axis=-1)
#         # r = f(cartesian_product_arrays(indices[z], *[np.arange(ind.size, dtype=int)[:, None] for ind in f.indices[z+1:]]))

#     cores[d - 1] = f(cartesian_product_arrays(indices[d-2], np.arange(f.indices[d-1].size, dtype=int)[:, None]))
#     for z in range(1, d+1):
#         cores[z-1] = cores[z-1].reshape(ranks[z-1], f.indices[z-1].size, ranks[z])

#     return cores, indices

# def random_right_indices(sizes: Sequence[int], r: int):
#     js = []
#     d = len(sizes)
#     for j in range(d):
#         indices = np.empty((r, d - j - 1), dtype=int)
#         for i in range(j + 1, d):
#             indices[:, i - j - 1] = np.random.randint(0, sizes[i], size=(r,))

#         js.append(indices)

#     return js

# def tt_cur_deim_iterative(f: TensorFunc, eps: float, max_iters: int = 2, kickrank: int = 2, val_size: int = 2500):
#     # randomly sample right indices
#     d = len(f.indices)
#     js = random_right_indices(f.shape, kickrank)

#     validation = []
#     f_indices = f.indices
#     for ind in f_indices:
#         validation.append(np.random.randint(0, ind.size, size=val_size))
#     validation = np.stack(validation, axis=-1)
#     real = f(validation)

#     while True:
#         itr = 0
#         indices = [[], []]
#         while True:
#             cores, indices[itr%2] = tt_cur_deim(f, js)
#             itr += 1

#             if itr >= max_iters:
#                 break

#             f = PermuteFunc(f.indices[::-1], f, list(reversed(range(d))))
#             js = [ind[:,::-1] for ind in indices[(itr+1)%2][::-1]]

#         # check the error
#         tt = pt.TensorTrain()
#         for i, core in enumerate(cores):
#             core_indices = []
#             if i != 0:
#                 core_indices.append(pt.Index(f"s_{i-1}", core.shape[0]))

#             core_indices.append(f.indices[i])

#             if i != len(cores) - 1:
#                 core_indices.append(pt.Index(f"s_{i}", core.shape[2]))

#             if i == 0:
#                 core = core.squeeze(0)
            
#             if i == len(cores) - 1:
#                 core = core.squeeze(-1)

#             tt.add_node(f"G{i}", pt.Tensor(core, core_indices))

#             if i != 0:
#                 tt.add_edge(f"G{i-1}", f"G{i}")

#         tt_indices = tt.free_indices()
#         perm = [tt_indices.index(ind) for ind in f_indices]
#         # print(perm)
#         estimate = tt.evaluate(tt_indices, validation[:, perm]).reshape(-1)

#         # print("evaluate time:", time.time() - eval_start)
#         logger.debug("%s", tt)
#         # print(estimate.shape, real.shape)
#         err = np.linalg.norm(real - estimate) / np.linalg.norm(real)
#         print(err)
#         if err < eps:
#             return tt
        
#         # adaptively increase the ranks
#         new_js = random_right_indices(f.shape, kickrank)
#         js = [np.concat([old, new], axis=0) for old, new in zip(js, new_js)]

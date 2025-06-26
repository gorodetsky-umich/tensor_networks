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
    # else:
    #     print(f"WARN: rank reduced from {prev_rank} to {real_rank}")

    q = q[:, :real_rank]
    return teneva._maxvol(
        q,
        tau=1.01,
        dr_min=dr_min,
        dr_max=dr_max,
        tau0=1.01,
        k0=100,
    )


@profile
def root_to_leaves(
    tensor_func: TensorFunc,
    parent_indices: Sequence[Index],
    node_indices: Sequence[Index],
    node: DimTreeNode,
) -> None:
    """Update the indices by propagating info from root to leaves."""
    free_indices = sorted(tensor_func.indices)
    # root will have everything, we need to separate them
    up_indices, up_ranges = [], []
    down_indices, down_ranges = [], []

    # for ind in node.info.indices:
    #     if ind in node_indices:
    #         up_ranges.append([[i] for i in range(ind.size)])
    #         up_indices.append(free_indices.index(ind))

    # for ind in node.info.indices:
    #     up_indices.append(free_indices.index(ind))

    # up_ranges.append(node.values.up_vals)

    # the indices in the DimTreeNode are up indices
    # when traversing from root to leaves, we need to consider the down indices of the root and the up indices of the siblings
    if node.conn.parent is not None:
        # we need adjust this order: parent free, not in parent, children
        # for ind in node.conn.parent.info.free_indices:
        #     down_indices.append(free_indices.index(ind))
        #     down_ranges.append([[i] for i in range(ind.size)])

        # for ind in free_indices:
        #     if ind not in node.conn.parent.info.indices:
        #         down_indices.append(free_indices.index(ind))

        # if node.conn.parent.conn.parent is not None:
        #     down_ranges.append(node.conn.parent.values.down_vals)

        # for c in node.conn.parent.conn.children:
        #     if c.info.node != node.info.node:
        #         for ind in c.info.indices:
        #             down_indices.append(free_indices.index(ind))

        #         down_ranges.append(c.values.up_vals)

        # up_down_indices = set(down_indices + up_indices)
        # assert up_down_indices == set(range(len(free_indices))), (
        #     f"down indices {down_indices} and up indices {up_indices} "
        #     f"do not cover all free indices {free_indices}"
        # )
        for ind in node.info.down_indices:
            if ind in node.conn.parent.info.free_indices:
                down_ranges.append([[i] for i in range(ind.size)])

        if node.conn.parent.conn.parent is not None:
            down_ranges.append(node.conn.parent.values.down_vals)

        for c in node.conn.parent.conn.children:
            if c.info.node != node.info.node:
                down_ranges.append(c.values.up_vals)

        down_vals = [sum(list(g), []) for g in product(*down_ranges)]
        # up_vals = [sum(list(g), []) for g in product(*up_ranges)]
        v = construct_matrix(
            tensor_func, (node.info.up_indices, node.values.up_vals), (node.info.down_indices, down_vals)
        )
        ind, _ = select_indices(v)
        node.values.down_vals = [down_vals[i] for i in ind]


@profile
def leaves_to_root(
    tensor_func: TensorFunc,
    node: DimTreeNode,
    net: "pt.TreeNetwork",
) -> None:
    """Update the down index values by sweeping from leaves to the root."""
    # free_indices = tensor_func.indices

    # down_indices = []
    # for ind in free_indices:
    #     if ind not in node.info.indices:
    #         down_indices.append(free_indices.index(ind))

    # up_indices = []
    up_ranges = []
    up_sizes = []

    for ind in node.info.up_indices:
        if ind in node.info.free_indices:
            up_sizes.append(ind.size)
            up_ranges.append(np.arange(ind.size)[:,None].tolist())
        
    for c in node.conn.children:
        up_sizes.append(len(c.values.up_vals))
        up_ranges.append(c.values.up_vals)
    # for ind in node.info.free_indices:
    #     up_indices.append(free_indices.index(ind))
    #     up_ranges.append([[i] for i in range(ind.size)])
    #     up_sizes.append(ind.size)

    # for c in node.conn.children:
    #     for ind in c.info.indices:
    #         up_indices.append(free_indices.index(ind))
    #     up_ranges.append(c.values.up_vals)
    #     up_sizes.append(len(c.values.up_vals))

    # up_down_indices = set(down_indices + up_indices)
    # assert up_down_indices == set(range(len(free_indices))), (
    #     f"down indices {down_indices} and up indices {up_indices} "
    #     f"do not cover all free indices {free_indices}"
    # )


    # FIXME: check the up val regime when there are free indices on the intermediate cores
    up_vals = [sum(list(g), []) for g in product(*up_ranges)]
    v = construct_matrix(
        tensor_func,
        (node.info.down_indices, node.values.down_vals),
        (node.info.up_indices, up_vals),
    )
    ind, b = select_indices(v)
    node.values.up_vals = [up_vals[i] for i in ind]
    # free indices, children indices, parent index
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
            up_vals.append(np.random.randint(0, ind.size-1, rank))
        c.values.up_vals = np.stack(up_vals, axis=-1).tolist()

        down_vals = []
        for ind in c.info.down_indices:
            down_vals.append(np.random.randint(0, ind.size-1, rank))
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
            n_indices = net.node_tensor(n.info.node).indices
            p_indices = net.node_tensor(n.conn.parent.info.node).indices
            root_to_leaves(f, p_indices, n_indices, n)

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
        else:
            # increase the ranks
            incr_ranks(tree, net)


if __name__ == "__main__":
    total_start = time.time()
    # # test root_to_leaves
    n = 25
    # # create the network structure
    # net = pt.TreeNetwork()
    # a = pt.Tensor(
    #     np.empty((0, 0)),
    #     [pt.Index(0, n), pt.Index("k", 2)],
    # )
    # net.add_node("A", a)
    # e = pt.Tensor(
    #     np.empty((0, 0)),
    #     [pt.Index(1, n), pt.Index("l", 2)],
    # )
    # net.add_node("E", e)
    # f = pt.Tensor(
    #     np.empty((0, 0, 0)),
    #     [pt.Index("k", 2), pt.Index("l", 2), pt.Index("m", 12)],
    # )
    # net.add_node("F", f)
    # net.add_edge("A", "F")
    # net.add_edge("E", "F")
    # b = pt.Tensor(
    #     np.empty((0, 0)),
    #     [pt.Index("j", 2), pt.Index(2, n)],
    # )
    # net.add_node("B", b)
    # c = pt.Tensor(
    #     np.empty((0, 0)),
    #     [pt.Index("i", 2), pt.Index(3, n)],
    # )
    # net.add_node("C", c)
    # d = pt.Tensor(
    #     np.empty((0, 0, 0)),
    #     [pt.Index("i", 2), pt.Index("j", 2), pt.Index("m", 12)],
    # )
    # net.add_node("D", d)
    # net.add_edge("B", "D")
    # net.add_edge("D", "C")
    # net.add_edge("D", "F")

    # print("=====init net=====")
    # print(net)
    # # create the target function
    f = FuncHilbert(
        [
            pt.Index(0, n, range(1, n + 1)),
            pt.Index(1, n, range(1, n + 1)),
            pt.Index(2, n, range(1, n + 1)),
            pt.Index(3, n, range(1, n + 1)),
        ]
    )
    # # create the DimTreeNode
    # tree = net.dimension_tree("A")
    # cross(f, net, "A", 0.1)
    # print("=====result net=====")
    # print(net)
    # indices = [Index(i, n, range(1, n + 1)) for i in range(4)]
    # net = pt.TreeNetwork()
    # net.add_node(
    #     "A", pt.Tensor(np.empty((0, 0, 0)), indices[:2] + [Index("i", 1)])
    # )
    # net.add_node(
    #     "B",
    #     pt.Tensor(
    #         np.empty((0, 0, 0)), [indices[-2], Index("i", 1), Index("j", 1)]
    #     ),
    # )
    # net.add_node(
    #     "C", pt.Tensor(np.empty((0, 0)), [indices[-1], Index("j", 1)])
    # )
    # net.add_edge("A", "B")
    # net.add_edge("B", "C")
    # net.cross(f, 0.1)
    # print(net)
    p = 25
    indices = [Index(i, p*(i+1),range(1,p*(i+1)+1)) for i in range(4)]
    tensor_func = FuncHilbert(indices)

    net = pt.TreeNetwork()
    net.add_node("A", pt.Tensor(np.empty((0, 0, 0)), [Index("i", 1), Index("j", 1), indices[2]]))
    net.add_node("B", pt.Tensor(np.empty((0, 0)), [indices[0], Index("k", 1)]))
    net.add_node("C", pt.Tensor(np.empty((0, 0)), [indices[1], Index("j", 1)]))
    net.add_node("D", pt.Tensor(np.empty((0, 0, 0)), [Index("i", 1), Index("k", 1), Index("l", 1)]))
    net.add_node("E", pt.Tensor(np.empty((0, 0)), [Index("l", 1), indices[3]]))
    net.add_edge("A", "C")
    net.add_edge("A", "D")
    net.add_edge("D", "B")
    net.add_edge("D", "E")
    print(net)
    net.cross(tensor_func, 0.1)
    print("end:", time.time() - total_start)

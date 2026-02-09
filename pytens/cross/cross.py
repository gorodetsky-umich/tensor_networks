"""Cross Approximation."""

from enum import Enum, auto
from typing import Optional, Sequence, Tuple
import logging
import copy

import numpy as np
from line_profiler import profile
import pydantic
from tntorch.maxvol import py_maxvol

import pytens.algs as pt
from pytens.cross.funcs import TensorFunc
from pytens.types import DimTreeNode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CrossAlgo(Enum):
    """Enumeration of cross algorithms."""

    MAXVOL = auto()
    DEIM = auto()


class ConvergenceCheck(Enum):
    """Enumeration of convergence check methods."""

    # check norm changes between iterations as the criteria for convergence
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
        description="Configure the rank increment between iterations",
    )
    max_rank: Optional[int] = pydantic.Field(
        default=None,
        description="Configure the maximum rank that is used in cross",
    )
    max_iters: Optional[int] = pydantic.Field(
        default=None,
        description="Limit the maximum number of sweeps over the entire tree",
    )
    validation_size: int = pydantic.Field(
        default=1000,
        description="Configure the number of validation points",
    )
    convergence: ConvergenceCheck = pydantic.Field(
        default=ConvergenceCheck.NORM,
        description="Configure how to check the algorithm convergence",
    )


class CrossResult(pydantic.BaseModel):
    """Class to record cross approximation results."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # The resulting tensor network
    net: "pt.TensorNetwork"
    # The used rows and columns are stored in the dimension tree
    dim_tree: DimTreeNode
    # The ranks and corresponding errors info during the cross algo
    ranks_and_errors: Sequence[Tuple[int, float]]


@profile
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
    q = v
    q, _ = np.linalg.qr(q)
    return py_maxvol(q)


def _deim(u: np.ndarray):
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


def _select_indices_deim(v: np.ndarray):
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


def _cartesian_product_arrays(*arrays):
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


class CrossApproximation:
    """The engine for cross approximation run"""

    def __init__(
        self, tensor_func: TensorFunc, config: CrossConfig = CrossConfig()
    ):
        self._config = config
        self._tensor_func = tensor_func

    @profile
    def _construct_matrix(self, rows, cols) -> np.ndarray:
        """
        Constructs a matrix from the tensor function by
        evaluating it on the provided row and column indices.
        """
        row_idx, row_vals = rows
        col_idx, col_vals = cols
        args = _cartesian_product_arrays(col_vals, row_vals).astype(
            int, copy=False
        )
        indices = col_idx + row_idx
        perm = [indices.index(ind) for ind in self._tensor_func.indices]
        args = args[:, perm]
        # print("constructing", len(args))
        return self._tensor_func(args).reshape(len(col_vals), len(row_vals))

    def _select_indices(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._config.cross_algo == CrossAlgo.MAXVOL:
            ind, b = _select_indices_maxvol(v)
        elif self._config.cross_algo == CrossAlgo.DEIM:
            b, ind = _select_indices_deim(v)
        else:
            raise ValueError(f"unsupported algo {self._config.cross_algo}")

        return ind, b

    @profile
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
            # print(
            #     (node.up_info.indices, node.up_info.vals),
            #     (node.down_info.indices, down_vals),
            # )
            v = self._construct_matrix(
                (node.up_info.indices, node.up_info.vals),
                (node.down_info.indices, down_vals),
            )

            ind, _ = self._select_indices(v)
            # print(ind)
            node.down_info.vals = down_vals[ind, :]
            node.down_info.rank = len(ind)

    @profile
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
        ind, b = self._select_indices(v)
        # print(ind)
        node.up_info.vals = up_vals[ind, :]
        node.up_info.rank = len(ind)
        # print("====>", node.values.up_vals)
        net.node_tensor(node.node).update_val_size(
            b.reshape(*up_sizes, -1).transpose(np.argsort(node.perm))
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
            up_vals = [
                np.random.randint(0, ind.size, [self._config.kickrank, 1])
                for ind in tree.indices
            ]
            up_vals = np.concatenate(up_vals, axis=-1)
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

    def _create_validation_set(self):
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
    ):
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
    ):
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
        return root_matrix.T.reshape(*f_sizes, *c_sizes).transpose(
            np.argsort(tree.perm)
        )

    @profile
    def cross(  # pylint: disable=R0913,R0917
        self,
        net: "pt.TensorNetwork",
        root: Optional["pt.NodeName"] = None,
        validation: Optional[np.ndarray] = None,
        eps: float = 0.1,
        initialization: Optional[np.ndarray] = None,
        known: Optional[np.ndarray] = None,
    ) -> CrossResult:
        """Cross approximation for the given network structure."""
        if root is None:
            root = list(net.network.nodes)[0]

        assert root is not None
        tree = net.dimension_tree(root)
        if initialization is None:
            tree.increment_ranks(1, self._config.max_rank)
            up_vals = [np.random.randint(0, ind.size) for ind in tree.indices]
            tree.add_values(np.asarray([up_vals]))
        else:
            tree.increment_ranks(len(initialization), self._config.max_rank)
            tree.add_values(initialization)

        converged = False

        if self._config.convergence == ConvergenceCheck.VALID_ERROR:
            if validation is None:
                validation = self._create_validation_set()

            real = self._tensor_func(validation)

        f_sizes = [ind.size for ind in tree.free_indices]
        f_vals = _cartesian_product_arrays(
            *[np.arange(sz)[:, None] for sz in f_sizes]
        )

        tree_nodes = tree.preorder()
        ranks_and_errs = {}
        trial = 0
        while not converged:
            old_net = copy.deepcopy(net)
            self._iterate_tree_nodes(net, tree_nodes)

            # get the value for the root node
            root_val = self._get_root_value(tree, f_sizes, f_vals)
            net.node_tensor(tree.node).update_val_size(root_val)

            if self._config.convergence == ConvergenceCheck.NORM:
                diff_net = net - old_net
                err = diff_net.norm() / net.norm()

            elif self._config.convergence == ConvergenceCheck.VALID_ERROR:
                assert validation is not None
                estimate = net.evaluate(
                    self._tensor_func.indices, validation
                ).reshape(-1)

                err = np.linalg.norm(real - estimate) / np.linalg.norm(real)

            else:
                raise RuntimeError("unknown termination criteria")

            ranks_and_errs[len(tree.up_info.vals)] = float(err)
            logger.debug("step: %s, error: %s", trial, err)
            if err <= eps or (
                self._config.max_iters is not None
                and trial >= self._config.max_iters
            ):
                break

            trial += 1
            self._incr_ranks(tree, known=known)

        # print(net)
        ranks_and_errs = list(sorted(list(ranks_and_errs.items())))
        # print(ranks_and_errs)
        return CrossResult(
            net=net, dim_tree=tree, ranks_and_errors=ranks_and_errs
        )

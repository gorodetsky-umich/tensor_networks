"""Tensor Train"""

import copy
import logging
import typing
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Self,
    Tuple,
    Union,
    cast,
)

import networkx as nx
import numpy as np
from sklearn.utils.extmath import randomized_svd  # type: ignore

from pytens.algs import Tensor, TensorNetwork, TreeNetwork, _svals_at_node
from pytens.ht import HierarchicalTucker
from pytens.types import Index, NodeName, SVDConfig
from pytens.utils import delta_svd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TensorTrain(TreeNetwork):
    """Class for tensor trains"""

    @property
    def backbone_nodes(self) -> List[NodeName]:
        """Return all backbone nodes in the tensor train."""
        return list(self.network.nodes)

    @staticmethod
    def rand_tt(
        indices: Sequence[Index], ranks: Optional[Sequence[int]] = None
    ) -> "TensorTrain":
        """Create a tensor train with the given indices."""
        net = TensorTrain()
        if ranks is None:
            ranks = [1] * (len(indices) - 1)
        for i, ind in enumerate(indices):
            if i == 0:
                core_indices = [ind, Index(f"s{i + 1}", ranks[i])]
            elif i == len(indices) - 1:
                core_indices = [Index(f"s{i}", ranks[i - 1]), ind]
            else:
                core_indices = [
                    Index(f"s{i}", ranks[i - 1]),
                    ind,
                    Index(f"s{i + 1}", ranks[i]),
                ]
            core_size = [ind.size for ind in core_indices]
            core_tensor = Tensor(np.random.random(core_size), core_indices)
            net.add_node(f"G{i}", core_tensor)

            if i != 0:
                net.add_edge(f"G{i}", f"G{i - 1}")

        return net

    def are_adjacent(self, indices: Sequence[Index]) -> bool:
        """Check whether the specified indices are next to each other"""
        end = self.end_nodes()[0]
        ind_nodes = [self.node_by_free_index(ind.name) for ind in indices]
        ind_nodes = list(set(ind_nodes))
        ind_dists = sorted([self.distance(end, node) for node in ind_nodes])
        # if there are adjacent, the distance should be consecutive
        dist_diffs = [x - y for x, y in zip(ind_dists[1:], ind_dists[:-1])]
        return all(d == 1 for d in dist_diffs)

    def linear_nodes(self, start: Optional[NodeName] = None) -> List[NodeName]:
        """Return nodes in linear order starting from start."""
        if start is None:
            start = self.end_nodes()[0]
        visited: List[NodeName] = [start]
        current = start
        while True:
            neighbors = [
                n for n in self.network.neighbors(current) if n not in visited
            ]
            if not neighbors:
                break
            current = neighbors[0]
            visited.append(current)
        return visited

    def __add__(self, other: Any) -> Self:
        """Add two tensor trains.

        New tensor has same names as self
        """
        if isinstance(other, TensorTrain):
            pass
        elif isinstance(other, TreeNetwork):
            return super().__add__(other)
        else:
            raise NotImplementedError

        assert nx.is_isomorphic(self.network, other.network)

        new_tens = copy.deepcopy(self)
        free_indices = self.free_indices()
        for _, (node1, node2) in enumerate(
            zip(self.network.nodes, other.network.nodes)
        ):
            logger.debug("Adding: Node %r and Node %r", node1, node2)

            tens1 = self.node_tensor(node1)
            tens2 = other.node_tensor(node2)
            new_tens.set_node_tensor(
                node1, tens1.concat_fill(tens2, free_indices)
            )

        return new_tens

    def __mul__(self, other: Self) -> Self:
        """Multiply two tensor trains.

        New tensor has same names as self
        """
        if isinstance(other, TensorTrain):
            pass
        elif isinstance(other, TreeNetwork):
            return super().__add__(other)
        else:
            raise NotImplementedError

        assert nx.is_isomorphic(self.network, other.network)

        new_tens = copy.deepcopy(self)
        free_indices = self.free_indices()
        for _, (node1, node2) in enumerate(
            zip(self.network.nodes, other.network.nodes)
        ):
            logger.debug("Multiplying: Node %r and Node %r", node1, node2)

            tens1 = self.node_tensor(node1)
            tens2 = other.node_tensor(node2)
            new_tens.set_node_tensor(node1, tens1.mult(tens2, free_indices))

        return new_tens

    def svals_by_merge(
        self,
        indices: Sequence[Index],
        max_rank: int = 100,
        rand: bool = True,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Compute the singular values for a tensor train."""
        ind_nodes = [self.node_by_free_index(ind.name) for ind in indices]
        nodes = self.swap(ind_nodes)
        if len(nodes) > 1:
            path = nx.shortest_path(self.network, nodes[0], nodes[1])
            # contract the network along the path
            n = self.merge_along_path(path)
        else:
            n = self.node_by_free_index(indices[0].name)

        self.orthonormalize(n)

        return _svals_at_node(
            self.node_tensor(n), indices, max_rank, rand, random_seed
        )

    def ttize(self) -> None:
        """Split the indices into multiple tensor train carriages."""
        # going from one end to the other end
        ends = self.end_nodes()
        path = nx.shortest_path(self.network, ends[0], ends[1])

        free_indices = self.free_indices()
        prev_node = None
        for n in path:
            n_indices = self.node_tensor(n).indices
            n_free = set(n_indices) & set(free_indices)
            if len(n_free) <= 1:
                prev_node = n
                continue

            curr_node = n
            for ind in list(n_free)[:-1]:
                curr_indices = self.node_tensor(curr_node).indices
                if prev_node is None:
                    left_indices = [ind]
                else:
                    path_ind = self.get_contraction_index(curr_node, prev_node)
                    left_indices = path_ind + [ind]
                lefts = [curr_indices.index(i) for i in left_indices]
                prev_node, curr_node = self.qr(curr_node, lefts)

            prev_node = curr_node

    @staticmethod
    def tt_svd(
        data: np.ndarray, indices: Sequence[Index], eps: float = 0.1
    ) -> "TensorTrain":
        """Decompose a data tensor into a tensor train via SVD truncation."""
        tt = TensorTrain()
        norm = np.linalg.norm(data)
        tt.add_node("G0", Tensor(data, list(indices)))
        left_node: Optional[NodeName] = None
        right_node: NodeName = "G0"

        for ind in indices[:-1]:
            left_inds = tt.node_tensor(right_node).indices
            lefts = [left_inds.index(ind)]
            if left_node is not None:
                lefts.append(
                    left_inds.index(
                        tt.get_contraction_index(left_node, right_node)[0]
                    )
                )
            [left_node, s, right_node], _ = tt.svd(
                right_node,
                lefts,
                SVDConfig(delta=norm * eps / ((len(indices) - 1) ** 0.5)),
            )
            tt.merge(right_node, s)

        return tt

    def svals_nbr(
        self,
        node1: NodeName,
        node2: NodeName,
        max_rank: int = 100,
        orthonormal: bool = False,
    ) -> np.ndarray:
        """Compute the singular values for two neighbor nodes."""
        random_seed = 42
        if not orthonormal:
            self.orthonormalize(node1)
        tensor = self.node_tensor(node1)
        left_ind = self.get_contraction_index(node1, node2)[0]
        left = tensor.indices.index(left_ind)
        perm = [left] + [i for i in range(len(tensor.indices)) if i != left]
        tensor_val = tensor.value.transpose(perm).reshape(left_ind.size, -1)
        _, s, _ = randomized_svd(
            tensor_val, max_rank, random_state=random_seed
        )
        return cast(np.ndarray, s)

    def flatten(self) -> "TensorTrain":
        """Return self (tensor train is already flat)."""
        return self

    def ends(self) -> List[NodeName]:
        """Compute the end nodes for the current tensor train."""
        res = []
        for n in self.network.nodes:
            if len(self.node_tensor(n).indices) == 2:
                res.append(n)

        return res

    @staticmethod
    def is_valid_tt(net: TensorNetwork) -> bool:
        """Check whether a given tensor train is valid"""
        # each node must have exactly one free index
        if len(net.free_indices()) != len(net.network.nodes):
            return False

        # network must be a path graph: all nodes have at most 2 neighbors
        # and exactly 2 end nodes (degree 1), except for single-node networks
        degrees = [
            len(list(net.network.neighbors(node)))
            for node in net.network.nodes
        ]
        if any(d > 2 for d in degrees):
            return False
        end_count = sum(1 for d in degrees if d == 1)
        if len(degrees) > 1 and end_count != 2:
            return False

        return True

    def to_ht(self) -> HierarchicalTucker:
        """Convert a tensor train to a hierarchical tucker"""
        ht = HierarchicalTucker()
        ht.network = copy.deepcopy(self.network)

        if len(self.network.nodes) == 1:
            return ht

        ends = self.end_nodes()
        assert len(ends) == 2
        path = nx.shortest_path(self.network, ends[0], ends[1])

        def _merge_pair(
            n1: NodeName, n2: NodeName, connecting_inds: List[Index]
        ) -> NodeName:
            contract_edges = ht.get_contraction_index(n1, n2)
            rs = []
            for n in (n1, n2):
                indices = ht.node_tensor(n).indices
                split_pos = [
                    indices.index(ind)
                    for ind in indices
                    if ind not in connecting_inds and ind not in contract_edges
                ]
                _, r = ht.qr(n, split_pos)
                rs.append(r)
            return ht.merge(rs[0], rs[1])

        def _to_ht(
            nodes: Sequence[NodeName], connecting_inds: List[Index]
        ) -> NodeName:
            if len(nodes) == 1:
                return nodes[0]
            if len(nodes) == 2:
                return _merge_pair(nodes[0], nodes[1], connecting_inds)

            mid = len(nodes) // 2
            contract_ind = ht.get_contraction_index(
                nodes[mid - 1], nodes[mid]
            )[0]
            connecting_inds.append(contract_ind)
            r1 = _to_ht(nodes[:mid], connecting_inds)
            r2 = _to_ht(nodes[mid:], connecting_inds)
            return _merge_pair(r1, r2, connecting_inds)

        _to_ht(path, [])
        return ht


def rand_tt(indices: List[Index], ranks: List[int]) -> TensorTrain:
    """Return a random tt."""

    dim = len(indices)
    assert len(ranks) + 1 == len(indices)

    tt = TensorTrain()

    r = [Index("r1", ranks[0])]
    tt.add_node(
        0,
        Tensor(np.random.randn(indices[0].size, ranks[0]), [indices[0], r[0]]),
    )

    core = 1
    for ii, index in enumerate(indices[1:-1]):
        r.append(Index(f"r{ii + 2}", ranks[ii + 1]))
        tt.add_node(
            core,
            Tensor(
                np.random.randn(ranks[ii], index.size, ranks[ii + 1]),
                [r[ii], index, r[ii + 1]],
            ),
        )
        core += 1
        tt.add_edge(ii, ii + 1)

    tt.add_node(
        dim - 1,
        Tensor(
            np.random.randn(ranks[-1], indices[-1].size), [r[-1], indices[-1]]
        ),
    )
    tt.add_edge(dim - 2, dim - 1)

    return tt


def tt_rank1(indices: List[Index], vals: List[np.ndarray]) -> TensorTrain:
    """Return a random rank 1 TT tensor."""

    dim = len(indices)

    tt = TensorTrain()

    r = [Index("r1", 1)]
    # print("vals[0] ", vals[0][:, np.newaxis])
    new_tens = Tensor(vals[0][:, np.newaxis], [indices[0], r[0]])
    tt.add_node(0, new_tens)
    # print("new_tens = ", new_tens.indices)

    core = 1
    for ii, index in enumerate(indices[1:-1]):
        r.append(Index(f"r{ii + 2}", 1))
        new_tens = Tensor(
            vals[ii + 1][np.newaxis, :, np.newaxis], [r[ii], index, r[ii + 1]]
        )
        tt.add_node(core, new_tens)
        tt.add_edge(core - 1, core)
        core += 1

    tt.add_node(dim - 1, Tensor(vals[-1][np.newaxis, :], [r[-1], indices[-1]]))
    tt.add_edge(dim - 2, dim - 1)
    # print("tt_rank1 = ", tt)
    return tt


def tt_separable(indices: List[Index], funcs: List[np.ndarray]) -> TensorTrain:
    """Rank 2 function formed by sums of functions of individual dimensions."""

    dim = len(indices)

    tt = TensorTrain()
    ranks = []
    for ii, index in enumerate(indices):
        ranks.append(Index(f"r_{ii + 1}", 2))
        if ii == 0:
            val = np.ones((index.size, 2))
            val[:, 0] = funcs[ii]

            tt.add_node(ii, Tensor(val, [index, ranks[-1]]))
        elif ii < dim - 1:
            val = np.zeros((2, index.size, 2))
            val[0, :, 0] = 1.0
            val[1, :, 0] = funcs[ii]
            val[1, :, 1] = 1.0
            tt.add_node(ii, Tensor(val, [ranks[-2], index, ranks[-1]]))
        else:
            val = np.ones((2, index.size))
            val[1, :] = funcs[ii]
            tt.add_node(ii, Tensor(val, [ranks[-2], index]))

        if ii > 0:
            tt.add_edge(ii - 1, ii)

    return tt


def tt_right_orth(tn: TensorTrain, node: int) -> TensorTrain:
    """Right orthogonalize all but first core.

    Tree tensor network as a TT and right orthogonalize

    A right orthogonal core has the r_{k-1} x nk r_k matrix
    R(Gk(ik)) = ( Gk(1) Gk(2) · · · Gk(nk) )
    having orthonormal rows so that

    sum G_k(i) G_k(i)^T  = I

    Modifies the input tensor network

    Assumes nodes have integer names
    """
    # pylint: disable=C0103
    # above disables the snake case complaints for variables like R
    val1 = tn.value(node)
    if val1.ndim == 3:
        # print("val1.shape = ", val1.shape)
        r, n, b = val1.shape
        # val1 = np.reshape(val1, (r, n * b), order="F")
        val1 = np.reshape(val1, (r, n * b))
        # print("val1.T.shape = ", val1.T.shape)
        q, R = np.linalg.qr(val1.T, mode="reduced")
        if q.shape[1] < r:
            newq = np.zeros((q.shape[0], r))
            newq[:, : q.shape[1]] = q
            q = newq
            newr = np.zeros((r, R.shape[1]))
            newr[: R.shape[0], :] = R
            R = newr

        # print("q.shape = ", q.shape)
        # print("r.shape = ", R.shape)
        # print("r = ", r)
        # print("q shape = ", q.shape)
        # new_val = np.reshape(q.T, (r, n, b), order="F")
        new_val = np.reshape(q.T, (r, n, b))
        tn.node_tensor(node).update_val_size(new_val)
    else:
        q, R = np.linalg.qr(val1.T)
        new_val = q.T
        tn.node_tensor(node).update_val_size(new_val)

    val2 = tn.value(node - 1)
    # new_val2 = np.einsum("...i,ij->...j", val2, R.T)
    new_val2 = np.dot(val2, R.T)
    tn.node_tensor(node - 1).update_val_size(new_val2)

    return tn


def eps_to_rank(s: np.ndarray, eps: float) -> int:
    """Tranlates the matrix approximation error \
    to rank in a truncated SVD"""
    tmp = (np.sqrt(np.cumsum(np.square(s[::-1])))[::-1]) <= eps
    res: int = int(np.argmax(tmp))
    if res == 0 and not tmp[0]:
        return int(s.shape[0])
    if res == 0 and tmp[0]:
        return 1
    return res


def gram_eig_and_svd(
    gl: np.ndarray, gr: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """ Implements eigenvalue decomposition + svd to \
        the gram matrices of a TT-core and returns the \
        low-rank factors """
    pos_tol = 1e-15

    eigl, vl = np.linalg.eigh(gl)
    eigr, vr = np.linalg.eigh(gr)
    eigl = np.abs(eigl)
    eigr = np.abs(eigr)

    eigl12 = np.sqrt(eigl)
    eigr12 = np.sqrt(eigr)

    threshold = np.ceil(np.log10(np.max(eigl12) * 1e-8 + pos_tol))
    eigl12 = np.round(eigl12, min(-int(threshold), 16))
    threshold = np.ceil(np.log10(np.max(eigr12) * 1e-8 + pos_tol))
    eigr12 = np.round(eigr12, min(-int(threshold), 16))

    maskl = eigl12 == 0
    maskr = eigr12 == 0

    eiglm12 = np.zeros_like(eigl12)
    eigrm12 = np.zeros_like(eigr12)
    eiglm12[~maskl] = 1 / eigl12[~maskl]
    eigrm12[~maskr] = 1 / eigr12[~maskr]

    # eiglm12 = np.nan_to_num(eiglm12, nan=0, posinf=0, neginf=0)
    # eigrm12 = np.nan_to_num(eigrm12, nan=0, posinf=0, neginf=0)

    tmp = (eigl12[:, np.newaxis] * vl.T) @ (vr * eigr12[np.newaxis, :])

    u, s, v = np.linalg.svd(tmp)
    rk = min(tmp.shape[0], tmp.shape[1], eps_to_rank(s, delta))

    u = u[:, :rk]
    s = s[:rk]
    v = v[:rk, :]

    curr_val = vl @ (eiglm12[:, np.newaxis] * u)

    next_val = (s[:, np.newaxis] * v * eigrm12[np.newaxis, :]) @ vr.T
    return curr_val, next_val


def tt_gramsvd_round(tn: TensorTrain, eps: float) -> TensorTrain:
    """
    Description: Modifies the input tensor network and returns the
    rounded version by implementing the Gram-SVD based rounding
    approach [1].

    [1] - H. Al Daas, G. Ballard and L. Manning, "Parallel Tensor-
    Train Rounding using Gram SVD," 2022 IEEE International Parallel
    and Distributed Processing Symposium (IPDPS), Lyon, France, 2022,
    pp. 930-940, doi: 10.1109/IPDPS53621.2022.00095."""

    def next_gram(
        gram_now: np.ndarray, core_next: np.ndarray, order: str = "lr"
    ) -> np.ndarray:
        """ Calculates the Gram matrix corresponding to the next \
        TT-core using the Gram matrix corresponding to the \
        current core. For example, if order == "lr" (left to \
        right sweep), the function outputs a matrix of size \
        R_{k} x R{k} from a matrix of size R_{k-1} x R_{k-1}"""
        snext = core_next.shape
        if order == "lr":
            tmp = (gram_now.T @ core_next.reshape((snext[0], -1))).reshape(
                (-1, snext[-1])
            )
            return np.asarray(tmp.T @ core_next.reshape((-1, snext[-1])))

        if order == "rl":
            tmp = (core_next.reshape((-1, snext[-1])) @ gram_now).reshape(
                (-1, snext[-2] * snext[-1])
            )
            return np.asarray(
                np.dot(tmp, core_next.reshape((-1, snext[-2] * snext[-1])).T)
            )

        raise ValueError(f"Invalid order: {order}. Use 'lr' or 'rl'.")

    dim = tn.dim()
    gr_list = [tn.value(dim - 1) @ tn.value(dim - 1).T]
    # print("pre backward sweep 1")
    # print("tn" , tn)
    # Collect gram matrices from right to left
    for i in range(dim - 2, -1, -1):
        # print("i = ", i)
        # print(tn.value(i))
        gr_list.append(next_gram(gr_list[-1], tn.value(i), "rl"))
    # print("end")

    norm = np.sqrt(gr_list[-1])[0, 0]
    delta = eps * norm / (dim - 1) ** 0.5
    gr_list = gr_list[::-1]

    # print("pre backward sweep 2")
    for i in range(dim - 1):
        sh = list(tn.value(i).shape)
        shp1 = list(tn.value(i + 1).shape)
        gl = tn.value(i).reshape((-1, sh[-1])).T @ tn.value(i).reshape(
            (-1, sh[-1])
        )

        curr_val, next_val = gram_eig_and_svd(gl, gr_list[i + 1], delta)

        curr_val = tn.value(i).reshape((-1, sh[-1])) @ curr_val
        next_val = next_val @ tn.value(i + 1).reshape((shp1[0], -1))

        rk = curr_val.shape[-1]
        sh[-1] = rk
        shp1[0] = rk
        curr_val = curr_val.reshape(sh)
        next_val = next_val.reshape(shp1)
        tn.node_tensor(i).update_val_size(curr_val)
        tn.node_tensor(i + 1).update_val_size(next_val)

    return tn


def tt_svd_round(tn: TensorTrain, eps: float) -> TensorTrain:
    """Round a tensor train.

    Nodes should be integers 0,1,2,...,dim-1

    orthogonalize determines of QR is used to orthogonalize the
    cores. If orthogonalize=False, the Gram-SVD rounding algo-
    rithm is used.
    """
    # pylint: disable=C0103
    # above disables the snake case complaints for variables like R
    # norm2 = tn.norm()
    dim = tn.dim()
    delta = None
    # delta = eps / np.sqrt(dim - 1) * norm2
    # cores = []
    # for node, data in tn.network.nodes(data=True):
    #     cores.append(node)

    # print("DIM = ", dim)
    out = tt_right_orth(tn, dim - 1)
    for jj in range(dim - 2, 0, -1):
        # print(f"orthogonalizing core {cores[jj]}")
        out = tt_right_orth(out, jj)

    # print("ON FORWARD SWEEP")
    core_list = list(out.network.nodes(data=True))
    node = core_list[0][0]
    data = core_list[0][1]
    value = out.value(node)
    trunc_svd = delta_svd(value, eps / np.sqrt(dim - 1), with_normalizing=True)
    delta = trunc_svd.delta
    assert delta is not None
    assert trunc_svd.v is not None
    assert trunc_svd.u is not None

    v = np.dot(np.diag(trunc_svd.s), trunc_svd.v)
    r2 = trunc_svd.u.shape[1]
    new_core = np.reshape(trunc_svd.u, (value.shape[0], r2))

    data["tensor"].update_val_size(new_core)

    # print("In here")
    val_old = out.node_tensor(node + 1).value
    next_val = np.einsum("ij,jk...->ik...", v, val_old)
    out.node_tensor(node + 1).update_val_size(next_val)

    for node, data in core_list[1:-1]:
        value = data["tensor"].value
        r1, n, r2a = value.shape
        val = np.reshape(value, (r1 * n, r2a))
        trunc_svd = delta_svd(val, delta)
        assert trunc_svd.v is not None
        assert trunc_svd.u is not None
        v = np.dot(np.diag(trunc_svd.s), trunc_svd.v)
        r2 = trunc_svd.u.shape[1]
        new_core = np.reshape(trunc_svd.u, (r1, n, r2))
        data["tensor"].update_val_size(new_core)

        val_old = out.node_tensor(node + 1).value
        next_val = np.einsum("ij,jk...->ik...", v, val_old)
        out.node_tensor(node + 1).update_val_size(next_val)

    return out


# Rounding sum of TT cores
def get_indices(
    maximum: int, periodicity: int, consecutive: int, start: int
) -> np.ndarray:
    """
    Gets the column indices of a matrix when right multiplied
    by the horizontal unfolding (or its transpose) of a TT-sum
    (H(X) or H(X).T). The indices correspond to the non-zero
    parts of H and helps to avoid unnecessary computation.
    """
    indices = np.asarray(
        np.concatenate(
            [
                np.arange(i, i + consecutive)
                for i in range(start, maximum, periodicity)
            ]
        )
    )
    return indices


def multiply_core_unfolding(  # pylint: disable=R0912
    mat: np.ndarray,
    cores_list: list,
    v_unfolding: bool,
    left_multiply: bool,
    transpose: bool,
) -> np.ndarray:
    """
    Multiplies a dense matrix 'mat' with a sparse block-diagonal
    core in a TT-sum.
    -   the summands are given as a Python list of tensor trains
        ('cores_list' in the arguments).
    -   'v_unfolding' indicates Vertical unfolding of a TT-core.
        i.e., a Rk-1 cross nk cross Rk core will be reshaped as
        Rk-1 * nk cross Rk matrix if v_unfolding is True. If
        False, the core will be reshaped as Rk-1 cross nk * Rk.
    -   'left_multiply' decides if 'cores_list' is left multiplied
        or right multiplied to 'mat'
    -   'transpose' denotes if we take a transpose of 'cores_list'
        before multiplication.
    """
    rows, cols = mat.shape
    n_cores = len(cores_list)
    if left_multiply:
        rk = [s.shape[-1] for s in cores_list]
        rk_cumsum = np.cumsum([0] + rk)
        rk_sum = np.sum(rk)
        if cores_list[0].ndim == 2:
            rk1 = [1 for s in cores_list]
        else:
            rk1 = [s.shape[0] for s in cores_list]
        rk1_sum = np.sum(rk1)
        rk1_cumsum = np.cumsum([0] + rk1)
        n = cores_list[0].shape[1]

        if v_unfolding and (not transpose):
            assert rows == rk_sum, f"Dimension mismatch {rows} != {rk_sum}"
            res = np.zeros((rk1_sum * n, cols))
            for i in range(n_cores):
                res[rk1_cumsum[i] * n : rk1_cumsum[i + 1] * n, :] = (
                    cores_list[i].reshape((-1, rk[i]))
                    @ mat[rk_cumsum[i] : rk_cumsum[i + 1], :]
                )
            return res

    else:
        rk = [s.shape[0] for s in cores_list]
        rk_cumsum = np.cumsum([0] + rk)
        rk_sum = np.sum(rk)
        if cores_list[0].ndim == 2:
            rk1 = [1 for s in cores_list]
        else:
            rk1 = [s.shape[-1] for s in cores_list]
        rk1_sum = np.sum(rk1)
        rk1_cumsum = np.cumsum([0] + rk1)
        n = cores_list[0].shape[1]

        if v_unfolding and (not transpose):
            assert cols == rk_sum * n, (
                f"Dimension mismatch {cols} != {rk_sum * n}"
            )
            res = np.zeros((rows, rk1_sum))
            for i in range(n_cores):
                res[:, rk1_cumsum[i] : rk1_cumsum[i + 1]] = mat[
                    :, rk_cumsum[i] * n : rk_cumsum[i + 1] * n
                ] @ cores_list[i].reshape((-1, rk1[i]))
            return res

        if (not v_unfolding) and (transpose):
            assert cols == rk1_sum * n, (
                f"Dimension mismatch {cols} != {rk1_sum * n}"
            )
            res = np.zeros((rows, rk_sum))
            for i in range(n_cores):
                ind = get_indices(cols, rk1_sum, rk1[i], rk1_cumsum[i])
                res[:, rk_cumsum[i] : rk_cumsum[i + 1]] = (
                    mat[:, ind] @ (cores_list[i].reshape((rk[i], -1))).T
                )
            return res

        if (not v_unfolding) and (not transpose):
            assert cols == rk_sum, f"Dimension mismatch {cols} != {rk_sum}"
            res = np.zeros((rows, n * rk1_sum))
            for i in range(n_cores):
                ind = get_indices(rk1_sum * n, rk1_sum, rk1[i], rk1_cumsum[i])
                res[:, ind] = mat[
                    :, rk_cumsum[i] : rk_cumsum[i + 1]
                ] @ cores_list[i].reshape((rk[i], -1))
            return res

    raise ValueError("Invalid options")


def next_gram_sum(
    gram_now: np.ndarray, core_next: list[np.ndarray], order: str = "rl"
) -> np.ndarray:
    """
    Let's say that we are dealing with 's' summands in our TT sum.

    gram_now is a sigma r_i times sigma r_i matrix (i from 1 to s)
    where r_i represents the sum of rank of a particular TT core of
    the summands. For example, it could be the sum for the last TT
    core of every summand.

    core_next is the list (of size s) of adjacent TT-core of all
    summands. For example, if gram_now corresponds to the last TT core
    of all the summands, then assuming order = rl, core_next will be a
    list of the penultimate cores of all the summands.

    order: 'lr' means left to right and 'rl' means right to left.
    """

    # shnext = [s.shape for s in core_next]
    if order == "rl":
        rk1_sum, _, rk_sum = np.sum([list(s.shape) for s in core_next], axis=0)
        n = core_next[0].shape[1]
        tmp = multiply_core_unfolding(gram_now, core_next, True, True, False)
        tmp = tmp.reshape((rk1_sum, n * rk_sum))
        return multiply_core_unfolding(tmp, core_next, False, False, True)

    if order == "lr":
        rk_sum, _, rk1_sum = np.sum([list(s.shape) for s in core_next], axis=0)
        n = core_next[0].shape[1]
        tmp = multiply_core_unfolding(gram_now, core_next, False, False, False)
        tmp = tmp.reshape((rk_sum * n, rk1_sum)).T
        return multiply_core_unfolding(tmp, core_next, True, False, False)

    raise ValueError(
        "Invalid argument for order. order should either be lr or rl"
    )


def tt_sum_gramsvd_round(
    factors_list: list[TensorTrain],
    eps: float = 1e-14,
) -> TensorTrain:
    """Gram-rounding of sum of tensor trains."""

    def core_info(k: int) -> tuple[list, list]:
        cores = [f.value(k) for f in factors_list]
        rk = [s.shape[0] for s in cores]
        rk1 = [s.shape[-1] for s in cores]
        n = cores[0].shape[1]
        if cores[0].ndim == 3:
            return cores, [np.sum(rk), n, np.sum(rk1)]
        return cores, [np.sum(rk), n]

    dim = factors_list[0].dim()

    ttsum = copy.deepcopy(factors_list[0])

    gr_list = [
        np.concatenate([f.value(dim - 1) for f in factors_list], axis=0)
    ]

    ttsum.node_tensor(dim - 1).update_val_size(gr_list[-1])
    gr_list = [gr_list[-1] @ gr_list[-1].T]

    gl = np.concatenate([f.value(0) for f in factors_list], axis=1)
    ttsum.node_tensor(0).update_val_size(gl)

    # Collect gram matrices from right to left
    for i in range(dim - 2, 0, -1):
        gr_list.append(
            next_gram_sum(
                gr_list[-1], [f.value(i) for f in factors_list], "rl"
            )
        )

    gr_list.append(np.sum((ttsum.value(0) @ gr_list[-1]) * ttsum.value(0)))
    norm = np.sqrt(gr_list[-1])
    delta = eps * norm / (dim - 1) ** 0.5

    gr_list = gr_list[::-1]

    for i in range(dim - 1):
        sh = list(ttsum.value(i).shape)
        core_next, shp1 = core_info(i + 1)

        gl = ttsum.value(i).reshape((-1, sh[-1])).T @ ttsum.value(i).reshape(
            (-1, sh[-1])
        )

        curr_val, next_val = gram_eig_and_svd(gl, gr_list[i + 1], delta)

        curr_val = ttsum.value(i).reshape((-1, sh[-1])) @ curr_val
        if i == (dim - 2):
            next_val = next_val @ ttsum.value(dim - 1)
        else:
            next_val = multiply_core_unfolding(
                next_val, core_next, False, False, False
            )

        rk = curr_val.shape[-1]
        sh[-1] = rk
        shp1[0] = rk

        curr_val = curr_val.reshape(sh)
        next_val = next_val.reshape(shp1)

        ttsum.node_tensor(i).update_val_size(curr_val)
        ttsum.node_tensor(i + 1).update_val_size(next_val)

    return ttsum


class TTRandRound:
    """
    Implementation of randomized rounding algorithms for Tensor Trains.

    Reference:
    [1] - Daas et. al, "Randomized algorithms for rounding in the Tensor-
    Train format." arxiv preprint arxiv:2110.04393 (2021). Available at:
    https://arxiv.org/abs/2110.04393.
    """

    def __init__(
        self, y: Union[TensorTrain, List[TensorTrain]], target_ranks: List
    ):
        self.y = y
        self.target_ranks = target_ranks

        if isinstance(y, List) and isinstance(y[0], TensorTrain):
            self.ns = len(y)
            self.d = y[0].network.number_of_nodes()

        elif isinstance(y, TensorTrain):
            self.ns = 1
            self.d = y.network.number_of_nodes()

        else:
            raise ValueError(
                f"Invalid type for y ({type(y)}). \
                             Argument y only accepts a list of \
                             TensorNetworks or a TensorNetwork"
            )

    def init_rand_mat(self, ranks: Optional[List] = None) -> List[np.ndarray]:
        """Generates a list of random TT-cores. Individual entries
        of the cores are Gaussian RVs and are normalized according
        to the size of the core"""
        if ranks is None:
            ranks = self.target_ranks

        sh = self.y[0].shape() if isinstance(self.y, list) else self.y.shape()
        r = []
        # Initialize random TT-tensor with specified variance
        for i in range(self.d):
            if i == 0:
                curr_shp = [sh[i], ranks[i]]
            elif i == self.d - 1:
                curr_shp = [ranks[i - 1], sh[i]]
            else:
                curr_shp = [ranks[i - 1], sh[i], ranks[i]]
            r.append(np.random.randn(*curr_shp) / np.sqrt(np.prod(curr_shp)))
        return r

    def partial_contraction(
        self, tt: TensorNetwork, y: List[np.ndarray], direction: str = "rl"
    ) -> List[np.ndarray]:
        """
        Partial contraction of TT cores. Returns a list of contracted cores
        w_i (by combining corresponding cores of, x[:i] and y[:i] for lr and
        x[i:] and y[i:] for rl contraction)
        """
        w = []
        if direction == "rl":
            for i in range(self.d - 1, 0, -1):
                x = tt.value(i)
                sx = x.shape
                sy = y[i].shape
                # tmp = np.einsum('ijk,ljm->ilkm', x[i], y[i])
                if i == self.d - 1:
                    w.append(x @ y[i].T)
                    continue
                tmp = (x.reshape((-1, sx[-1])) @ w[-1]).reshape((sx[0], -1))
                tmp = tmp @ y[i].reshape((sy[0], -1)).T
                w.append(tmp)

            w = w[::-1]
            return w

        raise ValueError("Invalid option")

    def rand_then_orth(self) -> TensorTrain:
        """Implements Algorithm 3.2 in reference [1]"""

        if isinstance(self.y, TensorNetwork):
            r = self.init_rand_mat()
            w = self.partial_contraction(self.y, r, "rl")
            x_approx: np.ndarray = self.y.value(0)
            res = copy.deepcopy(self.y)

            for i in range(self.d - 1):
                sx = list(x_approx.shape)
                zn = x_approx.reshape((-1, x_approx.shape[-1]))
                yn = zn @ w[i]
                q, _ = np.linalg.qr(yn)
                x_approx = q.reshape(sx[:-1] + [q.shape[-1]])
                res.node_tensor(i).update_val_size(x_approx)
                sy = list(self.y.value(i + 1).shape)
                x_approx = (
                    q.T @ zn @ self.y.value(i + 1).reshape((sy[0], -1))
                ).reshape([q.shape[-1]] + sy[1:])

            res.node_tensor(self.d - 1).update_val_size(x_approx)
            return res

        raise ValueError(
            "It seems that this function is \
                        being used to round a TT-sum"
        )

    def rto_rounding_ttsum(self) -> TensorTrain:
        """Implements Algorithm 3.4 in reference [1]"""

        if isinstance(self.y, List):
            r = self.init_rand_mat()
            tmp0 = []
            w = []
            res = copy.deepcopy(self.y[0])

            for y in self.y:
                tmp0.append(y.value(0))
                w.append(self.partial_contraction(y, r))
            x_approx = np.concatenate(tmp0, axis=1)

            del tmp0

            for i in range(self.d - 1):
                sx = list(x_approx.shape)
                rk = []
                rkp1 = []
                w_curr = []

                # Setup
                for j in range(self.ns):
                    sh = self.y[j].value(i).shape
                    rk.append(sh[-1])
                    rkp1.append(self.y[j].value(i + 1).shape[-1])
                    w_curr.append(w[j][i])

                # rksum = np.sum(rk)
                rkp1sum = np.sum(rkp1)
                rkcumsum = np.cumsum([0] + rk)

                # Start
                zn = x_approx.reshape((-1, sx[-1]))
                yn = zn @ np.concatenate(w_curr, axis=0)
                q, _ = np.linalg.qr(yn)
                self.target_ranks[i] = min(self.target_ranks[i], q.shape[-1])
                mn = q.T @ zn
                x_approx = q.reshape((sx[:-1] + [self.target_ranks[i]]))
                res.node_tensor(i).update_val_size(x_approx)
                xnp1 = []
                shp1 = []
                for j in range(self.ns):
                    shp1 = self.y[j].value(i + 1).shape
                    tmp = mn[:, rkcumsum[j] : rkcumsum[j + 1]] @ self.y[
                        j
                    ].value(i + 1).reshape((shp1[0], -1))
                    xnp1.append(tmp.reshape((-1, rkp1[j])))

                if i < self.d - 2:
                    x_approx = np.concatenate(xnp1, axis=1).reshape(
                        (self.target_ranks[i], shp1[1], rkp1sum)
                    )
                else:
                    x_approx = np.sum(xnp1, axis=0).reshape(
                        (self.target_ranks[i], shp1[1])
                    )
                    res.node_tensor(self.d - 1).update_val_size(x_approx)

            return res

        raise ValueError(
            "It seems that this function is being used \
                             to round a single TT"
        )

    def round(self) -> TensorNetwork:
        """Executes rounding"""
        if isinstance(self.y, List):
            res = self.rto_rounding_ttsum()
        else:
            res = self.rand_then_orth()
        return res


def tt_randomized_round(y: TensorTrain, target_ranks: List) -> TensorNetwork:
    """Executes randomized rounding for a TT TensorNetwork"""

    rand_setup = TTRandRound(y, target_ranks)
    return rand_setup.rand_then_orth()


def tt_sum_randomized_round(
    y: List[TensorTrain], target_ranks: List
) -> TensorTrain:
    """Executes randomized rounding for a TT TensorNetwork"""

    rand_setup = TTRandRound(y, target_ranks)
    return rand_setup.rto_rounding_ttsum()


def tt_rand_precond_svd_round(
    tn: Union[TensorTrain, List[TensorTrain]],
    eps: float,
    rank_bound: list[int],
) -> TensorNetwork:
    """
    Uses randomized rounding as a preconditioner to lower the ranks of
    the Tensor Train to the specified target rank before truncating the
    ranks further to a specified tolerance (eps) using SVD.

    Issues right now:
        - Total error accumulated post rounding is unknown due to initi-
        al rank-based truncation.
        - Need to adjust the eps in SVD-based truncation so that total
        error stays consistent with the global prespecified tolerance.
    """

    rand_rounded_tn = TTRandRound(y=tn, target_ranks=rank_bound)
    res = rand_rounded_tn.round()
    dim = rand_rounded_tn.d

    for i in range(dim - 1, 0, -1):
        tens_curr = res.value(i)
        sh = list(tens_curr.shape)
        tens_next = res.value(i - 1)

        delta = eps / (dim - 1) ** 0.5

        trunc_svd = delta_svd(tens_curr.reshape((sh[0], -1)), delta, True)
        assert trunc_svd.v is not None

        tens_curr = trunc_svd.v.reshape([-1] + sh[1:])
        if i == 1:
            tens_next = np.einsum(
                "jk,kl->jl",
                tens_next,
                trunc_svd.u * trunc_svd.s[np.newaxis, :],
            )
        else:
            tens_next = np.einsum(
                "ijk,kl->ijl",
                tens_next,
                trunc_svd.u * trunc_svd.s[np.newaxis, :],
            )

        res.node_tensor(i).update_val_size(tens_curr)
        res.node_tensor(i - 1).update_val_size(tens_next)

    return res


def ttop_rank1(
    indices_in: List[Index],
    indices_out: List[Index],
    cores: List[np.ndarray],
    rank_name_prefix: str,
) -> TensorNetwork:
    """Rank 1 TT-op with op in the first dimension."""
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    tt_op = TensorNetwork()

    rank_indices = [Index(f"{rank_name_prefix}_r1", 1)]
    a1_tens = Tensor(
        cores[0][:, :, np.newaxis],
        [indices_out[0], indices_in[0], rank_indices[0]],
    )
    tt_op.add_node(0, a1_tens)
    for ii in range(1, dim):
        rank_indices.append(Index(f"{rank_name_prefix}_r{ii + 1}", 1))
        if ii < dim - 1:
            eye = cores[ii][np.newaxis, :, :, np.newaxis]
            eye_tens = Tensor(
                eye,
                [
                    rank_indices[ii - 1],
                    indices_out[ii],
                    indices_in[ii],
                    rank_indices[ii],
                ],
            )
            tt_op.add_node(ii, eye_tens)
        else:
            eye = cores[ii][np.newaxis, :, :]
            eye_tens = Tensor(
                eye, [rank_indices[ii - 1], indices_out[ii], indices_in[ii]]
            )
            tt_op.add_node(ii, eye_tens)
        if ii == 1:
            tt_op.add_edge(ii - 1, ii)
        else:
            tt_op.add_edge(ii - 1, ii)

    return tt_op


def ttop_rank2(
    indices_in: List[Index],
    indices_out: List[Index],
    cores_r1: List[np.ndarray],
    cores_r2: List[np.ndarray],
    rank_name_prefix: str,
) -> TensorNetwork:
    """Rank 2 Sum of two ttops"""
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    tt_op = TensorNetwork()

    rank_indices = [Index(f"{rank_name_prefix}_r1", 2)]

    core = np.zeros((indices_out[0].size, indices_in[0].size, 2))
    core[:, :, 0] = cores_r1[0]
    core[:, :, 1] = cores_r2[0]

    a1_tens = Tensor(core, [indices_out[0], indices_in[0], rank_indices[0]])

    tt_op.add_node(0, a1_tens)
    for ii in range(1, dim):
        rank_indices.append(Index(f"{rank_name_prefix}_r{ii + 1}", 2))
        if ii < dim - 1:
            core = np.zeros((2, indices_out[ii].size, indices_in[ii].size, 2))
            core[0, :, :, 0] = cores_r1[ii]
            core[1, :, :, 1] = cores_r2[ii]

            ai_tens = Tensor(
                core,
                [
                    rank_indices[ii - 1],
                    indices_out[ii],
                    indices_in[ii],
                    rank_indices[ii],
                ],
            )
            tt_op.add_node(ii, ai_tens)
        else:
            core = np.zeros((2, indices_out[ii].size, indices_in[ii].size))
            core[0, :, :] = cores_r1[ii]
            core[1, :, :] = cores_r2[ii]
            ai_tens = Tensor(
                core, [rank_indices[ii - 1], indices_out[ii], indices_in[ii]]
            )
            tt_op.add_node(ii, ai_tens)
        tt_op.add_edge(ii - 1, ii)

    return tt_op


def ttop_sum(
    indices_in: List[Index],
    indices_out: List[Index],
    cores: List[List[np.ndarray]],
    rank_name_prefix: str,
) -> TensorNetwork:
    """Sum of ttops"""
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    tt_op = TensorNetwork()

    num_sum = len(cores)
    rank_indices = [Index(f"{rank_name_prefix}_r1", num_sum)]

    core = np.zeros((indices_out[0].size, indices_in[0].size, num_sum))
    for ii in range(num_sum):
        core[:, :, ii] = cores[ii][0]

    a1_tens = Tensor(core, [indices_out[0], indices_in[0], rank_indices[0]])

    tt_op.add_node(0, a1_tens)
    for ii in range(1, dim):
        rank_indices.append(Index(f"{rank_name_prefix}_r{ii + 1}", num_sum))
        if ii < dim - 1:
            core = np.zeros(
                (num_sum, indices_out[ii].size, indices_in[ii].size, num_sum)
            )
            for jj in range(num_sum):
                core[jj, :, :, jj] = cores[jj][ii]

            ai_tens = Tensor(
                core,
                [
                    rank_indices[ii - 1],
                    indices_out[ii],
                    indices_in[ii],
                    rank_indices[ii],
                ],
            )
            tt_op.add_node(ii, ai_tens)
        else:
            core = np.zeros(
                (num_sum, indices_out[ii].size, indices_in[ii].size)
            )
            for jj in range(num_sum):
                core[jj, :, :] = cores[jj][ii]

            ai_tens = Tensor(
                core, [rank_indices[ii - 1], indices_out[ii], indices_in[ii]]
            )
            tt_op.add_node(ii, ai_tens)
        tt_op.add_edge(ii - 1, ii)

    return tt_op


def tt_sum(
    tt_in: List[TensorTrain],
) -> TensorTrain:
    """Sum a set of tensor trains."""

    tt_out = TensorTrain()
    dim = tt_in[0].dim()
    for ii, node in enumerate(tt_in[0].network.nodes):
        inds = tt_in[0].node_tensor(node).indices
        core_values = [tt.value(node) for tt in tt_in]

        if ii == 0:
            new_value = np.hstack(core_values)
            index_left = Index(inds[0].name, inds[0].size)
            index_right = Index("rank_0", new_value.shape[1])
            new_inds = [index_left, index_right]

        elif ii == dim - 1:
            new_value = np.vstack(core_values)
            index_left = Index(f"rank_{ii - 1}", new_value.shape[0])
            index_right = Index(inds[1].name, inds[1].size)
            new_inds = [index_left, index_right]

        else:
            rank_left = np.sum([v.shape[0] for v in core_values])
            rank_right = np.sum([v.shape[2] for v in core_values])
            new_shape = (rank_left, core_values[0].shape[1], rank_right)
            new_value = np.zeros(new_shape)
            on_rank_left = 0
            on_rank_right = 0
            for core_value in core_values:
                increment_left = core_value.shape[0]
                increment_right = core_value.shape[2]
                new_value[
                    on_rank_left : on_rank_left + increment_left,
                    :,
                    on_rank_right : on_rank_right + increment_right,
                ] = core_value
                on_rank_left += increment_left
                on_rank_right += increment_right

            index_left = Index(f"rank_{ii - 1}", new_value.shape[0])
            index_middle = Index(inds[1].name, inds[1].size)
            index_right = Index(f"rank_{ii}", new_value.shape[2])
            new_inds = [index_left, index_middle, index_right]

        tt_out.add_node(ii, Tensor(new_value, new_inds))
        if ii > 0:
            tt_out.add_edge(ii - 1, ii)

    return tt_out


def ttop_sum_apply(
    tt_in: TensorNetwork,
    indices_in: List[Index],
    indices_out: List[Index],
    cores: List[List[Callable[[np.ndarray], np.ndarray]]],
    rank_name_prefix: str,
) -> TensorNetwork:
    """Apply sum of rank1 tt ops to a tt."""

    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    tt_out = TensorNetwork()
    num_sum = len(cores)

    node_list = list(tt_in.network.nodes())
    ii = 0
    v = tt_in.value(node_list[ii])
    rank_indices = [Index(f"{rank_name_prefix}_r1", num_sum * v.shape[1])]
    core = np.zeros((indices_out[ii].size, v.shape[1] * num_sum))
    indices = [indices_out[ii], rank_indices[ii]]
    on_ind = 0
    for jj in range(num_sum):
        new_core = cores[jj][ii](v)
        new_core = np.reshape(new_core, (core.shape[0], -1))
        core[:, on_ind : on_ind + new_core.shape[1]] = new_core
        on_ind += new_core.shape[1]
    tt_out.add_node(ii, Tensor(core, indices))

    for ii, node_tt in enumerate(node_list[1:], start=1):
        v = tt_in.value(node_tt)

        if ii < dim - 1:
            rank_indices.append(
                Index(f"{rank_name_prefix}_r{ii + 1}", v.shape[2] * num_sum)
            )

            core = np.zeros(
                (
                    num_sum * v.shape[0],
                    indices_out[ii].size,
                    num_sum * v.shape[2],
                )
            )

            indices = [rank_indices[ii - 1], indices_out[ii], rank_indices[ii]]
            on_ind1 = 0
            on_ind2 = 0
            for jj in range(num_sum):
                # new_core = np.einsum('jk,mkp->mjp', cores[jj][ii], v)
                new_core = cores[jj][ii](v)
                shape = new_core.shape
                new_core = np.reshape(new_core, (shape[0], shape[1], shape[2]))
                n1 = new_core.shape[0]
                n2 = new_core.shape[2]
                core[on_ind1 : on_ind1 + n1, :, on_ind2 : on_ind2 + n2] = (
                    new_core
                )
                on_ind1 += n1
                on_ind2 += n2
        else:
            core = np.zeros((num_sum * v.shape[0], indices_out[ii].size))
            indices = [rank_indices[ii - 1], indices_out[ii]]
            on_ind = 0
            for jj in range(num_sum):
                new_core = cores[jj][ii](v)
                core[on_ind : on_ind + new_core.shape[0], :] = new_core
                on_ind += new_core.shape[0]

        tt_out.add_node(ii, Tensor(core, indices))
        tt_out.add_edge(ii - 1, ii)

    return tt_out


def ttop_apply(ttop: TensorNetwork, tt_in: TensorNetwork) -> TensorNetwork:
    """Apply a ttop to a tt tensor.

    # tt overwritten, same free_indices as before
    """
    tt = copy.deepcopy(tt_in)
    dim = tt.dim()
    for ii, (node_op, node_tt) in enumerate(
        zip(ttop.network.nodes(), tt.network.nodes())
    ):
        op = ttop.value(node_op)
        v = tt.value(node_tt)
        # print(f"op shape: {node_op}", op.shape)
        # print(f"v shape: {node_tt}", v.shape)
        if ii == 0:
            new_core = np.einsum("ijk,jl->ilk", op, v)
            n = v.shape[0]
            new_core = np.reshape(new_core, (n, -1))
        elif ii < dim - 1:
            new_core = np.einsum("ijkl,mkp->mijpl", op, v)
            shape = new_core.shape
            new_core = np.reshape(
                new_core, (shape[0] * shape[1], shape[2], shape[3] * shape[4])
            )
        else:
            new_core = np.einsum("ijk,mk->mij", op, v)
            shape = new_core.shape
            new_core = np.reshape(new_core, (shape[0] * shape[1], -1))

        tt.set_node_tensor(
            node_tt, tt.node_tensor(node_tt).update_val_size(new_core)
        )

    # print("After op = ")
    # print(tt)
    return tt


@typing.no_type_check
def gmres(  # pylint: disable=R0913,R0917
    op,  # function from in to out
    rhs: TensorNetwork,
    x0: TensorNetwork,
    eps: float = 1e-5,
    round_eps: float = 1e-10,
    maxiter: int = 100,
) -> Tuple[TensorNetwork, float]:
    """Perform GMRES.
    VERY HACKY
    """
    # pylint: disable=C0103
    # above disables the snake case complaints for variables like R
    r0 = rhs + op(x0).scale(-1.0)
    r0 = tt_svd_round(r0, round_eps)
    beta = r0.norm()

    r0.scale(1.0 / beta)
    # print("r0 norm = ", r0.norm())

    v = [r0]

    # print("beta = ", beta)
    # print("v0 norm = ", v[0].norm())

    y = []
    H = None
    for jj in range(maxiter):
        # print(f"jj = {jj}")
        delta = round_eps

        w = op(v[-1])
        w = tt_svd_round(w, delta)

        if H is None:
            H = np.zeros((jj + 2, jj + 1))
        else:
            m, n = H.shape
            newH = np.zeros((m + 1, n + 1))
            newH[:m, :n] = H
            H = newH
        # print(f"H shape = {H.shape}")
        # print("inner w = ", w.inner(v[0]))
        # # print("w = ", w)
        # warr = w.contract().value.flatten()
        # varr = v[0].contract().value.flatten()
        # warr_next = warr - np.dot(warr, varr) * varr
        # print("check inner =", np.dot(warr, warr), w.inner(w))
        # print("H should be = ", np.dot(warr, varr), w.inner(v[0]))
        # # exit(1)
        # print("in arrays = ", np.dot(varr, varr), np.dot(warr_next, varr))
        for ii in range(jj + 1):
            # print("ii = ", ii)
            H[ii, jj] = w.inner(v[ii])
            vv = copy.deepcopy(v[ii])
            vv.scale(-H[ii, jj])
            w = w + vv
        # print("inner w = ", w.inner(v[0]), w.inner(w))
        # print("H = ", H)
        # exit(1)
        w = tt_svd_round(w, round_eps)
        H[jj + 1, jj] = w.norm()
        v.append(w.scale(1.0 / H[jj + 1, jj]))
        # for ii in range(jj+2):
        #     print(f"inner {-1, ii} = ", v[-1].inner(v[ii]))

        # exit(1)
        # + 1e-14

        # print(H)

        e = np.zeros((H.shape[0]))
        e[0] = beta
        yy, resid, _, _ = np.linalg.lstsq(H, e)
        y.append(yy)
        # print(f"Iteration {jj}: resid = {resid}")
        if np.abs(resid) < eps:
            break

        # if resid < eps:
        #     break
    # exit(1)
    x = copy.deepcopy(x0)
    # print("len y = ", len(y[-1]))
    # print("len v = ", len(v))
    for ii, (vv, yy) in enumerate(zip(v, y[-1])):
        x = x + vv.scale(yy)
    x = tt_svd_round(x, round_eps)
    r0 = rhs + op(x).scale(-1.0)
    resid = r0.norm()
    # print("resid = ", resid)
    # exit(1);
    return x, resid

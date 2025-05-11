"""Some utility functions."""

from typing import Optional, Callable, Sequence, Tuple, Set, List
from dataclasses import dataclass
import itertools

import numpy as np


@dataclass
class TruncSVD:
    """Store a truncated SVD."""

    u: np.ndarray
    s: np.ndarray
    v: np.ndarray
    remaining_delta: float
    delta: Optional[float] = None


def delta_svd(
    data: np.ndarray, delta: float, with_normalizing: bool = False
) -> TruncSVD:
    """
    Performs delta-truncated SVD similar to that of the `TTSVD`_ algorithm.

    Given an unfolding matrix of a tensor, its norm and the number
    of dimensions of the original tensor, this function computes the truncated
    SVD of `data`. The truncation error is determined using the dimension
    dependent _delta_ formula from `TTSVD`_ paper.

    Parameters
    ----------
    data: Matrix for which the truncated SVD will be performed.
    delta: threshold for singular values
    with_normalizing: if True, returns delta * norm of the tensor

    Returns
    -------
    u:obj:`numpy.array`
       Column-wise orthonormal matrix of left-singular vectors. _Truncate d_
    s:obj:`numpy.array`
        Array of singular values. _Truncated_
    v:obj:`numpy.ndarray`
        Row-wise orthonormal matrix of right-singular vectors. _Truncated_

    .. _TTSVD:
        https://epubs.siam.org/doi/epdf/10.1137/090752286

    Note:
    ----
    if run with with_normalize=False
    """

    # delta = (eps / ((dimensions - 1) ** (0.5))) * dataNorm

    m, n = data.shape
    if m > 10 * n:  # tall and skinny
        # print("Tall and skinny ")
        q, r = np.linalg.qr(data)
        u, s, v = np.linalg.svd(r)
        u = q @ u
    else:
        try:
            u, s, v = np.linalg.svd(data, False, True)
        except np.linalg.LinAlgError:
            # print("Numpy svd did not converge, using qr+svd")
            q, r = np.linalg.qr(data)
            u, s, v = np.linalg.svd(r)
            u = q @ u

    if with_normalizing:
        norm = np.sqrt(np.sum(s**2))
        delta = delta * norm

    slist = list(s * s)
    slist.reverse()
    truncpost = []
    for idx, element in enumerate(np.cumsum(slist)):
        if element <= delta**2:
            truncpost.append(idx)

        else:
            break

    truncation_rank = max(len(s) - len(truncpost), 1)
    used_delta = np.cumsum(slist)[truncpost[-1]] if len(truncpost) > 0 else 0.0
    if with_normalizing:
        return TruncSVD(
            u[:, :truncation_rank],
            s[:truncation_rank],
            v[:truncation_rank, :],
            float(np.sqrt(delta**2 - used_delta)),
            delta,
        )
    return TruncSVD(
        u[:, :truncation_rank],
        s[:truncation_rank],
        v[:truncation_rank, :],
        float(np.sqrt(delta**2 - used_delta)),
        None,
    )


class TensorFunc:
    def __init__(self, f: Callable, dims: Sequence[int]):
        self.f = f
        self.dims = dims

    def __call__(self, *args) -> np.ndarray:
        res = []
        for a in itertools.product(*args):
            # print(a)
            res.append(self.f(*a))

        return np.array(res)

    @property
    def shape(self):
        return self.dims


def check_convergence(
    tensor_func: TensorFunc,
    index_to_args: Callable,
    tensor_approx: np.ndarray,
    shape: Tuple[int, int],
    eps: float,
) -> Tuple[bool, Tuple[int, int]]:
    """Check convergence of the 2D cross approximation"""
    # randomly sample points from the function and evaluate the error
    points = []
    p, q = shape
    err_sq, nrm_sq = 0, 0
    ij = 0, 0
    max_diff = 0
    while len(points) < p + q:
        point = (np.random.randint(0, p), np.random.randint(0, q))
        if point in points:
            continue

        points.append(point)
        v = tensor_func(*index_to_args(point))
        err = v - tensor_approx[*point]
        err_sq += err**2
        nrm_sq += v**2

        if max_diff < np.abs(err):
            max_diff = np.abs(err)
            ij = point

    print(np.sqrt(err_sq), np.sqrt(nrm_sq))
    if np.sqrt(err_sq) <= eps * np.sqrt(nrm_sq):
        print("convergence check pass")
        return True, ij

    print("convergence check fail")
    return False, ij


def cross_approx(
    tensor_func: TensorFunc,
    index_to_args: Callable,
    shape: Tuple[int, int],
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, Set[int], Set[int]]:
    """
    Implementation of 2D cross approximation from [TODO: insert the paper link]
    """
    k = 0
    tensor_approx = np.zeros(shape)
    p, q = shape
    u = np.zeros((p, 1))
    v = np.zeros((q, 1))
    j = 0  # np.random.randint(q)
    rows, cols = set(), set()
    nrm = 0

    while True:
        uk = tensor_func(*index_to_args((None, j))) - tensor_approx[:, j]
        uk = uk.reshape(p, 1)
        print("uk", uk)
        i = np.argmax(np.abs(uk))
        if uk[i] == 0:
            print("zero, checking convergence")
            ok, ij = check_convergence(
                tensor_func, index_to_args, tensor_approx, shape, eps
            )
            if ok:
                return u, v.T, rows, cols

            i, j = ij
            continue

        print("choosing i =", i)
        rows.add(i)
        cols.add(j)

        vk = tensor_func(*index_to_args((i, None))) - tensor_approx[i, :]
        # print("vk", vk)
        # if vk[j] == 0:
        #     print("zero, checking convergence")
        #     ok, ij = check_convergence(
        #         tensor_func, index_to_args, tensor_approx, shape, eps
        #     )
        #     if ok:
        #         return u, v, rows, cols

        #     i, j = ij
        #     k += 1
        #     continue

        gamma = vk[j]
        vk = vk / gamma
        print("vk", vk)
        for x in np.flip(np.argsort(np.abs(vk))):
            print("choice", x)
            if x != j:
                j = x
                break

        print("choosing j =", j)
        vk = vk.reshape(q, 1)

        err_sq = np.linalg.norm(uk) ** 2 * np.linalg.norm(vk) ** 2
        err = np.sqrt((min(p, q) - k) * err_sq)
        print(u.shape, uk.shape, v.shape, vk.shape)
        nrm = np.sqrt(nrm**2 + 2 * (u.T @ uk).T @ (v.T @ vk) + err_sq)
        if k == 0:
            u = uk
            v = vk
        else:
            u = np.concat([u, uk], axis=1)
            v = np.concat([v, vk], axis=1)

        tensor_approx += uk @ vk.T
        print(gamma, err, nrm)
        if abs(gamma) <= eps:
            ok, ij = check_convergence(
                tensor_func, index_to_args, tensor_approx, shape, eps
            )
            if ok:
                return u, v.T, rows, cols

            i, j = ij

        k += 1


def flatten_lists(xss):
    """Flatten nested lists."""

    if isinstance(xss, list):
        return [x for xs in xss for x in flatten_lists(xs)]

    return xss

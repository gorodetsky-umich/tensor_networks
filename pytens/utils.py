"""Some utility functions."""

from typing import Optional
from dataclasses import dataclass
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

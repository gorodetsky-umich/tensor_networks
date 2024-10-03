"""Some utility functions."""

from typing import Tuple
import numpy as np


def delta_svd(
    data: np.ndarray, delta: float, with_normalizing=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs delta-truncated SVD similar to that of the `TTSVD`_ algorithm.

    Given an unfolding matrix of a tensor, its norm and the number of dimensions
    of the original tensor, this function computes the truncated SVD of `data`.
    The truncation error is determined using the dimension dependant _delta_ formula
    from `TTSVD`_ paper.

    Parameters
    ----------
    data: Matrix for which the truncated SVD will be performed.
    delta: threshold for singular values

    Returns
    -------
    u:obj:`numpy.ndarray`
        Column-wise orthonormal matrix of left-singular vectors. _Truncate d_
    s:obj:`numpy.array`
        Array of singular values. _Truncated_
    v:obj:`numpy.ndarray`
        Row-wise orthonormal matrix of right-singular vectors. _Truncated_

    .. _TTSVD:
        https://epubs.siam.org/doi/epdf/10.1137/090752286

    Todo
    ----
    - input checking
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
            print("Numpy svd did not converge, using qr+svd")
            q, r = np.linalg.qr(data)
            u, s, v = np.linalg.svd(r)
            u = q @ u

    if with_normalizing:
        norm = np.sqrt(np.sum(s**2))
        delta = delta * norm

    slist = list(s * s)
    slist.reverse()
    truncpost = [
        idx for idx, element in enumerate(np.cumsum(slist)) if element <= delta**2
    ]
    truncation_rank = max(len(s) - len(truncpost), 1)
    if with_normalizing:
        return (
            u[:, :truncation_rank],
            s[:truncation_rank],
            v[:truncation_rank, :],
            delta,
        )
    return u[:, :truncation_rank], s[:truncation_rank], v[:truncation_rank, :]

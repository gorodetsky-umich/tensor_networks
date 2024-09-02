import numpy as np

def deltaSVD(data, delta):
    """
    Performs delta-truncated SVD similar to that of the `TTSVD`_ algorithm.

    Given an unfolding matrix of a tensor, its norm and the number of dimensions
    of the original tensor, this function computes the truncated SVD of `data`.
    The truncation error is determined using the dimension dependant _delta_ formula
    from `TTSVD`_ paper.

    Parameters
    ----------
    data:obj:`numpy.array`
        Matrix for which the truncated SVD will be performed.
    dataNorm:obj:`float`
        Norm of the matrix. This parameter is used to determine the truncation bound.
    dimensions:obj:`int`
        Number of dimensions of the original tensor. This parameter is used to determine
        the truncation bound.
    eps:obj:`float`, optional
        Relative error upper bound for TT-decomposition.

    Returns
    -------
    u:obj:`numpy.ndarray`
        Column-wise orthonormal matrix of left-singular vectors. _Truncated_
    s:obj:`numpy.array`
        Array of singular values. _Truncated_
    v:obj:`numpy.ndarray`
        Row-wise orthonormal matrix of right-singular vectors. _Truncated_

    .. _TTSVD:
        https://epubs.siam.org/doi/epdf/10.1137/090752286
    """

    # TODO: input checking

    # delta = (eps / ((dimensions - 1) ** (0.5))) * dataNorm
    try:
        u, s, v = np.linalg.svd(data, False, True)
    except np.linalg.LinAlgError:
        print("Numpy svd did not converge, using qr+svd")
        q, r = np.linalg.qr(data)
        u, s, v = np.linalg.svd(r)
        u = q @ u
    slist = list(s * s)
    slist.reverse()
    truncpost = [
        idx for idx, element in enumerate(np.cumsum(slist)) if element <= delta**2
    ]
    truncationRank = max(len(s) - len(truncpost), 1)
    return u[:, :truncationRank], s[:truncationRank], v[:truncationRank, :]

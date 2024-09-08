import copy
import numpy as np
import pytens

def old_upwind(indices, h, cos_psi):
    """Only valid for 1D!"""

    def get_1d_stencil_plus(N: int):
        """1D stencil."""
        A = np.zeros((N, N))
        for ii in range(1, N-1):
            A[ii, ii] = 1
            A[ii, ii-1] = -1

        A /= h
        return A

    def get_1d_stencil_minus(N: int):
        """1D stencil."""
        A = np.zeros((N, N))
        for ii in range(1, N-1):
            A[ii, ii+1] = 1
            A[ii, ii] = -1
        A /= h
        return A


    stencil_plus = get_1d_stencil_plus(indices.space.size)
    stencil_minus = get_1d_stencil_minus(indices.space.size)


    eyev = np.eye(cos_psi.shape[0])
    eyev_plus = copy.deepcopy(eyev)
    eyev_plus[cos_psi < 0.0 + 1e-14, :] = 0.0
    eyev_minus = copy.deepcopy(eyev)
    eyev_minus[cos_psi > 0.0, :] = 0.0

    ind = [indices.space, indices.theta, indices.psi]
    ind_out = [pytens.Index(f'{i.name}p', i.size) for i in ind]
    ttop = pytens.ttop_rank2(
        ind,
        ind_out,
        [stencil_plus,
         np.eye(indices.theta.size),
         eyev_plus
         ],
        [stencil_minus,
         np.eye(indices.theta.size),
         eyev_minus
         ],
        "A")
    return ttop

import copy
import numpy as np
import pytens

def old_upwind_1d(indices, h, cos_psi):
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
    ttop = pytens.ttop_sum(
        ind,
        ind_out,
        [
            [
                stencil_plus,
                np.eye(indices.theta.size),
                eyev_plus
            ],
            [
                stencil_minus,
                np.eye(indices.theta.size),
                eyev_minus
            ],
        ],
        "A")
    return ttop


def old_upwind_2d(disc, indices):
    """Only valid for 2D!"""

    def get_1d_stencil_plus(N: int, h: float):
        """1D stencil."""
        A = np.zeros((N, N))
        for ii in range(1, N-1):
            A[ii, ii] = 1
            A[ii, ii-1] = -1

        A /= h
        return A

    def get_1d_stencil_minus(N: int, h: float):
        """1D stencil."""
        A = np.zeros((N, N))
        for ii in range(1, N-1):
            A[ii, ii+1] = 1
            A[ii, ii] = -1
        A /= h
        return A


    nx = disc.x.shape[0]
    eyex = np.eye(nx)
    ny = disc.y.shape[0]
    eyey = np.eye(ny)
    dx = disc.x[1] - disc.x[0]
    dy = disc.y[1] - disc.y[0]

    eye_psi = np.eye(disc.psi.shape[0])
    cos_psi = np.cos(disc.psi)
    sin_psi = np.sin(disc.psi)

    stencil_right = np.einsum('ij,kl->ikjl', get_1d_stencil_plus(nx, dx), eyey)
    stencil_right = np.reshape(stencil_right, (indices.space.size,
                                               indices.space.size))
    stencil_left = np.einsum('ij,kl->ikjl', get_1d_stencil_minus(nx, dx), eyey)
    stencil_left = np.reshape(stencil_left, (indices.space.size,
                                             indices.space.size))

    eye_right = copy.deepcopy(eye_psi)
    eye_right[cos_psi < 0.0 + 1e-15] = 0.0

    eye_left = copy.deepcopy(eye_psi)
    eye_left[cos_psi > 0.0] = 0.0

    stencil_up = np.einsum('ij,kl->ikjl', eyex, get_1d_stencil_plus(ny, dy))
    stencil_up = np.reshape(stencil_up, (indices.space.size,
                                            indices.space.size))
    stencil_down = np.einsum('ij,kl->ikjl', eyex, get_1d_stencil_minus(ny, dy))
    stencil_down = np.reshape(stencil_down, (indices.space.size,
                                             indices.space.size))

    eye_up = copy.deepcopy(eye_psi)
    eye_up[sin_psi < 0.0 + 1e-15] = 0.0

    eye_down = copy.deepcopy(eye_psi)
    eye_down[sin_psi > 0.0] = 0.0

    eye_theta = np.eye(indices.theta.size)

    ind = [indices.space, indices.theta, indices.psi]
    ind_out = [pytens.Index(f'{i.name}p', i.size) for i in ind]
    ttop_x = pytens.ttop_sum(
        ind,
        ind_out,
        [
            [
                stencil_right,
                eye_theta,
                eye_right
            ],
            [
                stencil_left,
                eye_theta,
                eye_left,
            ],
        ],
        "A",
    )
    ttop_y = pytens.ttop_sum(
        ind,
        ind_out,
        [
            [
                stencil_up,
                eye_theta,
                eye_up,
            ],
            [
                stencil_down,
                eye_theta,
                eye_down,
            ],
        ],
        "B"
    )
    return ttop_x, ttop_y

import copy
import numpy as np
import pytens

def old_upwind_1d(indices, h, cos_psi):
    """Only valid for 1D!"""

    def stencil_plus_op(v):
        o = np.zeros(v.shape)
        o[1:, :] = (v[1:, :] - v[:-1, :]) / h
        return o

    def stencil_minus_op(v):
        o = np.zeros(v.shape)
        o[1:-1, :] = (-v[1:-1, :] + v[2:, :]) / h
        return o    
    
    ind_plus = cos_psi < 0.0+1e-14
    def ev_plus(v):
        o = copy.deepcopy(v)
        o[:, ind_plus] = 0.0
        return o
    ind_minus = cos_psi > 0.0
    def ev_minus(v):
        o = copy.deepcopy(v)
        o[:, ind_minus] = 0.0
        return o    
    
    ind = [indices.space, indices.theta, indices.psi]
    ind_out = [pytens.Index(f'{i.name}p', i.size) for i in ind]

    def ttop(tt_in: pytens.TensorNetwork):
        return pytens.ttop_sum_apply(
            tt_in,
            ind,
            ind_out,
            [
                [
                    stencil_plus_op,
                    lambda v: v,
                    ev_plus,
                ],
                [
                    stencil_minus_op,
                    lambda v: v,
                    ev_minus
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

    cos_psi = np.cos(disc.psi)
    sin_psi = np.sin(disc.psi)

    stenc_right_x = get_1d_stencil_plus(nx, dx)
    def stencil_right(v):
        o = np.reshape(v, (nx, -1))
        o = np.dot(stenc_right_x, o)
        return o.reshape(v.shape)

    stenc_left_x = get_1d_stencil_minus(nx, dx)
    def stencil_left(v):
        o = np.reshape(v, (nx, -1))
        o = np.dot(stenc_left_x, o)
        return o.reshape(v.shape)    
    
    ind_plus = cos_psi < 0.0 + 1e-14
    def eye_right(v):
        o = copy.deepcopy(v)
        o[:, ind_plus] = 0.0
        return o
    ind_minus = cos_psi > 0.0
    def eye_left(v):
        o = copy.deepcopy(v)
        o[:, ind_minus] = 0.0
        return o

    stenc_up_y = get_1d_stencil_plus(ny, dy)
    def stencil_up(v):
        o = np.reshape(v, (nx, ny, -1))
        o = np.einsum('ij,kjm->kim', stenc_up_y, o)
        # o = np.dot(stenc_right_x, o)
        return o.reshape(v.shape)
    
    stenc_down_y = get_1d_stencil_minus(ny, dy)
    def stencil_down(v):
        o = np.reshape(v, (nx, ny, -1))
        o = np.einsum('ij,kjm->kim', stenc_down_y, o)
        return o.reshape(v.shape)

    ind_up = sin_psi < 0.0 + 1e-14
    def eye_up(v):
        o = copy.deepcopy(v)
        o[:, ind_up] = 0.0
        return o
    
    ind_down = sin_psi > 0.0
    def eye_down(v):
        o = copy.deepcopy(v)
        o[:, ind_down] = 0.0
        return o
    
    ind = [indices.space, indices.theta, indices.psi]
    ind_out = [pytens.Index(f'{i.name}p', i.size) for i in ind]

    def ttop_x(tt_in: pytens.TensorNetwork):
        return pytens.ttop_sum_apply(
            tt_in,
            ind,
            ind_out,
            [
                [
                    stencil_right,
                    lambda v: v,
                    eye_right,
                ],
                [
                    stencil_left,
                    lambda v: v,
                    eye_left,
                ],
            ],
            "A",
        )

    def ttop_y(tt_in: pytens.TensorNetwork):
        return pytens.ttop_sum_apply(
            tt_in,
            ind,
            ind_out,
            [
                [
                    stencil_up,
                    lambda v: v,
                    eye_up,
                ],
                [
                    stencil_down,
                    lambda v: v,
                    eye_down,
                ],
            ],
            "B"
        )
    return ttop_x, ttop_y

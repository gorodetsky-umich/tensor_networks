"""Solver utilities."""
from typing import Dict, Any, Optional, List

import copy
from dataclasses import dataclass
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

import numpy as np

from load_config import Config
import pytens

@dataclass
class Indices:
    """Indices involved in the problem"""
    dimension: int
    space: pytens.Index
    theta: pytens.Index
    psi: pytens.Index

    @classmethod
    def from_config(cls, config: Config) -> 'Indices':
        """Build indices from config."""

        n = np.prod(config.geometry.n)
        pos = pytens.Index('x', n)
        theta = pytens.Index('theta', config.angles.n_theta)
        psi = pytens.Index('psi', config.angles.n_psi)
        out = cls(config.geometry.dimension, pos,
                  theta, psi)
        return out

    def __repr__(self) -> str:
        out = (f"Indices({self.dimension!r},"
               f"{self.space!r}, {self.theta!r}, {self.psi})")
        return out


class Solution:

    time: float
    sol: Optional[pytens.TensorNetwork]
    aux: Dict[Any, Any]

    def __init__(self, time, sol) -> None:
        self.time = time
        self.sol = sol
        self.aux = {}

@dataclass
class Discretization:

    dimension: int
    theta: np.ndarray
    psi: np.ndarray
    x: np.ndarray
    y: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, config: Config) -> 'Discretization':

        x = np.linspace(
            config.geometry.lb[0],
            config.geometry.ub[0],
            config.geometry.n[0]
        )

        theta = np.linspace(0, np.pi, config.angles.n_theta)
        psi = np.linspace(0, 2*np.pi, config.angles.n_psi)
        if config.geometry.dimension == 2:
            y = np.linspace(
                config.geometry.lb[1],
                config.geometry.ub[1],
                config.geometry.n[1]
            )
            return cls(2, theta, psi, x, y)
        else:
            return cls(1, theta, psi, x, None)


    def get_min_h(self) -> float:

        if self.dimension == 1:
            return self.x[1] - self.x[0]
        elif self.dimension == 2:
            dx = self.x[1] - self.x[0]
            dy = self.y[1] - self.y[0]
            if dx < dy:
                return dx
            return dy


def upwind_stencil_plus(v, h):
    """Assumes moving left to right."""
    o = np.zeros(v.shape)
    o[1:, :] = (v[1:, :] - v[:-1, :]) / h
    return o

def upwind_stencil_minus(v, h):
    """Assumes moving left to right. """
    o = np.zeros(v.shape)
    o[1:-1, :] = (-v[1:-1, :] + v[2:, :]) / h
    return o

def id_mask(ind):
    def ret(v):
        o = copy.deepcopy(v)
        o[:, ind] = 0.0
        return o
    return ret

def upwind_transport_1d(indices: Indices, disc: Discretization):

    cos_psi = np.cos(disc.psi)

    ev_plus = id_mask(cos_psi < 0.0+1e-14)
    ev_minus = id_mask(cos_psi > 0.0)
    h = disc.get_min_h()

    ind = [indices.space, indices.theta, indices.psi]
    ind_out = [pytens.Index(f'{i.name}p', i.size) for i in ind]
    def ttop(tt_in: pytens.TensorNetwork):
        return pytens.ttop_sum_apply(
            tt_in,
            ind,
            ind_out,
            [
                [
                    lambda v: upwind_stencil_plus(v, h),
                    lambda v: v,
                    lambda v: ev_plus(v),
                ],
                [
                    lambda v: upwind_stencil_minus(v, h),
                    lambda v: v,
                    lambda v: ev_minus(v)
                ],
            ],
            "A")
    return ttop

def upwind_transport_2d(indices: Indices, disc: Discretization):

    cos_psi = np.cos(disc.psi)
    sin_psi = np.sin(disc.psi)

    eye_right = id_mask(cos_psi < 0.0)
    eye_left = id_mask(cos_psi > 0.0)
    eye_up = id_mask(sin_psi < 0.0)
    eye_down = id_mask(sin_psi > 0.0)

    nx = disc.x.shape[0]
    ny = disc.y.shape[0]
    dx = disc.x[1] - disc.x[0]
    dy = disc.y[1] - disc.y[0]
    def stencil_right(v):
        return upwind_stencil_plus(
            v.reshape((nx, -1)), dx
        ).reshape(v.shape)

    def stencil_left(v):
        return upwind_stencil_minus(
            v.reshape((nx, -1)), dx
        ).reshape(v.shape)

    def stencil_up(v):
        return upwind_stencil_plus(
            v.reshape((nx, ny, -1)).
            transpose(1, 0, 2), dy
        ).reshape(ny, nx,-1).transpose(1, 0, 2)

    def stencil_down(v):
        return upwind_stencil_minus(
            v.reshape((nx, ny, -1)).
            transpose(1, 0, 2), dy
        ).reshape(ny, nx,-1).transpose(1, 0, 2)

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


def get_transport_term(indices: Indices,
                       discretization: Discretization,
                       config: Config):

    ones = np.ones(indices.space.size)
    sin_theta = np.sin(discretization.theta)
    cos_psi = np.cos(discretization.psi)

    ind = [indices.space, indices.theta, indices.psi]
    omega_x = pytens.tt_rank1(ind, [ones, sin_theta, cos_psi])
    if config.solver.stencil == 'upwind':

        if discretization.dimension == 1:
            ttop = upwind_transport_1d(indices, discretization)
            def op(time, sol):
                op1 = ttop(sol)
                op2 = op1 * omega_x
                return op2
        else:

            sin_psi = np.sin(discretization.psi)
            omega_y = pytens.tt_rank1(ind, [ones, sin_theta, sin_psi])
            tt_op_x, tt_op_y = upwind_transport_2d(indices, discretization)
            def op(time, sol):
                op1_x = tt_op_x(sol)
                op2_x = op1_x * omega_x

                op1_y = tt_op_y(sol)
                op2_y = op1_y * omega_y
                return op2_x + op2_y
    else:
        raise NotImplementedError(
            'Stencil other than upwind is not yet implemented'
        )

    return op

"""Solver utilities."""
from typing import Dict, Any, Optional, List

import copy
from dataclasses import dataclass


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
            from old_stuff import old_upwind_1d
            h = discretization.get_min_h()
            ttop = old_upwind_1d(indices, h, cos_psi)
            def op(time, sol):
                # op1 = pytens.ttop_apply(ttop, copy.deepcopy(sol))
                op1 = ttop(sol)
                op2 = op1 * omega_x
                return op2
        else:
            from old_stuff import old_upwind_2d
            sin_psi = np.sin(discretization.psi)
            omega_y = pytens.tt_rank1(ind, [ones, sin_theta, sin_psi])
            tt_op_x, tt_op_y = old_upwind_2d(discretization, indices)
            def op(time, sol):
                # op1_x = pytens.ttop_apply(tt_op_x, copy.deepcopy(sol))
                op1_x = tt_op_x(sol)
                op2_x = op1_x * omega_x
                op1_y = tt_op_y(sol)
                # op1_y = pytens.ttop_apply(tt_op_y, copy.deepcopy(sol))
                op2_y = op1_y * omega_y
                return op2_x + op2_y
    else:
        raise NotImplementedError(
            'Stencil other than upwind is not yet implemented'
        )

    return op

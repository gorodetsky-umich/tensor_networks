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

import problems


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

def get_ic(indices: Indices,
           discretization: Discretization,
           config: Config):

    if config.problem == 'hohlraum':
        if indices.dimension == 1:
            x = np.zeros((config.geometry.n[0]))
            x[0] = 1
            theta = np.ones((config.angles.n_theta))
            psi = np.ones((config.angles.n_psi))
            ind = [indices.space, indices.theta, indices.psi]
            f = pytens.tt_rank1(ind, [x, theta, psi])
        else:
            xy = np.zeros((config.geometry.n[0], config.geometry.n[1]))
            xy[0, :] = 1.0
            xy[:, 0] = 1.0
            xy = xy.flatten()
            theta = np.ones((config.angles.n_theta))
            psi = np.ones((config.angles.n_psi))
            ind = [indices.space, indices.theta, indices.psi]
            f = pytens.tt_rank1(ind, [xy, theta, psi])
    else:
        raise NotImplementedError("2D hohlraum")

    return Solution(0.0, f)


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

def get_plots(disc: Discretization,
                      config: Config):

    if config.problem == 'hohlraum':
        if disc.dimension == 1:
            def plot_intensity(ax, sol):
                y = sol.aux['mean_intensity']
                ax.plot(disc.x[1:],
                        y[1:], '-', color='gray', alpha=0.2)
                ax.set_xlabel(r'Space $x$')
                ax.set_ylabel(r'Mean intensity $J$')
                return ax

            def plot_all_sols(sols: List[Solution]):
                fig, axs = plt.subplot_mosaic(
                    [
                        ['numerical', 'analytical', 'color'],
                        ['error', 'error', 'color2']
                    ],
                    width_ratios = [0.45, 0.45, 0.1],
                    layout='constrained')

                times = np.array([
                    s.time for s in sols if 'mean_intensity' in s.aux
                ])
                out = np.zeros((disc.x.shape[0], len(times)))
                on_index = 0
                for sol in sols:
                    if 'mean_intensity' in sol.aux:
                        out[:, on_index] = sol.aux['mean_intensity'] + 1e-16
                        on_index += 1

                tt, xx = np.meshgrid(times, disc.x[1:])
                levels = np.linspace(0, 1, 10)
                pos = axs['numerical'].contourf(tt, xx, out[1:, :], levels)
                axs['numerical'].set_xlabel('Time')
                axs['numerical'].set_ylabel('Space')
                axs['numerical'].set_title('Numerical')
                # fig.colorbar(pos, ax=ax)

                out_a = np.zeros((disc.x.shape[0], len(times)))
                on_index = 0
                for sol in sols:
                    if 'mean_intensity' in sol.aux:
                        on_time = sol.time
                        # assumes speed = 1
                        analytic = 0.5 * (1 - disc.x / (on_time + 1e-16))
                        analytic[disc.x > on_time] = 1e-16
                        out_a[:, on_index] = analytic
                        on_index += 1
                axs['analytical'].contourf(tt, xx, out_a[1:, :], levels)
                axs['analytical'].set_xlabel('Time')
                axs['analytical'].set_ylabel('Space')
                axs['analytical'].set_title('Analytical')
                fig.colorbar(pos, cax=axs['color'])

                err = np.abs(out - out_a) + 1e-16
                # err = np.log10(np.abs(out - out_a) + 1e-16)
                # err += 1e-16
                cmap = plt.colormaps["bone"]
                pos = axs['error'].contourf(tt, xx, err[1:, :], cmap=cmap)
                axs['error'].set_xlabel('Time')
                axs['error'].set_ylabel('Space')
                axs['error'].set_title('Error')
                fig.colorbar(pos, cax=axs['color2'])
                return fig, axs

            return None, plot_intensity, plot_all_sols
        else:
            def plot_intensity(sol):
                fig, axs = plt.subplot_mosaic(
                    [
                        ['numerical', 'analytical', 'color'],
                        ['error', 'error', 'color2']
                    ],
                    width_ratios = [0.45, 0.45, 0.1],
                    layout='constrained')

                y = sol.aux['mean_intensity']
                xx, yy = np.meshgrid(disc.x, disc.y)
                out = y.reshape((disc.x.shape[0], disc.y.shape[0]))
                levels = np.linspace(0, 1, 10)
                pos = axs['numerical'].contourf(xx[1:, 1:],
                                                yy[1:, 1:],
                                                out.T[1:, 1:],
                                                levels=levels)
                axs['numerical'].set_xlabel(r'$x$')
                axs['numerical'].set_ylabel(r'$y$')

                jleft = np.zeros((disc.y.shape[0], disc.x.shape[0])) + 1e-16
                jbottom = np.zeros((disc.y.shape[0], disc.x.shape[0])) + 1e-16

                ind_x = disc.x < sol.time
                xx_eta_x = xx[:, ind_x]
                yy_eta_x = yy[:, ind_x]
                term1 = yy_eta_x / np.sqrt(sol.time**2 - xx_eta_x**2)
                term1[term1 > 1] = 1
                eta_x = np.arccos(term1)
                jleft[:, ind_x] = 0.5 - \
                    (np.pi - eta_x) * xx_eta_x  /(2 * np.pi * sol.time) \
                    - 0.5 / np.pi * \
                    np.arcsin(
                        xx_eta_x * np.sin(eta_x) /
                        np.sqrt(xx_eta_x**2 + yy_eta_x**2 + 1e-16)
                    )

                ind_y = disc.y < sol.time
                xx_eta_y = xx[ind_y, :]
                yy_eta_y = yy[ind_y, :]
                term2 = xx_eta_y / np.sqrt(sol.time**2 - yy_eta_y**2)
                term2[term2 > 1] = 1
                eta_y = np.arccos(term2)
                jbottom[ind_y, :] = 0.5 - \
                    (np.pi - eta_y) * yy_eta_y  /(2 * np.pi * sol.time) \
                    - 0.5 / np.pi * \
                    np.arcsin(
                        yy_eta_y * np.sin(eta_y) /
                        np.sqrt(xx_eta_y**2 + yy_eta_y**2 + 1e-16)
                    )

                j = jleft[1:, 1:] + jbottom[1:, 1:]

                axs['analytical'].contourf(xx[1:, 1:],
                                           yy[1:, 1:],
                                           j,
                                           levels=levels)

                fig.colorbar(pos, cax=axs['color'])

                err = np.abs(out.T[1:, 1:] - j) + 1e-16
                cmap = plt.colormaps["bone"]
                pos = axs['error'].contourf(xx[1:, 1:],
                                            yy[1:, 1:],
                                            err, cmap=cmap)
                axs['error'].set_xlabel('Time')
                axs['error'].set_ylabel('Space')
                axs['error'].set_title('Error')
                fig.colorbar(pos, cax=axs['color2'])
                return fig, axs

            return plot_intensity, None, None
    else:
        raise NotImplementedError('Cannot plot for other problems')
    pass

def plot_ranks_compressions(disc: Discretization, sols: List[Solution]):
    """Plot the ranks assuming a TT compression.

    TODO:
    Add general function to tensor network to get number of parmeters!
    """

    max_ranks = [np.max(s.sol.ranks()) for s in sols]
    mean_ranks = [np.mean(s.sol.ranks()) for s in sols]

    dim = disc.dimension
    if dim == 1:
        total_unknowns = disc.x.shape[0] * disc.theta.shape[0] \
            * disc.psi.shape[0]
    elif dim == 2:
        total_unknowns = disc.x.shape[0] * disc.y.shape[0] * \
            disc.theta.shape[0] * disc.psi.shape[0]


    storage = [None] * len(sols)
    for ii, sol in enumerate(sols):
        ranks = sol.sol.ranks()
        num_params = ranks[0] * disc.x.shape[0] + \
            ranks[0] * ranks[1] * disc.theta.shape[0] + \
            ranks[1] * disc.psi.shape[0]
        storage[ii] = num_params

    compression_ratio = total_unknowns / np.array(storage)
    inds = np.arange(1, len(sols)+1)
    cumulative_unknowns = total_unknowns * inds
    cumulative_storage = np.cumsum(storage)
    cumulative_compression = cumulative_unknowns / cumulative_storage

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(max_ranks, '--k', label='max')
    axs[0].plot(mean_ranks, '-k', label='mean')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Rank')
    axs[0].legend()

    axs[1].semilogy(compression_ratio, '--k', label='per step')
    axs[1].semilogy(cumulative_compression, '-k', label='cumulative')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Compression Ratio')
    axs[1].legend()
    plt.tight_layout()
    return fig, axs



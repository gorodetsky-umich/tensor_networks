"""Solver utilities."""
from typing import Dict, Any, Optional, List
import pathlib
import copy
from dataclasses import dataclass
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

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


def get_rhs(indices: Indices,
            discretization: Discretization,
            config: Config):

    h = discretization.get_min_h()
    ones = np.ones(indices.space.size)
    sin_theta = np.sin(discretization.theta)
    cos_psi = np.cos(discretization.psi)

    ind = [indices.space, indices.theta, indices.psi]
    omega_x = pytens.tt_rank1(ind, [ones, sin_theta, cos_psi])
    if config.solver.stencil == 'upwind':

        if discretization.dimension == 1:
            from old_stuff import old_upwind_1d
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

def mean_intensity(indices: Indices,
                   disc: Discretization,
                   sol: Solution):

    sint = np.sin(disc.theta) / disc.theta.shape[0] * np.pi
    one =  np.ones((disc.psi.shape[0]))/ disc.psi.shape[0] * 2 * np.pi
    # print("tt = ", tt)
    integral = sol.sol.integrate(
        [indices.theta, indices.psi],
        [sint, one]
    ).contract().value

    norm = np.sum(np.sin(disc.theta))* \
           np.sum(np.ones((disc.psi.shape[0]))) / \
           disc.theta.shape[0] / disc.psi.shape[0] * \
           np.pi * 2 * np.pi

    integral /=  norm

    # print("integral = ", integral)
    return integral

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
                # print("xx shape = ", xx.shape)
                # print("yy shape = ", yy.shape)
                out = y.reshape((disc.x.shape[0],
                                 disc.y.shape[0]))
                levels = np.linspace(0, 1, 10)
                pos = axs['numerical'].contourf(xx[1:, 1:],
                                                yy[1:, 1:],
                                                out.T[1:, 1:],
                                                levels=levels)
                # ax.plot(disc.x[1:],
                #         y[1:], '-', color='gray', alpha=0.2)
                axs['numerical'].set_xlabel(r'$x$')
                axs['numerical'].set_ylabel(r'$y$')

                jleft = np.zeros((disc.y.shape[0], disc.x.shape[0]))
                jbottom = np.zeros((disc.y.shape[0], disc.x.shape[0]))
                dt = sol.time**2.0 - xx**2
                dt[dt < 1e-15] = 1e-14
                diff1 = yy / np.sqrt(dt)
                diff1[diff1 < 1] = 1
                eta_x = np.arccos(diff1)

                dt = sol.time**2.0 - yy**2
                dt[dt < 1e-14] = 1e-14
                diff2 = xx / np.sqrt(dt)
                diff2[diff2 < 1] = 1
                eta_y = np.arccos(diff2)


                # print(disc.x.shape)
                # print(disc.y.shape)
                # print(eta_x.shape)
                # print(eta_y.shape)
                ind_x = disc.x < sol.time
                jleft[:, ind_x] = 0.5 - \
                    (np.pi - eta_x[:, ind_x]) * xx[:, ind_x] / (2 * np.pi * sol.time) \
                    - 0.5 / np.pi * np.arcsin(xx[:, ind_x] * np.sin(eta_x[:, ind_x]) / np.sqrt(xx[:, ind_x]**2 + yy[:, ind_x]**2))

                ind_y = disc.y < sol.time
                jbottom[ind_y, :] = 0.5 - \
                    (np.pi - eta_y[ind_y, :]) * yy[ind_y, :] / (2 * np.pi * sol.time) \
                    - 0.5 / np.pi * np.arcsin(yy[ind_y, :] * np.sin(eta_y[ind_y, :]) / np.sqrt(xx[ind_y, :]**2 + yy[ind_y, :]**2))
                j = jleft + jbottom
                axs['analytical'].contourf(xx[1:, 1:],
                                           yy[1:, 1:],
                                           j[1:, 1:],
                                           levels=levels)
                                                              
                fig.colorbar(pos, cax=axs['color'])

                err = np.abs(out.T - j) + 1e-16
                # err = np.log10(np.abs(out - out_a) + 1e-16)
                # err += 1e-16
                cmap = plt.colormaps["bone"]
                pos = axs['error'].contourf(xx[1:, 1:],
                                            yy[1:, 1:],
                                            err[1:, 1:], cmap=cmap)
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

def main_loop(config: Config, logger, save_dir: pathlib.Path):

    sol = [None] * config.solver.num_steps

    indices = Indices.from_config(config)
    disc = Discretization.from_config(config)
    dt = config.solver.cfl * disc.get_min_h()
    logger.info("min_h = %f", disc.get_min_h())
    logger.info("dt = %f", dt)
    rhs = get_rhs(indices, disc, config)

    sol[0] = get_ic(indices, disc, config)

    sol[0].aux['mean_intensity'] = mean_intensity(
        indices, disc, sol[0]
    )

    time_step_plot, overlay_time_plot, final_plot = get_plots(disc, config)

    if overlay_time_plot is not None:
        fig, axs = plt.subplots(1, 1)
        axs = overlay_time_plot(axs, sol[0])

    if time_step_plot is not None:
        time_step_plot(sol[0])


    # plt.show()
    # plt.exit(1)

    # print(sol[0])
    time = 0.0
    for ii in (pbar := tqdm(range(1, config.solver.num_steps))):

        if ii % 1 == 0 and ii > 0:
            pbar.set_postfix(
                {
                    "time": sol[ii-1].time,
                    "ranks": sol[ii-1].sol.ranks()
                }
            )
        if ii % 1 == 0:
            logger.info("Step %d, ranks = %r", ii-1, sol[ii-1].sol.ranks())

        if config.solver.method == 'forward euler':
            sol[ii] = Solution(
                sol[ii-1].time + dt,
                sol[ii-1].sol + rhs(
                    sol[ii-1].time,
                    sol[ii-1].sol
                ).scale(-dt)
            )
        else:
            raise NotImplementedError("This solver not implemented")

        if ii % config.solver.round_freq == 0:
            sol[ii].sol = pytens.tt_round(
                sol[ii].sol,
                config.solver.round_tol
            )

        if ii % config.saving.plot_freq == 0:
            sol[ii].aux['mean_intensity'] = mean_intensity(
                indices, disc, sol[ii]
            )
            if overlay_time_plot is not None:
                axs = overlay_time_plot(axs, sol[ii])

            if time_step_plot is not None:
                 time_step_plot(sol[ii])

    plt.savefig(save_dir / f'{config.problem}_per_step_plot.pdf')

    logger.info("Final time: %r",sol[-1].time)
    plot_ranks_compressions(disc, sol)
    plt.savefig(save_dir / f'{config.problem}_ranks_compression.pdf')

    if final_plot is not None:
        fig, axs = final_plot(sol)
        plt.savefig(save_dir / f'{config.problem}_comparison_to_analytical.pdf')
    plt.show()

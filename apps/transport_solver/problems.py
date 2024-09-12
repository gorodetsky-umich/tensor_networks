"""Physics/Problem Specifications and utilities."""
import pathlib
from typing import Optional, List
import abc
from dataclasses import dataclass

import numpy as np

import pytens

from load_config import Config
import solver

def mean_intensity(indices: solver.Indices,
                   disc: solver.Discretization,
                   sol: solver.Solution) -> np.ndarray:

    sint = np.sin(disc.theta) / disc.theta.shape[0] * np.pi
    one =  np.ones((disc.psi.shape[0]))/ disc.psi.shape[0] * 2* np.pi
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


class Problem(abc.ABC):

    def __init__(self, config: Config) -> None:
        self.config = config
        self.indices = solver.Indices.from_config(config)

    @abc.abstractmethod
    def ic(self) -> pytens.TensorNetwork:
        pass

    @abc.abstractmethod
    def compute_aux_quantity(self, sol):
        pass

    @abc.abstractmethod
    def analytic_sol(self):
        pass

    @abc.abstractmethod
    def plot_all_sols(self):
        pass


class Hohlraum(Problem):

    def ic(self) -> pytens.TensorNetwork:
        if self.config.geometry.dimension == 1:
            x = np.zeros((self.config.geometry.n[0]))
            x[0] = 1
            theta = np.ones((self.config.angles.n_theta))
            psi = np.ones((self.config.angles.n_psi))
            ind = [self.indices.space, self.indices.theta, self.indices.psi]
            f = pytens.tt_rank1(ind, [x, theta, psi])
        elif self.config.geometry.dimension == 2:
            xy = np.zeros((self.config.geometry.n[0],
                           self.config.geometry.n[1]))
            xy[0, :] = 1.0
            xy[:, 0] = 1.0
            xy = xy.flatten()
            theta = np.ones((self.config.angles.n_theta))
            psi = np.ones((self.config.angles.n_psi))
            ind = [self.indices.space, self.indices.theta, self.indices.psi]
            f = pytens.tt_rank1(ind, [xy, theta, psi])
        else:
            raise NotImplementedError("Hohlraum beyond 2d not implemented")
        return f

    def compute_aux_quantity(
            self,
            sol: solver.Solution,
            disc: solver.Discretization
    ) -> solver.Solution:
        """Compute auxiliary quantities."""
        if self.geometry.dimension == 1:
            sol.aux['mean_intensity'] = mean_intensity(
                self.indices, disc, sol
            )
        
        return sol

    def analytic_sol(self):
        pass

    def plot_all_sols_1d(self, sols: List[Solution],
                         disc: Discretization):
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
        return axs
        
    def plot_all_sols(self,
                      sols: List[solver.Solution],
                      disc: Discretization):

        if self.config.geometry.dimension == 1:
            return self.plot_all_sols_1d(sols, disc)
        else:
            return None
        

    def plot_per_step_overlay(self, sol, disc, ax=None,
                              save: Optional[pathlib.Path]=None):
        if ax is None:
            fig, ax = plt.figure()
        if disc.dimension == 1:
            y = sol.aux['mean_intensity']
            ax.plot(disc.x[1:], y[1:], '-', color='gray', alpha=0.2)
            ax.set_xlabel(r'Space $x$')
            ax.set_ylabel(r'Mean intensity $J$')
            
            if save:
                fig = ax.get_figure()
                if save:
                    fig.savefig(save)
        else:
            return None

        return ax

    def plot_per_step(self):
        pass


def load_problem(config: Config) -> Problem:
    """Load a problem."""
    if config.problem == 'hohlraum':
        return Hohlraum(config)

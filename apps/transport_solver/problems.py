import abc
import numpy as np

import pytens

from load_config import Config
import solver

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


def mean_intensity(indices: Indices,
                   disc: solver.Discretization,
                   sol: solver.Solution):

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


class Problem(abc.ABC):

    def __init__(self, config: Config) -> None:
        self.config = config
        self.indices = Indices.from_config(config)

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

    def compute_aux_quantity(self,
                             sol: solver.Solution,
                             disc: solver.Discretization) -> solver.Solution:
        """Compute auxiliary quantities."""
        sol.aux['mean_intensity'] = mean_intensity(
            self.indices, disc, sol)
        )
        return sol

    def analytic_sol(self):
        pass

    def plot_all_sols(self):
        pass

    def plot_per_step_overlay(self):
        pass

    def plot_per_step(self):
        pass


def load_problem(config: Config) -> Problem:
    """Load a problem."""
    if config.problem == 'hohlraum':
        return Hohlraum(config.geometry.dimension)

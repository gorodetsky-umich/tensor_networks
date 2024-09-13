"""Main Loop."""
import pathlib
import logging
from typing import List

from load_config import Config
import solver
import problems

def plot_ranks_compressions(disc: solver.Discretization,
                            sols: List[solver.Solution]):
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

def main_loop(config: Config, logger: logging.Logger, save_dir: pathlib.Path):

    sol = [None] * (config.solver.num_steps + 1)

    problem = problems.load_problem(config)
    disc = solver.Discretization.from_config(config)
    dt = config.solver.cfl * disc.get_min_h()

    logger.info("min_h = %f", disc.get_min_h())
    logger.info("dt = %f", dt)

    rhs = solver.get_transport_term(problem.indices, disc, config)
    sol[0] = solver.Solution(0.0, problem.ic())
    sol[0] = problem.compute_aux_quantity(sol[0], disc)

    ax_o = problem.plot_per_step_overlay(sol[0], disc)

    # time_step_plot, overlay_time_plot, final_plot = get_plots(disc, config)


    axs = problem.plot_per_step(sol[0], disc)

    time = 0.0
    for ii in (pbar := tqdm(range(1, config.solver.num_steps + 1))):

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
            sol[ii] = problem.compute_aux_quantity(sol[ii], disc)
            ax_o = problem.plot_per_step_overlay(
                sol[ii],
                disc,
                ax=ax_o
            )
            ax_o.get_figure.savefig(
                save_dir / f'{config.problem}_{ii}_overlay_{sol[ii].time}.pdf'
            )

            ax = problem.plot_per_step(sol[ii], disc)
            if ax != None:
                ax.get_figure().savefig(
                    save_dir / f'{config.problem}_{ii}_{sol[ii].time}.pdf'
                )


    logger.info("Final time: %r",sol[-1].time)
    plot_ranks_compressions(disc, sol)
    plt.savefig(save_dir / f'{config.problem}_ranks_compression.pdf')

    axs = problem.plot_all_sols(sol, disc)
    if axs != None:
        axs.get_figure().savefig(
            save_dir / f'{config.problem}_comparison_to_analytical.pdf'
        )
    plt.show()

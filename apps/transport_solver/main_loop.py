"""Main Loop."""
import pathlib
import logger

from load_config import Config
import solver
import problems

def main_loop(config: Config, logger: logging.Logger, save_dir: pathlib.Path):

    sol = [None] * (config.solver.num_steps + 1)

    problem = problems.load_problem(config)
    disc = Discretization.from_config(config)
    dt = config.solver.cfl * disc.get_min_h()

    logger.info("min_h = %f", disc.get_min_h())
    logger.info("dt = %f", dt)

    rhs = solver.get_transport_term(problem.indices, disc, config)
    sol[0] = solver.Solution(0.0, problem.ic())
    sol[0] = problem.compute_aux_quantity(sol[0])

    time_step_plot, overlay_time_plot, final_plot = get_plots(disc, config)

    fig_o = None
    if overlay_time_plot is not None:
        fig_o, axs_o = plt.subplots(1, 1)
        axs_o = overlay_time_plot(axs_o, sol[0])

    if time_step_plot is not None:
        fig, axs = time_step_plot(sol[0])

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
            sol[ii].aux['mean_intensity'] = mean_intensity(
                indices, disc, sol[ii]
            )
            if overlay_time_plot is not None:
                axs_o = overlay_time_plot(axs_o, sol[ii])

            if time_step_plot is not None:
                fig, _ = time_step_plot(sol[ii])
                fig.savefig(
                    save_dir / f'{config.problem}_{ii}_{sol[ii].time}.pdf'
                )

    if fig_o is not None:
        fig_o.savefig(save_dir / f'{config.problem}_per_step_plot.pdf')

    logger.info("Final time: %r",sol[-1].time)
    plot_ranks_compressions(disc, sol)
    plt.savefig(save_dir / f'{config.problem}_ranks_compression.pdf')

    if final_plot is not None:
        fig, axs = final_plot(sol)
        plt.savefig(save_dir / f'{config.problem}_comparison_to_analytical.pdf')
    plt.show()

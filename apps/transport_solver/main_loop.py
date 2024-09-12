"""Main Loop."""
import pathlib
import logging

from load_config import Config
import solver
import problems

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
            sol[ii] = problem.compute_aux_quantity(sol[ii], disc)
            ax_o = problem.plot_per_step_overlay(
                sol[ii],
                disc,
                ax=ax_o,
                save=save_dir / f'{config.problem}_{ii}_overlay_{sol[ii].time}.pdf'
            )

            if time_step_plot is not None:
                fig, _ = time_step_plot(sol[ii])
                fig.savefig(
                    save_dir / f'{config.problem}_{ii}_{sol[ii].time}.pdf'
                )

    if ax_o is not None:        
        ax_o.get_figure().savefig(
            save_dir / f'{config.problem}_overlay_final.pdf'
        )

        
    logger.info("Final time: %r",sol[-1].time)
    plot_ranks_compressions(disc, sol)
    plt.savefig(save_dir / f'{config.problem}_ranks_compression.pdf')

    axs = problem.plot_all_sols(sol, disc)
    if axs != None:
        axs.get_figure().savefig(save_dir / f'{config.problem}_comparison_to_analytical.pdf')
    plt.show()

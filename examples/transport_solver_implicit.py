"""An explicit solver for transport."""
from pytens import *
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def get_indices(N=[160, 100, 99]):

    x = Index('x', N[0], 1)
    u = Index('u', N[1], 1)
    v = Index('v', N[2], 1)

    return [x, u, v]


def get_1d_stencil(N: int, k: float, c: float, h: float):
    """1D stencil."""
    A = np.zeros((N, N))
    for ii in range(1, N-1):
        A[ii, ii] = 4 * k
        A[ii, ii+1] = c - 2 * k
        A[ii, ii-1] = -c - 2 * k

    A[0, 0] = 4 * k
    A[0, 1] = c - 2 * k
    A[0, N-1] = -c - 2 * k

    A[N-1, N-1] = 4 * k
    A[N-1, N-2] = -c - 2 * k 
    A[N-1, 0] = c - 2 * k

    # A[N-1, 0] = 1
    
    A /= (2 * h)
    return A

def source_term(indices, tt, disc, draw=False):

    x, u, v = indices
    # sint = np.ones((u.size, 1))
    # sint[:, 0] = np.sin(xdisc[1])
    sint = vector('ui', u, np.sin(disc[1]) / u.size * np.pi)
    one = vector('vi', v, np.ones((v.size))/ v.size * 2 * np.pi)

    int_over_vel_attached = tt.attach(sint).attach(one)
    integral = int_over_vel_attached.contract('o')

    norm = np.sum(np.sin(disc[1]))* np.sum(np.ones((disc[2].shape[0]))) / disc[1].shape[0] / disc[2].shape[0] * np.pi * 2 * np.pi

    integral.scale(1.0 / norm)

    u2 = np.ones((u.size))
    v2 = np.ones((v.size))
    mean_tt = tt_rank1('source', indices,
                       [-integral.value('o'),
                        np.ones((u.size)),
                        np.ones((v.size))])

    source = tt + mean_tt

    if draw:
        fig, axs = plt.subplots(1,1)
        int_over_vel_attached.draw(ax=axs)

        fig, axs = plt.subplots(1,1)
        integral.draw(ax=axs)    
    # source.inspect()
    # plt.show()
    # exit(1)
    return source


class Problem:

    def __init__(self, indices, c=1.0, kscale=5e-1, dtscale=0.8):
        self.indices = indices
        self.dim = len(indices)
        self.c = c
        self.k = c * kscale

        self.h = 2.0 / (indices[0].size)
        self.dt = dtscale * self.h
        
        self.stencil = get_1d_stencil(indices[0].size, self.k, self.c, self.h)

        indices_out = [Index(f'{i.name}p', i.size, i.ndim) for i in indices]
        self.ttop = ttop_1dim(self.indices, indices_out, self.stencil)
        

        self.disc = [np.linspace(-1.0, 1.0 - 2.0 / i.size, i.size) for i in self.indices]
        self.disc[1] = np.linspace(0, np.pi, self.indices[1].size)
        self.disc[2] = np.linspace(0, 2.0*np.pi, self.indices[2].size)        

        ## Now initial condition
        coeffs = 2.0 * np.ones((self.dim))
        funcs = [None] * 3
        funcs[0] = np.sin(self.disc[0] * np.pi)
        funcs[1] = self.disc[1]**2 * coeffs[1]
        funcs[2] = self.disc[2]**2 * coeffs[2]
        
        self.u0 = tt_separable('u0', self.indices, funcs)

    def plot_slices(self, sol, ax, alpha=1.0, label=None, color='k'):
        u0full = sol.contract('o').value('o')
        axs[0].plot(self.disc[0], u0full[:, 0, 2], '-',
                    color=color, alpha=alpha, label=label)
        axs[1].plot(self.disc[0], u0full[:, 99, 98], '-',
                    color=color, alpha=alpha)
        axs[2].plot(self.disc[0], u0full[:, 89, 4], '-',
                    color=color, alpha=alpha)
        axs[3].plot(self.disc[0], u0full[:, 83, 24], '-',
                    color=color, alpha=alpha)

        
if __name__ == "__main__":

    x, u, v = get_indices(N=[160, 100, 99])
    round_eps = 1e-10
    # num_steps = 2230
    # num_steps = 5230
    num_steps = 10
    dtscale = 1e1 # 0.8 for explicit
    
    problem = Problem([x, u, v], dtscale=dtscale)

    fig, axs = plt.subplots(1, 1)
    problem.ttop.draw(ax=axs)

    fig, axs = plt.subplots(1, 1)
    problem.u0.draw(ax=axs)    

    op = problem.ttop.attach(problem.u0)
    fig, axs = plt.subplots(1, 1)
    op.draw(ax=axs)

    fig, axs = plt.subplots(4, 1)
    problem.plot_slices(problem.u0, axs)

    # source_term([x, u, v], problem.u0, problem.disc, draw=True)

    # plt.show()
    # exit(1)

    use_source = False
    solve = True
    if solve:

        d = 1
        # d = 10 / dt
        
        sol = [None] * num_steps
        sol[0] = problem.u0
        print("initial ranks = ", sol[0].ranks())
        for ii in range(1, num_steps):

            # op = problem.ttop.apply(sol[ii-1]).scale(-problem.dt).rename(f'u{ii}')

            if use_source is False:
                op = lambda ttin: ttin + problem.ttop.apply(ttin).scale(problem.dt)
                x0 = copy.deepcopy(sol[ii-1])
            else:
                def op(ttin):
                    o = ttin + problem.ttop.apply(ttin).scale(problem.dt)
                    o = o + source_term([x, u, v], ttin, problem.disc).scale(problem.dt * d)
                    return o
                x0 = copy.deepcopy(sol[ii-1])

            xf, resid = gmres(op, sol[ii-1], x0, 1e-10, round_eps, maxiter=30)
            # outer_iter = 0
            # while resid > 1e-3:
            #     x0 = xf
            #     xf, resid = gmres(op, sol[ii-1], xf, 1e-10, round_eps, maxiter=30)
            #     outer_iter += 1
            #     if outer_iter > 5:
            #         break

            print(f"Iteration {ii}, gmres resid = {resid}")
            # temp = sol[ii-1] + op
            sol[ii] = xf

            
            # print("Op ranks = ", op.ranks())

            # source = sol[ii-1].source_term(xdisc).scale(-dt * d)
            # sol[ii] = sol[ii-1] + op + source
            
            print(f"Sol[{ii}] ranks = {sol[ii].ranks()}")

            # if ii % 1 == 0:
            #     sol_copy = copy.deepcopy(sol[ii]).round(round_eps)
            #     sol[ii] = copy.deepcopy(sol_copy)
            #     print(f"\tRounded Sol[{ii}] ranks = {sol[ii].ranks()}")

            if ii % 2 == 0:                
                problem.plot_slices(sol[ii], axs, alpha=0.1)
        
        fig, axs = plt.subplots(4, 1)
        problem.plot_slices(sol[0], axs, label='ic', color='k')
        problem.plot_slices(sol[num_steps-1], axs, label='final', color='r')
        axs[0].legend()
        


    plt.show()
    

    

"""An explicit solver for transport. with the new interface"""
from pytens.algs2 import *
import copy
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx


def get_indices(N=[160, 100, 99]):

    x = Index('x', N[0])
    u = Index('u', N[1])
    v = Index('v', N[2])

    return [x, u, v]


def get_1d_stencil(N: int, k: float, c: float, h: float):
    """1D stencil."""
    A = np.zeros((N, N))
    for ii in range(1, N-1):
        A[ii, ii] = 4 * k
        A[ii, ii+1] = c - 2 * k
        A[ii, ii-1] = -c - 2 * k

    # NEW BC should be zero        
    # A[0, 0] = 4 * k
    # A[0, 1] = c - 2 * k
    # A[0, N-1] = -c - 2 * k

    # A[N-1, N-1] = 4 * k
    # A[N-1, N-2] = -c - 2 * k 
    # NEW BC should be zero
    # A[N-1, 0] = c - 2 * k
    
    A /= (2 * h)
    return A

def source_term_just_integral(indices, tt ,disc):
    x, u, v = indices
    # sint = np.ones((u.size, 1))
    # sint[:, 0] = np.sin(xdisc[1])
    sint = vector('ui', u, np.sin(disc[1]) / u.size * np.pi)
    one = vector('vi', v, np.ones((v.size))/ v.size * 2 * np.pi)

    int_over_vel_attached = tt.attach(sint).attach(one)
    integral = int_over_vel_attached.contract().value

    norm = np.sum(np.sin(disc[1]))* np.sum(np.ones((disc[2].shape[0]))) / disc[1].shape[0] / disc[2].shape[0] * np.pi * 2 * np.pi

    integral /= norm

    return integral

class Problem:

    def __init__(self, indices, c=1.0, kscale=5e-1, dtscale = 0.8):
        self.indices = indices
        self.dim = len(indices)
        self.c = c
        self.k = c * kscale

        # self.h = 2.0 / (indices[0].size)
        self.h = 1.0 / (indices[0].size)        
        self.dt = dtscale * self.h
        
        self.stencil = get_1d_stencil(indices[0].size, self.k, self.c, self.h)

        indices_out = [Index(f'{i.name}p', i.size) for i in indices]
        self.ttop = ttop_rank1(self.indices, indices_out,
                               [self.stencil,
                                np.eye(indices[1].size),
                                np.eye(indices[2].size)],
                               "A")



        # self.disc = [np.linspace(-1.0, 1.0 - 2.0 / i.size, i.size) for i in self.indices]
        self.disc = [np.linspace(0.0, 1.0 - 1.0 / i.size, i.size) for i in self.indices]        
        self.disc[1] = np.linspace(0, np.pi, self.indices[1].size)
        self.disc[2] = np.linspace(0, 2.0*np.pi, self.indices[2].size)        

        ones = np.ones((self.disc[0].shape[0]))
        sinu = np.sin(self.disc[1])
        cosv = np.cos(self.disc[2])
        # sinu = np.ones((self.disc[1].shape[0])) * 1e-2
        # cosv = np.ones((self.disc[2].shape[0]))
        self.omega = tt_rank1([x, u, v], [ones, sinu, cosv])
        # self.omega = tt_rank1(
        
        ## Now initial condition
        funcs = [None] * 3
        funcs[0] = np.zeros((self.disc[0].shape[0]))
        funcs[0][0] = 1
        # inds = cosv < 1.0
        funcs[1] = np.ones((self.disc[1].shape[0]))
        funcs[2] = np.ones((self.disc[2].shape[0]))
        funcs[2][cosv < 0.0+1e-15] = 0.0
        self.u0 = tt_rank1(self.indices, funcs)

        # coeffs = 2.0 * np.ones((self.dim))
        # funcs[0] = np.sin(self.disc[0] * np.pi)
        # funcs[1] = self.disc[1]**2 * coeffs[1]
        # funcs[2] = self.disc[2]**2 * coeffs[2]
        # self.u0 = tt_separable('u0', self.indices, funcs)

    def plot_slices(self, sol, ax, alpha=1.0, label=None, color='k'):
        u0full = sol.contract().value

        u = np.sin(self.disc[1][0])
        v = np.cos(self.disc[2][2])
        title = f'{u}_{v}'
        axs[0].plot(self.disc[0], u0full[:, 0, 2], '-',
                    color=color, alpha=alpha, label=label)
        axs[0].set_title(title)

        try:
            axs[1].plot(self.disc[0], u0full[:, 99, 98], '-',
                        color=color, alpha=alpha)
            u = np.sin(self.disc[1][99])
            v = np.cos(self.disc[2][98])
            title = f'{u}_{v}'
            axs[1].set_title(title)

            axs[2].plot(self.disc[0], u0full[:, 89, 4], '-',
                        color=color, alpha=alpha)
            u = np.sin(self.disc[1][89])
            v = np.cos(self.disc[2][4])
            title = f'{u}_{v}'
            axs[2].set_title(title)        

            axs[3].plot(self.disc[0], u0full[:, 83, 24], '-',
                        color=color, alpha=alpha)
            u = np.sin(self.disc[1][83])
            v = np.cos(self.disc[2][24])
            title = f'{u}_{v}'
            axs[3].set_title(title)
        except:
            pass
        plt.tight_layout()
        
if __name__ == "__main__":

    # x, u, v = get_indices(N=[160, 100, 99])
    x, u, v = get_indices(N=[160, 100, 99])    
    round_eps = 1e-10
    # num_steps = 2230
    # num_steps = 5230
    # num_steps = 120
    num_steps = 240
    
    # problem = Problem([x, u, v], kscale=0.5, dtscale=0.8)
    problem = Problem([x, u, v], kscale=0.5, dtscale=0.8)
    # problem = Problem([x, u, v], kscale=0.5, dtscale=0.2)
    # plt.figure()
    final_time = num_steps * problem.dt
    print("final time = ", final_time)

    
    fig, axs = plt.subplots(1, 1)
    problem.ttop.draw(ax=axs)

    fig, axs = plt.subplots(1, 1)
    problem.u0.draw(ax=axs)    

    op = problem.ttop.attach(problem.u0)
    fig, axs = plt.subplots(1, 1)
    op.draw(ax=axs)

    fig, axs = plt.subplots(4, 1)
    problem.plot_slices(problem.omega, axs)
    
    plt.close('all')    
    
    u0 = problem.u0.contract().value
    plt.figure()
    plt.contourf(u0[0, :, :])
    plt.colorbar()
    plt.title('u0')
    
    omega = problem.omega.contract().value

    plt.figure()
    plt.contourf(omega[5, :, :])
    plt.colorbar()
    plt.title('Omega')
    # source_term([x, u, v], problem.u0, problem.disc, draw=True)

    # plt.show()
    # exit(1)
    
    fig, axs = plt.subplots(4, 1)
    problem.plot_slices(problem.u0, axs)

    solve = True
    if solve:

        d = 1
        # d = 10 / dt
        
        sol = [None] * num_steps
        sol[0] = problem.u0
        print("initial ranks = ", sol[0].ranks())
        for ii in range(1, num_steps):

            # op1 = problem.ttop.apply(sol[ii-1])
            op1 = ttop_apply(problem.ttop, copy.deepcopy(sol[ii-1]))

            # op1.inspect()
            # print("\n")
            # problem.omega.inspect()
            # exit(1)
            
            op2 = op1 * problem.omega
            # op2 = op1
            
            op = op2.scale(-problem.dt)

            temp = sol[ii-1] + op
            sol[ii] = temp
            
            # print("Op ranks = ", op.ranks())

            # source = sol[ii-1].source_term(xdisc).scale(-dt * d)
            # sol[ii] = sol[ii-1] + op + source
            
            print(f"Sol[{ii}] ranks = {sol[ii].ranks()}")

            if ii % 1 == 0:
                sol_copy = tt_round(copy.deepcopy(sol[ii]), round_eps)
                sol[ii] = copy.deepcopy(sol_copy)
                print(f"\tRounded Sol[{ii}] ranks = {sol[ii].ranks()}")

            if ii % 5 == 0:
                problem.plot_slices(sol[ii], axs, alpha=0.1)
        
        fig, axs = plt.subplots(4, 1)
        problem.plot_slices(sol[0], axs, label='ic', color='k')
        problem.plot_slices(sol[num_steps-1], axs, label='final', color='r')
        axs[0].legend()


    plt.savefig('last_step.pdf')
    val = source_term_just_integral([x, u, v], sol[-1], problem.disc)
    
    plt.figure()
    plt.plot(problem.disc[0], val, label='numeric')
    analytic = 0.5 * (1.0 - problem.disc[0] / final_time)
    analytic[analytic < 0.0] = 0.0
    plt.plot(problem.disc[0], analytic, '--r', label='analytic')
    # plt.show()
    # plt.exit(1)
    plt.ylabel('J')
    plt.xlabel('x')
    plt.legend()

    plt.show()
    

    

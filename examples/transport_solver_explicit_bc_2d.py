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
    integral = int_over_vel_attached.contract('o')

    norm = np.sum(np.sin(disc[1]))* np.sum(np.ones((disc[2].shape[0]))) / disc[1].shape[0] / disc[2].shape[0] * np.pi * 2 * np.pi

    integral.scale(1.0 / norm)

    return integral
    
    
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

# def rhs(T: np.ndarray,
#         J: np.ndarray,
#         a: float = 1.0,
#         rho: float = 1.0,
#         kappa: float= 1.0,
#         cv: float = 1.0):
#     out = rho * kappa * (4 * np.pi * J - a * T**4)
#     return out

class Problem:

    def __init__(self, indices, c=1.0, kscale=5e-1, dtscale = 0.8):
        self.indices = indices
        self.dim = len(indices)
        self.c = c
        self.k = c * kscale

        # self.h = 2.0 / (indices[0].size)
        N1d = int(np.sqrt(indices[0].size))
        self.h = 1.0 / N1d
        self.dt = dtscale * self.h
        print("N1d = ", N1d)
        self.stencil = get_1d_stencil(N1d, self.k, self.c, self.h)
        print("stencil size = ", self.stencil.shape)

        stencil_x = np.einsum('ij,kl->ikjl', self.stencil, np.eye(N1d))
        stencil_y = np.einsum('ij,kl->ikjl', np.eye(N1d), self.stencil)
        stencil_x = np.reshape(stencil_x, (indices[0].size,
                                           indices[0].size))

        stencil_y = np.reshape(stencil_y, (indices[0].size,
                                           indices[0].size))
        np.set_printoptions(linewidth=np.inf)
        print("stencil_x")
        print(stencil_x)
        print("stencil_y")
        print(stencil_y)

        self.stencil_x = stencil_x
        self.stencil_y = stencil_y

        
        # exit(1)
        
        

        indices_out = [Index(f'{i.name}p', i.size, i.ndim) for i in indices]
        self.ttop_x = ttop_1dim(self.indices, indices_out, self.stencil_x)
        self.ttop_y = ttop_1dim(self.indices, indices_out, self.stencil_y)



        # self.disc = [np.linspace(-1.0, 1.0 - 2.0 / i.size, i.size) for i in self.indices]
        self.disc = [np.linspace(0.0, 1.0 - 2.0 / i.size, i.size) for i in self.indices]        
        self.disc[1] = np.linspace(0, np.pi, self.indices[1].size)
        self.disc[2] = np.linspace(0, 2.0*np.pi, self.indices[2].size)        

        ones = np.ones((self.disc[0].shape[0]))
        sinu = np.sin(self.disc[1])
        cosv = np.cos(self.disc[2])
        sinv = np.sin(self.disc[2])
        # sinu = np.ones((self.disc[1].shape[0])) * 1e-2
        # cosv = np.ones((self.disc[2].shape[0]))
        self.omega_x = tt_rank1('omega', [x, u, v], [ones, sinu, cosv])
        self.omega_y = tt_rank1('omega', [x, u, v], [ones, sinu, sinv])
        
        # self.omega = tt_rank1(
        
        ## Now initial condition
        funcs = [None] * 3

        n = int(np.sqrt(self.disc[0].shape[0]))
        xx = np.zeros(n)
        yy = np.ones(n)
        xx[0] = 1
        # yy[0] = 1

        ## x = 0
        xy = np.outer(xx, yy).flatten()

        ## y = 0
        yx = np.outer(yy, xx).flatten()

        # inds = cosv < 1.0
        # funcs[0] = xy # + yx
        # funcs[0] = yx
        funcs[0] = xy + yx        
        funcs[0][0] = 1.0
        funcs[1] = np.ones((self.disc[1].shape[0]))
        funcs[2] = np.ones((self.disc[2].shape[0]))
        self.u0a = tt_rank1('u0a', self.indices, copy.deepcopy(funcs))


        
        funcs[0] = np.ones((self.disc[0].shape[0]))
        funcs[2][cosv < 0.0+1e-15] = 0.0
        funcs[2][sinv < 0.0 + 1e-15] = 0.0
        # funcs[2][sinv < 0.0+1e-15] = 0.0
        self.mask = tt_rank1('u0b', self.indices, copy.deepcopy(funcs))
        
        # self.u0b = tt_rank1('u0b', self.indices, copy.deepcopy(funcs))

        self.u0 = self.u0a * self.mask
        # self.u0 = self.u0b

        

        # coeffs = 2.0 * np.ones((self.dim))
        # funcs[0] = np.sin(self.disc[0] * np.pi)
        # funcs[1] = self.disc[1]**2 * coeffs[1]
        # funcs[2] = self.disc[2]**2 * coeffs[2]
        # self.u0 = tt_separable('u0', self.indices, funcs)

    def plot_slices(self, sol, ax, alpha=1.0, label=None, color='k'):
        u0full = sol.contract('o').value('o')

        u = np.sin(self.disc[1][0])
        v = np.cos(self.disc[2][2])
        title = f'{u}_{v}'
        
        slice_jj = u0full[:, 0, 2]
        n2 = slice_jj.shape[0]
        n = int(np.sqrt(n2))
        slice_jj = np.reshape(slice_jj, (n, n))
        axs[0].imshow(slice_jj)
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        
        #                color=color, alpha=alpha, label=label)
        # axs[0].set_title(title)

        # try:
        #     axs[1].plot(self.disc[0], u0full[:, 99, 98], '-',
        #                 color=color, alpha=alpha)
        #     u = np.sin(self.disc[1][99])
        #     v = np.cos(self.disc[2][98])
        #     title = f'{u}_{v}'
        #     axs[1].set_title(title)

        #     axs[2].plot(self.disc[0], u0full[:, 89, 4], '-',
        #                 color=color, alpha=alpha)
        #     u = np.sin(self.disc[1][89])
        #     v = np.cos(self.disc[2][4])
        #     title = f'{u}_{v}'
        #     axs[2].set_title(title)        

        #     axs[3].plot(self.disc[0], u0full[:, 83, 24], '-',
        #                 color=color, alpha=alpha)
        #     u = np.sin(self.disc[1][83])
        #     v = np.cos(self.disc[2][24])
        #     title = f'{u}_{v}'
        #     axs[3].set_title(title)
        # except:
        #     pass
        plt.tight_layout()
        
if __name__ == "__main__":

    # x, u, v = get_indices(N=[160, 100, 99])
    # x, u, v = get_indices(N=[1024, 100, 99])
    x, u, v = get_indices(N=[4096, 100, 99])

    round_eps = 1e-5
    # num_steps = 2230
    # num_steps = 5230
    num_steps = 30
    
    problem = Problem([x, u, v], kscale=0.5, dtscale=0.8)
    # plt.figure()
    final_time = num_steps * problem.dt
    print("final time = ", final_time)

    

    # fig, axs = plt.subplots(1, 1)
    # problem.plot_slices(problem.u0, ax = axs)

    fig, axs = plt.subplots(2, 1)
    problem.plot_slices(problem.u0, axs)

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

            op1 = problem.ttop_x.apply(sol[ii-1])
            op1b = op1 * problem.omega_x
            
            op2 = problem.ttop_y.apply(sol[ii-1])
            op2b = op2 * problem.omega_y
            
            # op1.inspect()
            # print("\n")
            # problem.omega.inspect()
            # exit(1)
            op2 = op2b + op1b
            op2.rename('blah')
            # op2 = op1
            
            op = op2.scale(-problem.dt).rename(f'u{ii}')

            temp = sol[ii-1] + op
            if use_source is False:
                sol[ii] = temp
            else:
                source = source_term([x, u, v], sol[ii-1], problem.disc)
                source = source.scale(-problem.dt * d)
                sol[ii] = temp + source
            
            # print("Op ranks = ", op.ranks())

            # source = sol[ii-1].source_term(xdisc).scale(-dt * d)
            # sol[ii] = sol[ii-1] + op + source
            
            print(f"Sol[{ii}] ranks = {sol[ii].ranks()}")

            if ii % 1 == 0:
                sol_copy = copy.deepcopy(sol[ii]).round(round_eps)
                sol[ii] = copy.deepcopy(sol_copy)
                print(f"\tRounded Sol[{ii}] ranks = {sol[ii].ranks()}")

            if ii % 5 == 0:
                problem.plot_slices(sol[ii], axs, alpha=0.1)
        
        fig, axs = plt.subplots(2, 1)
        problem.plot_slices(sol[0], axs, label='ic', color='k')
        problem.plot_slices(sol[num_steps-1], axs, label='final', color='r')
        axs[0].legend()


    integral = source_term_just_integral([x, u, v], sol[-1], problem.disc)
    val = integral.contract('o').value('o')

    plt.figure()
    n = int(np.sqrt(val.shape[0]))
    plt.imshow(val.reshape((n, n)))
    # plt.plot(problem.disc[0], val, label='numeric')
    # analytic = 0.5 * (1.0 - problem.disc[0] / final_time)
    # analytic[analytic < 0.0] = 0.0
    # plt.plot(problem.disc[0], analytic, '--r', label='analytic')
    # plt.show()
    # plt.exit(1)
    plt.ylabel('x')
    plt.xlabel('y')
    plt.legend()
    plt.savefig('last_step.pdf')

    plt.show()
    

    

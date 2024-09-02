"""An explicit solver for transport."""
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

def get_1d_stencil_plus(N: int, h: float):
    """1D stencil."""
    A = np.zeros((N, N))
    for ii in range(1, N-1):
        A[ii, ii] = 1
        A[ii, ii-1] = -1

    A /= h

    # A[1, 1] = 1
    # A[1, 0] = -1
    # for ii in range(2, N):
    #     A[ii, ii] = 3
    #     A[ii, ii-1] = -4
    #     A[ii, ii-2] = 1
    # A[1, :] /= h
    # A[2:, :] /= (2*h)
    
    return A

def get_1d_stencil_minus(N: int, h: float):
    """1D stencil."""
    A = np.zeros((N, N))
    for ii in range(1, N-1):
        A[ii, ii+1] = 1
        A[ii, ii] = -1
    A /= h

    # for ii in range(1, N-2):
    #     A[ii, ii] = -3
    #     A[ii, ii+1] = 4
    #     A[ii, ii+2] = -1
    # # A[1, :] /= h
    # A[2:, :] /= (2*h)
    
    return A

def get_1d_stencil(N: int, k: float, c: float, h: float):
    """1D stencil."""
    A = np.zeros((N, N))
    for ii in range(1, N-1):
        A[ii, ii] = 4 * k
        A[ii, ii+1] = c - 2 * k
        A[ii, ii-1] = -c - 2 * k

    A /= (2 * h)
    return A

def source_term_just_integral(indices, tt, disc):
    x, u, v = indices
    sint = np.sin(disc[1]) / u.size * np.pi
    one =  np.ones((v.size))/ v.size * 2 * np.pi
    # print("tt = ", tt)
    integral = tt.integrate([u, v], [sint, one]).contract().value
    assert(integral.ndim == 1)

    norm = np.sum(np.sin(disc[1]))* np.sum(np.ones((disc[2].shape[0]))) / disc[1].shape[0] / disc[2].shape[0] * np.pi * 2 * np.pi

    integral /=  norm

    return integral

def source_term(indices, tt, disc, draw=False):

    x, u, v = indices
    # sint = np.ones((u.size, 1))
    # sint[:, 0] = np.sin(xdisc[1])
    integral = source_term_just_integral(indices, tt, disc)

    # print("integral.shape = ", integral.shape)
    # print("integral = ", integral)
    # u2 = np.ones((u.size))
    # v2 = np.ones((v.size))
    mean_tt = tt_rank1(indices,
                       [-integral,
                        np.ones((u.size)),
                        np.ones((v.size))])
    # mean_tt = tt_rank1(indices,
    #                    [-np.ones((x.size)) * 0.5,
    #                     np.ones((u.size)),
    #                     np.ones((v.size))])    

    # print("mean tt", mean_tt)
    # print(mean_tt.contract().value)
    # print("tt = ", tt.contract().value)
    source = tt + mean_tt
    # print("source unmasked = ", source.contract().value)
    # Set boundary of source to be zero
    funcs = [None] * 3
    funcs[0] = np.ones((disc[0].shape[0])) # * 1e-18
    funcs[0][0] = 0
    funcs[0][-1] = 0
    funcs[1] = np.ones((disc[1].shape[0]))
    funcs[2] = np.ones((disc[2].shape[0]))
    mask = tt_rank1(indices, funcs)

    # mask_arr = mask.contract().value
    # print(mask_arr[:, 5, :])
    # exit(1)
    
    source = source * mask

    # print("source = ", source.contract().value[:, 10, 10])
    # exit(1)
    # # check = source.contract('o').value('o')
    # check = mask.contract('o').value('o')    
    # plt.figure()
    # plt.plot(check[:, 25, 25])
    # plt.show()

    # exit(1)
    if draw:
        fig, axs = plt.subplots(1,1)
        int_over_vel_attached.draw(ax=axs)

    # source.inspect()
    # plt.show()
    # exit(1)
    return source

class Problem:

    def __init__(self, indices, c=1.0, dtscale = 0.8):
        self.indices = indices
        self.dim = len(indices)
        self.c = c

        # self.h = 2.0 / (indices[0].size)
        self.h = 1.0 / (indices[0].size)        
        self.dt = dtscale * self.h

        # self.disc = [np.linspace(0.0, 1.0 - 1.0 / i.size, i.size)  \
        #              for i in self.indices]
        self.disc = [None] * 3
        self.disc[0] = np.linspace(0.0, 1.0 - self.h, self.indices[0].size)
        self.disc[1] = np.linspace(0, np.pi, self.indices[1].size)
        self.disc[2] = np.linspace(0, 2.0*np.pi, self.indices[2].size)
        
        self.stencil_plus = get_1d_stencil_plus(indices[0].size, self.h)
        self.stencil_minus = get_1d_stencil_minus(indices[0].size, self.h)

        ones = np.ones((self.disc[0].shape[0]))
        sinu = np.sin(self.disc[1])
        cosv = np.cos(self.disc[2])
        
        eyev = np.eye(self.indices[2].size)
        eyev_plus = copy.deepcopy(eyev)
        eyev_plus[cosv < 0.0 + 1e-14, :] = 0.0
        eyev_minus = copy.deepcopy(eyev)
        eyev_minus[cosv > 0.0, :] = 0.0
        # plt.figure()
        # plt.imshow(eyev_plus)

        # plt.figure()
        # plt.imshow(eyev_minus)

        # plt.show()
        # exit(1)
        
        indices_out = [Index(f'{i.name}p', i.size) for i in indices]
        self.ttop = ttop_rank2(self.indices,
                               indices_out,
                               [self.stencil_plus,
                                np.eye(indices[1].size),
                                eyev_plus
                                ],
                               [self.stencil_minus,
                                np.eye(indices[1].size),
                                eyev_minus
                                ],
                               "A")

        # Original stencil
        self.stencil = get_1d_stencil(indices[0].size, self.c * 5e-1, self.c, self.h)
        self.ttop2 = ttop_rank1(self.indices, indices_out,
                               [self.stencil,
                                np.eye(indices[1].size),
                                np.eye(indices[2].size)],
                               "A")        

        # self.ttop = self.ttop2
        # print(cosv[0])        
        # print(self.ttop[:, :, 0, 0, 0, 0])
        # print(self.ttop2[:, :, 0, 0, 0, 0])
        # print("\n")
        # for ind in range(cosv.shape[0]):
        #     # ind = 20
        #     print(cosv[ind])
        #     # print(self.ttop[:, :, 0, 0, ind, ind])
        #     # print(self.ttop2[:, :, 0, 0, ind, ind])
        #     err = self.ttop[:, :, 0, 0, ind, ind].value - self.ttop2[:, :, 0, 0, ind, ind].value
        #     print("err = ", np.linalg.norm(err))
        #     for ind2 in range(self.disc[1].shape[0]):
        #         err = self.ttop[:, :, ind2, ind2, ind, ind].value - self.ttop2[:, :, ind2, ind2, ind, ind].value
        #         print("err = ", np.linalg.norm(err))
                
        # print("ttop = ", self.ttop)
        # print("ttop2 = ", self.ttop2
              # )
        # self.ttop = self.ttop2        
        # exit(1)

        print("indices = ", self.indices)
        self.omega = tt_rank1(self.indices, [ones, sinu, cosv])

        ## Now initial condition
        funcs = [None] * 3
        funcs[0] = np.zeros((self.disc[0].shape[0]))
        funcs[0][0] = 1
        # inds = cosv < 1.0
        funcs[1] = np.ones((self.disc[1].shape[0]))
        funcs[2] = np.ones((self.disc[2].shape[0]))
        # funcs[2][cosv < 0.0+1e-15] = 0.0
        self.u0 = tt_rank1(self.indices, funcs)


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

    x, u, v = get_indices(N=[160, 100, 99])
    # x, u, v = get_indices(N=[320, 100, 99])    
    round_eps = 1e-5
    # num_steps = 2230
    # num_steps = 5230
    # num_steps = 400
    # num_steps = 200
    # num_steps = 80
    # num_steps = 120  
    # num_steps = 10
    num_steps = 50
    
    problem = Problem([x, u, v], dtscale=2)

    final_time = num_steps * problem.dt
    print("final time = ", final_time)

    # exit(1)
    # print(problem.u0)
    u0 = problem.u0.contract().value
    print("u0 shape = ", u0.shape)
    plt.figure()
    plt.contourf(u0[0, :, :])
    plt.colorbar()
    plt.title('u0')
    
    # omega = problem.omega.contract().value

    # source = source_term([x, u, v], problem.u0, problem.disc)
    # plt.figure()
    # plt.contourf(source.contract().value[:, 0, :])
    # plt.colorbar()
    # plt.title('source')
    
    # plt.show()
    # exit(1)

    print("Initial condition = ", problem.u0)

    # exit(1)
    
    fig, axs = plt.subplots(4, 1)
    problem.plot_slices(problem.u0, axs)

    use_source = True
    solve = True
    if solve:
        d = 100
        sol = [None] * num_steps
        sol[0] = problem.u0
        for ii in range(1, num_steps):
            # print("\n\n\n")

            ######
            # Explicit
            ####
            # op1 = ttop_apply(problem.ttop, copy.deepcopy(sol[ii-1]))
            # op2 = op1 * problem.omega
            # op = op2.scale(-problem.dt * problem.c)
            # sol[ii] = sol[ii-1] + op            
            # if use_source is True:
            #     source = source_term([x, u, v], sol[ii-1], problem.disc)
            #     source = source.scale(-problem.dt * d)
            #     sol[ii] = sol[ii] + source
            #####
            # Implicit
            #####
            def op(ttin):
            
                o = ttop_apply(problem.ttop, copy.deepcopy(ttin))
                o = o * problem.omega
                o = o.scale(problem.dt)
                o = ttin + o

                if use_source is True:
                    source =  source_term([x, u, v], ttin, problem.disc).scale(problem.dt * d)
                    o = o + source

                return o
            x0 = copy.deepcopy(sol[ii-1])
            xf, resid = gmres(op, sol[ii-1], x0, 1e-10, round_eps, maxiter=20) 
            # xf, resid = gmres(op, sol[ii-1], x0, 1e-10, round_eps, maxiter=40)

            print(f"Iteration {ii}, gmres resid = {resid}")            

            sol[ii] = xf



                # plt.figure()
                # plt.contourf(source.contract().value[:, 0, :])
                # plt.colorbar()
                # plt.title('source')
                
                # plt.show()
                # exit(1)

            print(f"Sol[{ii}] ranks = {sol[ii].ranks()}")
            print(f"Sol[{ii}] = ", sol[ii])

            if ii % 1 == 0:
                # sol_copy = tt_round(copy.deepcopy(sol[ii]), round_eps)
                # sol_copy = 
                sol[ii] = tt_round(sol[ii], round_eps)
                # print(f"\tRounded Sol[{ii}] ranks = {sol[ii].ranks()}")

            # print(f"Sol_rounded[{ii}] = ", sol[ii])
            if ii % 5 == 0:
                problem.plot_slices(sol[ii], axs, alpha=0.1)
        
        fig, axs = plt.subplots(4, 1)
        problem.plot_slices(sol[0], axs, label='ic', color='k')
        problem.plot_slices(sol[num_steps-1], axs, label='final', color='r')
        axs[0].legend()
        
    # exit(1)

    integral = source_term_just_integral([x, u, v], sol[-1], problem.disc)

    plt.figure()
    plt.plot(problem.disc[0], integral, label='numeric')
    analytic = 0.5 * (1.0 - problem.disc[0] / final_time)
    analytic[analytic < 0.0] = 0.0
    plt.plot(problem.disc[0], analytic, '--r', label='analytic')
    # plt.show()
    # plt.exit(1)
    plt.ylabel('J')
    plt.xlabel('x')
    plt.legend()

    plt.show()
    

    

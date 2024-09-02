"""Tensor Network Solvers."""
from pytens import *

import numpy as np
import matplotlib.pyplot as plt
import timeit

np.random.seed(4)

def tt_inner_timer(r, n, d, num=100):

    indices = [Index(f'x{ii}', n, 1) for ii in range(d)]
    tt_ranks = [1] + [r for ii in range(d-1)] + [1]
    
    A = rand_tt('A', indices, tt_ranks)
    B = rand_tt('B', indices, tt_ranks)

    def inner():
        return A.inner(B)
    
    out = timeit.timeit(inner, number=num)# , globals=globals())
    return out
    
    

if __name__ == "__main__":

    num = 1
    n = 20
    d = 20
    # Rank scaling
    ranks = np.array([10, 20, 40, 80, 160, 320, 640])
    times = np.zeros(len(ranks))
    for ii in range(len(ranks)):
        times[ii] = tt_inner_timer(ranks[ii], n, d, num=num)

    plt.figure()
    plt.plot(np.log10(ranks), np.log10(times), '-', label='Contraction')
    plt.plot(np.log10(ranks), 3*np.log10(ranks), '--', label=r'r^3')
    plt.plot(np.log10(ranks), 4*np.log10(ranks), '--k', label=r'r^4')
    plt.xlabel('Log rank')
    plt.ylabel('Log time')
    plt.legend()


    num = 10
    d = 20
    r = 20
    # Mode size scaling
    nums = np.array([5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560])
    times = np.zeros(len(nums))
    for ii in range(len(nums)):
        times[ii] = tt_inner_timer(r, nums[ii], d, num=num)

    plt.figure()
    plt.plot(np.log10(nums), np.log10(times), '-', label='Contraction')
    plt.plot(np.log10(nums), np.log10(nums), '--', label=r'n')
    plt.plot(np.log10(nums), 2*np.log10(nums), '--', label=r'n^2')
    plt.xlabel('Log n')
    plt.ylabel('Log time')
    plt.legend()


    num = 2
    r = 5
    n = 5
    # Dimension scaling
    ds = np.array([5, 10, 20, 40, 80, 160, 320, 640])
    times = np.zeros(len(ds))
    for ii in range(len(ds)):
        times[ii] = tt_inner_timer(r, n, ds[ii], num=num)

    plt.figure()
    plt.plot(np.log10(ds), np.log10(times), '-', label='Contraction')
    plt.plot(np.log10(ds), np.log10(ds), '--', label=r'd')
    plt.plot(np.log10(ds), 2*np.log10(ds), '--', label=r'd^2')
    plt.xlabel('Log dim')
    plt.ylabel('Log time')
    plt.legend()
    plt.show()
    


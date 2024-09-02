"""Tensor Network Solver Capabilities and Test"""
# from pytens import *
from pytens.algs2 import *
import pytens.algs as algs_old
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(4)

if __name__ == "__main__":
    
    x = Index('x', 5)
    u = Index('u', 10)
    v = Index('v', 20)

    TT = rand_tt([x, u, v], [2, 2])
    
    print(TT)
    indices = TT.all_indices()
    # print(indices)
    free_indices = TT.free_indices()
    print("Full indices = ", free_indices)
    print("TT ranks = ", TT.ranks())

    fig, axs = plt.subplots(1, 1)    
    TT.draw(ax=axs)

    # ttcon = TT.contract()
    val = TT[2:4, 5, 3]
    print(val)

    tt_ranks2 = [3, 4]    
    TT2 = rand_tt([x, u, v], tt_ranks2)
    TT3 = TT.attach(TT2)
    print(TT3)
    
    fig, axs = plt.subplots(1, 1)
    TT3.draw(ax=axs)

    integrate = TT.integrate([x], [1.0])
    print(integrate)
    
    fig, axs = plt.subplots(1, 1)
    integrate.draw(ax=axs)
    
    print("\n\n\n")

    print("\n---------------\n")
    print("Addition ")
    tt_add = TT + TT2
    print(tt_add)

    print("\n---------------\n")
    print("Elementwise multiplication ")
    tt_mult = TT * TT2
    print(tt_mult)


    print("\n---------------\n")
    print("TTop")
    x = Index('x', 10)
    xp = Index('xp', 10)
    y = Index('y', 5)
    yp = Index('yp', 5)
    z = Index('z', 3)
    zp = Index('zp', 3)

    xo = algs_old.Index('x', 10, 1)
    xpo = algs_old.Index('xp', 10, 1)
    yo = algs_old.Index('y', 5, 1)
    ypo = algs_old.Index('yp', 5, 1)
    zo = algs_old.Index('z', 3, 1)
    zpo = algs_old.Index('zp', 3, 1)    

    indices_in = [x, y, z]
    indices_out = [xp, yp, zp]    
    A = np.random.randn(10, 10)
    B = np.eye(5, 5)
    C = np.eye(3, 3)
    ttop = ttop_rank1(indices_in, indices_out, [A, B, C], "A")

    print(ttop)
    fig, axs = plt.subplots(1, 1)
    ttop.draw(ax=axs)

    tt = rand_tt([x, y, z], [3, 2])    
    ttout = ttop_apply(ttop, copy.deepcopy(tt))
    print(ttout)
    fig, axs = plt.subplots(1, 1)
    ttout.draw(ax=axs)


    print("\n\n\n---------------\n")
    print("GMRES")

    indices_in_o = [xo, yo, zo]
    indices_out_o = [xpo, ypo, zpo]    
    ttop = ttop_rank1(indices_in, indices_out, [A, B, C], "A")
    ttop_old = algs_old.ttop_1dim(indices_in_o, indices_out_o, A)

    ttop_arr = ttop.contract().value
    ttop_old_arr = ttop_old.contract('o').value('o')
    err = np.linalg.norm(ttop_arr - ttop_old_arr)
    assert err < 1e-10, f"Error = {err}"


    ff_rand = [np.random.randn(10), np.random.randn(5), np.random.randn(3)]
    tt_rand = tt_separable([x, y, z], ff_rand)
    tt_rand_old = algs_old.tt_separable("f", [xo, yo, zo], ff_rand)

    tt_rand_arr = tt_rand.contract().value
    tt_rand_old_arr = tt_rand_old.contract('o').value('o')
    err = np.linalg.norm(tt_rand_arr - tt_rand_old_arr)
    assert err < 1e-10, f"Error = {err}"

    ttop_new_apply = ttop_apply(ttop, tt_rand)
    ttop_old_apply = ttop_old.apply(tt_rand_old)

    arr1 = ttop_new_apply.contract().value
    arr2 = ttop_old_apply.contract('o').value('o')

    err = np.linalg.norm(arr1 - arr2)
    print(err)
    assert err < 1e-10, f"Error = {err}"

    
    
    # exit(1)
    # print("err = ", np.linalg.norm(err))
    # exit(1)
    
    x0 = rand_tt([x, y, z], [3, 2])
    op = lambda ttin: ttop_apply(ttop, ttin)
    # rhs = op(x0)
    rhs = tt
    rhs_arr = rhs.contract().value
    xf, resid = gmres(op, rhs, x0, 1e-8, 1e-12, maxiter=20)
    # print("resid = ", resid)
    # assert resid < 1e-5

    xff = copy.deepcopy(xf)
    resid_tt = rhs + op(xff).scale(-1.0)
    print("resid norm = ", resid_tt.norm())
    
    check = op(xf).contract().value

    should_be = rhs.contract().value
    print("did rhs change = ", np.linalg.norm(rhs_arr - should_be))
    
    err = check - should_be
    resid_tt_arr = resid_tt.contract().value
    print("is resid correct = ", np.linalg.norm(resid_tt_arr - err))

    print("resid_tt norm = ", resid_tt.norm())
    print("err norm = ", np.linalg.norm(err))
    
    
    print("error = ", np.linalg.norm(err))
    print("should_be = ", np.linalg.norm(should_be))
    assert np.linalg.norm(err) < 1e-10
    
    # plt.show()


    


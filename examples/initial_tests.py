"""Tensor Network Solver capbailities and test"""
from pytens import *
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(4)

if __name__ == "__main__":
    
    x = Index('x', 5, 1)
    u = Index('u', 10, 1)
    v = Index('v', 20, 1)
    
    tt_ranks = [1, 2, 2, 1]    
    TT = rand_tt('A', [x, u, v], tt_ranks)
    TT.inspect()


    fig, axs = plt.subplots(1, 1)    
    pos = TT.draw_layout()
    TT.draw_nodes(pos, ax=axs)
    TT.draw_edges(pos, ax=axs)


    print("\n---------------\n")
    print("CONTRACTION EXAMPLE")
    
    # example of contracting, yields a tensor
    ttcon = TT.contract('o')
    ttcon.inspect()

    fig, axs = plt.subplots(1, 1)
    ttcon.draw(ax=axs)

    print("\n---------------\n")
    print("Vector example")
    w = vector('w', x, np.random.randn(x.size))
    fig, axs = plt.subplots(1, 1)        
    w.draw(ax=axs)
    
    print("\n---------------\n")
    print("Attaching example")

    # Another TT
    tt_ranks2 = [1, 3, 4, 1]
    TT2 = rand_tt('B', [x, u, v], tt_ranks2)
    TT3 = TT.attach(TT2)
    TT3.inspect()
    
    fig, axs = plt.subplots(1, 1)
    pos = nx.planar_layout(TT3.network)    
    TT3.draw_nodes(pos, ax=axs)
    TT3.draw_edges(pos, ax=axs)

    print("\n---------------\n")
    print("Partial Integration")
    int_partial = TT.integrate([v], np.ones(1)).as_tt()
    int_partial.draw()

    print("\n---------------\n")
    print("Addition ")
    tt_add = TT + TT2
    tt_add.inspect()


    print("\n---------------\n")
    print("Elementwise multiplication ")
    tt_mult = TT * TT2
    tt_mult.inspect()


    print("\n---------------\n")
    print("Right orthogonalization")
    TTc = copy.deepcopy(TT)
    arr1 = TTc.contract('e').value('e')

    TTc = TTc.right_orthogonalize('A_3')
    node = TTc.value('A_3')    
    TTc.inspect()
    TTc = TTc.right_orthogonalize('A_2')
    

    print("\n---------------\n")
    print("Renaming")
    TT1_renamed = copy.deepcopy(TT)
    TT1_renamed.rename('t')
    TT1_renamed.inspect()

    # print(np.sum(arr * arr))
    print("\n---------------\n")
    print("Rounding")
    TTadd = TT + TT1_renamed
    TTadd = TTadd.rename('added')
    TTadd.round(1e-10)


    print("\n---------------\n")
    print("TTop")
    x = Index('x', 10, 1)
    xp = Index('xp', 10, 1)
    y = Index('y', 5, 1)
    yp = Index('yp', 5, 1)
    z = Index('z', 3, 1)
    zp = Index('zp', 3, 1)

    indices_in = [x, y, z]
    indices_out = [xp, yp, zp]
    A = np.random.randn(10, 10)
    ttop = ttop_1dim(indices_in, indices_out, A)

    tt = rand_tt('A', [x, y, z], [1, 3, 2, 1])

    plt.close('all')
    fig, axs = plt.subplots(1, 1)
    ttop.draw(ax=axs)

    combined = ttop.attach(tt)
    fig, axs = plt.subplots(1, 1)
    combined.draw(ax=axs)
    
    ttout = ttop.apply(tt)
    fig, axs = plt.subplots(1, 1)
    ttout.draw(ax=axs)
    ttout.inspect()


    print("\n---------------\n")
    print("GMRES")
    
    x0 = rand_tt('x0', [x, y, z], [1, 3, 2, 1])
    op = lambda ttin: ttop.apply(ttin)
    xf, resid = gmres(op, tt, x0, 1e-5, 1e-5, maxiter=20)
    print("resid = ", resid)
    assert resid < 1e-5
    xf.inspect()

    check = ttop.apply(xf).contract('o').value('o')
    should_be = tt.contract('o').value('o')

    err = check - should_be
    print("error = ", np.linalg.norm(err))
    
    # plt.show()

    

    


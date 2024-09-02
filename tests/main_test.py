"""Tensor Network Solvers."""
from pytens import *
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import unittest

np.random.seed(4)

class TestIndex(unittest.TestCase):

    def test_equality(self):

        x = Index('x', 5, 1)
        y = Index('x', 5, 1)
        z = Index('z', 5, 1)
        self.assertEqual(x, y)
        self.assertNotEqual(x, z)

class TestTT(unittest.TestCase):

    def setUp(self):
        self.x = Index('x', 5, 1)
        self.u = Index('u', 10, 1)
        self.v = Index('v', 20, 1)
        self.tt_ranks = [1, 2, 2, 1]    
        self.TT = rand_tt('A', [self.x, self.u, self.v], self.tt_ranks)
        self.tt_ranks2 = [1, 3, 4, 1]
        self.TT2 = rand_tt('B', [self.x, self.u, self.v], self.tt_ranks2)
        
    def test_ranks(self):
        tt_ranks = self.TT.ranks()
        self.assertEqual(tt_ranks[0], self.tt_ranks[1])
        self.assertEqual(tt_ranks[1], self.tt_ranks[2])
        tt_ranks2 = self.TT2.ranks()
        self.assertEqual(tt_ranks2[0], self.tt_ranks2[1])
        self.assertEqual(tt_ranks2[1], self.tt_ranks2[2])        

    def test_contraction_and_index(self):
        ttcon = self.TT.contract('o')
        ttarr = ttcon.value('o')
        self.assertEqual(ttarr.ndim, 3)
        self.assertEqual(ttarr.shape[0], self.x.size)
        self.assertEqual(ttarr.shape[1], self.u.size)
        self.assertEqual(ttarr.shape[2], self.v.size)

        val = self.TT[2:4, 5, 3].value('eval')
        val_should_be = ttarr[2:4, 5, 3]

        self.assertTrue(np.allclose(val_should_be, val, atol=1e-14, rtol=1e-14))

    def test_inner(self):
        inner_val = self.TT.inner(self.TT2)
        out1 = self.TT.contract('a').value('a')
        out2 = self.TT2.contract('a').value('a')

        self.assertTrue(np.allclose(inner_val, np.sum(out1*out2), atol=1e-14, rtol=1e-14))

    def test_integrate(self):
        integral = self.TT.integrate([self.x, self.u, self.v], np.ones(3)).contract('i').value('i')
        ttarr = self.TT.contract('a').value('a')
        self.assertTrue(np.allclose(integral, np.sum(ttarr), atol=1e-14, rtol=1e-14))

        int_partial = self.TT.integrate([self.v], np.ones(1)).contract('i').value('i')
        self.assertEqual(int_partial.ndim, 2)
        self.assertEqual(int_partial.shape[0], self.x.size)
        self.assertEqual(int_partial.shape[1], self.u.size)
        self.assertTrue(np.allclose(int_partial, np.sum(ttarr, axis=2), atol=1e-14, rtol=1e-14))

    def test_addition(self):
        tt_add = self.TT + self.TT2
        sum1 = tt_add.contract('eval').value('eval')

        out1 = self.TT.contract('a').value('a')
        out2 = self.TT2.contract('a').value('a')
        self.assertTrue(np.allclose(sum1, out1 + out2, atol=1e-14, rtol=1e-14))
        ranks = tt_add.ranks()
        self.assertEqual(ranks[0], self.tt_ranks[1] + self.tt_ranks2[1])
        self.assertEqual(ranks[1], self.tt_ranks[2] + self.tt_ranks2[2])
        
    def test_multiplication(self):
        tt_mult = self.TT * self.TT2
        mult1 = tt_mult.contract('eval').value('eval')

        out1 = self.TT.contract('a').value('a')
        out2 = self.TT2.contract('a').value('a')
        self.assertTrue(np.allclose(mult1, out1 * out2, atol=1e-14, rtol=1e-14))
        
        ranks = tt_mult.ranks()
        self.assertEqual(ranks[0], self.tt_ranks[1] * self.tt_ranks2[1])
        self.assertEqual(ranks[1], self.tt_ranks[2] * self.tt_ranks2[2])

    def test_right_orthogonalization(self):
        TTc = copy.deepcopy(self.TT)
        arr1 = TTc.contract('e').value('e')
        
        TTc = TTc.right_orthogonalize('A_3')
        node = TTc.value('A_3')    
        
        check = np.dot(node, node.T)
        should_be = np.eye(self.tt_ranks[2])

        self.assertTrue(np.allclose(check, should_be, atol=1e-14, rtol=1e-14))

        arr2 = TTc.contract('e').value('e')
        self.assertTrue(np.allclose(arr1, arr2, atol=1e-14, rtol=1e-14))

        TTc = TTc.right_orthogonalize('A_2')
        node = TTc.value('A_2')        
        check = np.dot(node[:, 0, :], node[:, 0, :].T)
        for ii in range(1, node.shape[1]):
            check += np.dot(node[:, ii, :], node[:, ii, :].T)
        should_be = np.eye(self.tt_ranks[1])
        self.assertTrue(np.allclose(check, should_be, atol=1e-14, rtol=1e-14))

        arr2 = TTc.contract('e').value('e')
        self.assertTrue(np.allclose(arr1, arr2, atol=1e-14, rtol=1e-14))

    def test_rename(self):
        # Testing renaming by renaming and then inner.
        TT1_renamed = copy.deepcopy(self.TT)
        TT1_renamed.rename('t')
        TTadd = self.TT + TT1_renamed
        TTadd = TTadd.rename('added')
        
        ttarr = self.TT.contract('o').value('o')
        should_be = 2 * np.sum(ttarr * ttarr)
        check = self.TT.inner(TTadd)
        self.assertTrue(np.allclose(check, should_be, atol=1e-14, rtol=1e-14))

    def test_rounding(self):

        TT1_renamed = copy.deepcopy(self.TT)
        TT1_renamed.rename('t')
        TTadd = self.TT + TT1_renamed
        TTadd = TTadd.rename('added')
        
        ttadd = TTadd.contract('o').value('o')
        TTadd.round(1e-10)
        # exit(1)
        new_ranks = TTadd.ranks()

        self.assertTrue(new_ranks[0], self.tt_ranks[1])
        self.assertTrue(new_ranks[1], self.tt_ranks[2])

        ttadd_rounded = TTadd.contract('a').value('a')
        self.assertTrue(np.allclose(ttadd_rounded, ttadd, atol=1e-14, rtol=1e-14))

    def test_gmres(self):

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

        x0 = rand_tt('x0', [x, y, z], [1, 3, 2, 1])
        op = lambda ttin: ttop.apply(ttin)
        xf, resid = gmres(op, tt, x0, 1e-5, 1e-5, maxiter=30)
        print("resid = ", resid)
        self.assertTrue(resid < 1e-5)


        check = ttop.apply(xf).contract('o').value('o')
        should_be = tt.contract('o').value('o')

        self.assertTrue(np.allclose(check, should_be,
                                    atol=1e-5, rtol=1e-5))
        

        
if __name__ == "__main__":
    unittest.main()

"""Tensor Network Solvers."""
from pytens.algs import *
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import unittest

np.random.seed(4)

class TestIndex(unittest.TestCase):

    def test_equality(self):

        x = Index('x', 5)
        y = Index('x', 5)
        z = Index('z', 5)
        self.assertEqual(x, y)
        self.assertNotEqual(x, z)

class TestTT(unittest.TestCase):

    def setUp(self):
        self.x = Index('x', 5)
        self.u = Index('u', 10)
        self.v = Index('v', 20)
        self.tt_ranks = [2, 2] 
        self.TT = rand_tt([self.x, self.u, self.v], self.tt_ranks)
        self.tt_ranks2 = [3, 4]
        self.TT2 = rand_tt([self.x, self.u, self.v], self.tt_ranks2)

    def test_ranks(self):
        tt_ranks = self.TT.ranks()
        self.assertEqual(tt_ranks[0], self.tt_ranks[0])
        self.assertEqual(tt_ranks[1], self.tt_ranks[1])
        tt_ranks2 = self.TT2.ranks()
        self.assertEqual(tt_ranks2[0], self.tt_ranks2[0])
        self.assertEqual(tt_ranks2[1], self.tt_ranks2[1])

    def test_contraction_and_index(self):
        ttcon = self.TT.contract()
        ttarr = ttcon.value
        self.assertEqual(ttarr.ndim, 3)
        self.assertEqual(ttarr.shape[0], self.x.size)
        self.assertEqual(ttarr.shape[1], self.u.size)
        self.assertEqual(ttarr.shape[2], self.v.size)
        self.assertEqual(ttcon.indices[0], self.x)
        self.assertEqual(ttcon.indices[1], self.u)
        self.assertEqual(ttcon.indices[2], self.v)

        val = self.TT[2:4, 5:7, 3].value
        val_should_be = ttarr[2:4, 5:7, 3]

        self.assertTrue(np.allclose(val_should_be, val, atol=1e-14, rtol=1e-14))

    def test_inner(self):
        inner_val = self.TT.inner(self.TT2)
        out1 = self.TT.contract().value
        out2 = self.TT2.contract().value

        self.assertTrue(np.allclose(inner_val, np.sum(out1*out2), atol=1e-14, rtol=1e-14))

    def test_integrate(self):
        integral = self.TT.integrate([self.x, self.u, self.v], np.ones(3)).contract().value
        ttarr = self.TT.contract().value
        self.assertTrue(np.allclose(integral, np.sum(ttarr), atol=1e-14, rtol=1e-14))

        int_partial = self.TT.integrate([self.v], np.ones(1)).contract().value
        self.assertEqual(int_partial.ndim, 2)
        self.assertEqual(int_partial.shape[0], self.x.size)
        self.assertEqual(int_partial.shape[1], self.u.size)
        self.assertTrue(np.allclose(int_partial, np.sum(ttarr, axis=2), atol=1e-14, rtol=1e-14))

    def test_addition(self):
        # print("ADD")
        tt_add = self.TT + self.TT2
        # print("ttadd = ", tt_add)
        sum1 = tt_add.contract().value

        out1 = self.TT.contract().value
        out2 = self.TT2.contract().value
        err = sum1 - out1 - out2
        # print("error = ", np.linalg.norm(err), np.linalg.norm(out1+out2))
        self.assertTrue(np.allclose(sum1, out1 + out2, atol=1e-14, rtol=1e-14))
        ranks = tt_add.ranks()
        self.assertEqual(ranks[0], self.tt_ranks[0] + self.tt_ranks2[0])
        self.assertEqual(ranks[1], self.tt_ranks[1] + self.tt_ranks2[1])
        
    def test_multiplication(self):

        # print("\n MULTIPLICATION")
        # print(self.TT)
        tt_mult = self.TT * self.TT2
        mult1 = tt_mult.contract().value

        out1 = self.TT.contract().value
        out2 = self.TT2.contract().value
        self.assertTrue(np.allclose(mult1, out1 * out2, atol=1e-14, rtol=1e-14))
        
        ranks = tt_mult.ranks()
        self.assertEqual(2, len(ranks))
        self.assertEqual(ranks[0], self.tt_ranks[0] * self.tt_ranks2[0])
        self.assertEqual(ranks[1], self.tt_ranks[1] * self.tt_ranks2[1])

    def test_right_orthogonalization(self):
        TTc = copy.deepcopy(self.TT)
        arr1 = TTc.contract().value
        
        TTc = tt_right_orth(TTc, 2)
        node = TTc.value(2)    
        
        check = np.dot(node, node.T)
        should_be = np.eye(self.tt_ranks[1])

        self.assertTrue(np.allclose(check, should_be, atol=1e-14, rtol=1e-14))

        arr2 = TTc.contract().value
        self.assertTrue(np.allclose(arr1, arr2, atol=1e-14, rtol=1e-14))

        TTc = tt_right_orth(TTc, 1)
        node = TTc.value(1)        
        check = np.dot(node[:, 0, :], node[:, 0, :].T)
        for ii in range(1, node.shape[1]):
            check += np.dot(node[:, ii, :], node[:, ii, :].T)
        should_be = np.eye(self.tt_ranks[0])
        self.assertTrue(np.allclose(check, should_be, atol=1e-14, rtol=1e-14))

        arr2 = TTc.contract().value
        self.assertTrue(np.allclose(arr1, arr2, atol=1e-14, rtol=1e-14))

    def test_rounding(self):

        # print("\nROUNDING")
        TTadd = self.TT + self.TT
        # TTadd = TTadd.rename('added')
        
        # print(TTadd)
        ttadd = TTadd.contract().value
        TTadd =  tt_round(TTadd, 1e-10)
        # # exit(1)
        new_ranks = TTadd.ranks()

        self.assertTrue(new_ranks[0], self.tt_ranks[0])
        self.assertTrue(new_ranks[1], self.tt_ranks[1])

        ttadd_rounded = TTadd.contract().value
        self.assertTrue(np.allclose(ttadd_rounded, ttadd, atol=1e-14, rtol=1e-14))

    def test_scale(self):

        TT = copy.deepcopy(self.TT)
        TT.scale(2.0)

        tt_arr1 = self.TT.contract().value
        tt_arr2 = TT.contract().value
  
        self.assertTrue(np.allclose(2*tt_arr1, tt_arr2, atol=1e-14, rtol=1e-14))

    def test_ttop(self):
        # print("\nTTOP APPLY")
        x = Index('x', 10)
        xp = Index('xp', 10)
        y = Index('y', 5)
        yp = Index('yp', 5)
        z = Index('z', 3)
        zp = Index('zp', 3)

        indices_in = [x, y, z]
        indices_out = [xp, yp, zp]
        A = np.random.randn(10, 10)
        e1 = np.eye(5, 5)
        e2 = np.eye(3, 3)
        ttop = ttop_rank1(indices_in, indices_out, [A, e1, e2], "A")

        ttop_arr = ttop.contract().value
        # print(ttop_arr.shape)
        
        tt = rand_tt([x, y, z], [3, 2])
        tt_arr = tt.contract().value
        # print(tt_arr.shape)

        should_be = np.einsum('ijklmn,jln->ikm', ttop_arr, tt_arr)
        check = ttop_apply(ttop, tt).contract().value

        self.assertTrue(np.allclose(check, should_be,
                                    atol=1e-14, rtol=1e-14))


        A1 = np.random.randn(10, 10)
        A2 = np.random.randn(10, 10)
        e1 = np.eye(5, 5)
        e2 = np.random.randn(5, 5)
        f1 = np.eye(3, 3)
        f2 = np.random.randn(3, 3)
        
        ttop = ttop_rank2(indices_in, indices_out,
                          [A1, e1, f1],
                          [A2, e2, f2],
                          "A")
        
        ttop_arr = ttop.contract().value
        # print(ttop_arr.shape)
        
        tt = rand_tt([x, y, z], [3, 2])
        tt_arr = tt.contract().value
        # print(tt_arr.shape)

        should_be = np.einsum('ijklmn,jln->ikm', ttop_arr, tt_arr)
        check = ttop_apply(ttop, tt).contract().value

        err = np.linalg.norm(should_be - check)
        # print("error = ", err)
        self.assertTrue(np.allclose(check, should_be,
                                    atol=1e-14, rtol=1e-14))        

    # @unittest.skip('other stuff first')
    def test_gmres(self):

        x = Index('x', 10)
        xp = Index('xp', 10)
        y = Index('y', 5)
        yp = Index('yp', 5)
        z = Index('z', 3)
        zp = Index('zp', 3)

        indices_in = [x, y, z]
        indices_out = [xp, yp, zp]
        A = np.random.randn(10, 10)
        ttop = ttop_rank1(indices_in, indices_out, [A, np.eye(5, 5), np.eye(3, 3)],
                          "A")
        tt = rand_tt([x, y, z], [3, 2])

        x0 = rand_tt([x, y, z], [3, 2])
        op = lambda ttin: ttop_apply(ttop, ttin)
        xf, resid = gmres(op, tt, x0, 1e-5, 1e-10, maxiter=30)
        # print("resid = ", resid)
        self.assertTrue(resid < 1e-5)


    #     check = ttop.apply(xf).contract('o').value('o')
    #     should_be = tt.contract('o').value('o')

    #     self.assertTrue(np.allclose(check, should_be,
    #                                 atol=1e-5, rtol=1e-5))
        

        
if __name__ == "__main__":
    unittest.main()

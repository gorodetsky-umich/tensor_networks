"""Tensor Network Solvers."""

import copy
import os
import unittest
import tempfile
import pickle

import numpy as np

from pytens.algs import *
from pytens.cross.cross import (
    CrossAlgo,
    CrossApproximation,
    CrossConfig,
    ConvergenceCheck,
)
from pytens.types import Index
from pytens.cross.funcs import CachedFunc
from tests.search_test import *

np.random.seed(4)


class TestIndex(unittest.TestCase):
    def test_equality(self):
        x = Index("x", 5)
        y = Index("x", 5)
        z = Index("z", 5)
        self.assertEqual(x, y)
        self.assertNotEqual(x, z)


class TestTT(unittest.TestCase):
    def setUp(self):
        self.x = Index("t", 5)
        self.u = Index("u", 10)
        self.v = Index("v", 20)
        self.tt_ranks = [2, 2]
        self.TT = TensorNetwork.rand_tt(
            [self.x, self.u, self.v], self.tt_ranks
        )
        self.tt_ranks2 = [3, 4]
        self.TT2 = TensorNetwork.rand_tt(
            [self.x, self.u, self.v], self.tt_ranks2
        )

    def test_save(self):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "test")
            with open(fname, "wb") as fp:
                out = pickle.dump(self.TT, fp, pickle.HIGHEST_PROTOCOL)

            with open(fname, mode="rb") as f:
                new_tt = pickle.load(f)
                tt_ranks = new_tt.ranks()
                self.assertEqual(tt_ranks[0], self.tt_ranks[0])
                self.assertEqual(tt_ranks[1], self.tt_ranks[1])

                eval_here = new_tt[0, 2, 4].value
                eval_orig = self.TT[0, 2, 4].value
                err = np.abs(eval_here - eval_orig)
                self.assertTrue(err < 1e-14)

    def test_to_dict(self):
        tt_dict = self.TT.to_dict()

        new_tt = TensorNetwork.from_dict(tt_dict)

        tt_ranks = new_tt.ranks()
        self.assertEqual(tt_ranks[0], self.tt_ranks[0])
        self.assertEqual(tt_ranks[1], self.tt_ranks[1])

        eval_here = new_tt[0, 2, 4].value
        eval_orig = self.TT[0, 2, 4].value
        err = np.abs(eval_here - eval_orig)
        self.assertTrue(err < 1e-14)

    def test_to_separated_dict(self):
        metadata, numpy_arrays = self.TT.to_separated_dict()

        new_tt = TensorNetwork.from_separated_dict(metadata, numpy_arrays)

        tt_ranks = new_tt.ranks()
        self.assertEqual(tt_ranks[0], self.tt_ranks[0])
        self.assertEqual(tt_ranks[1], self.tt_ranks[1])

        eval_here = new_tt[0, 2, 4].value
        eval_orig = self.TT[0, 2, 4].value
        err = np.abs(eval_here - eval_orig)
        self.assertTrue(err < 1e-14)

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

        self.assertTrue(
            np.allclose(val_should_be, val, atol=1e-14, rtol=1e-14)
        )

    def test_inner(self):
        inner_val = self.TT.inner(self.TT2)
        out1 = self.TT.contract().value
        out2 = self.TT2.contract().value

        self.assertTrue(
            np.allclose(inner_val, np.sum(out1 * out2), atol=1e-14, rtol=1e-14)
        )

    def test_integrate(self):
        integral = (
            self.TT.integrate([self.x, self.u, self.v], np.ones(3))
            .contract()
            .value
        )
        ttarr = self.TT.contract().value
        self.assertTrue(
            np.allclose(integral, np.sum(ttarr), atol=1e-14, rtol=1e-14)
        )

        int_partial = self.TT.integrate([self.v], np.ones(1)).contract().value
        self.assertEqual(int_partial.ndim, 2)
        self.assertEqual(int_partial.shape[0], self.x.size)
        self.assertEqual(int_partial.shape[1], self.u.size)
        self.assertTrue(
            np.allclose(
                int_partial, np.sum(ttarr, axis=2), atol=1e-14, rtol=1e-14
            )
        )

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

    def test_sum_multiple_tt(self):
        TT1 = TensorNetwork.rand_tt([self.x, self.u, self.v], [2, 2])
        TT2 = TensorNetwork.rand_tt([self.x, self.u, self.v], [4, 3])
        TT3 = TensorNetwork.rand_tt([self.x, self.u, self.v], [8, 12])
        TT4 = TensorNetwork.rand_tt([self.x, self.u, self.v], [3, 4])

        tts = [TT1, TT2, TT3, TT4]
        tt_sum_1 = tt_sum(tts)
        tt_sum_2 = TT1 + TT2 + TT3 + TT4

        out1 = tt_sum_1.contract().value
        out2 = tt_sum_2.contract().value
        # err = out1 - out2
        # print("error = ", np.linalg.norm(err), np.linalg.norm(out1+out2))
        self.assertTrue(np.allclose(out1, out2, atol=1e-14, rtol=1e-14))
        ranks = tt_sum_1.ranks()
        self.assertEqual(ranks[0], 2 + 4 + 8 + 3)
        self.assertEqual(ranks[1], 2 + 3 + 12 + 4)

    def test_multiplication(self):
        # print("\n MULTIPLICATION")
        # print(self.TT)
        tt_mult = self.TT * self.TT2
        mult1 = tt_mult.contract().value

        out1 = self.TT.contract().value
        out2 = self.TT2.contract().value
        self.assertTrue(
            np.allclose(mult1, out1 * out2, atol=1e-14, rtol=1e-14)
        )

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
        TTadd = tt_svd_round(TTadd, 1e-5)
        # # exit(1)
        new_ranks = TTadd.ranks()

        self.assertTrue(new_ranks[0], self.tt_ranks[0])
        self.assertTrue(new_ranks[1], self.tt_ranks[1])

        ttadd_rounded = TTadd.contract().value
        self.assertTrue(
            np.allclose(ttadd_rounded, ttadd, atol=1e-13, rtol=1e-13)
        )

    def test_gramsvd_rounding(self):
        # print("\nROUNDING")
        TTadd = self.TT + self.TT
        # TTadd = TTadd.rename('added')

        # print(TTadd)
        ttadd = TTadd.contract().value
        TTadd = tt_gramsvd_round(TTadd, 1e-5)
        # # exit(1)
        new_ranks = TTadd.ranks()

        self.assertTrue(new_ranks[0], self.tt_ranks[0])
        self.assertTrue(new_ranks[1], self.tt_ranks[1])

        ttadd_rounded = TTadd.contract().value
        self.assertTrue(
            np.allclose(ttadd_rounded, ttadd, atol=1e-13, rtol=1e-13)
        )

    def test_gram_rounding_ttsum(self):
        # print("\nROUNDING")
        s = 3
        TTadd = self.TT
        for _ in range(s - 1):
            TTadd = TTadd + self.TT

        sum_list = [copy.deepcopy(self.TT) for _ in range(s)]

        # TTadd = TTadd.rename('added')

        # print(TTadd)
        ttadd = TTadd.contract().value
        TTadd = tt_sum_gramsvd_round(sum_list, 1e-5)
        # # exit(1)
        new_ranks = TTadd.ranks()

        self.assertTrue(new_ranks[0], self.tt_ranks[0])
        self.assertTrue(new_ranks[1], self.tt_ranks[1])

        ttadd_rounded = TTadd.contract().value
        self.assertTrue(
            np.allclose(ttadd_rounded, ttadd, atol=1e-13, rtol=1e-13)
        )

    def test_rand_rounding(self):
        # print("\nROUNDING")
        TTadd = self.TT + self.TT

        # Target ranks
        target = [2, 2]

        # print(TTadd)
        ttadd = TTadd.contract().value

        TTadd = tt_randomized_round(y=TTadd, target_ranks=target)

        # # exit(1)
        new_ranks = TTadd.ranks()

        self.assertTrue(new_ranks[0], self.tt_ranks[0])
        self.assertTrue(new_ranks[1], self.tt_ranks[1])

        ttadd_rounded = TTadd.contract().value
        self.assertTrue(
            np.allclose(ttadd_rounded, ttadd, atol=1e-13, rtol=1e-13)
        )

    def test_rand_rounding_ttsum(self):
        # print("\nROUNDING")
        s = 3
        TTadd = self.TT
        for _ in range(s - 1):
            TTadd = TTadd + self.TT

        sum_list = [copy.deepcopy(self.TT) for _ in range(s)]

        # Target ranks
        target = [2, 2]

        # print(TTadd)
        ttadd = TTadd.contract().value

        TTadd = tt_sum_randomized_round(y=sum_list, target_ranks=target)

        # # exit(1)
        new_ranks = TTadd.ranks()

        self.assertTrue(new_ranks[0], self.tt_ranks[0])
        self.assertTrue(new_ranks[1], self.tt_ranks[1])

        ttadd_rounded = TTadd.contract().value
        self.assertTrue(
            np.allclose(ttadd_rounded, ttadd, atol=1e-13, rtol=1e-13)
        )

    def test_scale(self):
        TT = copy.deepcopy(self.TT)
        TT.scale(2.0)

        tt_arr1 = self.TT.contract().value
        tt_arr2 = TT.contract().value

        self.assertTrue(
            np.allclose(2 * tt_arr1, tt_arr2, atol=1e-14, rtol=1e-14)
        )

    def test_ttop(self):
        # print("\nTTOP APPLY")
        x = Index("x", 10)
        xp = Index("xp", 10)
        y = Index("y", 5)
        yp = Index("yp", 5)
        z = Index("z", 3)
        zp = Index("zp", 3)

        indices_in = [x, y, z]
        indices_out = [xp, yp, zp]
        A = np.random.randn(10, 10)
        e1 = np.eye(5, 5)
        e2 = np.eye(3, 3)
        ttop = ttop_rank1(indices_in, indices_out, [A, e1, e2], "A")

        ttop_arr = ttop.contract().value
        # print(ttop_arr.shape)

        tt = TensorNetwork.rand_tt([x, y, z], [3, 2])
        tt_arr = tt.contract().value
        # print(tt_arr.shape)

        should_be = np.einsum("ijklmn,jln->ikm", ttop_arr, tt_arr)
        check = ttop_apply(ttop, tt).contract().value

        self.assertTrue(np.allclose(check, should_be, atol=1e-14, rtol=1e-14))

        A1 = np.random.randn(10, 10)
        A2 = np.random.randn(10, 10)
        e1 = np.eye(5, 5)
        e2 = np.random.randn(5, 5)
        f1 = np.eye(3, 3)
        f2 = np.random.randn(3, 3)

        ttop = ttop_rank2(
            indices_in, indices_out, [A1, e1, f1], [A2, e2, f2], "A"
        )

        ttop_arr = ttop.contract().value
        # print(ttop_arr.shape)

        tt = TensorNetwork.rand_tt([x, y, z], [3, 2])
        tt_arr = tt.contract().value
        # print(tt_arr.shape)

        should_be = np.einsum("ijklmn,jln->ikm", ttop_arr, tt_arr)
        check = ttop_apply(ttop, tt).contract().value

        err = np.linalg.norm(should_be - check)
        # print("error = ", err)
        self.assertTrue(np.allclose(check, should_be, atol=1e-13, rtol=1e-13))

        out = ttop_sum_apply(
            tt,
            indices_in,
            indices_out,
            [
                [
                    lambda v: np.dot(A1, v),
                    lambda v: np.einsum("jk,mkp->mjp", e1, v),
                    lambda v: np.einsum("ij,mj->mi", f1, v),
                ],
                [
                    lambda v: np.dot(A2, v),
                    lambda v: np.einsum("jk,mkp->mjp", e2, v),
                    lambda v: np.einsum("ij,mj->mi", f2, v),
                ],
            ],
            "A",
        )
        check2 = out.contract().value
        err2 = np.linalg.norm(should_be - check2)
        self.assertTrue(np.allclose(check2, should_be, atol=1e-13, rtol=1e-13))
        # print("out = ", out)

    # @unittest.skip('other stuff first')
    def test_gmres(self):
        x = Index("x", 10)
        xp = Index("xp", 10)
        y = Index("y", 5)
        yp = Index("yp", 5)
        z = Index("z", 3)
        zp = Index("zp", 3)

        indices_in = [x, y, z]
        indices_out = [xp, yp, zp]
        A = np.random.randn(10, 10)
        ttop = ttop_rank1(
            indices_in, indices_out, [A, np.eye(5, 5), np.eye(3, 3)], "A"
        )
        tt = TensorNetwork.rand_tt([x, y, z], [3, 2])

        x0 = TensorNetwork.rand_tt([x, y, z], [3, 2])
        op = lambda ttin: ttop_apply(ttop, ttin)
        _, resid = gmres(op, tt, x0, 1e-5, 1e-10, maxiter=30)
        # print("resid = ", resid)
        self.assertTrue(resid < 1e-5)

    #     check = ttop.apply(xf).contract('o').value('o')
    #     should_be = tt.contract('o').value('o')

    #     self.assertTrue(np.allclose(check, should_be,
    #                                 atol=1e-5, rtol=1e-5))

    def test_optimize(self):
        # print("\nROUNDING")
        TTadd = self.TT + self.TT
        # TTadd = TTadd.rename('added')

        # print(TTadd)
        indices = TTadd.free_indices()
        ttadd = TTadd.contract().value
        TTadd.round(0, 1e-5)
        # # exit(1)
        new_ranks = TTadd.ranks()

        self.assertTrue(new_ranks[0], self.tt_ranks[0])
        self.assertTrue(new_ranks[1], self.tt_ranks[1])

        ttadd_rounded = TTadd.contract()
        rounded_indices = TTadd.free_indices()
        perm = [rounded_indices.index(ind) for ind in indices]
        ttadd_rounded = ttadd_rounded.permute(perm).value
        self.assertTrue(
            np.allclose(ttadd_rounded, ttadd, atol=1e-12, rtol=1e-12)
        )


class TestTree(unittest.TestCase):
    def setUp(self):
        np.random.seed(100)
        self.x = Index("x", 5)
        self.u = Index("u", 10)
        self.v = Index("v", 20)
        self.tree = rand_tree([self.x, self.u, self.v], [1, 2, 3, 4, 5])

    def test_tree_split(self):
        original = self.tree.contract().value
        original_free = self.tree.free_indices()

        self.tree.svd(4, [0, 2])
        after_split = self.tree.contract().value
        after_split_free = self.tree.free_indices()
        permutation = [after_split_free.index(i) for i in original_free]
        after_split = after_split.transpose(permutation)

        self.assertTrue(
            np.allclose(original, after_split, atol=1e-5, rtol=1e-5)
        )

    def test_tree_split_free(self):
        original = self.tree.contract().value
        original_free = self.tree.free_indices()

        self.tree.svd(3, [0, 1])
        after_split = self.tree.contract().value
        after_split_free = self.tree.free_indices()
        permutation = [after_split_free.index(i) for i in original_free]
        after_split = after_split.transpose(permutation)

        self.assertTrue(
            np.allclose(original, after_split, atol=1e-5, rtol=1e-5)
        )

    def test_tree_merge(self):
        original = self.tree.contract().value
        original_free = self.tree.free_indices()

        self.tree.merge(2, 3)
        after_merge = self.tree.contract().value
        after_merge_free = self.tree.free_indices()
        permutation = [after_merge_free.index(i) for i in original_free]
        after_merge = after_merge.transpose(permutation)

        self.assertTrue(
            np.allclose(original, after_merge, atol=1e-5, rtol=1e-5)
        )

    def test_tree_orthonorm(self):
        original = self.tree.contract().value
        original_indices = self.tree.free_indices()

        root = 3
        root = self.tree.orthonormalize(root)
        after_orthonormal = self.tree.contract().value
        after_orthonormal_indices = self.tree.free_indices()
        permutation = [
            after_orthonormal_indices.index(i) for i in original_indices
        ]
        after_orthonormal = after_orthonormal.transpose(permutation)

        self.assertTrue(
            np.allclose(after_orthonormal, original, atol=1e-5, rtol=1e-5)
        )

        # check each neighbor of root becomes isometry
        nbrs = list(self.tree.network.neighbors(root))
        for n in nbrs:
            self.tree.network.remove_edge(root, n)
            reachable_nodes = nx.descendants(self.tree.network, n)
            reachable_graph = self.tree.network.subgraph(
                [n] + list(reachable_nodes)
            )
            subnet = TensorNetwork()
            subnet.network = reachable_graph
            self.assertTrue(subnet.norm(), 1)

            self.tree.network.add_edge(root, n)

    def test_tree_canonicalize(self):
        x = np.random.randn(3, 4, 5)
        single_node1 = TensorNetwork()
        indices1 = [Index("i", 3), Index("j", 4), Index("k", 5)]
        single_node1.add_node("x", Tensor(x, indices1))

        single_node2 = TensorNetwork()
        indices2 = [Index("j", 4), Index("i", 3), Index("k", 5)]
        single_node2.add_node("y", Tensor(x.transpose(1, 0, 2), indices2))

        self.assertEqual(
            single_node1.canonical_structure(),
            single_node2.canonical_structure(),
        )

        # test symmetry
        tree1 = TensorNetwork()
        u = np.random.randn(2, 3, 4)
        u_indices = [Index("iu", 2), Index("ju", 3), Index("ku", 4)]
        v = np.random.randn(4, 5, 6)
        v_indices = [Index("iv", 4), Index("jv", 5), Index("kv", 6)]
        root = np.random.randn(2, 4, 3)
        root_indices = [Index("iu", 2), Index("iv", 4), Index("f", 3)]
        tree1.add_node("root", Tensor(root, root_indices))
        tree1.add_node("u", Tensor(u, u_indices))
        tree1.add_node("v", Tensor(v, v_indices))
        tree1.add_edge("root", "u")
        tree1.add_edge("root", "v")

        tree2 = TensorNetwork()
        root_indices2 = [Index("iv", 4), Index("iu", 2), Index("f", 3)]
        tree2.add_node("root", Tensor(root.transpose(1, 0, 2), root_indices2))
        u_indices2 = [
            Index("ju", 3),
            Index("ku", 4),
            Index("iu", 2),
        ]
        tree2.add_node("u", Tensor(u.transpose(1, 2, 0), u_indices2))
        v_indices2 = [
            Index("kv", 6),
            Index("iv", 4),
            Index("jv", 5),
        ]
        tree2.add_node("v", Tensor(v.transpose(2, 0, 1), v_indices2))
        tree2.add_edge("root", "u")
        tree2.add_edge("root", "v")

        self.assertEqual(
            tree1.canonical_structure(), tree2.canonical_structure()
        )

        tt1 = TensorNetwork()
        u1 = np.random.randn(2, 3)
        u1_indices = [Index("iu", 2), Index("uv", 3)]
        v1 = np.random.randn(3, 4, 5)
        v1_indices = [Index("uv", 3), Index("jv", 4), Index("vw", 5)]
        w1 = np.random.randn(5, 6)
        w1_indices = [Index("vw", 5), Index("jw", 6)]
        tt1.add_node("u", Tensor(u1, u1_indices))
        tt1.add_node("v", Tensor(v1, v1_indices))
        tt1.add_node("w", Tensor(w1, w1_indices))
        tt1.add_edge("u", "v")
        tt1.add_edge("v", "w")

        tt2 = TensorNetwork()
        u2 = np.random.randn(4, 3)
        u2_indices = [Index("iu", 4), Index("uv", 3)]
        v2 = np.random.randn(3, 2, 5)
        v2_indices = [Index("uv", 3), Index("jv", 2), Index("vw", 5)]
        w2 = np.random.randn(5, 6)
        w2_indices = [Index("vw", 5), Index("jw", 6)]
        tt2.add_node("u", Tensor(u2, u2_indices))
        tt2.add_node("v", Tensor(v2, v2_indices))
        tt2.add_node("w", Tensor(w2, w2_indices))
        tt2.add_edge("u", "v")
        tt2.add_edge("v", "w")

        self.assertNotEqual(
            tt1.canonical_structure(), tt2.canonical_structure()
        )

    def test_add1(self):
        x = np.random.randn(2, 13, 14)
        x_tensor = Tensor(x, [Index("a", 2), Index("i", 13), Index("j", 14)])
        u = np.random.randn(2, 15)
        u_tensor = Tensor(u, [Index("a", 2), Index("k", 15)])
        net1 = TensorNetwork()
        net1.add_node("x", x_tensor)
        net1.add_node("u", u_tensor)
        net1.add_edge("x", "u")
        t1 = net1.contract()

        y = np.random.randn(3, 13, 14)
        y_tensor = Tensor(y, [Index("b", 3), Index("i", 13), Index("j", 14)])
        v = np.random.randn(3, 15)
        v_tensor = Tensor(v, [Index("b", 3), Index("k", 15)])
        net2 = TensorNetwork()
        net2.add_node("y", y_tensor)
        net2.add_node("v", v_tensor)
        net2.add_edge("y", "v")
        t2 = net2.contract()

        t12 = t1.value + t2.value
        t12_net = net1 + net2
        t12_net.round("x", t12_net.norm() * 1e-10)
        # network free inds are changed after rounding
        tensor12 = t12_net.contract()
        perm = [tensor12.indices.index(ind) for ind in net1.free_indices()]
        tensor12 = tensor12.permute(perm)
        self.assertTrue(np.allclose(tensor12.value, t12))

    def test_add2(self):
        x = np.random.randn(1, 2, 3, 4)
        x_tensor = Tensor(
            x, [Index("a", 1), Index("b", 2), Index("c", 3), Index("d", 4)]
        )
        u1 = np.random.randn(1, 13)
        u1_tensor = Tensor(u1, [Index("a", 1), Index("i", 13)])
        u2 = np.random.randn(2, 14)
        u2_tensor = Tensor(u2, [Index("b", 2), Index("j", 14)])
        u3 = np.random.randn(3, 15)
        u3_tensor = Tensor(u3, [Index("c", 3), Index("k", 15)])
        u4 = np.random.randn(4, 16)
        u4_tensor = Tensor(u4, [Index("d", 4), Index("l", 16)])
        net1 = TensorNetwork()
        net1.add_node("x", x_tensor)
        net1.add_node("u1", u1_tensor)
        net1.add_node("u2", u2_tensor)
        net1.add_node("u3", u3_tensor)
        net1.add_node("u4", u4_tensor)
        net1.add_edge("x", "u1")
        net1.add_edge("x", "u2")
        net1.add_edge("x", "u3")
        net1.add_edge("x", "u4")
        t1 = net1.contract()

        y = np.random.randn(2, 3, 4, 5)
        y_tensor = Tensor(
            y, [Index("e", 2), Index("f", 3), Index("g", 4), Index("h", 5)]
        )
        v1 = np.random.randn(2, 13)
        v1_tensor = Tensor(v1, [Index("e", 2), Index("i", 13)])
        v2 = np.random.randn(3, 14)
        v2_tensor = Tensor(v2, [Index("f", 3), Index("j", 14)])
        v3 = np.random.randn(4, 15)
        v3_tensor = Tensor(v3, [Index("g", 4), Index("k", 15)])
        v4 = np.random.randn(5, 16)
        v4_tensor = Tensor(v4, [Index("h", 5), Index("l", 16)])
        net2 = TensorNetwork()
        net2.add_node("y", y_tensor)
        net2.add_node("v1", v1_tensor)
        net2.add_node("v2", v2_tensor)
        net2.add_node("v3", v3_tensor)
        net2.add_node("v4", v4_tensor)
        net2.add_edge("y", "v1")
        net2.add_edge("y", "v2")
        net2.add_edge("y", "v3")
        net2.add_edge("y", "v4")
        t2 = net2.contract()

        t12 = t1.value + t2.value
        net12 = net1 + net2
        net12.round("x", net12.norm() * 1e-10)
        # network free inds are changed after rounding
        tensor12 = net12.contract()
        perm = [tensor12.indices.index(ind) for ind in net1.free_indices()]
        tensor12 = tensor12.permute(perm)
        self.assertTrue(
            np.allclose(t12, tensor12.value, rtol=1e-10, atol=1e-10)
        )

    def test_add3(self):
        x = np.random.randn(13, 14, 2, 5)
        x_tensor = Tensor(
            x, [Index("i", 13), Index("j", 14), Index("a", 2), Index("b", 5)]
        )
        u1 = np.random.randn(2, 15)
        u1_tensor = Tensor(u1, [Index("d", 2), Index("k", 15)])
        u2 = np.random.randn(5, 16)
        u2_tensor = Tensor(u2, [Index("b", 5), Index("m", 16)])
        u3 = np.random.randn(2, 3, 2)
        u3_tensor = Tensor(u3, [Index("a", 2), Index("c", 3), Index("d", 2)])
        u4 = np.random.randn(3, 17)
        u4_tensor = Tensor(u4, [Index("c", 3), Index("l", 17)])
        net1 = TensorNetwork()
        net1.add_node("x", x_tensor)
        net1.add_node("u1", u1_tensor)
        net1.add_node("u2", u2_tensor)
        net1.add_node("u3", u3_tensor)
        net1.add_node("u4", u4_tensor)
        net1.add_edge("x", "u3")
        net1.add_edge("x", "u2")
        net1.add_edge("u3", "u1")
        net1.add_edge("u3", "u4")
        t1 = net1.contract()

        y = np.random.randn(13, 14, 1, 2)
        y_tensor = Tensor(
            y, [Index("i", 13), Index("j", 14), Index("aa", 1), Index("bb", 2)]
        )
        v1 = np.random.randn(3, 15)
        v1_tensor = Tensor(v1, [Index("dd", 3), Index("k", 15)])
        v2 = np.random.randn(2, 16)
        v2_tensor = Tensor(v2, [Index("bb", 2), Index("m", 16)])
        v3 = np.random.randn(1, 2, 3)
        v3_tensor = Tensor(
            v3, [Index("aa", 1), Index("cc", 2), Index("dd", 3)]
        )
        v4 = np.random.randn(2, 17)
        v4_tensor = Tensor(v4, [Index("cc", 2), Index("l", 17)])
        net2 = TensorNetwork()
        net2.add_node("y", y_tensor)
        net2.add_node("v1", v1_tensor)
        net2.add_node("v2", v2_tensor)
        net2.add_node("v3", v3_tensor)
        net2.add_node("v4", v4_tensor)
        net2.add_edge("y", "v2")
        net2.add_edge("y", "v3")
        net2.add_edge("v3", "v1")
        net2.add_edge("v3", "v4")
        t2 = net2.contract()

        t12 = t1.value + t2.value
        net12 = net1 + net2
        net12.round("x", net12.norm() * 1e-10)
        # network free inds are changed after rounding
        tensor12 = net12.contract()
        perm = [tensor12.indices.index(ind) for ind in net1.free_indices()]
        tensor12 = tensor12.permute(perm)
        self.assertTrue(
            np.allclose(t12, tensor12.value, rtol=1e-10, atol=1e-10)
        )

    def test_add4(self):
        x = np.random.randn(13, 14, 2, 5)
        x_tensor = Tensor(
            x, [Index("i", 13), Index("j", 14), Index("a", 2), Index("b", 5)]
        )
        u1 = np.random.randn(2, 15)
        u1_tensor = Tensor(u1, [Index("d", 2), Index("k", 15)])
        u2 = np.random.randn(5, 16)
        u2_tensor = Tensor(u2, [Index("b", 5), Index("m", 16)])
        u3 = np.random.randn(2, 3, 2)
        u3_tensor = Tensor(u3, [Index("a", 2), Index("c", 3), Index("d", 2)])
        u4 = np.random.randn(3, 17)
        u4_tensor = Tensor(u4, [Index("c", 3), Index("l", 17)])
        net1 = TensorNetwork()
        net1.add_node("x", x_tensor)
        net1.add_node("u1", u1_tensor)
        net1.add_node("u2", u2_tensor)
        net1.add_node("u3", u3_tensor)
        net1.add_node("u4", u4_tensor)
        net1.add_edge("x", "u3")
        net1.add_edge("x", "u2")
        net1.add_edge("u3", "u1")
        net1.add_edge("u3", "u4")
        t1 = net1.contract()

        t11 = t1.value + t1.value
        net11 = net1 + net1
        net11.round("x", net11.norm() * 1e-10)
        # network free inds are changed after rounding
        tensor11 = net11.contract()
        perm = [tensor11.indices.index(ind) for ind in net1.free_indices()]
        tensor11 = tensor11.permute(perm)
        self.assertTrue(
            np.allclose(tensor11.value, t11, rtol=1e-10, atol=1e-10)
        )
        self.assertEqual(net11.get_contraction_index("u3", "u1")[0].size, 2)
        self.assertEqual(net11.get_contraction_index("x", "u2")[0].size, 5)
        self.assertEqual(net11.get_contraction_index("x", "u3")[0].size, 2)
        self.assertEqual(net11.get_contraction_index("u4", "u3")[0].size, 3)

    def test_mul1(self):
        x = np.random.randn(2, 13, 14)
        x_tensor = Tensor(x, [Index("a", 2), Index("i", 13), Index("j", 14)])
        u = np.random.randn(2, 15)
        u_tensor = Tensor(u, [Index("a", 2), Index("k", 15)])
        net1 = TensorNetwork()
        net1.add_node("x", x_tensor)
        net1.add_node("u", u_tensor)
        net1.add_edge("x", "u")
        t1 = net1.contract()

        y = np.random.randn(3, 13, 14)
        y_tensor = Tensor(y, [Index("b", 3), Index("i", 13), Index("j", 14)])
        v = np.random.randn(3, 15)
        v_tensor = Tensor(v, [Index("b", 3), Index("k", 15)])
        net2 = TensorNetwork()
        net2.add_node("y", y_tensor)
        net2.add_node("v", v_tensor)
        net2.add_edge("y", "v")
        t2 = net2.contract()

        t12 = t1.value * t2.value
        t12_net = net1 * net2
        t12_net.round("x", t12_net.norm() * 1e-10)
        # network free inds are changed after rounding
        tensor12 = t12_net.contract()
        perm = [tensor12.indices.index(ind) for ind in net1.free_indices()]
        tensor12 = tensor12.permute(perm)
        self.assertTrue(
            np.allclose(tensor12.value, t12, rtol=1e-10, atol=1e-10)
        )

    def test_mul2(self):
        x = np.random.randn(1, 2, 3, 4)
        x_tensor = Tensor(
            x, [Index("a", 1), Index("b", 2), Index("c", 3), Index("d", 4)]
        )
        u1 = np.random.randn(1, 13)
        u1_tensor = Tensor(u1, [Index("a", 1), Index("i", 13)])
        u2 = np.random.randn(2, 14)
        u2_tensor = Tensor(u2, [Index("b", 2), Index("j", 14)])
        u3 = np.random.randn(3, 15)
        u3_tensor = Tensor(u3, [Index("c", 3), Index("k", 15)])
        u4 = np.random.randn(4, 16)
        u4_tensor = Tensor(u4, [Index("d", 4), Index("l", 16)])
        net1 = TensorNetwork()
        net1.add_node("x", x_tensor)
        net1.add_node("u1", u1_tensor)
        net1.add_node("u2", u2_tensor)
        net1.add_node("u3", u3_tensor)
        net1.add_node("u4", u4_tensor)
        net1.add_edge("x", "u1")
        net1.add_edge("x", "u2")
        net1.add_edge("x", "u3")
        net1.add_edge("x", "u4")
        t1 = net1.contract()

        y = np.random.randn(2, 3, 4, 5)
        y_tensor = Tensor(
            y, [Index("e", 2), Index("f", 3), Index("g", 4), Index("h", 5)]
        )
        v1 = np.random.randn(2, 13)
        v1_tensor = Tensor(v1, [Index("e", 2), Index("i", 13)])
        v2 = np.random.randn(3, 14)
        v2_tensor = Tensor(v2, [Index("f", 3), Index("j", 14)])
        v3 = np.random.randn(4, 15)
        v3_tensor = Tensor(v3, [Index("g", 4), Index("k", 15)])
        v4 = np.random.randn(5, 16)
        v4_tensor = Tensor(v4, [Index("h", 5), Index("l", 16)])
        net2 = TensorNetwork()
        net2.add_node("y", y_tensor)
        net2.add_node("v1", v1_tensor)
        net2.add_node("v2", v2_tensor)
        net2.add_node("v3", v3_tensor)
        net2.add_node("v4", v4_tensor)
        net2.add_edge("y", "v1")
        net2.add_edge("y", "v2")
        net2.add_edge("y", "v3")
        net2.add_edge("y", "v4")
        t2 = net2.contract()

        t12 = t1.value * t2.value
        net12 = net1 * net2
        net12.round("x", net12.norm() * 1e-10)
        # print(net12)
        # network free inds are changed after rounding
        tensor12 = net12.contract()
        perm = [tensor12.indices.index(ind) for ind in net1.free_indices()]
        tensor12 = tensor12.permute(perm)
        self.assertTrue(
            np.allclose(t12, tensor12.value, rtol=1e-10, atol=1e-10)
        )

    def test_mul3(self):
        x = np.random.randn(13, 14, 2, 5)
        x_tensor = Tensor(
            x, [Index("i", 13), Index("j", 14), Index("a", 2), Index("b", 5)]
        )
        u1 = np.random.randn(2, 15)
        u1_tensor = Tensor(u1, [Index("d", 2), Index("k", 15)])
        u2 = np.random.randn(5, 16)
        u2_tensor = Tensor(u2, [Index("b", 5), Index("l", 16)])
        u3 = np.random.randn(2, 3, 2)
        u3_tensor = Tensor(u3, [Index("a", 2), Index("c", 3), Index("d", 2)])
        u4 = np.random.randn(3, 17)
        u4_tensor = Tensor(u4, [Index("c", 3), Index("m", 17)])
        net1 = TensorNetwork()
        net1.add_node("u0", x_tensor)
        net1.add_node("u1", u1_tensor)
        net1.add_node("u2", u2_tensor)
        net1.add_node("u3", u3_tensor)
        net1.add_node("u4", u4_tensor)
        net1.add_edge("u0", "u3")
        net1.add_edge("u0", "u2")
        net1.add_edge("u3", "u1")
        net1.add_edge("u3", "u4")
        t1 = net1.contract()

        y = np.random.randn(13, 14, 1, 2)
        y_tensor = Tensor(
            y, [Index("i", 13), Index("j", 14), Index("aa", 1), Index("bb", 2)]
        )
        v1 = np.random.randn(3, 15)
        v1_tensor = Tensor(v1, [Index("dd", 3), Index("k", 15)])
        v2 = np.random.randn(2, 16)
        v2_tensor = Tensor(v2, [Index("bb", 2), Index("l", 16)])
        v3 = np.random.randn(1, 2, 3)
        v3_tensor = Tensor(
            v3, [Index("aa", 1), Index("cc", 2), Index("dd", 3)]
        )
        v4 = np.random.randn(2, 17)
        v4_tensor = Tensor(v4, [Index("cc", 2), Index("m", 17)])
        net2 = TensorNetwork()
        net2.add_node("v0", y_tensor)
        net2.add_node("v1", v1_tensor)
        net2.add_node("v2", v2_tensor)
        net2.add_node("v3", v3_tensor)
        net2.add_node("v4", v4_tensor)
        net2.add_edge("v0", "v2")
        net2.add_edge("v0", "v3")
        net2.add_edge("v3", "v1")
        net2.add_edge("v3", "v4")
        t2 = net2.contract()

        t12 = t1.value * t2.value
        net12 = net1 * net2
        net12.round("u0", net12.norm() * 1e-10)
        # network free inds are changed after rounding
        tensor12 = net12.contract()
        perm = [tensor12.indices.index(ind) for ind in net1.free_indices()]
        tensor12 = tensor12.permute(perm)
        self.assertTrue(
            np.allclose(t12, tensor12.value, rtol=1e-10, atol=1e-10)
        )


class TestCross(unittest.TestCase):
    """Test suite for cross approximation"""

    class FuncAckley(CachedFunc):
        """Source: https://www.sfu.ca/~ssurjano/ackley.html"""

        def __init__(self, indices: List[Index]):
            inds = []
            for ind in indices:
                new_ind = ind.with_new_rng(
                    np.linspace(-32.768, 32.768, ind.size)
                )
                inds.append(new_ind)
            super().__init__(inds)
            self.name = "Ackley"

        def _run(self, args: np.ndarray):
            y1 = np.sqrt(np.sum(args**2, axis=1) / args.shape[1])
            y1 = -20 * np.exp(-0.2 * y1)

            y2 = np.sum(np.cos(2 * np.pi * args), axis=1)
            y2 = -np.exp(y2 / args.shape[1])

            y3 = 20 + np.exp(1.0)

            return y1 + y2 + y3

    class FuncPathological(CachedFunc):
        """
        Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
                benchmark functions for global optimization problems". Journal of
                Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
                ("87. Pathological Function"; Continuous, Differentiable,
                Non-separable, Non-scalable, Multimodal).
        """

        def __init__(self, indices: List[Index]):
            inds = []
            for ind in indices:
                new_ind = ind.with_new_rng(np.linspace(-100, 100, ind.size))
                inds.append(new_ind)

            super().__init__(inds)
            # self.low = -100
            # self.range = 100 * 2
            self.name = "Pathological"

        def _run(self, args: np.ndarray):
            x1 = args[:, :-1]
            x2 = args[:, 1:]

            y1 = (np.sin(np.sqrt(100.0 * x1**2 + x2**2))) ** 2 - 0.5
            y2 = 1.0 + 0.001 * (x1**2 - 2.0 * x1 * x2 + x2**2) ** 2

            return np.sum(0.5 + y1 / y2, axis=1)

    def test_cross_two_nodes(self):
        """Cross approximation for a matrix"""

        indices = [Index("i", 8), Index("j", 10)]
        func = TestCross.FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1])
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)
        res = cross_engine.cross(net, eps=1e-4)

        # check results on a validation set
        validation = np.dstack(
            np.meshgrid(*[range(ind.size) for ind in indices])
        ).reshape(-1, len(indices))
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )

    def test_cross_three_nodes(self):
        """Cross approximation for a three dimensional TT"""

        indices = [Index("i", 8), Index("j", 10), Index("k", 12)]
        func = TestCross.FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1, 1])
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)
        res = cross_engine.cross(net, eps=1e-4)

        # check results on a validation set
        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )

    def test_cross_tt(self):
        """Cross approximation for a four dimensional TT"""

        indices = [
            Index("i", 8),
            Index("j", 10),
            Index("k", 12),
            Index("l", 20),
        ]
        func = TestCross.FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1, 1, 1])
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)
        res = cross_engine.cross(net, eps=1e-4)

        # check results on a validation set
        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )

    def test_cross_ht(self):
        """Cross approximation for a four dimensional HT"""

        indices = [
            Index("i", 8),
            Index("j", 10),
            Index("k", 12),
            Index("l", 20),
        ]
        func = TestCross.FuncAckley(indices)
        net = TensorNetwork.rand_ht(func.indices, 1)
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)
        res = cross_engine.cross(net, eps=1e-4)

        # check results on a validation set
        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )

    def test_cross_tucker(self):
        """Cross approximation for a four dimensional Tucker"""

        indices = [
            Index("i", 8),
            Index("j", 10),
            Index("k", 12),
            Index("l", 20),
        ]
        func = TestCross.FuncAckley(indices)
        net = TensorNetwork.rand_tucker(func.indices, 1)
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)
        res = cross_engine.cross(net, eps=1e-4)

        # check results on a validation set
        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )

    def test_cross_deim(self):
        """Cross approximation for a four dimensional TT with DEIM"""

        indices = [
            Index("i", 8),
            Index("j", 10),
            Index("k", 12),
            Index("l", 20),
        ]
        func = TestCross.FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1] * (len(indices) - 1))
        cross_config = CrossConfig(kickrank=2, cross_algo=CrossAlgo.DEIM)
        cross_engine = CrossApproximation(func, cross_config)
        res = cross_engine.cross(net, eps=1e-4)

        # check results on a validation set
        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )

    def test_cross_tucker_deim(self):
        """Cross approximation for a four dimensional Tucker with DEIM"""

        indices = [
            Index("i", 8),
            Index("j", 10),
            Index("k", 12),
            Index("l", 20),
        ]
        func = TestCross.FuncAckley(indices)
        net = TensorNetwork.rand_tucker(func.indices, 1)
        cross_config = CrossConfig(kickrank=2, cross_algo=CrossAlgo.DEIM)
        cross_engine = CrossApproximation(func, cross_config)
        res = cross_engine.cross(net, eps=1e-4)

        # check results on a validation set
        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )

    def test_cross_tt_deim_valid_error(self):
        """
        Cross approximation for Tensor Train with DEIM and
        converge at validation set
        """

        indices = [
            Index("i", 8),
            Index("j", 10),
            Index("k", 12),
            Index("l", 20),
            Index("m", 20),
            Index("n", 8),
            Index("o", 8),
            Index("p", 8),
        ]
        func = TestCross.FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1] * (len(indices) - 1))
        cross_config = CrossConfig(
            kickrank=2,
            cross_algo=CrossAlgo.DEIM,
            convergence=ConvergenceCheck.VALID_ERROR,
        )
        cross_engine = CrossApproximation(func, cross_config)
        res = cross_engine.cross(net, eps=1e-4)

        # check results on a validation set
        validation = np.stack(
            np.meshgrid(
                *[np.random.randint(0, ind.size, size=5) for ind in indices]
            ),
            axis=-1,
        ).reshape(-1, len(indices))
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )

    def test_cross_tt_maxvol_valid_error(self):
        """
        Cross approximation for Tensor Train with DEIM and
        converge at validation set
        """

        indices = [
            Index("i", 8),
            Index("j", 10),
            Index("k", 12),
            Index("l", 20),
            Index("m", 20),
            Index("n", 8),
            Index("o", 8),
            Index("p", 8),
        ]
        func = TestCross.FuncPathological(indices)
        net = TensorNetwork.rand_tt(func.indices, [1] * (len(indices) - 1))
        cross_config = CrossConfig(
            kickrank=2,
            cross_algo=CrossAlgo.DEIM,
            convergence=ConvergenceCheck.VALID_ERROR,
        )
        cross_engine = CrossApproximation(func, cross_config)
        res = cross_engine.cross(net, eps=1e-4)

        # check results on a validation set
        validation = np.stack(
            np.meshgrid(
                *[np.random.randint(0, ind.size, size=5) for ind in indices]
            ),
            axis=-1,
        ).reshape(-1, len(indices))
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )

    def test_cross_tt_provided_validation(self):
        """Cross approximation for a four dimensional TT"""

        indices = [
            Index("i", 8),
            Index("j", 10),
            Index("k", 12),
            Index("l", 20),
        ]
        func = TestCross.FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1, 1, 1])
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)

        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        res = cross_engine.cross(net, eps=1e-4, validation=validation)

        # check results on a validation set
        real_val = func(validation)
        approx_val = res.net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )


if __name__ == "__main__":
    unittest.main()

"""Tensor Network Solvers."""

import copy
import os
import unittest
import tempfile
import pickle

import numpy as np

from pytens.algs import *
from pytens.cross.cross import CrossAlgo, CrossApproximation, CrossConfig
from pytens.types import Index
from pytens.cross.funcs import FuncAckley
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


class TestCross(unittest.TestCase):
    """Test suite for cross approximation"""

    def test_cross_two_nodes(self):
        """Cross approximation for a matrix"""

        indices = [Index("i", 8), Index("j", 10)]
        func = FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1])
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)

        validation = np.dstack(
            np.meshgrid(*[range(ind.size) for ind in indices])
        ).reshape(-1, len(indices))
        cross_engine.cross(
            net, list(net.network.nodes)[0], validation, eps=1e-4
        )

        real_val = func(validation)
        approx_val = net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )
        
    def test_cross_three_nodes(self):
        """Cross approximation for a three dimensional TT"""

        indices = [Index("i", 8), Index("j", 10), Index("k", 12)]
        func = FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1, 1])
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)

        validation = np.stack(np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1).reshape(-1, len(indices))
        cross_engine.cross(
            net, list(net.network.nodes)[0], validation, eps=1e-4
        )

        real_val = func(validation)
        approx_val = net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )
        
    def test_cross_tt(self):
        """Cross approximation for a four dimensional TT"""

        indices = [Index("i", 8), Index("j", 10), Index("k", 12), Index("l", 20)]
        func = FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1, 1, 1])
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)

        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        cross_engine.cross(
            net, list(net.network.nodes)[0], validation, eps=1e-4
        )

        real_val = func(validation)
        approx_val = net.evaluate(func.indices, validation)
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
        func = FuncAckley(indices)
        net = TensorNetwork.rand_ht(func.indices, 1)
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)

        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        cross_engine.cross(
            net, list(net.network.nodes)[0], validation, eps=1e-4
        )

        real_val = func(validation)
        approx_val = net.evaluate(func.indices, validation)
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
        func = FuncAckley(indices)
        net = TensorNetwork.rand_tucker(func.indices, 1)
        cross_config = CrossConfig(kickrank=2)
        cross_engine = CrossApproximation(func, cross_config)

        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        cross_engine.cross(
            net, list(net.network.nodes)[0], validation, eps=1e-4
        )

        real_val = func(validation)
        approx_val = net.evaluate(func.indices, validation)
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
        func = FuncAckley(indices)
        net = TensorNetwork.rand_tt(func.indices, [1] * (len(indices) - 1))
        cross_config = CrossConfig(kickrank=2, cross_algo=CrossAlgo.DEIM)
        cross_engine = CrossApproximation(func, cross_config)

        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        cross_engine.cross(
            net, list(net.network.nodes)[0], validation, eps=1e-4
        )

        real_val = func(validation)
        approx_val = net.evaluate(func.indices, validation)
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
        func = FuncAckley(indices)
        net = TensorNetwork.rand_tucker(func.indices, 1)
        cross_config = CrossConfig(kickrank=2, cross_algo=CrossAlgo.DEIM)
        cross_engine = CrossApproximation(func, cross_config)

        validation = np.stack(
            np.meshgrid(*[range(ind.size) for ind in indices]), axis=-1
        ).reshape(-1, len(indices))
        cross_engine.cross(
            net, list(net.network.nodes)[0], validation, eps=1e-4
        )

        real_val = func(validation)
        approx_val = net.evaluate(func.indices, validation)
        self.assertTrue(
            np.linalg.norm(real_val - approx_val) / np.linalg.norm(real_val)
            <= 1e-4
        )


if __name__ == "__main__":
    unittest.main()

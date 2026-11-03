"""Accuracy and evaluation-budget regressions for tree cross approximation."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from pytens.algs import Tensor, TensorNetwork, rand_tt, rand_ht, rand_tucker
from pytens.cross.cross import (
    ConvergenceCheck,
    CrossAlgo,
    CrossApproximation,
    CrossConfig,
)
from pytens.cross.evaluation import CrossEvaluator
from pytens.cross.func_interface import TensorFunc
from pytens.cross.runner import TTCrossRunner, HTCrossRunner, TuckerCrossRunner
from pytens.types import Index


class ArrayFunction(TensorFunc):
    def __init__(self, data, column=False):
        super().__init__(
            [
                Index(str(i), n, tuple(range(n)))
                for i, n in enumerate(data.shape)
            ]
        )
        self.data = data
        self.column = column
        self.batches = []

    def run(self, args):
        self.batches.append(len(args))
        values = self.data[tuple(args.astype(int).T)]
        return values[:, None] if self.column else values


def grid(shape):
    return np.indices(shape).reshape(len(shape), -1).T


class TestCrossRegressions(unittest.TestCase):
    def setUp(self):
        np.random.seed(19)

    def fit(self, data, config=None, column=False, eps=1e-12):
        f = ArrayFunction(data, column)
        net = rand_tt(f.indices, [1] * (data.ndim - 1))
        result = CrossApproximation(f, config).cross(
            net,
            validation=grid(data.shape),
            eps=eps,
            initialization=np.zeros((1, data.ndim), dtype=int),
        )
        return f, result

    def test_zero_both_modes(self):
        for mode in ConvergenceCheck:
            with self.subTest(mode=mode):
                _, result = self.fit(
                    np.zeros((3, 4)), CrossConfig(convergence=mode)
                )
                self.assertTrue(result.converged)
                self.assertEqual(len(result.history), 1)
                self.assertEqual(result.history[0].validation_rmse, 0)
                self.assertEqual(result.ranks_and_errors, [(1, 0.0)])

    def test_rank_cap_is_not_success(self):
        for mode in ConvergenceCheck:
            with self.subTest(mode=mode):
                _, result = self.fit(
                    np.eye(4),
                    CrossConfig(
                        max_rank=1,
                        convergence=mode,
                        stagnation_patience=2,
                    ),
                )
                self.assertFalse(result.converged)
                self.assertEqual(result.stop_reason, "rank_limit")
                self.assertEqual(len(result.history), 3)
                self.assertAlmostEqual(
                    result.history[-1].validation_error, np.sqrt(3 / 4)
                )

    def test_exact_sweep_budget_and_history(self):
        _, result = self.fit(np.eye(4), CrossConfig(max_rank=1, max_iters=1))
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "max_iters")
        self.assertEqual(len(result.history), 1)
        _, result = self.fit(np.eye(4), CrossConfig(max_iters=8))
        self.assertTrue(result.converged)
        self.assertGreater(len(result.history), 1)
        self.assertEqual(len(result.history), len(result.ranks_and_errors))
        self.assertEqual(
            [h.sweep for h in result.history],
            list(range(1, len(result.history) + 1)),
        )
        self.assertEqual(result.history[-1].ranks, [4])

    def test_column_outputs_and_tiny_values(self):
        for scale in [1.0, 1e-200]:
            _, result = self.fit(
                scale * np.arange(1.0, 17.0).reshape(4, 4), column=True
            )
            self.assertTrue(result.converged)
            self.assertLess(result.history[-1].validation_error, 1e-12)

    def test_physical_order_and_root_invariance(self):
        for sizes in [(2, 2, 2), (2, 3, 4)]:
            coords = np.indices(sizes)
            data = (1 + coords[0]) * (2 + 3 * coords[1]) * (4 + 5 * coords[2])
            f = ArrayFunction(data)
            a, b, c = f.indices
            for algo in CrossAlgo:
                for reverse in [False, True]:
                    for root in ["root", "leaf"]:
                        with self.subTest(
                            sizes=sizes, algo=algo, reverse=reverse, root=root
                        ):
                            bond = Index("bond", 1)
                            net = TensorNetwork()
                            net.add_node(
                                "root", Tensor(np.ones((a.size, 1)), [a, bond])
                            )
                            inds = [c, b, bond] if reverse else [b, c, bond]
                            net.add_node(
                                "leaf",
                                Tensor(np.ones([i.size for i in inds]), inds),
                            )
                            net.add_edge("root", "leaf")
                            result = CrossApproximation(
                                f,
                                CrossConfig(
                                    max_rank=1,
                                    cross_algo=algo,
                                ),
                            ).cross(
                                net,
                                root=root,
                                validation=grid(sizes),
                                eps=1e-12,
                            )
                            self.assertTrue(result.converged)
                            np.testing.assert_allclose(
                                net.evaluate(f.indices, grid(sizes)),
                                data.reshape(-1),
                                rtol=1e-12,
                            )
                            self.assertEqual(
                                set(net.free_indices()), set(f.indices)
                            )
                            for ind in net.free_indices():
                                self.assertEqual(
                                    tuple(ind.space), tuple(range(ind.size))
                                )

    def test_function_order_for_initialization_and_known(self):
        class InspectEngine(CrossApproximation):
            def _iterate_tree_nodes(engine, net, nodes):
                if not hasattr(engine, "seen"):
                    engine.seen = nodes[1].up_info.vals.copy()
                return super(InspectEngine, engine)._iterate_tree_nodes(
                    net, nodes
                )

        data = np.arange(1.0, 25.0).reshape(2, 3, 4)
        f = ArrayFunction(data)
        net = rand_tt(f.indices[::-1], [1, 1])
        engine = InspectEngine(f, CrossConfig(max_iters=4))
        point = np.array([[1, 2, 3]])
        result = engine.cross(
            net,
            root=list(net.network)[0],
            validation=grid(data.shape),
            initialization=point,
            known=point,
            eps=1e-12,
        )
        subtree = result.dim_tree.preorder()[1]
        order = [f.indices.index(i) for i in subtree.up_info.indices]
        np.testing.assert_array_equal(engine.seen, point[:, order])
        self.assertTrue(result.converged)

    def test_batching_and_cache_preserve_accuracy(self):
        data = np.arange(1.0, 25.0).reshape(2, 3, 4)
        runs = []
        for cache_size in [0, 1000]:
            np.random.seed(42)
            f, result = self.fit(
                data, CrossConfig(cache_size=cache_size, batch_size=5)
            )
            self.assertTrue(result.converged)
            self.assertLessEqual(max(f.batches), 5)
            self.assertEqual(sum(f.batches), result.evaluations.evaluated)
            self.assertEqual(
                result.evaluations.requested,
                result.evaluations.evaluated + result.evaluations.cache_hits,
            )
            runs.append(result)
        self.assertEqual(runs[1].evaluations.evaluated, data.size)
        self.assertLess(
            runs[1].evaluations.evaluated, runs[0].evaluations.evaluated
        )
        np.testing.assert_allclose(
            runs[0].net.evaluate(f.indices, grid(data.shape)),
            runs[1].net.evaluate(f.indices, grid(data.shape)),
        )

    def test_cache_eviction_duplicates_and_reset(self):
        f = ArrayFunction(np.array([[1.0, 2.0], [3.0, 4.0]]))
        evaluator = CrossEvaluator(f, batch_size=3, cache_size=1)
        points = np.array([[0, 0], [0, 0], [1, 1]])
        np.testing.assert_array_equal(evaluator(points), [1, 1, 4])
        np.testing.assert_array_equal(evaluator(points), [1, 1, 4])
        self.assertEqual(evaluator.stats.evaluated, 3)
        self.assertEqual(len(evaluator.cache), 1)
        engine = CrossApproximation(f, CrossConfig(cache_size=100))
        for value in [2.0, 5.0]:
            f.data[:] = value
            net = rand_tt(f.indices, [1])
            result = engine.cross(
                net, validation=grid(f.data.shape), eps=1e-12
            )
            self.assertTrue(result.converged)
            np.testing.assert_allclose(
                net.evaluate(f.indices, grid(f.data.shape)), value
            )
            self.assertEqual(result.evaluations.evaluated, 4)

    def test_no_network_copy_in_default_mode(self):
        with patch(
            "pytens.cross.cross.copy.deepcopy", wraps=copy.deepcopy
        ) as copied:
            self.fit(np.ones((3, 4)))
        self.assertEqual(copied.call_count, 0)

    def test_exact_cache_keys_at_and_beyond_uint64(self):
        class DiscreteFunction(TensorFunc):
            def __init__(self, sizes):
                super().__init__(
                    [Index(str(i), n) for i, n in enumerate(sizes)]
                )

            def __call__(self, points):
                return points @ np.arange(1, points.shape[1] + 1)

        for sizes in [[2] * 64, [1] + [2] * 64, [2] * 256, [257] * 16]:
            with self.subTest(sizes=sizes):
                f = DiscreteFunction(sizes)
                points = np.random.default_rng(83).integers(
                    np.asarray(sizes), size=(20, len(sizes))
                )
                # Include both endpoints and repeat some requests.
                points[0] = 0
                points[1] = np.asarray(sizes) - 1
                points[-3:] = points[:3]
                evaluator = CrossEvaluator(f, batch_size=7, cache_size=100)
                np.testing.assert_array_equal(evaluator(points), f(points))
                evaluations = evaluator.stats.evaluated
                np.testing.assert_array_equal(
                    evaluator(points[::-1]), f(points[::-1])
                )
                self.assertEqual(evaluator.stats.evaluated, evaluations)
                self.assertEqual(evaluations, len(np.unique(points, axis=0)))

    def test_cache_preserves_value_dtype(self):
        f = ArrayFunction(np.ones((3, 3)))
        points = grid((3, 3))
        for data in [
            np.arange(9, dtype=np.int64) + 2**60,
            np.arange(9, dtype=np.float32),
            np.arange(9) + 1j * np.arange(9)[::-1],
        ]:
            f.data = data.reshape(3, 3)
            evaluator = CrossEvaluator(f, batch_size=4, cache_size=20)
            for _ in range(2):
                values = evaluator(points)
                self.assertEqual(values.dtype, data.dtype)
                np.testing.assert_array_equal(values, data)

    def test_cache_eviction_uses_last_access_in_batch(self):
        f = ArrayFunction(np.arange(4.0).reshape(2, 2))
        evaluator = CrossEvaluator(f, batch_size=10, cache_size=1)
        evaluator(np.array([[1, 1]]))
        # The existing entry is touched after the miss and must survive.
        evaluator(np.array([[0, 0], [1, 1]]))
        evaluations = evaluator.stats.evaluated
        np.testing.assert_array_equal(evaluator(np.array([[1, 1]])), [3.0])
        self.assertEqual(evaluator.stats.evaluated, evaluations)
        self.assertEqual(len(evaluator.cache), 1)

    def test_small_cache_randomized_requests(self):
        f = ArrayFunction(np.arange(64.0).reshape(8, 8))
        rng = np.random.default_rng(44)
        for capacity in [1, 5, 100]:
            evaluator = CrossEvaluator(f, batch_size=7, cache_size=capacity)
            for _ in range(20):
                points = rng.integers(8, size=(30, 2))
                points[-5:] = points[:5]
                np.testing.assert_array_equal(evaluator(points), f(points))
                self.assertLessEqual(len(evaluator.cache), capacity)
            self.assertEqual(
                evaluator.stats.requested,
                evaluator.stats.evaluated + evaluator.stats.cache_hits,
            )

    def test_reject_invalid_outputs_and_inputs(self):
        for values in [np.full((2, 2), np.nan), np.full((2, 2), np.inf)]:
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.fit(values)
        f = ArrayFunction(np.ones((2, 2)))
        f.run = lambda args: np.ones((len(args), 2))
        with self.assertRaisesRegex(ValueError, "shape"):
            CrossApproximation(f).cross(rand_tt(f.indices, [1]))
        for points in [
            np.empty((0, 2)),
            np.array([[0.5, 0]]),
            np.array([[2, 0]]),
        ]:
            f = ArrayFunction(np.ones((2, 2)))
            with self.assertRaises(ValueError):
                CrossApproximation(f).cross(
                    rand_tt(f.indices, [1]), validation=points
                )
        for config in [
            dict(max_iters=0),
            dict(max_rank=0),
            dict(batch_size=0),
            dict(kickrank=-1),
            dict(validation_size=0),
        ]:
            with self.assertRaises(ValueError):
                CrossConfig(**config)

    def test_runners_expose_result_and_warn(self):
        f = ArrayFunction(np.eye(4))
        runner = TTCrossRunner(
            CrossConfig(max_rank=1, max_iters=1, cache_size=100)
        )
        with self.assertWarnsRegex(RuntimeWarning, "max_iters"):
            runner.run(f, eps=1e-12, validation=grid(f.data.shape))
        self.assertFalse(runner.last_result.converged)
        for cls in [TTCrossRunner, HTCrossRunner, TuckerCrossRunner]:
            f = ArrayFunction(np.ones((2, 3, 4, 2)))
            runner = cls(CrossConfig(batch_size=7))
            runner.run(f, eps=1e-12, validation=grid(f.data.shape))
            self.assertTrue(runner.last_result.converged)
            self.assertLessEqual(max(f.batches), 7)

    def test_small_dense_trees(self):
        f = ArrayFunction(np.random.default_rng(23).normal(size=(3, 3, 3)))
        points = grid(f.data.shape)
        for factory in [
            lambda: rand_tt(f.indices, [1, 1]),
            lambda: rand_ht(f.indices, 1),
            lambda: rand_tucker(f.indices, 1),
        ]:
            result = CrossApproximation(f, CrossConfig(max_iters=12)).cross(
                factory(),
                validation=points,
                eps=1e-11,
            )
            self.assertTrue(result.converged)
            np.testing.assert_allclose(
                result.net.evaluate(f.indices, points),
                f.data.reshape(-1),
                atol=1e-11,
            )


if __name__ == "__main__":
    unittest.main()

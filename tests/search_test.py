"""Test file for the tensor network structure search module."""

import unittest
import time


import numpy as np
import json

from pytens.algs import TreeNetwork, Index, Tensor
from pytens.cross.funcs import FUNCS, FuncData
from pytens.search.configuration import SearchConfig
from pytens.search.state import ISplit, OSplit, SearchState
from pytens.search.search import SearchEngine


class TestConfig(unittest.TestCase):
    """Test configuration properties"""

    def test_config_load(self):
        config_str = json.dumps(
            {
                "synthesizer": {
                    "action_type": "isplit",
                },
                "rank_search": {
                    "search_mode": "all",
                    "k": 3,
                },
            }
        )
        config = SearchConfig.load(config_str)
        self.assertEqual(config.synthesizer.action_type, "isplit")
        self.assertEqual(config.rank_search.search_mode, "all")
        self.assertEqual(config.rank_search.k, 3)


class TestAction(unittest.TestCase):
    """Test action properties."""

    def test_isplit_equality(self):
        """Check the correctness of __eq__ for ISplit."""
        a1 = ISplit("n1", [0, 1])
        a3 = ISplit("n1", [0])
        a4 = ISplit("n2", [0, 1])
        self.assertNotEqual(a1, a3)
        self.assertNotEqual(a1, a4)

    def test_osplit_equality(self):
        """Check the correctness of __eq__ for OSplit."""
        a1 = OSplit([Index("I0", 1), Index("I1", 2)])
        a2 = OSplit([Index("I0", 1)])
        a3 = OSplit([Index("I1", 2), Index("I0", 1)])
        self.assertNotEqual(a1, a2)
        self.assertEqual(a1, a3)

    def test_osplit_inequality(self):
        """Check the correctness of __lt__ for OSplit."""
        a1 = OSplit([Index("I0", 1), Index("I1", 2)])
        a2 = OSplit([Index("I0", 1)])
        a3 = OSplit([Index("I2", 2), Index("I0", 1)])
        self.assertLess(a2, a1)
        self.assertLess(a1, a3)

    def test_isplit_execution(self):
        """Check the correctness of ISplit execution."""
        data = np.random.randn(3, 4, 5, 6)
        indices = [Index("i", 3), Index("j", 4), Index("k", 5), Index("l", 6)]
        tensor = Tensor(data, indices)
        net = TreeNetwork()
        net.add_node("G", tensor)

        ac = ISplit("G", [0, 1])
        (u, s, v), _ = ac.svd(net)
        self.assertEqual(net.value(u).shape, (3, 4, 12))
        self.assertEqual(net.value(s).shape, (12, 12))
        self.assertEqual(net.value(v).shape, (12, 5, 6))

        net.merge(v, s)
        ac = ISplit("G", [0])
        (u, s, v), _ = ac.svd(net)
        self.assertEqual(net.value(u).shape, (3, 3))
        self.assertEqual(net.value(s).shape, (3, 3))
        self.assertEqual(net.value(v).shape, (3, 4, 12))

    def test_osplit_execution(self):
        """Check the correctness of OSplit execution."""
        data = np.random.randn(3, 4, 5, 6)
        indices = [Index("i", 3), Index("j", 4), Index("k", 5), Index("l", 6)]
        tensor = Tensor(data, indices)
        net = TreeNetwork()
        net.add_node("G", tensor)

        ac = OSplit([Index("i", 3), Index("k", 5)])
        (u, s, v), _ = ac.svd(net)
        self.assertEqual(net.value(u).shape, (3, 5, 15))
        self.assertEqual(net.value(s).shape, (15, 15))
        self.assertEqual(net.value(v).shape, (15, 4, 6))

        net.merge(v, s)
        ac = OSplit([Index("i", 3)])
        (u, s, v), _ = ac.svd(net)
        self.assertEqual(net.value(u).shape, (3, 3))
        self.assertEqual(net.value(s).shape, (3, 3))
        self.assertEqual(net.value(v).shape, (3, 5, 15))


class TestState(unittest.TestCase):
    """Test search state properties."""

    def test_legal_actions(self):
        """Conflict actions should be removed."""
        data = np.random.randn(3, 4, 5)
        indices = [Index("i", 3), Index("j", 4), Index("k", 5)]
        tensor = Tensor(data, indices)
        net = TreeNetwork()
        net.add_node("G", tensor)
        init_state = SearchState(net, net.norm() * 0.1)

        self.assertListEqual(
            init_state.get_legal_actions(),
            [
                ISplit("G", [0]),
                ISplit("G", [1]),
                ISplit("G", [2]),
            ],
        )

        self.assertListEqual(
            init_state.get_legal_actions(True),
            [
                OSplit([Index("i", 3)]),
                OSplit([Index("j", 4)]),
                OSplit([Index("k", 5)]),
            ],
        )

        ac = ISplit("G", [0])
        new_st = init_state.take_action(ac)
        assert new_st is not None
        expected = {}
        for node in new_st.network.network.nodes:
            for i, ind in enumerate(new_st.network.node_tensor(node).indices):
                if ind not in expected:
                    expected[ind] = ISplit(node, [i])
                    print(node, i)

        for ac in new_st.get_legal_actions():
            print(ac)

        self.assertListEqual(new_st.get_legal_actions(), list(expected.values()))

        ac = OSplit([Index("i", 3)])
        new_st = init_state.take_action(ac)
        assert new_st is not None
        self.assertListEqual(
            new_st.get_legal_actions(True),
            [
                OSplit([Index("j", 4)]),
                OSplit([Index("k", 5)]),
            ],
        )


class TestSearch(unittest.TestCase):
    """Test the general functionality of all search strategies."""

    def setUp(self):
        """Create the inital tensor network for testing."""
        np.random.seed(1)

        data = np.random.randn(3, 4, 5)
        indices = [Index("i", 3), Index("j", 4), Index("k", 5)]
        tensor = Tensor(data, indices)
        self.net = TreeNetwork()
        self.net.add_node("G", tensor)

        return super().setUp()

    def test_dfs(self):
        config = SearchConfig()
        config.engine.eps = 0.5
        config.engine.verbose = True
        search_engine = SearchEngine(config=config)
        result = search_engine.dfs(self.net)
        assert result is not None
        self.assertEqual(result.stats.count, 8)

        assert result.best_state is not None
        bn = result.best_state.network
        assert bn is not None
        err_norm = float(
            np.linalg.norm(self.net.contract().value - bn.contract().value)
        )
        self.assertLessEqual(err_norm, 0.5 * self.net.norm())
        self.assertLessEqual(bn.cost(), self.net.cost())

    def test_bfs(self):
        config = SearchConfig()
        config.engine.eps = 0.5
        config.engine.verbose = True
        search_engine = SearchEngine(config=config)
        result = search_engine.bfs(self.net)
        assert result is not None
        self.assertEqual(result.stats.count, 7)

        assert result.best_state is not None
        bn = result.best_state.network
        assert bn is not None
        err_norm = float(
            np.linalg.norm(self.net.contract().value - bn.contract().value)
        )
        self.assertLessEqual(err_norm, 0.5 * self.net.norm())
        self.assertLessEqual(bn.cost(), self.net.cost())

    def test_partition(self):
        config = SearchConfig()
        config.engine.eps = 0.5
        config.engine.verbose = True
        search_engine = SearchEngine(config=config)
        result = search_engine.partition_search(self.net)
        assert result is not None
        self.assertEqual(result.stats.count, 7)

        assert result.best_state is not None
        bn = result.best_state.network
        assert bn is not None
        err_norm = float(
            np.linalg.norm(self.net.contract().value - bn.contract().value)
        )
        self.assertLessEqual(
            err_norm,
            0.5 * self.net.norm(),
        )
        self.assertLessEqual(bn.cost(), self.net.cost())

    def test_partition_all(self):
        config = SearchConfig()
        config.engine.eps = 0.5
        config.engine.verbose = True
        config.rank_search.search_mode = "all"
        search_engine = SearchEngine(config=config)
        result = search_engine.partition_search(self.net)
        assert result is not None
        self.assertEqual(result.stats.count, 7)

        assert result.best_state is not None
        bn = result.best_state.network
        assert bn is not None
        err_norm = float(
            np.linalg.norm(self.net.contract().value - bn.contract().value)
        )
        self.assertLessEqual(err_norm, float(0.5 * self.net.norm()))
        self.assertLessEqual(bn.cost(), self.net.cost())

    def test_partition_cross(self):
        for f in FUNCS[:1]:
            for n in [25]:
                domain = np.linspace(-32.5, 32.5, n)
                indices = [Index(f"I{i}", n, domain) for i in range(4)]
                tensor_func = f(indices)
                grid = np.meshgrid(*[np.arange(0, n) for _ in range(4)])
                all_args = np.stack(grid, axis=0).reshape(4, -1).T
                real_val = tensor_func(all_args).reshape(n, n, n, n)
                # start = time.time()
                config = SearchConfig()
                config.engine.eps = 0.1
                config.engine.verbose = True
                search_engine = SearchEngine(config=config)
                net = TreeNetwork()
                indices = [Index(f"I{i}", n) for i in range(4)]
                tensor = Tensor(np.zeros([n] * 4), indices)
                net.add_node("G", tensor)
                result = search_engine.partition_search(tensor_func)
                # end = time.time()
                assert result.best_state is not None
                bn = result.best_state.network
                assert bn is not None
                # print(bn)
                err = real_val - bn.contract().value
                # print(
                #     tensor_func.name,
                #     eps,
                #     n,
                #     end - start,
                #     net.cost() / bn.cost(),
                #     np.linalg.norm(err) / np.linalg.norm(real_val),
                #     np.max(np.abs(err)) / np.max(np.abs(real_val)),
                # )
                rtol = np.linalg.norm(err) / np.linalg.norm(real_val)
                self.assertLessEqual(float(rtol), 0.5)

    def test_partition_cross_data(self):
        n = 25
        grid = np.meshgrid(*[np.arange(0, n) for _ in range(4)])
        all_args = np.stack(grid, axis=0).reshape(4, -1).T
        real_val = 1.0 / np.sum(all_args + 1, axis=1)
        real_val = real_val.reshape(n, n, n, n)
        indices = [Index(f"I{i}", n, range(n)) for i in range(4)]
        tensor_func = FuncData(indices, real_val)

        config = SearchConfig()
        config.engine.eps = 1e-1
        config.engine.verbose = True
        search_engine = SearchEngine(config=config)
        result = search_engine.partition_search(tensor_func)
        assert result.best_state is not None
        bn = result.best_state.network
        err = np.linalg.norm(real_val - bn.contract().value) / np.linalg.norm(
            real_val
        )
        self.assertLessEqual(float(err), 1e-1)


class TestTopDownSearch(unittest.TestCase):
    def test_top_down_cross(self):
        n = 15
        grid = np.meshgrid(*[np.arange(0, n) for _ in range(4)])
        all_args = np.stack(grid, axis=0).reshape(4, -1).T
        real_val = 1.0 / np.sum(all_args + 1, axis=1)
        real_val = real_val.reshape(n, n, n, n)
        indices = [Index(f"I{i}", n, range(n)) for i in range(4)]
        tensor_func = FuncData(indices, real_val)

        config = SearchConfig()
        config.engine.eps = 1e-1
        config.engine.verbose = True
        config.engine.decomp_algo = "cross"
        config.cross.init_eps = 0.1
        config.cross.init_struct = "tt"
        config.topdown.search_algo = "merge"
        config.topdown.merge_mode = "all"
        search_engine = SearchEngine(config=config)
        result = search_engine.top_down(tensor_func)
        assert result.best_state is not None
        err = result.stats.re_f
        # print("final cost", result.best_state.network.cost())
        # print(result.stats.search_end - result.stats.search_start)
        # print(result.best_state.network)
        # print(err)
        # print(result.stats.init_cross_evals)
        # print(result.stats.search_cross_evals)
        self.assertLessEqual(float(err), 2e-1)

    def test_top_down_cross_init_reshape(self):
        n = 25
        grid = np.meshgrid(*[np.arange(0, n) for _ in range(4)])
        all_args = np.stack(grid, axis=0).reshape(4, -1).T
        real_val = 1.0 / np.sum(all_args + 1, axis=1)
        real_val = real_val.reshape(n, n, n, n)
        indices = [Index(f"I{i}", n, range(n)) for i in range(4)]
        tensor_func = FuncData(indices, real_val)

        config = SearchConfig()
        config.engine.eps = 1e-1
        config.engine.verbose = True
        config.engine.decomp_algo = "cross"
        config.cross.init_eps = 0.1
        config.cross.init_struct = "tt"
        config.topdown.search_algo = "merge"
        config.topdown.merge_mode = "all"
        search_engine = SearchEngine(config=config)
        result = search_engine.top_down(tensor_func)
        assert result.best_state is not None
        err = result.stats.re_f
        print("final cost", result.best_state.network.cost())
        print(result.stats.search_end - result.stats.search_start)
        print(result.best_state.network)
        print(err)
        print(result.stats.init_cross_evals)
        print(result.stats.search_cross_evals)
        self.assertLessEqual(float(err), 2e-1)


if __name__ == "__main__":
    np.random.seed(1234)
    test = TestTopDownSearch()
    test.test_top_down_cross_init_reshape()

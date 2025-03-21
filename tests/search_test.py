"""Test file for the tensor network structure search module."""
import unittest

import numpy as np
import json

from pytens.algs import TensorNetwork, Index, Tensor
from pytens.search.configuration import SearchConfig
from pytens.search.state import ISplit, OSplit, SearchState
from pytens.search.search import SearchEngine
from pytens.search.hierarchical.top_down import TopDownSearch, Rule

class TestConfig(unittest.TestCase):
    """Test configuration properties"""

    def test_config_load(self):
        config_str = json.dumps({
            "synthesizer": {
                "action_type": "isplit",
            },
            "rank_search": {
                "fit_mode": "all",
                "k": 3,
            },
        })
        config = SearchConfig.load(config_str)
        self.assertEqual(config.synthesizer.action_type, "isplit")
        self.assertEqual(config.rank_search.fit_mode, "all")
        self.assertEqual(config.rank_search.k, 3)

class TestAction(unittest.TestCase):
    """Test action properties."""

    def test_isplit_equality(self):
        """Check the correctness of __eq__ for ISplit."""
        a1 = ISplit("n1", [0,1])
        a3 = ISplit("n1", [0])
        a4 = ISplit("n2", [0,1])
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
        data = np.random.randn(3,4,5,6)
        indices = [Index("i", 3), Index("j", 4), Index("k", 5), Index("l", 6)]
        tensor = Tensor(data, indices)
        net = TensorNetwork()
        net.add_node("G", tensor)

        ac = ISplit("G", [0,1])
        (u, s, v), _ = ac.execute(net)
        self.assertEqual(net.value(u).shape, (3, 4, 12))
        self.assertEqual(net.value(s).shape, (12, 12))
        self.assertEqual(net.value(v).shape, (12, 5, 6))

        net.merge(v, s)
        ac = ISplit("G", [0])
        (u, s, v), _ = ac.execute(net)
        self.assertEqual(net.value(u).shape, (3, 3))
        self.assertEqual(net.value(s).shape, (3, 3))
        self.assertEqual(net.value(v).shape, (3, 4, 12))

    def test_osplit_execution(self):
        """Check the correctness of OSplit execution."""
        data = np.random.randn(3,4,5,6)
        indices = [Index("i", 3), Index("j", 4), Index("k", 5), Index("l", 6)]
        tensor = Tensor(data, indices)
        net = TensorNetwork()
        net.add_node("G", tensor)

        ac = OSplit([Index("i", 3), Index("k", 5)])
        (u, s, v), _ = ac.execute(net)
        self.assertEqual(net.value(u).shape, (3, 5, 15))
        self.assertEqual(net.value(s).shape, (15, 15))
        self.assertEqual(net.value(v).shape, (15, 4, 6))

        net.merge(v, s)
        ac = OSplit([Index("i", 3)])
        (u, s, v), _ = ac.execute(net)
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
        net = TensorNetwork()
        net.add_node("G", tensor)
        init_state = SearchState(net, net.norm() * 0.1)

        self.assertListEqual(init_state.get_legal_actions(), [
            ISplit("G", [0]),
            ISplit("G", [1]),
            ISplit("G", [2]),
        ])

        self.assertListEqual(init_state.get_legal_actions(True), [
            OSplit([Index("i", 3)]),
            OSplit([Index("j", 4)]),
            OSplit([Index("k", 5)]),
        ])

        ac = ISplit("G", [0])
        for new_st in init_state.take_action(ac, config=SearchConfig()):
            self.assertListEqual(new_st.get_legal_actions(), [
                ISplit("n0", [0]),
                ISplit("n0", [1]),
                ISplit("n0", [2]),
                ISplit("G", [0]),
            ])

        ac = OSplit([Index("i", 3)])
        for new_st in init_state.take_action(ac, config=SearchConfig()):
            self.assertListEqual(new_st.get_legal_actions(True), [
                OSplit([Index("j", 4)]),
                OSplit([Index("k", 5)]),
            ])

class TestSearch(unittest.TestCase):
    """Test the general functionality of all search strategies."""
    def setUp(self):
        """Create the inital tensor network for testing."""
        np.random.seed(1)

        data = np.random.randn(3, 4, 5)
        indices = [Index("i", 3), Index("j", 4), Index("k", 5)]
        tensor = Tensor(data, indices)
        self.net = TensorNetwork()
        self.net.add_node("G", tensor)

        return super().setUp()
    
    def test_dfs(self):
        config = SearchConfig()
        config.engine.eps = 0.5
        config.engine.verbose = True
        search_engine = SearchEngine(config=config)
        result = search_engine.dfs(self.net)
        self.assertEqual(result.stats.count, 8)

        bn = result.best_network
        self.assertLessEqual(np.linalg.norm(self.net.contract().value - bn.contract().value), 0.5 * self.net.norm())
        self.assertLessEqual(bn.cost(), self.net.cost())

    def test_bfs(self):
        config = SearchConfig()
        config.engine.eps = 0.5
        config.engine.verbose = True
        search_engine = SearchEngine(config=config)
        result = search_engine.bfs(self.net)
        self.assertEqual(result.stats.count, 7)

        bn = result.best_network
        self.assertLessEqual(np.linalg.norm(self.net.contract().value - bn.contract().value), 0.5 * self.net.norm())
        self.assertLessEqual(bn.cost(), self.net.cost())

    def test_partition(self):
        config = SearchConfig()
        config.engine.eps = 0.5
        config.engine.verbose = True
        search_engine = SearchEngine(config=config)
        result = search_engine.partition_search(self.net)
        self.assertEqual(result.stats.count, 7)

        bn = result.best_network
        self.assertLessEqual(np.linalg.norm(self.net.contract().value - bn.contract().value), 0.5 * self.net.norm())
        self.assertLessEqual(bn.cost(), self.net.cost())

    def test_partition_all(self):
        config = SearchConfig()
        config.engine.eps = 0.5
        config.engine.verbose = True
        config.rank_search.fit_mode = "all"
        search_engine = SearchEngine(config=config)
        result = search_engine.partition_search(self.net)
        self.assertEqual(result.stats.count, 7)

        bn = result.best_network
        self.assertLessEqual(np.linalg.norm(self.net.contract().value - bn.contract().value), 0.5 * self.net.norm())
        self.assertLessEqual(bn.cost(), self.net.cost())

class TestTopDownSearch(unittest.TestCase):
    # def test_hint(self):
    #     config = SearchConfig()
    #     search_engine = TopDownSearch(config)
    #     search_engine.samples = [TopDownSearch.Sample([Index("i1", 2), Index("i2", 2), Index("i3", 2), Index("i4", 15), Index("i5", 17)], [
    #         [Index("i1", 2), Index("i2", 2), Index("i5", 17), Index("s", 9)],
    #         [Index("i3", 2), Index("i4", 15), Index("s", 9)],
    #     ])]

    #     new_sample = TopDownSearch.Sample([Index("i1", 4), Index("i2", 2), Index("i3", 3), Index("i4", 5), Index("i5", 17)], [
    #         [Index("i1", 4), Index("i5", 17), Index("s2", 9)],
    #         [Index("i3", 3), Index("i4", 5), Index("s1", 6)],
    #         [Index("i2", 2), Index("s1", 6), Index("s2", 9)],
    #     ])
    #     hints = search_engine.mine_hint(new_sample)
    #     self.assertDictEqual(hints, {
    #         (2, 2, 2, 15, 17): [[range(0,2), range(4,5)],[range(3,4)], [range(2,3)]],
    #     })

    #     new_sample = TopDownSearch.Sample([Index("i1", 2), Index("i2", 4), Index("i3", 3), Index("i4", 5), Index("i5", 17)], [
    #         [Index("i1", 2), Index("i5", 17), Index("s2", 9)],
    #         [Index("i3", 3), Index("i4", 5), Index("s1", 6)],
    #         [Index("i2", 4), Index("s1", 6), Index("s2", 9)],
    #     ])
    #     hints = search_engine.mine_hint(new_sample)
    #     self.assertDictEqual(hints, {
    #         (2, 2, 2, 15, 17): [[range(0,1), range(4,5)], [range(3,4)]],
    #         (4, 2, 3, 5, 17): [[range(4,5)],[range(2,3), range(3,4)]],
    #     })

    #     tensor = Tensor(np.random.randn(2,2,6,5,17), [Index("i1", 2), Index("i2", 2), Index("i3", 6), Index("i4", 5), Index("i5", 17)])
    #     out, applied = search_engine.with_hint(tensor)
    #     self.assertListEqual(out.indices, [Index("i5", 34), Index("i2",2), Index("i3", 6), Index("i4", 5)])
    #     self.assertTrue(np.allclose(out.value, tensor.value.transpose(0,4,1,2,3).reshape(-1, 2, 6, 5)))

    def test_rule(self):
        rule1 = Rule([2, 2, 2, 15, 17], [0, 4])
        rule2 = Rule([2, 2, 6, 5, 17], [0, 4])
        rule3 = Rule([2, 2, 5, 6, 17], [0, 1, 4])
        rule4 = Rule([2, 2, 2, 15, 17], [3])
        rule5 = Rule([4, 3, 2, 5, 17], [1, 3])
        self.assertTrue(rule1.match([2, 2, 2, 15, 17]))
        self.assertFalse(rule1.match([2, 2, 3, 15, 17]))
        self.assertTrue(rule1.match([2, 2, 4, 15, 17]))
        self.assertFalse(rule1.match([2, 2, 2, 3, 17]))

        rule12 = rule1.join(rule2)
        self.assertListEqual(rule12.pattern, [2, 2, 2, 5, 17])
        self.assertListEqual(rule12.interest_points, [0, 4])

        rule13 = rule1.join(rule3)
        self.assertListEqual(rule13.pattern, [2, 2, 1, 3, 17])
        self.assertListEqual(rule13.interest_points, [0, 4])

        self.assertIsNone(rule1.join(rule4))
        self.assertIsNone(rule1.join(rule5))
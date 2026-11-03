"""Randomized sum rounding on branching trees with shared physical cores."""

import copy

import numpy as np
import pytest

import pytens


def tree(rank, rng):
    shapes = {
        "center": ["a", "b", "c"],
        "left": ["a", "x", "d"],
        "tip": ["d", "u", "v"],
        "right": ["b", "y"],
        "top": ["c", "z"],
    }
    physical = dict(x=3, u=2, v=2, y=4, z=3)
    out = pytens.TensorNetwork()
    for n, names in shapes.items():
        indices = [pytens.Index(i, physical.get(i, rank)) for i in names]
        shape = tuple(i.size for i in indices)
        out.add_node(n, pytens.Tensor(rng.normal(size=shape), indices))
    for a, b in [
        ("center", "left"),
        ("center", "right"),
        ("center", "top"),
        ("left", "tip"),
    ]:
        out.add_edge(a, b)
    return out


def dense(tn):
    return tn.contract().permute_by_name(["x", "u", "v", "y", "z"]).value


@pytest.mark.parametrize("root", ["center", "tip", "left"])
def test_full_range_sum_without_assembly(root, monkeypatch):
    rng = np.random.default_rng(11)
    terms = [tree(2, rng), tree(3, rng), tree(1, rng)]
    # Core permutations must not change the operator's meaning.
    for n in terms[1].network:
        t = terms[1].node_tensor(n)
        terms[1].set_node_tensor(
            n, t.permute(list(reversed(range(t.value.ndim))))
        )
    before = [
        [t.node_tensor(n).value.copy() for n in t.network] for t in terms
    ]
    expected = sum(dense(t) for t in terms)

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not assemble a sum or a physical subtree")

    with monkeypatch.context() as guard:
        guard.setattr(pytens, "tree_sum", forbidden)
        guard.setattr(pytens.TensorNetwork, "contract", forbidden)
        out = pytens.tree_sum_rand_precond_svd_round(
            terms, 1e-12, [6] * 4, root=root, rng=np.random.default_rng(7)
        )
    np.testing.assert_allclose(dense(out), expected, rtol=1e-10, atol=1e-10)
    assert {frozenset(e) for e in out.network.edges} == {
        frozenset(e) for e in terms[0].network.edges
    }
    assert out.node_by_free_index("u") == out.node_by_free_index("v")
    for t, values in zip(terms, before):
        for n, value in zip(t.network, values):
            np.testing.assert_array_equal(t.node_tensor(n).value, value)


def test_cancellation_and_rank_caps():
    a = tree(2, np.random.default_rng(3))
    b = copy.deepcopy(a).scale(-1)
    out = pytens.tree_sum_rand_precond_svd_round(
        [a, b], 1e-10, [2] * 4, rng=np.random.default_rng(5)
    )
    np.testing.assert_allclose(dense(out), 0, atol=1e-10)
    out = pytens.tree_sum_rand_precond_svd_round(
        [a], 1e-10, [1] * 4, rng=np.random.default_rng(5)
    )
    assert max(out.ranks()) == 1
    assert np.isfinite(dense(out)).all()


def test_single_node_and_invalid_bounds():
    a = pytens.TensorNetwork()
    a.add_node("a", pytens.Tensor(np.arange(3.0), [pytens.Index("x", 3)]))
    out = pytens.tree_sum_rand_precond_svd_round([a, a], 0, [])
    np.testing.assert_array_equal(
        out.node_tensor("a").value, np.arange(3.0) * 2
    )
    with pytest.raises(ValueError, match="rank bound"):
        pytens.tree_sum_rand_precond_svd_round(
            [tree(2, np.random.default_rng(1))], 0, []
        )

"""Randomized rounding of tree sums without forming direct-sum cores."""

from collections.abc import Sequence
import math

import networkx as nx
import numpy as np

from .algs import Tensor, TensorNetwork, _TreeTerms, _apply_on_axis
from .types import NodeName


def tree_sum_rand_precond_svd_round(
    terms: Sequence[TensorNetwork],
    eps: float,
    rank_bound: Sequence[int],
    root: NodeName | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> TensorNetwork:
    """Sketch a tree sum, then SVD-truncate the compressed tree.

    Bounds follow ``terms[0].inner_indices()`` (the order of ``ranks()``).
    Terms must share node/edge names, bond names and physical-index placement;
    bond sizes and local index order may differ. Inputs are never modified.

    A common Gaussian tree supplies exterior sketches for all terms. A
    leaves-to-root sweep projects already compressed children, sums the local
    sketches, and obtains an isometric core by QR. Term-specific projection
    messages pass to the parent. Only the root core is summed explicitly,
    after all its bonds have been compressed. No physical subtree, block
    diagonal sum, or pairwise Gram matrix is assembled.

    Like TT randomized preconditioning, rank bounds control the initial
    projection; ``eps`` controls only the subsequent SVD error, not the total
    error relative to the input sum. Insufficient bounds can lose components.
    With rng=None, use NumPy's global RNG, respecting np.random.seed.
    """
    if not terms:
        raise ValueError("At least one term is required.")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative.")
    ref = terms[0]
    if not nx.is_tree(ref.network):
        raise ValueError("Randomized tree rounding requires a connected tree.")
    bonds = ref.inner_indices()
    if len(rank_bound) != len(bonds) or any(
        int(r) != r or r < 1 for r in rank_bound
    ):
        raise ValueError("Provide one positive integer rank bound per bond.")
    edges = {frozenset(e) for e in ref.network.edges}
    physical = set(ref.free_indices())
    for term in terms:
        if (
            set(term.network) != set(ref.network)
            or {frozenset(e) for e in term.network.edges} != edges
        ):
            raise ValueError(
                "All terms must have the same named tree topology."
            )
        if set(term.free_indices()) != physical:
            raise ValueError("All terms must have the same physical indices.")
        for n in ref.network:
            expected = ref.node_tensor(n).indices
            actual = term.node_tensor(n).indices
            if {i.name for i in actual} != {i.name for i in expected} or {
                i for i in actual if i in physical
            } != {i for i in expected if i in physical}:
                raise ValueError(
                    "Core index placement must match across terms."
                )
    tt = _TreeTerms(terms, root)
    bounds = {b.name: int(r) for b, r in zip(bonds, rank_bound, strict=True)}
    # Cap each sketch by the physical dimensions on either side of its cut.
    below = {}
    for n in reversed(tt.order):
        below[n] = math.prod(i.size for i in tt.free_indices(n)) * math.prod(
            below[c] for c in tt.children[n]
        )
    for n in tt.order[1:]:
        b = tt.up_bond[n]
        bounds[b] = min(
            bounds[b], tt.offsets[b][-1], below[n], below[tt.root] // below[n]
        )
    normal = np.random.standard_normal if rng is None else rng.standard_normal
    probes = {}
    for n in tt.nodes:
        shape = [
            bounds[i.name] if i.name in bounds else i.size
            for i in tt.indices[n]
        ]
        probes[n] = normal(shape) / math.sqrt(math.prod(shape))

    messages = {}

    def exterior(src, dst, term_index):
        """Contract a term/probe pair on one side of an edge; retain that bond."""
        keep = ref.get_contraction_index(src, dst)[0].name
        value = tt.cores[src][term_index]
        operations = []
        for neighbor in ref.network.neighbors(src):
            if neighbor == dst:
                continue
            bond = ref.get_contraction_index(src, neighbor)[0].name
            message = messages[neighbor, src, term_index]
            operations.append((tt.bond_axes[src][bond], message.T))
        # Shrink the largest axes before expanding any small ones.
        for axis, mat in sorted(
            operations, key=lambda op: op[1].shape[0] / value.shape[op[0]]
        ):
            value = _apply_on_axis(value, axis, mat)
        axis = tt.bond_axes[src][keep]
        contracted = [k for k in range(value.ndim) if k != axis]
        return np.tensordot(
            value, probes[src].conj(), axes=(contracted, contracted)
        )

    # Two directed sweeps avoid recursive cached closures retaining entire
    # input sums until cyclic garbage collection during time stepping.
    for n in reversed(tt.order[1:]):
        parent = tt.parent[n]
        for k in range(len(terms)):
            messages[n, parent, k] = exterior(n, parent, k)
    for n in tt.order:
        for child in tt.children[n]:
            for k in range(len(terms)):
                messages[n, child, k] = exterior(n, child, k)

    projections = {}
    result = TensorNetwork()
    result.network = ref.network.copy()

    def reduced(n, term_index):
        value = tt.cores[n][term_index]
        operations = []
        for child in tt.children[n]:
            bond = tt.up_bond[child]
            operations.append(
                (tt.bond_axes[n][bond], projections[child][term_index])
            )
        for axis, mat in sorted(
            operations, key=lambda op: op[1].shape[0] / value.shape[op[0]]
        ):
            value = _apply_on_axis(value, axis, mat)
        return value

    for n in reversed(tt.order):
        if n == tt.root:
            value = reduced(n, 0).copy()
            for k in range(1, len(terms)):
                value += reduced(n, k)
        else:
            bond = tt.up_bond[n]
            axis = tt.bond_axes[n][bond]
            sketch = None
            for k in range(len(terms)):
                local = reduced(n, k)
                block = _apply_on_axis(
                    local, axis, messages[tt.parent[n], n, k].T
                )
                if sketch is None:
                    sketch = block
                else:
                    sketch += block
            shape = np.moveaxis(sketch, axis, -1).shape
            # moveaxis, rather than a swap, keeps all other core axes ordered.
            q, _ = np.linalg.qr(
                np.moveaxis(sketch, axis, -1).reshape(-1, shape[-1])
            )
            value = np.moveaxis(q.reshape(*shape[:-1], q.shape[1]), -1, axis)
            projections[n] = []
            for k in range(len(terms)):
                local = np.moveaxis(reduced(n, k), axis, -1)
                projections[n].append(
                    q.conj().T @ local.reshape(-1, local.shape[-1])
                )
        result.set_node_tensor(
            n,
            Tensor(
                value,
                [
                    i.with_new_size(value.shape[k])
                    for k, i in enumerate(tt.indices[n])
                ],
            ),
        )
        for child in tt.children[n]:
            del projections[child]
    messages.clear()
    norm = float(np.linalg.norm(result.node_tensor(tt.root).value))
    result.round(tt.root, atol=eps * norm, orthonormal=True)
    return result

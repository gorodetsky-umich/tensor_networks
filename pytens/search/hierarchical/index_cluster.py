"""Various index clustering algorithms."""

from __future__ import annotations

import copy
import itertools
import logging
import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from pytens.cross.cross import CrossApproximation, CrossConfig
from pytens.algs import rand_tt
from pytens.search.state import OSplit
from pytens.types import Index, IndexOp, IndexSplit, NodeName, SValsParams

if TYPE_CHECKING:
    import pytens.algs as pt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def eff_rank(svals: np.ndarray) -> float:
    """Compute the effective rank of a spectrum via entropy."""
    s = svals  # ** 2
    s = s[s > 1e-8]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


# Helpers for SVDIndexCluster._tree_score.
#
# Invariant throughout: exactly one node (the core) is not orthonormal, so
# `svals_at(core, ..., with_orthonormal=False)` is exact. The network is
# orthonormalized once, and the core is only ever moved by single merge + QR
# steps: `_shift_core` hands it to a neighbor, `_swap_core` lets it keep its
# free indices and take the neighbor's position.


@dataclass
class _TreeScoreState:
    """State shared by the `_tree_score` helpers.

    Attributes:
        orig: The untouched input network; its topology tells which nodes
            lie behind a neighbor, since names other than the core never
            leave their original positions.
        free: The free indices being paired.
        comb_corr: Scores computed so far, keyed by index pair.
        origins_done: Nodes whose free indices have already met every
            other node's.
    """

    orig: pt.TensorNetwork
    free: Sequence[Index]
    comb_corr: Dict[Sequence[Index], float] = field(default_factory=dict)
    origins_done: Set[NodeName] = field(default_factory=set)

    def free_on_node(
        self, net: pt.TensorNetwork, node: NodeName
    ) -> List[Index]:
        """Free indices carried by `node`."""
        inds = net.node_tensor(node).indices
        return [ind for ind in inds if ind in self.free]

    def has_unscored_behind(self, root: NodeName, exclude: NodeName) -> bool:
        """Whether any node reachable from `root` in the original tree
        without passing through `exclude` has not been an origin yet."""
        keep = [n for n in self.orig.network.nodes if n != exclude]
        subgraph = self.orig.network.subgraph(keep)
        behind = nx.node_connected_component(subgraph, root)
        return not behind <= self.origins_done

    def score_pairs(
        self,
        net: pt.TensorNetwork,
        core: NodeName,
        pair_inds: Sequence[Index],
    ) -> None:
        """Score every not-yet-scored index pair among `pair_inds` on
        `core`, which must be the core of `net`."""
        for ind_pair in itertools.combinations(pair_inds, 2):
            key = tuple(ind_pair)
            if key in self.comb_corr or key[::-1] in self.comb_corr:
                continue

            svals = net.svals_at(
                core, ind_pair, max_rank=100, with_orthonormal=False
            )
            if len(svals) >= 2:
                self.comb_corr[key] = eff_rank(svals)
            else:
                self.comb_corr[key] = 1

            logger.debug(
                "indices: %s, eff rank: %s, norm: %s, svals: %s, score: %s",
                ind_pair,
                eff_rank(svals),
                sum(svals**2),
                svals,
                self.comb_corr[key],
            )


def _split_merged(
    net: pt.TensorNetwork,
    merged: NodeName,
    q_name: NodeName,
    r_name: NodeName,
    q_inds: Sequence[Index],
) -> None:
    """QR the merged node so that Q (orthonormal) holds `q_inds` and R (the
    new core) holds the rest, then name them `q_name` and `r_name`."""
    merged_inds = net.node_tensor(merged).indices
    lefts = [merged_inds.index(ind) for ind in q_inds]
    q, r = net.qr(merged, lefts)
    nx.relabel_nodes(net.network, {q: q_name, r: r_name}, copy=False)


def _shift_core(net: pt.TensorNetwork, core: NodeName, nxt: NodeName) -> None:
    """Hand the core over to the neighbor `nxt`; both names keep their
    positions and indices.

    The core is QR-split on its own (its indices vs. the bond to `nxt`) and
    only R is contracted into `nxt`. Merging first and splitting the merged
    tensor instead would return a bond of size `min(rows, cols)`, which
    inflates it from `r` to `d * r` on every shift.
    """
    nxt_inds = net.node_tensor(nxt).indices
    core_inds = net.node_tensor(core).indices
    lefts = [i for i, ind in enumerate(core_inds) if ind not in nxt_inds]
    _, r = net.qr(core, lefts)
    net.merge(nxt, r)


def _swap_core(
    state: _TreeScoreState,
    net: pt.TensorNetwork,
    core: NodeName,
    nxt: NodeName,
) -> None:
    """Merge the core into `nxt`, score the pairs on the merged node, then
    split so that the core keeps its free indices but takes the position of
    `nxt`, while `nxt` moves to the core's old position."""
    core_inds = net.node_tensor(core).indices
    core_free = state.free_on_node(net, core)
    nxt_inds = net.node_tensor(nxt).indices
    nxt_free = state.free_on_node(net, nxt)

    net.merge(core, nxt)
    state.score_pairs(net, core, core_free + nxt_free)

    # Q gets nxt's free indices plus the core's other bonds, so it attaches
    # where the core used to be; R gets the core's free indices plus nxt's
    # other bonds.
    q_inds = nxt_free + [
        ind
        for ind in core_inds
        if ind not in nxt_inds and ind not in core_free
    ]
    _split_merged(net, core, nxt, core, q_inds)


def _carry_core(
    state: _TreeScoreState,
    net: pt.TensorNetwork,
    core: NodeName,
    pos: NodeName,
    owned: bool = False,
) -> None:
    """Carry the core's free indices through the tree so they meet every
    partner not scored yet. `pos` is the original node whose position the
    core currently occupies; that node now sits where the core came from,
    so it is the one neighbor not to descend into. `owned` says `net` is a
    private copy nobody reads after this call, so the last branch may
    consume it instead of copying."""
    candidates = []
    for nbr in net.network.neighbors(core):
        if nbr == pos:
            continue
        if state.has_unscored_behind(nbr, exclude=pos):
            candidates.append(nbr)

    for i, nbr in enumerate(candidates):
        if owned and i == len(candidates) - 1:
            branch = net
        else:
            branch = copy.deepcopy(net)

        _swap_core(state, branch, core, nbr)
        # after the swap, `nbr` sits where the core was
        _carry_core(state, branch, core, pos=nbr, owned=True)


def _tour_origins(
    state: _TreeScoreState,
    net: pt.TensorNetwork,
    origin: NodeName,
    parent: Optional[NodeName],
) -> None:
    """Visit every node as origin in DFS order, moving the core along the
    tour edges. `net` must have its core at `origin` on entry, and has it
    there again on exit."""
    state.origins_done.add(origin)
    state.score_pairs(net, origin, state.free_on_node(net, origin))
    _carry_core(state, net, origin, pos=origin)

    for child in list(net.network.neighbors(origin)):
        if child == parent:
            continue
        _shift_core(net, origin, child)
        _tour_origins(state, net, child, origin)
        _shift_core(net, child, origin)


class IndexCluster:
    """Base class for index clustering algorithms."""

    def __init__(self, threshold: int):
        self._threshold = threshold

    @abstractmethod
    def cluster(
        self, net: pt.TensorNetwork, ind_splits: Sequence[IndexOp]
    ) -> Sequence[Sequence[Sequence[Index]]]:
        """Cluster the given indices into groups."""
        raise NotImplementedError


class RandomIndexCluster(IndexCluster):
    """Randomly cluster indices into groups."""

    def __init__(self, threshold: int, rand: bool = True):
        super().__init__(threshold)
        self._rand = rand

    def cluster(
        self, net: pt.TensorNetwork, ind_splits: Sequence[IndexOp]
    ) -> Sequence[Sequence[Sequence[Index]]]:
        # randomly partition the indices into @threshold@ sets
        threshold = self._threshold
        indices = net.free_indices()
        ind_groups = []
        for split_op in ind_splits:
            if not isinstance(split_op, IndexSplit):
                continue

            assert split_op.result is not None
            ind_groups.append(split_op.result)

        for ind in indices:
            if not any(ind in g for g in ind_groups):
                ind_groups.append([ind])

        # seed_all(0)
        if self._rand:
            random.shuffle(ind_groups)

        q, r = divmod(len(ind_groups), threshold)
        group_sizes = [q + 1] * r + [q] * (threshold - r)
        sublists = []
        used_len = 0
        for gsize in group_sizes:
            if gsize == 0:
                continue

            ind_set = ind_groups[used_len : used_len + gsize]
            sublists.append([ind for inds in ind_set for ind in inds])
            used_len += gsize

        assert used_len == len(ind_groups)
        return [sublists]


class SVDIndexCluster(IndexCluster):
    """Cluster indices based on singular values."""

    def cluster(
        self, net: pt.TensorNetwork, ind_splits: Sequence[IndexOp]
    ) -> Sequence[Sequence[Sequence[Index]]]:
        """Consider all possible combinations of indices.

        For each combination, we calculate the correlation matrix of
        the reshaped tensor. If the correlation is high enough,
        we merge the indices.
        """
        indices = net.free_indices()
        threshold = self._threshold
        if len(indices) <= threshold:
            return [], []

        comb_corr = {}
        if len(net.network.nodes) == 1:
            comb_corr = self._single_node_corr(net, indices)
        else:
            comb_corr = self._tree_score(net, indices)

        sorted_comb_corr = sorted(
            comb_corr.items(), key=lambda x: x[1], reverse=False
        )
        logger.debug("sorted combs: %s", list(sorted_comb_corr))

        # start from the largest group and expand until the threshold
        group_size = len(indices) // threshold
        num_groups = min(threshold, len(indices) - threshold)

        # Idea 2: randomly sample a few clusters and pick the top k
        # Idea 1: start from the topmost, second topmost, etc..
        k_ind_sets = []
        for _ in range(5):
            index_sets = self._sample_index_sets(
                sorted_comb_corr, num_groups, group_size, threshold
            )
            k_ind_sets.append(index_sets)

        for index_sets in k_ind_sets:
            logger.debug("getting index clusters: %s", index_sets)

        return k_ind_sets

    @staticmethod
    def _sample_index_sets(
        comb_corr: List[Tuple[Sequence[Index], float]],
        num_groups: int,
        group_size: int,
        threshold: int,
    ) -> List[List[Index]]:
        """One random sample of index groupings from correlation pairs."""
        index_sets = []
        visited: Set[Index] = set()
        for i in range(num_groups):
            group: Set[Index] = set()
            for xs, _ in comb_corr:
                if random.random() < 0.1:
                    continue
                if xs[0] in visited and xs[0] not in group:
                    continue
                if xs[1] in visited and xs[1] not in group:
                    continue
                logger.debug("adding %s to group %s", xs, group)
                group.update(xs)
                visited.update(xs)
                if len(group) >= group_size and i != threshold - 1:
                    break
            if group:
                index_sets.append(list(group))
        return index_sets

    def _collect_inds(
        self,
        net: pt.TensorNetwork,
        visited: Set[NodeName],
        curr_node: NodeName,
    ) -> List[Index]:
        visited.add(curr_node)
        all_inds = []
        free_inds = net.free_indices()
        for ind in net.node_tensor(curr_node).indices:
            if ind in free_inds:
                all_inds.append(ind)

        for nbr in net.network.neighbors(curr_node):
            if nbr not in visited:
                nbr_inds = self._collect_inds(net, visited, nbr)
                all_inds.extend(nbr_inds)

        return all_inds

    def _tree_score(
        self, net: pt.TensorNetwork, indices: Sequence[Index]
    ) -> Dict[Sequence[Index], float]:
        """Score every pair of free indices with a single orthonormalization.

        Every node is visited as an origin along a DFS tour of the tree; the
        core is handed along the tour edges, and at each origin its free
        indices are carried through the rest of the tree so that they meet
        every partner not yet scored.
        """
        state = _TreeScoreState(orig=net, free=indices)

        start = net.end_nodes()[0]
        tmp_net = copy.deepcopy(net)
        tmp_net.orthonormalize(start)
        _tour_origins(state, tmp_net, start, parent=None)

        return state.comb_corr

    def _single_node_corr(
        self, net: pt.TensorNetwork, indices: Sequence[Index]
    ) -> Dict[Sequence[Index], float]:
        comb_corr: Dict[Sequence[Index], float] = {}
        # for single node networks, we can directly compute the SVDs
        for i, ind_i in enumerate(indices):
            for ind_j in indices[i + 1 :]:
                ac = OSplit([ind_i, ind_j])
                svd_params = SValsParams(
                    max_rank=100, orthonormal=True, random_seed=42
                )
                svals = ac.svals(net, svd_params=svd_params)
                if len(svals) >= 2:
                    comb_corr[tuple(ac.indices)] = eff_rank(svals)
                else:
                    comb_corr[tuple(ac.indices)] = 1

                logger.debug(
                    "indices: %s, eff rank: %s, norm: %s, svals: %s,"
                    " score: %s",
                    ac.indices,
                    eff_rank(svals),
                    sum(svals**2),
                    svals,
                    comb_corr[tuple(ac.indices)],
                )

        return comb_corr


class CrossIndexCluster(IndexCluster):
    """Cluster indices using cross approximation rank estimates."""

    def __init__(self, threshold: int, eps: float):
        super().__init__(threshold)

        self._eps = eps

    def cluster(
        self, net: pt.TensorNetwork, ind_splits: Sequence[IndexOp]
    ) -> Sequence[Sequence[Sequence[Index]]]:
        """
        Incrementally run cross until we find a low rank representation.

        Based on that order, we group the indices by the neighbors.
        """

        indices = net.free_indices()
        nodes = [net.node_by_free_index(ind.name) for ind in indices]
        # get the two ends where the nodes have only one nbr in nodes
        ends = []
        for n in nodes:
            nbrs = list(net.network.neighbors(n))
            if len(nbrs) == 1 or not all(nbr in nodes for nbr in nbrs):
                ends.append(n)

        ordered_indices = sorted(
            indices,
            key=lambda x: net.distance(
                ends[0], net.node_by_free_index(x.name)
            ),
        )

        # enumerate the indices one by one
        i = 0
        best_so_far = net
        while i < len(ordered_indices):
            max_so_far = max(ind.size for ind in net.all_indices())
            for j in range(i, len(ordered_indices)):
                indices = ordered_indices[:i]
                indices.append(ordered_indices[j])
                indices.extend(ordered_indices[i:j])
                indices.extend(ordered_indices[j + 1 :])
                tt = rand_tt(indices, [1] * (len(indices) - 1))
                # net_inds = [
                #     ind.with_new_rng(range(ind.size)) for ind in indices
                # ]
                cross_config = CrossConfig(
                    kickrank=5, max_rank=max_so_far, max_iters=max_so_far
                )
                cross_engine = CrossApproximation(
                    net.as_func(net.free_indices()), cross_config
                )
                res = cross_engine.cross(tt, tt.end_nodes()[0], eps=self._eps)
                if res.ranks_and_errors[-1][-1] <= self._eps:
                    logger.debug(
                        "cross result over indices %s is %s", indices, tt
                    )
                    tt_max = max(ind.size for ind in net.all_indices())
                    if tt_max < max_so_far:
                        best_so_far = tt
                        max_so_far = tt_max

            ordered_indices = best_so_far.free_indices()
            i += 1
            logger.debug("choosing the index prefix %s", ordered_indices[:i])

        logger.debug("best ordered tt is %s", best_so_far)
        nbr_cluster = RandomIndexCluster(self._threshold, rand=False)
        return nbr_cluster.cluster(best_so_far, [])

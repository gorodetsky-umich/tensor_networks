"""Top down reshaping search"""

import random
import math
import copy
from typing import Generator, List, Optional, Union, Tuple
import itertools

import sympy

from pytens.search.configuration import SearchConfig
from pytens.search.partition import PartitionSearch
from pytens.algs import TensorNetwork, NodeName
from pytens.types import IndexSplit, IndexMerge, Index
from pytens.search.hierarchical.error_dist import BaseErrorDist


def _create_split_target(
    factors: List[int], selected_factors: List[int]
) -> List[int]:
    remaining_factors = factors[:]
    for f in selected_factors:
        remaining_factors.remove(f)

    remaining_size = math.prod(remaining_factors)
    return list(selected_factors) + [remaining_size]


class SearchState:
    """Hierarchical search state"""

    def __init__(
        self,
        free_indices: List[Index],
        reshape_history: List[Union[IndexMerge, IndexSplit]],
        network: TensorNetwork,
        unused_delta: float,
    ):
        self.free_indices = free_indices
        self.reshape_history = reshape_history
        self.network = network
        self.unused_delta = unused_delta

    def merge_index(self, merge_op: IndexMerge) -> "SearchState":
        """Perform a merge operation on the given node."""
        new_st = copy.deepcopy(self)
        new_net = new_st.network
        new_net.merge_index(merge_op)

        new_indices = []
        for i in new_net.free_indices():
            if i not in self.network.free_indices():
                new_indices.append(i)

        for mi in merge_op.merging_indices:
            new_st.free_indices.remove(mi)

        merge_op.merge_result = new_indices[0]
        new_st.reshape_history.append(merge_op)
        new_st.free_indices.extend(new_indices)

        return new_st

    def split_index(self, split_op: IndexSplit) -> "SearchState":
        """Perform a split operation on the given node."""
        # print("applying split", split_op)
        new_st = copy.deepcopy(self)
        new_net = new_st.network
        new_net.split_index(split_op)
        # print(node)

        old_indices = self.network.free_indices()
        new_indices = []
        for i in new_net.free_indices():
            if i not in old_indices:
                new_indices.append(i)

        ind = split_op.splitting_index
        new_st.free_indices.remove(ind)
        new_st.free_indices.extend(new_indices)
        new_st.reshape_history.append(split_op)

        return new_st


class TopDownSearch:
    """Search for reshaped structures from top to bottom"""

    class SplitResult:
        """Return type for the _split_indices method."""

        def __init__(
            self, refactored: bool, split_info: dict, network: TensorNetwork
        ):
            self.ok = refactored
            self.split_info = split_info
            self.network = network

    def __init__(self, config: SearchConfig):
        self.config = config

    def search(
        self,
        net: TensorNetwork,
        error_dist: BaseErrorDist,
    ) -> Tuple[TensorNetwork, SearchState]:
        """Perform the topdown search starting from the given net"""
        remaining_delta = net.norm() * self.config.engine.eps
        best_network = net
        best_st = None
        init_st = SearchState(net.free_indices(), [], net, 0)
        for st in self._search_at_level(0, init_st, remaining_delta, error_dist):
            for n in st.network.network.nodes:
                network = copy.deepcopy(st.network)
                print("unused_delta", math.sqrt(st.unused_delta))
                network.round(n, delta=math.sqrt(st.unused_delta))
                if network.cost() < best_network.cost():
                    best_network = network
                    best_st = st

        return best_network, best_st

    def _get_merge_op(
        self, indices: List[Index], merge_candidates: List[Index]
    ) -> Generator[IndexMerge, None, None]:
        if self.config.topdown.enable_random:
            # yield one possible result
            merge_indices = sorted(random.sample(merge_candidates, k=2))
            merge_pos = [indices.index(ind) for ind in merge_indices]
            yield IndexMerge(
                merging_indices=merge_indices,
                merging_positions=merge_pos,
            )
        else:
            merge_len = len(merge_candidates) - 1
            if merge_len < 2:
                yield
                return

            for i in range(2, merge_len):
                for comb in itertools.combinations(merge_candidates, i):
                    merge_pos = [indices.index(ind) for ind in comb]
                    yield IndexMerge(
                        merging_indices=comb,
                        merging_positions=merge_pos,
                    )

    def _merge_indices(
        self, st: SearchState, node: NodeName
    ) -> Generator[SearchState, None, None]:
        indices = st.network.network.nodes[node]["tensor"].indices
        while len(indices) > self.config.topdown.group_threshold:
            merge_candidates = []
            for ind in indices:
                if ind in st.free_indices:
                    merge_candidates.append(ind)

            for merge_op in self._get_merge_op(indices, merge_candidates):
                yield st.merge_index(merge_op, node)

    def _get_split_op(
        self, st: SearchState, index: Index
    ) -> Generator[Optional[IndexSplit], None, None]:
        if index not in st.free_indices:
            yield
            return

        res = sympy.factorint(index.size)
        factors = [i for i, n in res.items() for _ in range(n)]
        if len(factors) == 1:
            yield
            return

        if self.config.topdown.enable_random:
            k = random.randint(0, len(factors) - 1)
            selected = random.sample(factors, k=k)
            yield IndexSplit(
                splitting_index=index,
                split_target=_create_split_target(factors, selected),
            )
        else:
            for k in range(0, len(factors) - 1):
                for selected in itertools.combinations(factors, r=k):
                    yield IndexSplit(
                        splitting_index=index,
                        split_target=_create_split_target(factors, selected),
                    )

    def _split_indices(
        self, st: SearchState, node: NodeName
    ) -> Generator[SearchState, None, None]:
        net = st.network
        indices = net.network.nodes[node]["tensor"].indices
        index_splits = itertools.product(
            *[self._get_split_op(st, ind) for ind in indices]
        )

        for index_split in index_splits:
            new_st = copy.deepcopy(st)
            # print(index_split)
            for split_op in index_split:
                if split_op is None:
                    continue

                new_st = new_st.split_index(split_op)
                # To avoid merge in the middle, we need to ensure that
                # none of the splits goes beyond the threshold
                new_net = new_st.network
                new_indices = new_net.network.nodes[node]["tensor"].indices
                ndims = len(new_indices)
                # if ndims > self.config.topdown.group_threshold:
                #     # continue
                #     yield from self._merge_indices(new_st, node)
                #     return

            yield new_st

    def _optimize_subnet(
        self,
        st: SearchState,
        nodes: List[NodeName],
        level: int,
        error_dist: BaseErrorDist,
        remaining_delta: float,
    ):
        """Optimize the children nodes in a given network"""
        node = nodes[0]
        print("before index splitting", node)
        print(st.network)
        for split_result in self._split_indices(st, node):
            print("after index splitting", node)
            print(split_result.network)
            curr_net = split_result.network
            n_indices = curr_net.network.nodes[node]["tensor"].indices
            assert len(n_indices) <= self.config.topdown.group_threshold
            # if len(n_indices) > self.config.topdown.group_threshold:
            #     # We may use some metric later, but let's start with random
            #     self._merge_indices(bn, n)

            split_result.network.orthonormalize(node)
            new_sn = TensorNetwork()
            new_sn.add_node(
                node, split_result.network.network.nodes[node]["tensor"]
            )
            new_st = SearchState(
                split_result.free_indices,
                split_result.reshape_history,
                new_sn,
                split_result.unused_delta,
            )
            for sn_st in self._search_at_level(
                level + 1, new_st, remaining_delta, error_dist
            ):
                optimized_st = copy.deepcopy(sn_st)
                if len(split_result.reshape_history) > 0:
                    split_op = split_result.reshape_history[-1]
                    split_info = {
                        split_op.splitting_index.name: [
                            ind.name for ind in split_op.split_result
                        ]
                    }
                else:
                    split_info = {}

                optimized_st.network = copy.deepcopy(split_result.network)
                optimized_st.network.replace_with(
                    node, sn_st.network, split_info
                )
                # optimized_st.unused_delta = math.sqrt(
                #     sn_st.unused_delta**2 + st.unused_delta**2
                # )
                # print("after replacement")
                # print(optimized_st.network)

                if len(nodes) == 1:
                    # print("nodes length is 1")
                    yield optimized_st
                    return

                yield from self._optimize_subnet(
                    optimized_st,
                    nodes[1:],
                    level,
                    error_dist,
                    remaining_delta,
                )

    def _search_at_level(
        self,
        level: int,
        st: SearchState,
        remaining_delta: float,
        error_dist: BaseErrorDist,
    ) -> Generator[SearchState, None, None]:
        print("Optimizing")
        print(st.network)
        search_engine = PartitionSearch(self.config)
        # decrease the delta budget exponentially
        delta, remaining_delta = error_dist.get_delta(level, remaining_delta)
        result = search_engine.search(st.network, delta=delta)
        bn = result.best_network
        print("best network")
        print(bn)

        next_nodes = list(bn.network.nodes)
        # distribute delta equally to all subnets
        remaining_delta = remaining_delta / math.sqrt(len(next_nodes))

        unused_delta = result.unused_delta**2 + st.unused_delta
        best_st = SearchState(
            st.free_indices, st.reshape_history, bn, unused_delta
        )
        if len(next_nodes) > 1:
            yield from self._optimize_subnet(
                best_st, next_nodes, level + 1, error_dist, remaining_delta
            )
        else:
            best_st.unused_delta = best_st.unused_delta + remaining_delta ** 2
            yield best_st

"""Top down reshaping search"""

import random
import math
import copy
from typing import Tuple
from collections.abc import Callable

import sympy

from pytens.search.configuration import SearchConfig
from pytens.search.partition import PartitionSearch
from pytens.algs import TensorNetwork, NodeName
from pytens.types import IndexSplit, IndexMerge
from pytens.search.hierarchical.error_dist import BaseErrorDist


class TopDownSearch:
    """Search for reshaped structures from top to bottom"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.free_indices = []
        self.reshape_info = []

    def search(
        self,
        net: TensorNetwork,
        error_dist: BaseErrorDist,
    ):
        """Perform the topdown search starting from the given net"""
        self.free_indices = net.free_indices()
        remaining_delta = net.norm() * self.config.engine.eps
        bn, remaining_delta = self._search_at_level(0, net, remaining_delta, error_dist)
        # remaining_delta = math.sqrt(remaining_delta ** 2 - net.norm() ** 2 + bn.norm() ** 2) # ** 0.5
        # print(remaining_delta)
        best_network = bn
        for n in bn.network.nodes:
            network = copy.deepcopy(bn)
            network.round(n, delta=remaining_delta)
            if network.cost() < best_network.cost():
                best_network = network
        return best_network

    def _random_group_indices(self, net: TensorNetwork, node: NodeName):
        indices = net.network.nodes[node]["tensor"].indices
        while len(indices) > self.config.topdown.group_threshold:
            n_free_indices = []
            for ind in indices:
                if ind in self.free_indices:
                    n_free_indices.append(ind)
            # print(n_free_indices)
            old_indices = indices
            merged_indices = sorted(random.sample(n_free_indices, k=2))
            merge_op = IndexMerge(merging_indices = merged_indices,
                                  merging_positions = [indices.index(ind) for ind in merged_indices])
            net.merge_index(merge_op)
            # print("after merging", merged_indices)
            # print(net)
            new_indices = []
            indices = net.network.nodes[node]["tensor"].indices
            # print(indices)
            for i in indices:
                if i not in old_indices:
                    new_indices.append(i)

            for mi in merged_indices:
                self.free_indices.remove(mi)
            
            merge_op.merge_result = new_indices[0]
            self.reshape_info.append(merge_op)
            self.free_indices.extend(new_indices)

    def _random_split_indices(self, net: TensorNetwork, node: NodeName):
        indices = net.network.nodes[node]["tensor"].indices
        refactored = False
        split_info = {}

        for ind in indices:
            if len(indices) > self.config.topdown.group_threshold:
                break

            if ind not in self.free_indices:
                continue

            res = sympy.factorint(ind.size)
            factors = []
            for i, n in res.items():
                factors.extend([i] * n)

            if len(factors) == 1:
                continue

            k = random.randint(0, len(factors) - 1)
            if k != 0:
                random_factors = random.sample(factors, k=k)
                remaining_factors = factors[:]
                for f in random_factors:
                    remaining_factors.remove(f)

                remaining_size = math.prod(remaining_factors)
                
                old_indices = net.free_indices()
                # print("splitting", ind, "into", random_factors)
                # print("current index id", TensorNetwork.next_index_id)
                split_op = IndexSplit(splitting_index = ind, split_target = random_factors + [remaining_size])
                net.split_index(split_op
                )
                self.free_indices.remove(ind)

                new_indices = []
                for i in net.free_indices():
                    if i not in old_indices:
                        new_indices.append(i)

                self.free_indices.extend(new_indices)
                split_info[ind.name] = [ind.name for ind in new_indices]
                
                # split_op.split_result = [ind.name for ind in new_indices]
                self.reshape_info.append(split_op)

                refactored = True

        return refactored, split_info

    def _search_at_level(
        self, level: int, net: TensorNetwork, remaining_delta: float, error_dist: BaseErrorDist,
    ) -> Tuple[TensorNetwork, float]:
        # print("optimizing")
        # print(net)
        search_engine = PartitionSearch(self.config)
        # decrease the delta budget exponentially
        delta, remaining_delta = error_dist.get_delta(level, remaining_delta)
        result = search_engine.search(net, delta=delta)
        bn = result.best_network
        unused_delta = result.unused_delta
        # print(net.norm() ** 2 - bn.norm() ** 2, delta ** 2)

        next_nodes = list(bn.network.nodes)
        # distribute delta equally to all subnets
        remaining_delta = remaining_delta / math.sqrt(len(next_nodes))

        for n in next_nodes:
            # print("before index splitting", n)
            # print(bn)
            ok, split_info = self._random_split_indices(bn, n)
            # print("after index splitting", n)
            # print(bn)
            n_indices = bn.network.nodes[n]["tensor"].indices
            if len(n_indices) > self.config.topdown.group_threshold:
                # We may use some metric later, but let's start with random
                self._random_group_indices(bn, n)

            if ok:
                bn.orthonormalize(n)
                new_sn = TensorNetwork()
                new_sn.add_node(n, bn.network.nodes[n]["tensor"])
                optimized_net, left_delta = self._search_at_level(level + 1, new_sn, remaining_delta, error_dist)

                # print("before replacement")
                # print(bn)
                bn.replace_with(n, optimized_net, split_info)
                # print("after replacement")
                # print(bn)
            else:
                left_delta = remaining_delta

            unused_delta = math.sqrt(unused_delta ** 2 + left_delta ** 2)

        return bn, unused_delta
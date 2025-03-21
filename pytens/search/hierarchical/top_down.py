"""Top down reshaping search"""

import random
import math
import copy
from typing import Generator, List, Optional, Union, Tuple, Dict, Sequence
import itertools
import time
import re

import sympy
import numpy as np

from pytens.search.configuration import SearchConfig
from pytens.search.partition import PartitionSearch
from pytens.algs import TensorNetwork, NodeName, Tensor
from pytens.types import IndexSplit, IndexMerge, Index
from pytens.search.hierarchical.error_dist import BaseErrorDist


def _create_split_target(
    factors: List[int], selected_factors: List[int]
) -> List[int]:
    remaining_factors = factors[:]
    for f in selected_factors:
        remaining_factors.remove(f)

    if len(remaining_factors) == 0:
        return list(selected_factors)

    remaining_size = math.prod(remaining_factors)
    return list(selected_factors) + [remaining_size]


# def _get_last_split_info(reshape_history: List[Union[IndexSplit, IndexMerge]]):
#     last_index = -1
#     while(len(reshape_history) >= abs(last_index)):
#         reshape_op = reshape_history[last_index]
#         if isinstance(reshape_op, IndexSplit):
#             # split_info = {
#             #             reshape_op.splitting_index.name: [
#             #                 ind.name for ind in reshape_op.split_result
#             #             ]
#             #         }
#             # return split_info
#             return reshape_op

#         last_index -= 1

#     return {}


def permute_unique(nums: List[int]) -> Generator[Tuple[int], None, None]:
    nums.sort()
    used = [False] * len(nums)

    def backtrack(pat):
        if len(pat) == len(nums):
            yield tuple(pat[:])
            return

        for i, num in enumerate(nums):
            if used[i]:
                continue
            if i > 0 and num == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            pat.append(num)
            yield from backtrack(pat)
            used[i] = False
            pat.pop()

    yield from backtrack([])


def split_into_chunks(lst, n):
    if n == 1:
        # When n is 1, the only chunk is the entire list
        yield [lst]
    else:
        # for indices in itertools.combinations(range(1, len(lst)), n-1):
        #     partition = []
        #     start_idx = 0
        #     for end_idx in indices + (len(lst),):
        #         partition.append(lst[start_idx:end_idx])
        #         start_idx = end_idx

        #     yield partition
        for i in range(1, len(lst) - n + 2):  # Ensure at least `n` chunks
            for tail in split_into_chunks(lst[i:], n - 1):
                yield [lst[:i]] + tail


def select_factors(
    factors: Dict[int, int], splits_allowed: int
) -> Generator[List[int], None, None]:
    """Select a suitable number of factors for reshaping"""
    # enumerate all possible choices for each factor
    factors_flat = [x for x, c in factors.items() for _ in range(c)]
    # partition the list into splits_allowed groups
    seen = set()
    for factors_perm in permute_unique(factors_flat):
        for chunks in split_into_chunks(factors_perm, splits_allowed + 1):
            chunk_factors = tuple([math.prod(chunk) for chunk in chunks])
            if chunk_factors not in seen:
                seen.add(chunk_factors)
                yield chunk_factors

    # seen = set()
    # for counts in itertools.product(*[range(v + 1) for v in factors.values()]):
    #     if sum(counts) != splits_allowed:
    #         continue

    #     print("selecting", list(counts), "from", factors)
    #     selected_factors, remaining_factors = [], []
    #     for i, (f, c) in enumerate(factors.items()):
    #         selected_factors.extend([f] * counts[i])
    #         remaining_factors.extend([f] * (c - counts[i]))

    #     for permuted_selection in permute_unique(selected_factors):
    #         if len(remaining_factors) == 0:
    #             shape = tuple(permuted_selection)
    #         else:
    #             remaining_size = math.prod(remaining_factors)
    #             shape = tuple(list(permuted_selection) + [remaining_size])

    #         if shape in seen:
    #             continue

    #         seen.add(shape)
    #         yield shape


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


def linear_temperature(start, alpha, k):
    return start - alpha * k


def exp_temperature(start, alpha, k):
    return start * (alpha**k)


def log_temperature(start, alpha, k):
    return start * alpha / math.log(1 + k)


def get_temperature(config: SearchConfig, k: int):
    if config.topdown.temp_schedule == "linear":
        return linear_temperature(
            config.topdown.init_temp, config.topdown.alpha, k
        )


def entropy(prob_dist):
    # To avoid issues with log(0), simply work only with positive probabilities.
    prob_nonzero = prob_dist[prob_dist > 0]
    return -np.sum(prob_nonzero * np.log2(prob_nonzero))


def total_correlation(data):
    total = np.sum(data)
    p = data / total
    h = entropy(p.flatten())
    h_sum = 0.0
    for i in range(len(data.shape)):
        p_i = np.sum(
            p, axis=tuple(x for x in range(len(data.shape)) if x != i)
        )
        h_i = entropy(p_i)
        h_sum += h_i

    total_corr = h_sum - h
    return total_corr


def pearson_correlation(x):
    # assume the data is in the shape of groups x num_samples x features
    x_mean = np.mean(x, axis=1, keepdims=True)
    x = (x - x_mean).reshape(x.shape[0], -1)
    return np.corrcoef(x)


class Rule:
    def __init__(self, pattern, interest_points):
        self.pattern = pattern
        self.interest_points = interest_points

    def __str__(self):
        pattern_str = []
        for i, x in enumerate(self.pattern):
            if i in self.interest_points:
                pattern_str.append(f"({x})")
            else:
                pattern_str.append(str(x))

        return str(pattern_str)

    def match(self, shape: Sequence[int]) -> bool:
        """Check whether this rule can be applied to the given shape."""
        if len(shape) != len(self.pattern):
            return False

        for i, j in zip(self.pattern, shape):
            if j % i != 0:
                return False

        return True

    def join(self, other: "Rule") -> Optional["Rule"]:
        """Find the lowest upper bound for both rules."""
        if len(self.pattern) != len(other.pattern):
            return None

        self_ips = set(self.interest_points)
        other_ips = set(other.interest_points)
        if not self_ips.issubset(other_ips) and not self_ips.issuperset(
            other_ips
        ):
            return None

        relax_ips = sorted(list(self_ips.intersection(other_ips)))
        relax_pattern = []
        for k, (i, j) in enumerate(zip(self.pattern, other.pattern)):
            if k in relax_ips:
                if i != j:
                    return None

                relax_pattern.append(i)
            else:
                relax_pattern.append(sympy.igcd(i, j))

        return Rule(relax_pattern, relax_ips)

    def get_op(
        self, indices: List[Index]
    ) -> Optional[Tuple[IndexMerge, IndexSplit]]:
        """Convert a rule into merge and split operations."""
        if not self.match([ind.size for ind in indices]):
            return None

        merge_inds = []
        for i in self.interest_points:
            merge_inds.append(indices[i])

        sep_sz = [ind.size for ind in merge_inds]
        merge_sz = math.prod(sep_sz)
        merge_result = Index(merge_inds[0].name, merge_sz)
        merge_op = IndexMerge(
            merging_indices=merge_inds, merge_result=merge_result
        )
        split_op = IndexSplit(
            splitting_index=merge_result,
            split_target=sep_sz,
            split_result=merge_inds,
        )
        return merge_op, split_op


def align_shapes(shape1, shape2):
    """Align the given shapes. They should have the same length"""
    j, k = 0, 0
    accj, acck = 1, 1
    equal_points = []
    while j < len(shape1) and k < len(shape2):
        if accj == acck:
            equal_points.append((j, k))
            accj *= shape1[j]
            j += 1
            acck *= shape2[k]
            k += 1
        elif accj < acck:
            accj *= shape1[j]
            j += 1
        else:
            acck *= shape2[k]
            k += 1

    equal_points.append((len(shape1), len(shape2)))
    return equal_points


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

    class Sample:
        def __init__(self, indices, nodes):
            self.indices = indices
            self.nodes = nodes
            self.shape = [ind.size for ind in indices]

    def __init__(self, config: SearchConfig):
        self.config = config

        self.memoization = {}
        self.samples = []
        self.hints = {}

    def mine_hint(self, available_hints, new_sample: Sample):
        
        # results is a mapping from shape to best structure
        rules = []
        for node_indices in new_sample.nodes:
            pattern = list(new_sample.shape)
            interest_points = []
            for i, ind in enumerate(node_indices):
                if ind in new_sample.indices:
                    interest_points.append(i)

            rule = Rule(pattern, interest_points)

            # check whether we can relax this rule by relaxing its context

            # a pattern should not start or end with *
            # Either the prefix is not * or the suffix is not *, otherwise we cannot correctly match the shapes
            interest_points = [i for i, c in enumerate(pattern) if c != "*"]

            if len(interest_points) == 0:
                continue

            # find the first number and the last number
            first = interest_points[0]
            last = interest_points[-1]
            if first + 1 <= len(pattern) / 2:
                rng = range(first)
            else:
                rng = range(first, len(new_sample.shape))

            for i in rng:
                pattern[i] = new_sample.shape[i]

            if last + 1 <= len(pattern) / 2:
                rng = range(last)
            else:
                rng = range(last, len(new_sample.shape))

            for i in rng:
                pattern[i] = new_sample.shape[i]

            # rule = Rule(pattern, interest_points, True)
            rules.append(rule)
            print(rule)

        # do pairwise comparison
        results = {}
        for sample in self.samples:
            if len(new_sample.shape) != len(sample.shape):
                continue

            equal_points = align_shapes(sample.shape, new_sample.shape)
            # print(equal_points)

            # check the indices position on nodes
            hints = []
            for node in sample.nodes:
                indices = []
                for ind in node:
                    if ind in sample.indices:
                        indices.append(sample.indices.index(ind))

                # check for overlapping between indices and partitioned sets
                hint = []
                for j, ep in enumerate(equal_points[1:]):
                    has_hint = True
                    for i in range(equal_points[j][0], ep[0]):
                        if i not in indices:
                            # this set is not completed included in the current node, no hint can be mined
                            has_hint = False
                            break

                    if has_hint:
                        hint.append(j + 1)

                if hint:
                    hints.append(hint)

            # for current hints, we should add here so that below we can check its validity
            # convert current hints to equal_point representation if possible
            for hint in available_hints.get(tuple(sample.shape), []):
                # if a range is split into two sets
                # find a subrange to replace it if there exists
                converted_hint = []
                for rng in hint:
                    start = list(rng)[0]
                    end = list(rng)[-1]
                    supported = True
                    # check whether this rng is a subset of a section
                    for j, ep in enumerate(equal_points[1:]):
                        if end < equal_points[j][0] or start >= ep[0]:
                            continue

                        # has intersection
                        if start <= equal_points[j][0] and ep[0] - 1 <= end:
                            # subset
                            converted_hint.append(j + 1)
                        else:
                            # superset
                            supported = False
                            break

                if supported and converted_hint not in hints:
                    hints.append(converted_hint)

            # print(hints)
            final_hints = []
            for node in new_sample.nodes:
                # verify the hints in the new_sample
                indices = []
                for ind in node:
                    if ind in new_sample.indices:
                        indices.append(new_sample.indices.index(ind))

                for hint in hints:
                    new_hint = []
                    for k in hint:
                        has_hint = True
                        for i in range(
                            equal_points[k - 1][1], equal_points[k][1]
                        ):
                            if i not in indices:
                                has_hint = False

                        if has_hint:
                            new_hint.append(k)

                    if new_hint and len(new_hint) > 1:
                        final_hints.append(new_hint)

            # print(final_hints)

            general_hint = []
            for hint in final_hints:
                general_hint.append(
                    [
                        range(equal_points[j - 1][0], equal_points[j][0])
                        for j in hint
                    ]
                )

            results[tuple(sample.shape)] = general_hint

        self.samples.append(new_sample)
        available_hints.update(results)
        return available_hints

    def with_hint(self, available_hints, tensor):
        # temporarily merge two indices into one
        applied = {}
        for shape, hints in available_hints.items():
            print(shape, hints)
            equal_points = align_shapes(
                shape, [ind.size for ind in tensor.indices]
            )

            # try to apply the hint if possible
            for hint in hints:
                merge = []
                for rng in hint:
                    start = list(rng)[0]
                    end = list(rng)[-1]
                    # check whether this rng is a subset of a section
                    within = False
                    for j, ep in enumerate(equal_points[1:]):
                        if end < equal_points[j][1] or start >= ep[1]:
                            continue

                        print(start, end, equal_points[j][0], ep[0])
                        # has intersection
                        if (
                            start == equal_points[j][0] and ep[0] - 1 <= end
                        ) or (
                            start <= equal_points[j][0] and ep[0] - 1 == end
                        ):
                            # subset
                            merge.extend(range(equal_points[j][1], ep[1]))

                    print(merge, tensor.indices)
                    if len(merge) > 1:
                        applied[tensor.indices[equal_points[j][1]].name] = [
                            tensor.indices[x] for x in merge
                        ]

        print(applied)

        merge_ops, split_ops = [], []
        for merged_name, merge_inds in applied.items():
            merge_result = Index(
                merged_name, math.prod(ind.size for ind in merge_inds)
            )
            merge_op = IndexMerge(
                merging_indices=merge_inds, merge_result=merge_result
            )
            merge_ops.append(merge_op)
            split_op = IndexSplit(
                splitting_index=merge_result,
                split_target=[ind.size for ind in merge_inds],
                split_result=merge_inds,
            )
            split_ops.append(split_op)
            # hd, tl = [], []
            # indices = tensor.indices[:]
            # for i, ind in enumerate(tensor.indices):
            #     if ind in merge_inds:
            #         hd.append(i)
            #         indices.remove(ind)
            #     else:
            #         tl.append(i)

            # print(hd + tl)
            # v = tensor.value.transpose(tuple(hd + tl)).reshape(-1, *[ind.size for ind in indices])
            # indices = [Index(merged_name, math.prod(ind.size for ind in merge_inds))] + indices
            # tensor = Tensor(v, indices)

        return merge_ops, split_ops

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
        # print(net.free_indices())
        for st in self._search_at_level(
            0, init_st, remaining_delta, error_dist, net.free_indices()
        ):
            # print(st.network)
            # print("init cost", st.network.cost())
            for n in st.network.network.nodes:
                network = copy.deepcopy(st.network)
                # print("unused_delta", math.sqrt(st.unused_delta))
                network.round(n, delta=math.sqrt(st.unused_delta))
                if network.cost() < best_network.cost():
                    best_network = network
                    best_st = st

            # print("best cost", best_network.cost())
            # import sys

            # sys.stdout.flush()
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
        if len(indices) > self.config.topdown.group_threshold:
            merge_candidates = []
            for ind in indices:
                if ind in st.free_indices:
                    merge_candidates.append(ind)

            for merge_op in self._get_merge_op(indices, merge_candidates):
                new_st = st.merge_index(merge_op)
                yield from self._merge_indices(new_st, node)
        else:
            yield st

    def _get_split_op(
        self, st: SearchState, index: Index, splits_allowed: int
    ) -> Generator[Optional[IndexSplit], None, None]:
        if index not in st.free_indices or splits_allowed <= 0:
            yield
            return

        res = sympy.factorint(index.size)
        factors = [i for i, n in res.items() for _ in range(n)]
        if len(factors) == 1:
            yield
            return

        if self.config.topdown.enable_random:
            k = random.randint(0, len(factors))
            selected = random.sample(factors, k=k)
            yield IndexSplit(
                splitting_index=index,
                split_target=_create_split_target(factors, selected),
            )
        else:
            # we always try our best to decompose the indices to
            # the maximum number and they subsume higher level reshapes
            # print("searching for factors of", index, "with quota", splits_allowed)
            for split_target in select_factors(res, splits_allowed):
                # for selected in itertools.combinations(factors, r=k):
                yield IndexSplit(
                    splitting_index=index,
                    split_target=split_target,
                )

    def _split_indices(
        self, st: SearchState, node: NodeName
    ) -> Generator[Tuple[bool, SearchState], None, None]:
        net = st.network
        indices = net.network.nodes[node]["tensor"].indices
        index_splits = []

        # distribute the allowed splits between indices
        splits_allowed = self.config.topdown.group_threshold - len(indices)
        # splits_allowed = min(1, splits_allowed)
        max_splits = []
        for ind in indices:
            if ind in st.free_indices:
                factors = sympy.factorint(ind.size)
                max_splits.append(sum(factors.values()) - 1)
            else:
                max_splits.append(0)

        splits_allowed = min(sum(max_splits), splits_allowed)
        for combination in itertools.product(
            *[range(x + 1) for x in max_splits]
        ):
            if sum(combination) == splits_allowed:
                print(
                    "splits distribution", combination, "for indices", indices
                )
                splits = []
                for ind_idx, ind in enumerate(indices):
                    splits.append(
                        self._get_split_op(st, ind, combination[ind_idx])
                    )
                # print(list(combination), list(splits))
                index_splits = itertools.chain(
                    index_splits, itertools.product(*splits)
                )

        # we sort the index_splits in the order of total correlations
        def score_split(index_split: List[IndexSplit]):
            start = time.time()
            print(index_split)
            tensor = net.network.nodes[node]["tensor"]
            reshapes = []
            for split_op in index_split:
                if split_op is not None:
                    start_pos = tensor.indices.index(split_op.splitting_index)
                    tensor = tensor.split_indices(split_op)
                    for pos, sz in enumerate(split_op.split_target):
                        perm = (start_pos + pos,)
                        for i in range(
                            start_pos, start_pos + len(split_op.split_target)
                        ):
                            if i != start_pos + pos:
                                perm += (i,)

                        for i in range(len(tensor.indices)):
                            if i < start_pos or i >= start_pos + len(
                                split_op.split_target
                            ):
                                perm += (i,)

                        print(split_op, perm)
                        reshapes.append(
                            tensor.value.transpose(*perm).reshape(
                                sz, split_op.splitting_index.size // sz, -1
                            )
                        )

            select = False
            for reshape_opt in reshapes:
                if np.mean(pearson_correlation(reshape_opt)) >= 0.5:
                    select = True
                    break

            print("score split", time.time() - start)
            return select

        # index_splits = sorted(index_splits, key=score_split, reverse=True)
        # index_splits = filter(score_split, index_splits)
        seen = set()
        for index_split in index_splits:
            if tuple(index_split) in seen:
                continue

            print(index_split, "not seen previously in", seen)
            seen.add(tuple(index_split))

            # we need to evaluate how long it takes in the scoring function
            # if not score_split(index_split):
            #     print("Skipping split op", index_split)
            #     continue

            refactored = False
            new_st = copy.deepcopy(st)
            print(splits_allowed, index_split)
            for split_op in index_split:
                if split_op is None:
                    continue

                split_op = copy.deepcopy(split_op)

                tmp_net = new_st.network
                tmp_indices = tmp_net.network.nodes[node]["tensor"].indices
                ndims = len(tmp_indices) + len(split_op.split_target) - 1
                if ndims > self.config.topdown.group_threshold:
                    continue

                new_st = new_st.split_index(split_op)
                refactored = True
                # To avoid merge in the middle, we need to ensure that
                # none of the splits goes beyond the threshold
                # new_net = new_st.network
                # new_indices = new_net.network.nodes[node]["tensor"].indices
                # ndims = len(new_indices)
                # if ndims > self.config.topdown.group_threshold:
                #     for merged_st in self._merge_indices(new_st, node):
                #         yield refactored, merged_st

                #     return

            yield refactored, new_st

    def _optimize_subnet_for_node(
        self,
        st: SearchState,
        node: NodeName,
        level: int,
        error_dist: BaseErrorDist,
        remaining_delta: float,
        curr_best: TensorNetwork,
    ):
        """Optimize the children nodes in a given network"""
        # if len(nodes) == 0:
        #     yield st
        #     return

        # node = nodes[0]
        # print("before index splitting", node)
        # print(st.network)
        for ok, split_result in self._split_indices(st, node):
            if not ok:
                # yield from self._optimize_subnet(
                #     st, nodes[1:], level, error_dist, remaining_delta
                # )
                continue

            # print("after index splitting", node)
            # print(split_result.network)
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
                level + 1,
                new_st,
                remaining_delta,
                error_dist,
                st.network.network.nodes[node]["tensor"].indices,
                curr_best,
            ):
                yield (split_result.network, sn_st)
                # optimized_st = copy.deepcopy(sn_st)
                # optimized_st.network = copy.deepcopy(split_result.network)
                # optimized_st.network.replace_with(
                #     node, sn_st.network, sn_st.reshape_history
                # )
                # # optimized_st.unused_delta = math.sqrt(
                # #     sn_st.unused_delta**2 + st.unused_delta**2
                # # )
                # # print("replacing", node)
                # # print("after replacement")
                # # print(optimized_st.network)
                # yield from self._optimize_subnet(
                #     optimized_st,
                #     nodes[1:],
                #     level,
                #     error_dist,
                #     remaining_delta,
                # )

    def _optimize_subnet(
        self,
        st: SearchState,
        nodes: List[NodeName],
        level: int,
        error_dist: BaseErrorDist,
        remaining_delta: float,
    ):
        """Optimize the children nodes in a given network"""
        if len(nodes) == 0:
            yield st
            return

        node = nodes[0]
        # print("before index splitting", node)
        # print(st.network)
        for ok, split_result in self._split_indices(st, node):
            if not ok:
                yield from self._optimize_subnet(
                    st, nodes[1:], level, error_dist, remaining_delta
                )
                continue

            # print("after index splitting", node)
            # print(split_result.network)
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
            # check whether we have seen this shape previously
            # let's first do the simple thing that memoize by shapes
            # create the index mapping
            # k = tuple(ind.size for ind in new_st.network.free_indices())
            # if k in self.memoization:
            #     memoized_st = self.memoization[k]
            #     optimized_st = copy.deepcopy(new_st)
            for sn_st in self._search_at_level(
                level + 1,
                new_st,
                remaining_delta,
                st.network.network.nodes[node]["tensor"].indices,
                error_dist,
            ):
                optimized_st = copy.deepcopy(sn_st)
                optimized_st.network = copy.deepcopy(split_result.network)
                optimized_st.network.replace_with(
                    node, sn_st.network, sn_st.reshape_history
                )
                # optimized_st.unused_delta = math.sqrt(
                #     sn_st.unused_delta**2 + st.unused_delta**2
                # )
                # print("replacing", node)
                # print("after replacement")
                # print(optimized_st.network)
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
        parent_indices,
        curr_best: Optional[TensorNetwork] = None,
    ) -> Generator[SearchState, None, None]:
        # print("Optimizing")
        # print(st.network)
        search_start = time.time()
        search_engine = PartitionSearch(self.config)
        # decrease the delta budget exponentially
        delta, remaining_delta = error_dist.get_delta(level, remaining_delta)
        print("calling partition search on")
        print(st.network)
        print(st.free_indices)
        search_engine.best_network = curr_best
        tensor = list(st.network.network.nodes(data=True))[0][1]["tensor"]
        merge_ops, split_ops = self.with_hint(
            self.hints.get(tuple(parent_indices), {}), tensor
        )
        for merge_op in merge_ops:
            st.network.merge_index(merge_op)
        print("after hint apply")
        print(st.network)

        result = search_engine.search(st.network, delta=delta)
        print("search time:", time.time() - search_start)
        # after the search, we mine the hints
        bn = result.best_network
        # restore the hint indices
        for split_op in split_ops:
            bn.split_index(split_op)

        print("after hint restore")
        print(bn)

        # update the hints
        self.hints[tuple(parent_indices)] = self.mine_hint(
            self.hints.get(tuple(parent_indices), {}),
            TopDownSearch.Sample(
                bn.free_indices(),
                [t["tensor"].indices for _, t in bn.network.nodes(data=True)],
            ),
        )
        print("best network")
        print(bn)
        # print(delta, remaining_delta, result.unused_delta)
        next_nodes = list(bn.network.nodes)
        # distribute delta equally to all subnets
        remaining_delta = remaining_delta / math.sqrt(len(next_nodes))

        unused_delta = result.unused_delta**2 + st.unused_delta
        best_st = SearchState(
            st.free_indices, st.reshape_history, bn, unused_delta
        )
        # For the last node, there can be two cases:
        # 1) this node can be split into two after reshaping
        # 2) this node cannot be split in any reshaping
        # In the enumerative case, we enumerate all possible cases;
        # In the random case, we randomly end this recursion.
        # if len(next_nodes) > 1 or (
        #     self.config.topdown.enable_random and random.random() < 0.5
        # ):
        #     for node in next_nodes:
        #         best_sn_st = None
        #         best_split = None
        #         for (split_net, sn_st) in self._optimize_subnet_for_node(
        #             best_st, node, level + 1, error_dist, remaining_delta
        #         ):
        #             if best_sn_st is None or sn_st.network.cost() < best_sn_st.network.cost():
        #                 best_sn_st = sn_st
        #                 best_split = split_net

        #         if best_sn_st is not None:
        #             best_st.network = best_split
        #             best_st.network.replace_with(node, best_sn_st.network, best_sn_st.reshape_history)
        #             best_st.reshape_history = best_sn_st.reshape_history
        #             best_st.unused_delta = best_sn_st.unused_delta
        #         else:
        #             best_st.unused_delta = best_st.unused_delta + remaining_delta**2

        #     yield best_st
        # else:
        #     best_st.unused_delta = best_st.unused_delta + remaining_delta**2
        #     yield best_st

        # enumerate nodes in the order of their scores
        for node in next_nodes:
            best_sn_st = None
            best_split = None
            for split_net, sn_st in self._optimize_subnet_for_node(
                best_st,
                node,
                level + 1,
                error_dist,
                remaining_delta,
                best_sn_st.network if best_sn_st is not None else None,
            ):
                print(
                    "local search result for",
                    best_st.network.network.nodes[node]["tensor"].indices,
                )
                print(sn_st.network.cost())
                print(sn_st.network)
                if (
                    best_sn_st is None
                    or sn_st.network.cost() < best_sn_st.network.cost()
                ):
                    best_sn_st = sn_st
                    best_split = split_net

            if best_sn_st is not None:
                print("find local optimal")
                best_st.network = best_split
                best_st.network.replace_with(
                    node, best_sn_st.network, best_sn_st.reshape_history
                )
                best_st.free_indices = best_sn_st.free_indices
                best_st.reshape_history = best_sn_st.reshape_history
                best_st.unused_delta = best_sn_st.unused_delta
            else:
                best_st.unused_delta = (
                    best_st.unused_delta + remaining_delta**2
                )

        # self.memoization[tuple(st.network.free_indices())] = best_st
        yield best_st

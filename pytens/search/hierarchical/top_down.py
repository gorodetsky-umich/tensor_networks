"""Top down reshaping search"""

import random
import math
import copy
from typing import Generator, List, Optional, Union, Tuple, Dict, Sequence
import itertools
import time
import enum
import pickle

import sympy
import numpy as np

from pytens.search.configuration import SearchConfig
from pytens.search.partition import PartitionSearch, BAD_SCORE
from pytens.algs import TreeNetwork, NodeName, Tensor
from pytens.types import IndexSplit, IndexMerge, Index
from pytens.search.hierarchical.error_dist import BaseErrorDist
from pytens.search.state import OSplit


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
        network: TreeNetwork,
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


class MatchStatus(enum.Enum):
    NOT_MATCH = 0
    MATCH_REFINE = 1
    MATCH_ABSTRACT = 2


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

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False

        return (
            self.pattern == other.pattern
            and self.interest_points == other.interest_points
        )

    def match(self, shape: Sequence[int]) -> MatchStatus:
        """Check whether this rule can be applied to the given shape.

        There could be three different results:
        - Match with a refined shape
        - Match with a abstract shape
        - Not Match
        """
        if len(shape) != len(self.pattern):
            return MatchStatus.NOT_MATCH

        for k, (i, j) in enumerate(zip(self.pattern, shape)):
            if k in self.interest_points and i != j:
                return MatchStatus.NOT_MATCH

            if j % i == 0:
                return False

        return True

    def subsume(self, other: "Rule") -> bool:
        """Check whether the current rule subsume the other one."""
        if len(other.pattern) != len(self.pattern):
            return False

        # other <= self
        if not set(self.interest_points).issubset(set(other.interest_points)):
            return False

        for k, (i, j) in enumerate(zip(self.pattern, other.pattern)):
            if k in self.interest_points and i != j:
                return False

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


class Sample:
    def __init__(self, indices, nodes):
        self.indices = indices
        self.nodes = nodes
        self.shape = [ind.size for ind in indices]


def mine_rules(available_rules: Sequence[Rule], new_sample: Sample):
    """Mine rules from the new sample result"""
    rules = []
    for node_indices in new_sample.nodes:
        pattern = list(new_sample.shape)
        interest_points = []
        for ind in node_indices:
            if ind in new_sample.indices:
                interest_points.append(new_sample.indices.index(ind))

        if len(interest_points) > 1:
            rule = Rule(pattern, interest_points)

            # check whether we can relax this rule by relaxing its context
            relaxed = False
            # removed = []
            for other in available_rules:
                relax_rule = rule.join(other)
                if (
                    relax_rule is not None
                    and relax_rule not in available_rules
                ):
                    rules.append(relax_rule)
                    # removed.append(other)
                    relaxed = True

            # if not relaxed:
            if rule not in available_rules:
                rules.append(rule)

            # for other in removed:
            #     available_rules.remove(other)

    available_rules.extend(rules)


def apply_rules(
    available_rules: Sequence[Rule], tensor: Tensor
) -> Sequence[Tuple[IndexMerge, IndexSplit]]:
    """Check for applicable rules and generate merge/split operations for these rules."""
    matches = []
    for rule in available_rules:
        if not rule.match([ind.size for ind in tensor.indices]):
            print(rule, "does not match", tensor.indices)
            continue

        print(rule, "matches", tensor.indices)
        subsumed = []
        top_found = False
        for matched_rule in matches:
            if matched_rule.subsume(rule):
                print(matched_rule, "subsumes", rule)
                # replace matched_rule with rule
                subsumed.append(matched_rule)

            if rule.subsume(matched_rule):
                print(rule, "subsumes", matched_rule)
                top_found = True
                break

        for matched_rule in subsumed:
            matches.remove(matched_rule)

        if not top_found:
            matches.append(rule)

    merge_ops, split_ops = [], []
    for rule in matches:
        ops = rule.get_op(tensor.indices)
        if ops is not None:
            merge_ops.append(ops[0])
            split_ops.append(ops[1])

    return merge_ops, split_ops


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


# we sort the index_splits in the order of total correlations
def score_split(
    net: TreeNetwork, node: NodeName, index_split: List[IndexSplit]
):
    start = time.time()
    # print(index_split)
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

                # print(split_op, perm)
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

    # print("score split", time.time() - start)
    return select


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def root(self, u):
        if u not in self.parent or self.parent[u] == u:
            return u

        return self.root(self.parent[u])

    def union(self, u, v):
        u_root = self.root(u)
        v_root = self.root(v)
        if u_root != v_root:
            self.parent[v_root] = u_root

    def size(self, root=None):
        if root is not None:
            results = self.groups(root)
            if len(results) == 0:
                return 0
            return len(results[self.root(root)])

        all_elmts = set()
        for x, xs in self.parent.items():
            all_elmts.add(x)
            all_elmts.add(xs)

        return len(all_elmts)

    def groups(self, root=None):
        results = {}
        if root is not None:
            expect_root = self.root(root)
        else:
            expect_root = None

        for u, v in self.parent.items():
            u_root = self.root(u)
            if root is not None and u_root != expect_root:
                continue

            if u_root not in results:
                results[u_root] = set()

            results[u_root].add(u)
            results[u_root].add(v)

        return results


class TopDownSearch:
    """Search for reshaped structures from top to bottom"""

    class SplitResult:
        """Return type for the _split_indices method."""

        def __init__(
            self, refactored: bool, split_info: dict, network: TreeNetwork
        ):
            self.ok = refactored
            self.split_info = split_info
            self.network = network

    def __init__(self, config: SearchConfig):
        self.config = config

        self.memoization = {}
        self.samples = []
        self.hints = {}

    def search(
        self,
        net: TreeNetwork,
        error_dist: BaseErrorDist,
    ) -> Tuple[TreeNetwork, SearchState]:
        """Perform the topdown search starting from the given net"""
        remaining_delta = net.norm() * self.config.engine.eps
        best_network = net
        best_st = None
        init_st = SearchState(net.free_indices(), [], copy.deepcopy(net), 0)
        # print(net.free_indices())
        for st in self._search_at_level(
            0, init_st, remaining_delta, error_dist, None
        ):
            nodes = list(st.network.network.nodes)
            # print(st.network)
            # print("init cost", st.network.cost())
            for n in nodes:
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

        if self.config.topdown.search_algo == "random":
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

    def merge_by_correlation(
        self,
        net: TreeNetwork,
        search_engine: PartitionSearch,
        delta: float,
        threshold: int = 4,
    ):
        """Consider all possible combinations of indices.

        For each combination, we calculate the correlation matrix of the reshaped tensor.
        If the correlation is high enough, we merge the indices.
        """

        tensor = list(net.network.nodes(data=True))[0][1]["tensor"]
        value = tensor.value
        indices = tensor.indices

        if len(indices) <= threshold:
            return [], []

        shape = [ind.size for ind in indices]

        # single_effranks = {}
        # single_corrs = {}
        # for i in range(len(indices)):
        #     perm = [i] + [x for x in range(len(indices)) if x != i]
        #     value_perm = value.transpose(perm)
        #     value_reshape = value_perm.reshape(indices[i].size, -1)

        #     if self.config.topdown.search_algo == "svd":
        #         # we can use these svd options to search for structure and see which operations are selected by the solver

        #         s = np.linalg.svdvals(value_reshape)
        #         # p = s / np.sum(s)
        #         # # Calculate entropy
        #         # entropy = -np.sum(p * np.log(p+1e-16))
        #         # # Compute effective rank
        #         # # eff_rank = np.exp(entropy)
        #         # single_effranks[i] = entropy
        #         # print(i, "effective rank", eff_rank)
        #         # Count singular values above the threshold
        #         # Calculate cumulative variance
        #         # cumulative_variance = np.cumsum(np.flip(s) ** 2)
        #         # rank = len(s) - np.searchsorted(cumulative_variance, delta ** 2)
        #         # single_ranks[i] = (value_reshape.shape[0] + value_reshape.shape[1]) * rank
        #         # print(i, "delta cost", single_ranks[i])
        #         # corr = np.corrcoef(value_reshape)
        #         # print(i, "correlation", np.mean(np.abs(corr)))
        #     elif self.config.topdown.search_algo == "correlation":
        #         mask = np.zeros(value_reshape.shape[0], dtype=bool)
        #         mask[np.random.choice(value_reshape.shape[0], size=min(50000, value_reshape.shape[0]), replace=False)] = True
        #         sample_data = value_reshape[mask]
        #         corr_res = np.corrcoef(sample_data+ np.random.random(sample_data.shape) * 1e-13)
        #         single_corrs[i] = np.linalg.slogdet(corr_res).logabsdet

        comb_corr = {}
        index_costs = {}
        useless_indices = []
        if self.config.topdown.search_algo == "svd":
            k = 1
            combs = list(itertools.combinations(indices, k))

            if len(indices) % 2 == 0 and k == len(indices) // 2:
                combs = combs[: len(combs) // 2]

            actions = []
            for comb in combs:
                # config = SearchConfig()
                # config.rank_search.k = 1
                # engine = PartitionSearch(config)
                # collect all actions up to size 2

                # for ind in comb:
                #     actions.append(OSplit([ind]))
                actions.append(OSplit(comb))

            # print([str(ac) for ac in actions])
            for res in search_engine.search(
                net, delta=delta, actions=actions, budget=len(actions)
            ):
                print([str(ac) for ac in res.best_actions])

                for ac in res.best_actions:
                    inds = [indices.index(ind) for ind in ac.indices]
                    index_costs[tuple(inds)] = min(
                        -ac.target_size, comb_corr.get(tuple(inds), 0)
                    )

            # for ac in actions:
            #     inds = [indices.index(ind) for ind in ac.indices]
            #     if tuple(inds) not in index_costs:
            #         index_costs[tuple(inds)] = -search_engine.constraint_engine.split_actions[ac][1][-1]

            # print(tuple(inds), comb_corr[tuple(inds)])
            search_engine.reset()
            # used_indices = [indices.index(x) for ac in res.best_actions for x in ac.indices]
            # if res.best_actions is None:
            #     # we cannot get better compression with these ops only
            #     # default to correlation
            #     comb_corr = None

            # else:
            #     for ac in res.best_actions:
            #         if len(ac.indices) == 2:
            #             k = tuple(indices.index(x) for x in ac.indices)
            #             comb_corr[k] = ac.target_size
            #     print([str(ac) for ac in res.best_actions])
            #     for ac in actions:
            #         k = tuple(indices.index(x) for x in ac.indices)
            #         if ac in res.best_actions:
            #             comb_corr[k] = min(res.best_solver_cost, comb_corr.get(k, res.best_solver_cost)) #min(ac.target_size, comb_corr.get(k, BAD_SCORE)) #
            #         else:
            #             comb_corr[k] = BAD_SCORE

        print(index_costs)
        if (
            self.config.topdown.search_algo == "correlation"
            or len(comb_corr) == 0
        ):
            for comb in itertools.combinations(range(len(indices)), 2):
                # if comb[0] in index_costs or comb[1] in index_costs:
                #     continue
                if comb in index_costs:
                    comb_corr[comb] = index_costs[comb]  # - net.cost()
                    continue

                if comb[0] in itertools.chain(*index_costs.keys()) or comb[
                    1
                ] in itertools.chain(*index_costs.keys()):
                    # impossible to choose these
                    continue

                perm = list(comb) + [
                    x for x in range(len(indices)) if x not in comb
                ]
                value_perm = value.transpose(perm)
                others = [x for i, x in enumerate(shape) if i not in comb]
                value_group = value_perm.reshape(-1, *others)
                corr_data = value_group.reshape(-1, np.prod(others))
                # chunk the data into groups
                # if corr_data.shape[0] < corr_data.shape[1]:
                #     corr_data = corr_data.transpose()
                # s = np.linalg.svdvals(corr_data)
                # # comb_corr[comb] = np.mean(-np.diff(s) / s[:-1])
                # # print(s)
                # # print(np.diff(np.diff(s)))
                # # Calculate probabilities
                # p = s / np.sum(s)
                # # Calculate entropy
                # entropy = -np.sum(p * np.log(p + 1e-16))
                # # Compute effective rank
                # # eff_rank = np.exp(entropy)
                # # if we can reduce the effective rank, then that's great
                # # if we cannot, then we should prefer smaller increase of effective ranks
                # comb_corr[comb] = entropy / (single_effranks[comb[0]] * single_effranks[comb[1]])
                # print(comb, comb_corr[comb])
                # print("effective rank difference", )
                # corr = np.corrcoef(corr_data)
                # print(comb, "correlation", np.mean(np.abs(corr)))
                # # Count singular values above the threshold
                # print (eff_rank, np.sum(s > 0.1 * s[0]))
                # Calculate cumulative variance
                # cumulative_variance = np.cumsum(np.flip(s) ** 2)

                # # Determine the effective rank
                # rank = np.searchsorted(cumulative_variance, delta ** 2)
                # cost = rank * (corr_data.shape[0] + corr_data.shape[1])
                # comb_corr[comb] = cost
                # elif self.config.topdown.search_algo == "random correlation":
                # mask = np.zeros(corr_data.shape[0], dtype=bool)
                # mask[np.random.choice(corr_data.shape[0], size=min(50000, corr_data.shape[0]), replace=False)] = True
                # sample_data = corr_data[mask]
                # comb_corr[comb] = np.mean(np.abs(np.corrcoef(sample_data)))
                # if np.isnan(comb_corr[comb]):
                #     comb_corr[comb] = 0
                # print(comb, comb_corr[comb])

                # corr_data = corr_data.transpose()
                # mask = np.zeros(corr_data.shape[0], dtype=bool)
                # mask[np.random.choice(corr_data.shape[0], size=min(50000, corr_data.shape[0]), replace=False)] = True
                # sample_data = corr_data[mask]
                # comb_corr[comb] = np.mean(np.abs(np.corrcoef(sample_data)))
                # if np.isnan(comb_corr[comb]):
                #     comb_corr[comb] = 0
                # corr = np.corrcoef(corr_data)
                # comb_corr[comb] = np.mean((corr))
                # if np.isnan(comb_corr[comb]):
                #     comb_corr[comb] = 0
                # print(comb, comb_corr[comb])

                mask = np.zeros(corr_data.shape[0], dtype=bool)
                mask[
                    np.random.choice(
                        corr_data.shape[0],
                        size=min(50000, corr_data.shape[0]),
                        replace=False,
                    )
                ] = True
                sample_data = corr_data[mask]
                corr_res = np.corrcoef(
                    sample_data + np.random.random(sample_data.shape) * 1e-13
                )
                # comb_cost = index_costs.get(comb[0], 0) + index_costs.get(comb[1], 0)
                # print(np.linalg.matrix_rank(corr_res))
                if self.config.topdown.aggregation == "mean":
                    comb_corr[comb] = -np.mean(np.abs(corr_res))
                elif self.config.topdown.aggregation == "det":
                    comb_corr[comb] = np.linalg.det(corr_res)
                elif self.config.topdown.aggregation == "norm":
                    comb_corr[comb] = np.linalg.norm(corr_res)
                elif self.config.topdown.aggregation == "sval":
                    comb_corr[comb] = np.linalg.svdvals(corr_res)[0]

                # compute the correlation by blocks of rows
                # total_sum, total_cnt = 0, 0
                # step = 75000 # max within the memory limit
                # for i in range(0, corr_data.shape[0], step):
                #     # for j in range(i, corr_data.shape[0], step):
                #     corr = np.corrcoef(corr_data[i:i+step])
                #     total_sum += abs(corr).sum()
                #     total_cnt += corr.size

                # comb_corr[comb] = total_sum / total_cnt
                print(comb, comb_corr[comb])
                import sys

                sys.stdout.flush()

        # vals = list(comb_corr.values())
        # print(np.quantile(vals, 0.25), np.quantile(vals, 0.5), np.quantile(vals, 0.75))
        # comb_corr = sorted(comb_corr.items(), key=lambda x: x[1] if x[1] >= np.quantile(vals, 0.5) else 1.0)
        comb_corr = sorted(comb_corr.items(), key=lambda x: x[1])
        # print(comb_corr)
        # consider different cases to merge indices from high to low
        # until it reaches the target threshold
        merged_indices = set()
        merged_groups = []
        groups = 0
        for comb, score in comb_corr:
            print(comb, score)
            if (
                len(comb) < 2
                or comb[0] in merged_indices
                or comb[1] in merged_indices
            ):
                continue

            print("adding", comb)
            merged_indices.add(comb[0])
            merged_indices.add(comb[1])
            merged_groups.append(comb)
            # merged_indices.union(comb[0], comb[1])
            if (
                len(indices) - len(merged_indices) + len(merged_groups)
                <= threshold
            ):
                break

        # merged_indices = DisjointSet()
        # merged_groups = []
        # for comb in comb_corr:
        #     if merged_indices.size(comb[0]) >= 3 or merged_indices.size(comb[1]) >= 3:
        #         continue

        #     print("adding", comb)
        #     merged_indices.union(comb[0], comb[1])
        #     merged_groups = merged_indices.groups()
        #     # merged_indices.union(comb[0], comb[1])
        #     if len(indices) - merged_indices.size() + len(merged_groups) <= threshold:
        #         break
        # merged_groups = merged_groups.values()

        # create index merge and split to restore
        merge_ops, split_ops = [], []
        for ig, elmts in enumerate(merged_groups):
            elmts = list(elmts)
            # print("group elements", elmts)
            merging_indices = [indices[i] for i in elmts]
            merging_sizes = [ind.size for ind in merging_indices]
            merge_result = Index(f"tmp_merge_{ig}", np.prod(merging_sizes))
            # search_engine.constraint_engine.split_actions[OSplit([merge_result])] = search_engine.constraint_engine.split_actions[OSplit(sorted(merging_indices))]
            merge_op = IndexMerge(
                merging_indices=merging_indices, merge_result=merge_result
            )
            merge_ops.append(merge_op)
            split_op = IndexSplit(
                splitting_index=merge_result,
                split_target=merging_sizes,
                split_result=merging_indices,
            )
            split_ops.append(split_op)

        print(
            len(merge_ops), "remaining order is", len(indices) - len(merge_ops)
        )
        import sys

        sys.stdout.flush()
        return merge_ops, split_ops

    def _split_indices_correlation(
        self, st: SearchState, indices: List[Index]
    ) -> Generator[Tuple[bool, SearchState], None, None]:
        # split each index into most smaller parts
        index_splits = []

        for ind in indices:
            if ind not in st.free_indices:
                continue

            factors = sympy.factorint(ind.size)
            factor_list = [i for i, n in factors.items() for _ in range(n)]
            if len(factor_list) == 1:
                continue

            split_ops = [
                IndexSplit(splitting_index=ind, split_target=factor_perm)
                for factor_perm in permute_unique(factor_list)
            ]
            index_splits.append(split_ops)

        return itertools.product(*index_splits)

    def _split_indices_allowed(self, st: SearchState, indices: List[Index]):
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
                # print(
                #     "splits distribution", combination, "for indices", indices
                # )
                splits = []
                for ind_idx, ind in enumerate(indices):
                    splits.append(
                        self._get_split_op(st, ind, combination[ind_idx])
                    )
                # print(list(combination), list(splits))
                index_splits = itertools.chain(
                    index_splits, itertools.product(*splits)
                )

        return index_splits

    def _split_indices(
        self, st: SearchState, node: NodeName
    ) -> Generator[Tuple[bool, SearchState], None, None]:
        net = st.network
        indices = net.network.nodes[node]["tensor"].indices
        index_splits = []

        if self.config.topdown.search_algo == "enumerate":
            index_splits = self._split_indices_allowed(st, indices)
        elif (
            self.config.topdown.search_algo == "correlation"
            or self.config.topdown.search_algo == "svd"
        ):
            index_splits = self._split_indices_correlation(st, indices)

        # index_splits = sorted(index_splits, key=score_split, reverse=True)
        # index_splits = filter(score_split, index_splits)
        seen = set()
        for index_split in index_splits:
            if tuple(index_split) in seen:
                continue

            # print(index_split, "not seen previously in", seen)
            seen.add(tuple(index_split))

            # we need to evaluate how long it takes in the scoring function
            # if not score_split(index_split):
            #     print("Skipping split op", index_split)
            #     continue

            refactored = False
            new_st = copy.deepcopy(st)
            # print(splits_allowed, index_split)
            for split_op in index_split:
                if split_op is None:
                    continue

                split_op = copy.deepcopy(split_op)

                tmp_net = new_st.network
                tmp_indices = tmp_net.network.nodes[node]["tensor"].indices
                ndims = len(tmp_indices) + len(split_op.split_target) - 1
                if (
                    self.config.topdown.search_algo == "enumerate"
                    and ndims > self.config.topdown.group_threshold
                ):
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
        curr_best: TreeNetwork,
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
            # curr_net = split_result.network
            # n_indices = curr_net.network.nodes[node]["tensor"].indices
            # assert len(n_indices) <= self.config.topdown.group_threshold
            # if len(n_indices) > self.config.topdown.group_threshold:
            #     # We may use some metric later, but let's start with random
            #     self._merge_indices(bn, n)

            split_result.network.orthonormalize(node)
            new_sn = TreeNetwork()
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
                # curr_best,
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
            new_sn = TreeNetwork()
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
        curr_best: Optional[TreeNetwork] = None,
    ) -> Generator[SearchState, None, None]:
        # print("Optimizing")
        # print(st.network)
        search_start = time.time()
        search_engine = PartitionSearch(self.config)
        # decrease the delta budget exponentially
        delta, remaining_delta = error_dist.get_delta(level, remaining_delta)
        # print("calling partition search on")
        # print(st.network)
        # print(st.free_indices)
        # search_engine.best_network = curr_best
        # tensor = list(st.network.network.nodes(data=True))[0][1]["tensor"]
        # merge_ops, split_ops = apply_rules(
        #     self.hints.get(tuple(parent_indices), {}), tensor
        # )
        if (self.config.topdown.search_algo in ["correlation", "svd"]) and (
            parent_indices is not None
            or self.config.topdown.merge_mode == "all"
        ):
            # merge indices by correlation
            begin_corr = time.time()
            merge_ops, split_ops = self.merge_by_correlation(
                st.network, search_engine, delta=delta
            )
            print("correlation time:", time.time() - begin_corr)

            for merge_op in merge_ops:
                st.network.merge_index(merge_op)
            # print("after index merge")
            # print(st.network)

        for result in search_engine.search(st.network, delta=delta):
            # print("search time:", time.time() - search_start)
            # after the search, we mine the hints
            bn = result.best_network

        if (self.config.topdown.search_algo in ["correlation", "svd"]) and (
            parent_indices is not None
            or self.config.topdown.merge_mode == "all"
        ):
            # restore the hint indices
            for split_op in split_ops:
                bn.split_index(split_op)

            # print("after hint restore")
            # print(bn)

        # update the hints
        # available_rules = self.hints.get(tuple(parent_indices), [])
        # bn_nodes = [
        #     t["tensor"].indices for _, t in bn.network.nodes(data=True)
        # ]
        # mine_rules(available_rules, Sample(bn.free_indices(), bn_nodes))
        # self.hints[tuple(parent_indices)] = available_rules
        # print("best network")
        # print(bn)
        # print(delta, remaining_delta, result.unused_delta)
        next_nodes = list(bn.network.nodes)
        # distribute delta equally to all subnets
        remaining_delta = remaining_delta / math.sqrt(len(next_nodes))

        unused_delta = result.unused_delta**2 + st.unused_delta
        best_st = SearchState(st.free_indices, st.reshape_history, bn, 0)
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
                None,
                # best_sn_st.network if best_sn_st is not None else None,
            ):
                # print(
                #     "local search result for",
                #     best_st.network.network.nodes[node]["tensor"].indices,
                # )
                # print(sn_st.network.cost())
                # print(sn_st.network)
                if (
                    best_sn_st is None
                    or sn_st.network.cost() < best_sn_st.network.cost()
                ):
                    best_sn_st = sn_st
                    best_split = split_net

            if best_sn_st is not None:
                # print("find local optimal")
                best_st.network = best_split
                best_st.network.replace_with(
                    node, best_sn_st.network, best_sn_st.reshape_history
                )
                best_st.free_indices = best_sn_st.free_indices
                best_st.reshape_history = best_sn_st.reshape_history
                unused_delta += best_sn_st.unused_delta
            else:
                unused_delta += remaining_delta**2

        best_st.unused_delta = unused_delta
        # self.memoization[tuple(st.network.free_indices())] = best_st
        yield best_st

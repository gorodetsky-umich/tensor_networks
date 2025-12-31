import copy
import itertools
import logging
import math
import operator
import random
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import sympy

from pytens.algs import TreeNetwork
from pytens.search.configuration import ReshapeOption, SearchConfig
from pytens.search.hierarchical.types import (
    HSearchState,
    IndexSplitResult,
    SubnetResult,
)
from pytens.search.stages.base import SearchStage, StageRunParams
from pytens.search.stages.stage_runner import StageRunner
from pytens.search.utils import SearchResult, index_partition
from pytens.types import Index, IndexSplit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _permute_unique(nums: List[int]) -> Sequence[Tuple[int, ...]]:
    nums.sort()
    used = [False] * len(nums)

    def backtrack(pat: List[int]) -> List[Tuple[int, ...]]:
        if len(pat) == len(nums):
            return [tuple(pat[:])]

        results = []
        for i, num in enumerate(nums):
            if used[i]:
                continue
            if i > 0 and num == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            pat.append(num)
            results.extend(backtrack(pat))
            used[i] = False
            pat.pop()

        return results

    return backtrack([])


def _split_into_chunks(
    lst: Sequence[int], n: int
) -> Sequence[List[Sequence[int]]]:
    if n == 1:
        # When n is 1, the only chunk is the entire list
        return [[lst]]

    results = []
    for i in range(1, len(lst) - n + 2):  # Ensure at least `n` chunks
        for tail in _split_into_chunks(lst[i:], n - 1):
            results.append([lst[:i]] + tail)

    return results


def _select_factors(
    factors: Dict[int, int], budget: int
) -> List[Sequence[int]]:
    """Select a suitable number of factors for reshaping"""
    # enumerate all possible choices for each factor
    factors_flat = [x for x, c in factors.items() for _ in range(c)]
    # partition the list into splits_allowed groups
    seen = set()
    results = []
    for factors_perm in _permute_unique(factors_flat):
        for chunks in _split_into_chunks(factors_perm, budget + 1):
            chunk_factors = tuple([math.prod(chunk) for chunk in chunks])
            if chunk_factors not in seen:
                seen.add(chunk_factors)
                results.append(chunk_factors)

    return results


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


class IndexSplitStage(SearchStage):
    """A step that enumerates different options to split indices"""

    def __init__(self, config: SearchConfig):
        super().__init__(config)

    def run(self, runner: StageRunner, params: StageRunParams) -> SearchResult:
        assert params.ctx is not None

        indices = params.ctx.indices
        best_res = None
        for split_result in self._split_indices(params.state, list(indices)):
            result = self._process_split_result(runner, params, split_result)
            res_cost = result.subnet_state.network.cost()

            if best_res is None:
                best_res = result
                continue

            best_cost = best_res.subnet_state.network.cost()
            if res_cost < best_cost:
                best_res = result

        assert best_res is not None            
        return best_res

    def _process_split_result(
        self,
        runner: StageRunner,
        params: StageRunParams,
        split_result: IndexSplitResult,
    ) -> SubnetResult:
        assert params.ctx is not None

        net = split_result.state.network
        new_sn = TreeNetwork()
        new_sn.network = nx.subgraph(net.network, params.ctx.nodes).copy()
        new_st = HSearchState(
            split_result.state.free_indices,
            split_result.state.reshape_history,
            new_sn,
        )

        # new_st.network.orthonormalize(node)
        logger.debug("splitted network: %s", new_st.network)

        # exclude actions that split single internal indices
        # exclusions = []
        # for ind in new_sn.free_indices():
        #     if ind not in split_result.state.free_indices:
        #         exclusions.append(ind)

        sn_params = StageRunParams(new_st, params.delta, [], params.ctx)
        sn_st = runner.run(sn_params).best_state
        assert sn_st is not None
        return SubnetResult(net, new_sn, sn_st)

    def _split_indices(
        self, st: HSearchState, indices: List[Index], compute_data: bool = True
    ) -> Sequence[IndexSplitResult]:
        if not self._config.topdown.reshape_enabled and compute_data:
            return [IndexSplitResult(st, [])]

        index_splits = self._split_indices_on_budget(st, indices, compute_data)

        seen = set()
        result_sts = []
        for index_split in index_splits:
            if tuple(index_split) in seen:
                continue

            seen.add(tuple(index_split))

            refactored = False
            new_st = copy.deepcopy(st)
            # if st.tensor_func is None:

            used_splits = []
            for split_op in index_split:
                split_op = copy.deepcopy(split_op)
                # tmp_indices = new_st.network.node_tensor(node).indices

                # ndims = len(tmp_indices) + len(split_op.shape) - 1
                # if (
                #     self.config.topdown.reshape_algo == ReshapeOption.ENUMERATE
                #     and ndims > self.config.topdown.group_threshold
                # ):
                #     continue

                new_st = new_st.split_index(split_op, compute_data)
                used_splits.append(split_op)
                refactored = True

            if refactored:
                result_sts.append(IndexSplitResult(new_st, used_splits))

        return result_sts

    def _split_indices_on_budget(
        self,
        st: HSearchState,
        indices: Sequence[Index],
        compute_data: bool = True,
    ) -> List[List[IndexSplit]]:
        # distribute the allowed splits between indices

        maxs = []
        for ind in indices:
            if ind in st.free_indices:
                factors = sympy.factorint(ind.size)
                maxs.append(sum(factors.values()) - 1)
            else:
                maxs.append(0)

        all_splits = []
        if self._config.topdown.reshape_algo in (
            ReshapeOption.RANDOM,
            ReshapeOption.ENUMERATE,
        ):
            budget = self._config.topdown.group_threshold - len(indices)
            budget = min(sum(maxs), budget)  # exhaust the budget as possible
            for ind_budget in itertools.product(*[range(x + 1) for x in maxs]):
                if sum(ind_budget) != budget:
                    continue

                splits = []
                for i, ind in enumerate(indices):
                    ind_splits = self._get_split_op(st, ind, ind_budget[i])
                    if len(ind_splits) != 0:
                        splits.append(ind_splits)

                all_splits.extend(list(itertools.product(*splits)))

            return all_splits

        if self._config.topdown.reshape_algo == ReshapeOption.CLUSTER:
            splits = []
            for i, ind in enumerate(indices):
                ind_splits = self._get_split_op(st, ind, maxs[i], compute_data)
                if len(ind_splits) != 0:
                    splits.append(ind_splits)

            all_splits.extend(list(itertools.product(*splits)))

        return all_splits

    def _split_scores(
        self, st: HSearchState, index: Index
    ) -> Dict[int, float]:
        # get the svals decay for each split points of the index size
        node = st.network.node_by_free_index(index.name)
        st.network.orthonormalize(node)

        split_scores = {}
        for n in sympy.divisors(index.size):
            if n in (1, index.size):
                continue

            tmp_net = TreeNetwork()
            tmp_net.network = copy.deepcopy(st.network.network)
            # get indices on one side of the node
            nbrs = list(tmp_net.network.neighbors(node))

            if not nbrs:
                linds, rinds = [], tmp_net.free_indices()
            else:
                linds, rinds = index_partition(tmp_net, node, nbrs[0])

            lres = Index(str(index.name) + "_0", n)
            rres = Index(str(index.name) + "_1", index.size // n)
            tmp_net.split_index(
                IndexSplit(
                    index=index,
                    shape=(n, index.size // n),
                    result=[lres, rres],
                )
            )

            target_inds = linds
            if index in linds:
                target_inds = rinds

            target_inds.append(lres)

            max_rank = 2
            s = tmp_net.random_svals(node, target_inds, max_rank=max_rank)
            split_scores[n] = s[0] / s[min(len(s), max_rank) - 1]
            # print(target_inds, n, split_scores[n])

        return split_scores

    def _get_split_op(
        self,
        st: HSearchState,
        index: Index,
        budget: int,
        compute_data: bool = True,
    ) -> Sequence[IndexSplit]:
        if index not in st.free_indices or budget <= 0:
            return []

        res = sympy.factorint(index.size)
        factors = [i for i, n in res.items() for _ in range(n)]
        if len(factors) == 1:
            return []

        if self.config.topdown.reshape_algo == ReshapeOption.RANDOM:
            k = np.random.randint(0, len(factors))
            selected = random.sample(factors, k=k)
            shape = _create_split_target(factors, selected)
            return [IndexSplit(index=index, shape=shape)]

        if compute_data:
            # filter out the top few
            scores = self._split_scores(st, index)
            # we always try our best to decompose the indices to
            # the maximum number and they subsume higher level reshapes
            shape_with_scores = set()
            for shape in _select_factors(res, budget):
                # Calculate cumulative product sizes and sum their scores
                cumulative_sizes = itertools.accumulate(
                    shape[1:-1], operator.mul, initial=shape[0]
                )
                # transform the shape such that the low scores are excluded
                cumulative_score = [scores[size] for size in cumulative_sizes]
                pos = 0
                while pos < len(shape) - 1:
                    if cumulative_score[pos] < 2:
                        shape = (
                            tuple(shape[:pos])
                            + (shape[pos] * shape[pos + 1],)
                            + tuple(shape[pos + 2 :])
                        )
                        cumulative_score.pop(pos)
                    else:
                        pos += 1

                    # print(shape)

                assert len(cumulative_score) + 1 == len(shape)
                score = sum(cumulative_score)
                shape_with_scores.add((score, shape))

            shape_with_scores = list(shape_with_scores)
            shape_with_scores.sort(reverse=True)
        else:
            shape_with_scores = [
                (1, shape) for shape in _select_factors(res, budget)
            ]

        split_ops = []
        for _, shape in shape_with_scores[:10]:
            # for selected in itertools.combinations(factors, r=k):
            split_ops.append(IndexSplit(index=index, shape=shape))

        return split_ops

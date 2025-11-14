"""Various index clustering algorithms."""

from abc import abstractmethod
from typing import Sequence
import random
import itertools
import copy

from line_profiler import profile
import networkx as nx

from pytens.algs import TreeNetwork, TensorTrain
from pytens.types import Index, SVDAlgorithm
from pytens.search.state import OSplit
from pytens.search.utils import seed_all


class IndexCluster:
    """Base class for index clustering algorithms."""

    def __init__(self, threshold: int):
        self._threshold = threshold

    @abstractmethod
    def cluster(self, net: TreeNetwork) -> Sequence[Sequence[Index]]:
        """Cluster the given indices into groups."""
        raise NotImplementedError


class RandomIndexCluster(IndexCluster):
    """Randomly cluster indices into groups."""

    def __init__(self, threshold: int, rand: bool = True):
        super().__init__(threshold)
        self._rand = rand

    def cluster(self, net: TreeNetwork) -> Sequence[Sequence[Index]]:
        # randomly partition the indices into @threshold@ sets
        threshold = self._threshold
        indices = net.free_indices()

        # seed_all(0)
        if self._rand:
            random.shuffle(indices)

        sublen = len(indices) // threshold
        sublists = []
        while len(sublists) < threshold < len(indices):
            used_len = len(sublists) * sublen
            if len(sublists) == threshold - 1:
                ind_set = indices[used_len:]
            else:
                ind_set = indices[used_len : used_len + sublen]

            sublists.append(ind_set)

        return sublists


class SVDIndexCluster(IndexCluster):
    """Cluster indices based on singular values."""

    @profile
    def cluster(self, net: TreeNetwork) -> Sequence[Sequence[Index]]:
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
        assert isinstance(net, TensorTrain), "clustering should happen on tensor trains"
        # remove duplicate node swapping
        ends = net.end_nodes()
        nodes = nx.shortest_path(net.network, ends[0], ends[1])
        for i, ni in enumerate(nodes):
            tmp_net = copy.deepcopy(net)
            tmp_net.orthonormalize(ni)
            i_inds = tmp_net.node_tensor(ni).indices
            i_free = [ind for ind in i_inds if ind in indices]
            for j, nj in enumerate(nodes[i+1:]):
                # swap n[i] and n[i+j-1]
                if j > 0:
                    tmp_net.swap_nbr([ni, nodes[i+j], nj], ni, nodes[i+j])

                j_inds = tmp_net.node_tensor(nj).indices
                j_free = [ind for ind in j_inds if ind in indices]
                ac = OSplit(i_free + j_free)

                # svd_algo = SVDAlgorithm.MERGE
                # we don't need to repeat the orthonormalization either
                # svals = ac.svals(copy.deepcopy(tmp_net), max_rank=2, algo=SVDAlgorithm.MERGE)
                merged_net = copy.deepcopy(tmp_net)
                merged_net.merge(ni, nj)
                svals = merged_net.svals_at(ni, ac.indices, max_rank=2, with_orthonormal=False)
                # print(svals)
                if len(svals) >= 2:
                    comb_corr[tuple(ac.indices)] = -svals[0] / svals[1]
                else:
                    comb_corr[tuple(ac.indices)] = 0

        comb_corr = sorted(comb_corr.items(), key=lambda x: x[1])

        # start from the largest group and expand until the threshold
        group_size = len(indices) // threshold
        index_sets = []
        visited = set()
        for i in range(threshold):
            group = set()
            for xs, _ in comb_corr:
                if xs[0] in visited and xs[0] not in group:
                    continue

                if xs[1] in visited and xs[1] not in group:
                    continue

                group.update(xs)
                visited.update(xs)

                if len(group) >= group_size and i != threshold - 1:
                    break

            if group:
                index_sets.append(sorted(list(group)))

        return index_sets

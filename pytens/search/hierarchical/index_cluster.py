"""Various index clustering algorithms."""

from abc import abstractmethod
from typing import Sequence, Dict
import random
import itertools
import copy

from line_profiler import profile
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

from pytens.algs import TreeNetwork, TensorTrain
from pytens.types import Index, IndexMerge, IndexOp, IndexSplit, SVDAlgorithm
from pytens.search.state import OSplit
from pytens.search.utils import seed_all


def eff_ranks(svals: np.ndarray):
    s = svals ** 2
    p = s / s.sum()
    return np.exp(-np.sum(p * np.log(p)))

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

    def cluster(self, net: TreeNetwork, ind_splits: Sequence[IndexOp]) -> Sequence[Sequence[Index]]:
        # randomly partition the indices into @threshold@ sets
        threshold = self._threshold
        indices = net.free_indices()
        ind_groups = []
        for split_op in ind_splits:
            assert isinstance(split_op, IndexSplit)
            assert split_op.result is not None
            ind_groups.append(split_op.result)

        for ind in indices:
            if not any(ind in g for g in ind_groups):
                ind_groups.append([ind])

        # assert isinstance(net, TensorTrain)
        # ends = net.end_nodes()
        # ind_groups.sort(key=lambda g: net.distance(net.node_by_free_index(g[0].name), ends[0]))
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

            ind_set = ind_groups[used_len:used_len+gsize]
            sublists.append([ind for inds in ind_set for ind in inds])
            used_len += gsize

        assert used_len == len(ind_groups)
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
        if isinstance(net, TensorTrain):
            comb_corr, similarity = self._tt_corr(net, indices)
            # model = SpectralClustering(n_clusters=threshold, affinity='precomputed_nearest_neighbors', random_state=42)
            # labels = model.fit_predict(similarity)
            # print(labels)
            # ind_sets = [[] for _ in range(threshold)]
            # for i, j in enumerate(labels):
            #     ind_sets[j].append(indices[i])

            # return ind_sets

        elif len(net.network.nodes) == 1:
            comb_corr = self._single_node_corr(net, indices)
        else:
            raise NotImplementedError(
                "SVD-based clustering is only implemented for TT and single-node networks."
            )

        comb_corr = sorted(comb_corr.items(), key=lambda x: x[1])

        # start from the largest group and expand until the threshold
        q, r = divmod(len(indices), threshold)
        group_size = len(indices) // threshold
        group_sizes = [q + 1] * r + [q] * (threshold - r)
        num_groups = min(threshold, len(indices) - threshold)
        index_sets = []
        visited = set()

        for i in range(num_groups):
            group = set()
            for xs, _ in comb_corr:
                # if (len(group) == 0 and (xs[0] in visited or xs[1] in visited)) or (len(group) > 0 and ((xs[0] not in group and xs[1] not in group) or (xs[1] in group and xs[0] in visited) or (xs[0] in group and xs[1] in visited))):
                #     continue

                # group.update(xs)
                # visited.update(xs)

                # if len(group) >= group_sizes[i]:
                #     break

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

        # print(index_sets)
        return index_sets

    def _tt_corr(self, net: TensorTrain, indices: Sequence[Index]) -> Dict[Sequence[Index], float]:
        comb_corr = {}
        # remove duplicate node swapping
        ends = net.end_nodes()
        nodes = nx.shortest_path(net.network, ends[0], ends[1])
        similarity = np.zeros((len(indices), len(indices)))
        for i, ni in enumerate(nodes):
            tmp_net = copy.deepcopy(net)
            tmp_net.orthonormalize(ni)
            i_inds = tmp_net.node_tensor(ni).indices
            i_free = [ind for ind in i_inds if ind in indices]
            for j, nj in enumerate(nodes[i+1:]):
                # swap n[i] and n[i+j-1]
                if j > 0:
                    tmp_net.swap_nbr([ni, nodes[i+j], nj], ni, nodes[i+j])

                # print(tmp_net)


                j_inds = tmp_net.node_tensor(nj).indices
                j_free = [ind for ind in j_inds if ind in indices]
                ac = OSplit(i_free + j_free)

                # svd_algo = SVDAlgorithm.MERGE
                # we don't need to repeat the orthonormalization either
                # svals = ac.svals(copy.deepcopy(tmp_net), max_rank=2, algo=SVDAlgorithm.MERGE)
                merged_net = copy.deepcopy(tmp_net)
                merged_net.merge(ni, nj)
                # print(merged_net)
                # print(ni)
                svals = merged_net.svals_at(ni, ac.indices, max_rank=2, with_orthonormal=False)
                # print(ac.indices, eff_ranks(svals), sum(svals ** 2), svals)
                i_idx = indices.index(i_free[0])
                j_idx = indices.index(j_free[0])
                similarity[i_idx, j_idx] = eff_ranks(svals)
                similarity[j_idx, i_idx] = eff_ranks(svals) 
                if len(svals) >= 2:
                    comb_corr[tuple(ac.indices)] = -svals[0] / svals[1]
                else:
                    comb_corr[tuple(ac.indices)] = 0

        return comb_corr, similarity

    def _single_node_corr(self, net: TreeNetwork, indices: Sequence[Index]) -> Dict[Sequence[Index], float]:
        comb_corr = {}
        # for single node networks, we can directly compute the SVDs
        for i, ind_i in enumerate(indices):
            for j, ind_j in enumerate(indices[i+1:]):
                ac = OSplit([ind_i, ind_j])
                svals = net.svals(ac.indices, max_rank=2, orthonormal=True)
                if len(svals) >= 2:
                    comb_corr[tuple(ac.indices)] = -svals[0] / svals[1]
                else:
                    comb_corr[tuple(ac.indices)] = 0

        return comb_corr

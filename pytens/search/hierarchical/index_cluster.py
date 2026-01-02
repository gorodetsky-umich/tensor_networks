"""Various index clustering algorithms."""

from abc import abstractmethod
from typing import Sequence, Dict
import random
import logging
import copy

from line_profiler import profile
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

from pytens.algs import TreeNetwork, TensorTrain
from pytens.cross.funcs import FuncTensorNetwork
from pytens.types import Index, IndexMerge, IndexOp, IndexSplit, SVDAlgorithm
from pytens.search.state import OSplit
from pytens.search.utils import seed_all
from pytens.cross.cross import cross

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def eff_rank(svals: np.ndarray):
    s = svals ** 2
    p = s / s.sum()
    return np.exp(-np.sum(p * np.log(p)))


def spectrum_similarity(s1, s2):
    p1 = s1**2 / np.sum(s1**2)
    p2 = s2**2 / np.sum(s2**2)
    return np.dot(p1, p2)


class IndexCluster:
    """Base class for index clustering algorithms."""

    def __init__(self, threshold: int):
        self._threshold = threshold

    @abstractmethod
    def cluster(
        self, net: TreeNetwork, ind_splits: Sequence[IndexOp]
    ) -> Sequence[Sequence[Sequence[Index]]]:
        """Cluster the given indices into groups."""
        raise NotImplementedError


class RandomIndexCluster(IndexCluster):
    """Randomly cluster indices into groups."""

    def __init__(self, threshold: int, rand: bool = True):
        super().__init__(threshold)
        self._rand = rand

    def cluster(
        self, net: TreeNetwork, ind_splits: Sequence[IndexOp]
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

            ind_set = ind_groups[used_len : used_len + gsize]
            sublists.append([ind for inds in ind_set for ind in inds])
            used_len += gsize

        assert used_len == len(ind_groups)
        return [sublists]


class SVDIndexCluster(IndexCluster):
    """Cluster indices based on singular values."""

    @profile
    def cluster(
        self, net: TreeNetwork, ind_splits: Sequence[IndexOp]
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
        elif isinstance(net, TensorTrain):
            comb_corr = self._tt_corr(net, indices)
            # model = SpectralClustering(n_clusters=threshold, affinity='precomputed_nearest_neighbors', random_state=42)
            # labels = model.fit_predict(similarity)
            # print(labels)
            # ind_sets = [[] for _ in range(threshold)]
            # for i, j in enumerate(labels):
            #     ind_sets[j].append(indices[i])

            # return ind_sets
            # clusters = self._cluster_dimensions(similarity, threshold, 3)
            # print("!!!!", clusters)

        else:
            raise NotImplementedError(
                "SVD-based clustering is only implemented for TT and single-node networks."
            )

        comb_corr = sorted(comb_corr.items(), key=lambda x: x[1], reverse=False)
        logger.debug("sorted combs: %s", list(comb_corr))

        # start from the largest group and expand until the threshold
        q, r = divmod(len(indices), threshold)
        group_size = len(indices) // threshold
        group_sizes = [q + 1] * r + [q] * (threshold - r)
        num_groups = min(threshold, len(indices) - threshold)

        # Idea 2: randomly sample a few clusters and pick the top k
        # Idea 1: start from the topmost, second topmost, etc..
        k_ind_sets = []
        for k in range(3):
            index_sets = []
            visited = set()
            for i in range(num_groups):
                group = set()
                for xs, _ in comb_corr[i+k if i == 0 else 0:]:
                    # if (len(group) == 0 and (xs[0] in visited or xs[1] in visited)) or (len(group) > 0 and ((xs[0] not in group and xs[1] not in group) or (xs[1] in group and xs[0] in visited) or (xs[0] in group and xs[1] in visited))):
                    #     continue

                    # group.update(xs)
                    # visited.update(xs)

                    # if len(group) >= group_size and i != threshold - 1:
                    #     break

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

            k_ind_sets.append(index_sets)

        for index_sets in k_ind_sets:
            logger.debug("getting index clusters: %s", index_sets)

        return k_ind_sets

    def _cluster_dimensions(
        self, singular_value_matrix, k_clusters, n_neighbors=5
    ):
        """
        Groups tensor dimensions into k clusters using Spectral Clustering.
        Uses KNN sparsification to handle 'flat' singular value distributions.

        Args:
            singular_value_matrix: (d x d) symmetric matrix of pairwise scores.
            k_clusters: The target number of groups.
            n_neighbors: Number of neighbors to keep for graph sparsification.
                        (Try d/k or slightly higher).
        """

        # 1. Preprocessing: Sparsify the graph
        # We convert the dense, noisy matrix into a KNN graph.
        # This forces a structure even if values are close.
        # We use 'precomputed' mode by manually zeroing out weak links first,
        # or rely on the clustering algo's built-in affinity.

        # A robust way: Zero out everything except the top N neighbors for each row
        d = singular_value_matrix.shape[0]
        affinity = np.zeros_like(singular_value_matrix)

        for i in range(d):
            # Get indices of top n_neighbors
            # argsort gives ascending, so we take the last n_neighbors
            top_indices = np.argsort(singular_value_matrix[i, :])[
                -n_neighbors:
            ]
            affinity[i, top_indices] = singular_value_matrix[i, top_indices]

        # Symmetrize (since KNN is directed: i might like j, but j might not like i)
        affinity = 0.5 * (affinity + affinity.T)

        # 2. Spectral Clustering
        # This embeds the graph into k-dims and runs k-means
        sc = SpectralClustering(
            n_clusters=k_clusters, affinity="precomputed", random_state=42
        )

        labels = sc.fit_predict(affinity)

        # Group the results
        clusters = {}
        for i in range(k_clusters):
            clusters[i] = np.where(labels == i)[0].tolist()

        return clusters

    def _tt_corr(
        self, net: TensorTrain, indices: Sequence[Index]
    ) -> Dict[Sequence[Index], float]:
        comb_corr = {}
        # remove duplicate node swapping
        ends = net.end_nodes()
        nodes = nx.shortest_path(net.network, ends[0], ends[1])
        # for i, ni in enumerate(nodes):
        #     if i == 0:
        #         net.orthonormalize(ni)

        #     s = net.svals(net.node_tensor(ni).indices[:1], orthonormal=ni)
        #     logger.debug(
        #         "svals for %s are %s", net.node_tensor(ni).indices[0], s
        #     )

        #     if i == 0:
        #         _, r = net.qr(ni, [0])
        #         net.merge(nodes[i + 1], r)
        #     elif i < len(nodes) - 1:
        #         _, r = net.qr(ni, [0, 2])
        #         net.merge(nodes[i + 1], r)

        for i, ni in enumerate(nodes):
            tmp_net = copy.deepcopy(net)
            tmp_net.orthonormalize(ni)
            i_inds = tmp_net.node_tensor(ni).indices
            i_free = [ind for ind in i_inds if ind in indices]
            for j, nj in enumerate(nodes[i + 1 :]):
                # swap n[i] and n[i+j-1]
                if j > 0:
                    tmp_net.swap_nbr([ni, nodes[i + j], nj], ni, nodes[i + j])

                logger.debug("after swapping nbrs: %s", tmp_net)

                j_inds = tmp_net.node_tensor(nj).indices
                j_free = [ind for ind in j_inds if ind in indices]
                ac = OSplit(i_free + j_free)

                # svd_algo = SVDAlgorithm.MERGE
                # we don't need to repeat the orthonormalization either
                # svals = ac.svals(copy.deepcopy(tmp_net), max_rank=2, algo=SVDAlgorithm.MERGE)
                merged_net = copy.deepcopy(tmp_net)
                merged_net.merge(ni, nj)
                logger.debug("after merge %s and %s: %s", ni, nj, merged_net)
                # print(ni)
                svals = merged_net.svals_at(
                    ni, ac.indices, max_rank=10, with_orthonormal=False
                )

                if len(svals) >= 2:
                    comb_corr[tuple(ac.indices)] = eff_rank(svals) #svals[0] / svals[1]
                else:
                    comb_corr[tuple(ac.indices)] = 1
                # for sval_idx in range(1, min(len(svals), 5)):
                #     comb_corr[tuple(ac.indices)].append(svals[sval_idx - 1] / svals[sval_idx])

                logger.debug(
                    "indices: %s, eff rank: %s, norm: %s, svals: %s, score: %s",
                    ac.indices,
                    eff_rank(svals),
                    sum(svals**2),
                    svals,
                    comb_corr[tuple(ac.indices)],
                )

        return comb_corr

    def _single_node_corr(
        self, net: TreeNetwork, indices: Sequence[Index]
    ) -> Dict[Sequence[Index], float]:
        comb_corr = {}
        # for single node networks, we can directly compute the SVDs
        for i, ind_i in enumerate(indices):
            for j, ind_j in enumerate(indices[i + 1 :]):
                ac = OSplit([ind_i, ind_j])
                svals = net.svals(ac.indices, max_rank=10, orthonormal=True)
                if len(svals) >= 2:
                    comb_corr[tuple(ac.indices)] = eff_rank(svals) #svals[0] / svals[1]
                else:
                    comb_corr[tuple(ac.indices)] = 1

        return comb_corr


class SVDNbrIndexCluster(SVDIndexCluster):
    @profile
    def cluster(
        self, net: TreeNetwork, ind_splits: Sequence[IndexOp]
    ) -> Sequence[Sequence[Index]]:
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
            comb_corr = self._split_scores(net, indices)
        elif len(net.network.nodes) == 1:
            comb_corr = self._single_node_corr(net, indices)
        else:
            raise NotImplementedError(
                "SVD-based clustering is only implemented for TT and single-node networks."
            )

        comb_corr = sorted(comb_corr.items(), key=lambda x: x[1])

        # sort the nodes
        nodes = [net.node_by_free_index(ind.name) for ind in indices]
        # get the two ends where the nodes have only one nbr in nodes
        ends = []
        for n in nodes:
            nbrs = list(net.network.neighbors(n))
            if len(nbrs) == 1 or not all(nbr in nodes for nbr in nbrs):
                ends.append(n)

        ordered_indices = list(
            sorted(
                indices,
                key=lambda x: net.distance(
                    ends[0], net.node_by_free_index(x.name)
                ),
            )
        )

        ind_sets = []
        used_len = 0
        for i, _ in sorted(list(comb_corr)[:threshold]):
            ind_set = ordered_indices[used_len : i + 1]
            ind_sets.append(ind_set)
            used_len = i + 1

        return ind_sets

    def _split_scores(
        self, net: TensorTrain, indices: Sequence[Index]
    ) -> Dict[int, float]:
        # remove duplicate node swapping
        ends = net.end_nodes()
        nodes = nx.shortest_path(net.network, ends[0], ends[1])
        scores = {}
        for i, ni in enumerate(nodes[:-1]):
            if i == 0:
                net.orthonormalize(ni)

            i_inds = net.node_tensor(ni).indices
            i_free = [i_inds[0]]
            if i > 0:
                i_free.append(i_inds[2])

            svals = net.svals_at(
                ni, i_free, max_rank=100, with_orthonormal=False
            )
            scores[i] = eff_rank(svals)  # svals[0] / svals[1]
            logger.debug(
                "indices: %s, svals: %s, norm: %s, score: %s",
                i_free,
                svals,
                sum(svals**2),
                scores[i],
            )

            if i < len(nodes) - 1:
                # move the orthogonality to the right
                _, r = net.qr(ni, [0, 2] if i != 0 else [0])
                net.merge(nodes[i + 1], r)

        return scores


class CrossIndexCluster(IndexCluster):
    def __init__(self, threshold: int, eps: float):
        super().__init__(threshold)

        self._eps = eps

    @profile
    def cluster(
        self, net: TreeNetwork, ind_splits: Sequence[IndexOp]
    ) -> Sequence[Sequence[Index]]:
        """
        Incrementally run cross until we find a low rank representation.

        Based on that order, we group the indices by the neighbors.
        """

        indices = net.free_indices()
        nodes = [net.node_by_free_index(ind.name) for ind in indices]
        # get the two ends where the nodes have only one nbr in nodes
        # TODO: handle single nodes
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
                tt = TensorTrain.rand_tt(indices)
                net_inds = [
                    ind.with_new_rng(range(ind.size)) for ind in indices
                ]
                res = cross(
                    FuncTensorNetwork(net_inds, net),
                    tt,
                    tt.end_nodes()[0],
                    eps=self._eps,
                    kickrank=5,
                    max_rank=max_so_far,
                    max_iters=max_so_far,
                )
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

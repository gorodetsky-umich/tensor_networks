"""Hierarchical Tucker"""

from collections.abc import Sequence
from typing import List

import numpy as np

from pytens.algs import Tensor, TreeNetwork
from pytens.types import Index, NodeName
from pytens.utils import delta_svd


class HierarchicalTucker(TreeNetwork):
    """Class for hierarchical tuckers."""

    @staticmethod
    def rand_ht(
        indices: List[Index], rank: int, child_each_level: int = 2
    ) -> "HierarchicalTucker":
        """Return a random hierarchical tucker."""
        ht = HierarchicalTucker()

        def build_child(
            pid: int, node_id: int, sub_indices: List[Index], rank: int = 1
        ) -> int:
            # print(node_id, sub_indices)
            if len(sub_indices) == 1:
                ind = sub_indices[0]
                val = np.random.random((rank, ind.size))
                node = Tensor(
                    val,
                    [
                        Index(f"R_{pid}_{node_id}", rank),
                        ind,
                    ],
                )
                ht.add_node(f"G{node_id}", node)
                return node_id + 1

            # partition the indices into groups hierarchically,
            # the leftovers are always in the last group
            ind_group_num = child_each_level
            ind_group_size = len(sub_indices) // ind_group_num
            last_group_size = (
                len(sub_indices) - (ind_group_num - 1) * ind_group_size
            )
            next_node_id = node_id + 1

            if pid == -1:
                val = np.random.random([rank] * child_each_level)
                indices = []
            else:
                val = np.random.random([rank] * (child_each_level + 1))
                indices = [Index(f"R_{pid}_{node_id}", rank)]

            for i in range(ind_group_num - 1):
                child_id = next_node_id
                indices.append(Index(f"R_{node_id}_{child_id}", rank))
                next_node_id = build_child(
                    node_id,
                    next_node_id,
                    sub_indices[i * ind_group_size : (i + 1) * ind_group_size],
                    rank,
                )
                ht.add_edge(f"G{child_id}", f"G{node_id}")

            child_id = next_node_id
            indices.append(Index(f"R_{node_id}_{child_id}", rank))
            next_node_id = build_child(
                node_id, next_node_id, sub_indices[-last_group_size:], rank
            )
            ht.add_edge(f"G{child_id}", f"G{node_id}")

            ht.set_node_tensor(f"G{node_id}", Tensor(val, indices))

            return next_node_id

        build_child(-1, 0, indices, rank)
        return ht

    def root(self) -> NodeName:
        """Find the root node for a hierarchical tucker."""

        free_inds = self.free_indices()

        n: NodeName
        for n in self.network.nodes:
            node_inds = self.node_tensor(n).indices
            if len(node_inds) != 2:
                continue

            if any(ind in free_inds for ind in node_inds):
                continue

            return n

        # impossible path
        raise ValueError("Invalid hierarchical tucker, cannot find the root.")

    @staticmethod
    def _build_ht_layers(
        free_inds: Sequence[Index],
    ) -> List[List[Sequence[Index]]]:
        """Build the binary dimension-tree layers from root to leaves."""
        layer: List[Sequence[Index]] = [free_inds]
        layers = [layer]
        while not all(len(ind_group) <= 1 for ind_group in layers[-1]):
            next_layer: List[Sequence[Index]] = []
            for ind_group in layers[-1]:
                if len(ind_group) > 1:
                    mid = len(ind_group) // 2
                    next_layer.append(ind_group[:mid])
                    next_layer.append(ind_group[mid:])
                else:
                    next_layer.append([])
            layers.append(next_layer)
        return layers

    @staticmethod
    def ht_svd(
        data: np.ndarray, free_inds: Sequence[Index], eps: float
    ) -> "HierarchicalTucker":
        """Create a hierarchical tucker for the given data using SVD."""

        ht = HierarchicalTucker()
        delta = np.linalg.norm(data) * eps / np.sqrt(2 * len(data.shape) - 3)

        ind_sizes = [ind.size for ind in free_inds]
        assert data.shape == tuple(ind_sizes)

        layers = HierarchicalTucker._build_ht_layers(free_inds)

        # from leaves to root
        leaf_inds = list(free_inds)[:]
        for level, layer in enumerate(layers[1:][::-1]):
            i = 0
            ind_cnt = 0
            leaf_inds_copy = leaf_inds[:]
            for i, inds in enumerate(layer):
                if len(inds) == 0:
                    ind_cnt += 1
                    continue

                if len(inds) == 1:
                    curr_data = np.moveaxis(data, ind_cnt, 0)
                    res = delta_svd(
                        curr_data.reshape(inds[0].size, -1), delta=delta
                    )
                    assert res.u is not None
                    leaf_ind = Index(f"s_{level}_{i}", res.u.shape[1])
                    ht.add_node(
                        f"n_{level}_{i}",
                        Tensor(res.u, [inds[0], leaf_ind]),
                    )
                    data = np.einsum("a...,ab->...b", curr_data, res.u)
                    data = np.moveaxis(data, -1, i)

                    ind_pos = leaf_inds.index(inds[0])
                    leaf_inds.pop(ind_pos)
                    leaf_inds.insert(ind_pos, leaf_ind)
                    ind_cnt += 1
                else:
                    # build transition nodes
                    # take two leaf inds and make the reshape
                    left_size = (
                        leaf_inds[ind_cnt].size * leaf_inds[ind_cnt + 1].size
                    )
                    curr_data = np.moveaxis(
                        data, [ind_cnt, ind_cnt + 1], [0, 1]
                    )
                    res = delta_svd(
                        curr_data.reshape(left_size, -1), delta=delta
                    )
                    assert res.u is not None
                    leaf_ind = Index(f"s_{level}_{i}", res.u.shape[1])
                    ht.add_node(
                        f"n_{level}_{i}",
                        Tensor(
                            res.u.reshape(
                                leaf_inds[ind_cnt].size,
                                leaf_inds[ind_cnt + 1].size,
                                res.u.shape[1],
                            ),
                            [
                                leaf_inds[ind_cnt],
                                leaf_inds[ind_cnt + 1],
                                leaf_ind,
                            ],
                        ),
                    )
                    data = np.einsum(
                        "a...,ab->...b",
                        curr_data.reshape(left_size, *curr_data.shape[2:]),
                        res.u,
                    )
                    data = np.moveaxis(data, -1, i)

                    idx0 = leaf_inds_copy.index(leaf_inds[ind_cnt])
                    idx1 = leaf_inds_copy.index(leaf_inds[ind_cnt + 1])
                    ht.add_edge(
                        f"n_{level}_{i}",
                        f"n_{level - 1}_{idx0}",
                    )
                    ht.add_edge(
                        f"n_{level}_{i}",
                        f"n_{level - 1}_{idx1}",
                    )

                    leaf_inds.pop(ind_cnt)
                    leaf_inds.pop(ind_cnt)
                    leaf_inds.insert(ind_cnt, leaf_ind)

                    ind_cnt += 1

        ht.add_node("root", Tensor(data, [leaf_inds[0], leaf_inds[1]]))

        if len(layers) > 1:
            ht.add_edge("root", f"n_{len(layers) - 2}_0")
            ht.add_edge("root", f"n_{len(layers) - 2}_1")

        return ht

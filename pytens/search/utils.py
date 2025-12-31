"""Utility function for structure search."""

import os
import random
from typing import Dict, List, Literal, Optional, Self, Tuple, Union

import numpy as np
import pydantic
import torch

from pytens.algs import Tensor, TreeNetwork
from pytens.cross.funcs import (
    CountingFunc,
    MergeFunc,
    PermuteFunc,
    SplitFunc,
    TensorFunc,
)
from pytens.search.state import OSplit, SearchState
from pytens.types import Index, IndexMerge, IndexPermute, IndexSplit, NodeName

DataTensor = Union[TreeNetwork, CountingFunc]


class SearchStats(pydantic.BaseModel):
    """Statistics collected during the search process"""

    best_cost: List[Tuple[float, float]] = []
    costs: List[Tuple[float, float]] = []
    errors: List[Tuple[float, float]] = []
    ops: List[Tuple[float, int]] = []
    unique: Dict[int, int] = {}

    # results
    count: int = 0
    preprocess_time: float = 0.0
    merge_time: float = 0.0
    merge_transform_time: float = 0.0
    cross_time: float = 0.0
    search_start: float = 0.0
    search_end: float = 0.0
    cr_core: float = 0.0
    cr_start: float = 0.0
    re_f: float = 0.0
    re_max: float = 0.0
    init_cross_evals: int = 0
    init_cross_size: int = 0
    search_cross_evals: int = 0

    def incr_unique(self, key: int):
        """Increment the unique counter."""
        self.unique[key] = self.unique.get(key, 0) + 1

    def merge(self, other: "SearchStats") -> None:
        """Merge the other search stats into the current one."""
        self.search_cross_evals += other.search_cross_evals
        self.preprocess_time += other.preprocess_time
        self.merge_time += other.merge_time
        self.merge_transform_time += other.merge_transform_time
        self.cross_time += other.cross_time


class SearchResult:
    """Result returned by the search process"""

    def __init__(
        self,
        stats=SearchStats(),
        best_state=None,
        unused_delta=0.0,
    ):
        self.stats = stats
        self.best_state = best_state
        self.unused_delta = unused_delta
        self.replay_traces = []

    def __lt__(self, other: Self) -> bool:
        if self.best_state is None:
            return False

        if other.best_state is None:
            return True

        if self.best_state < other.best_state:
            return True

        if self.best_state == other.best_state:
            return self.unused_delta > other.unused_delta

        return False

    def update_best_state(self, other: Self) -> Self:
        """Update the field of best_state if other is better"""
        assert other.best_state is not None
        if other < self:
            self.best_state = other.best_state
            self.unused_delta = other.unused_delta

        return self


def approx_error(tensor: Tensor, net: TreeNetwork) -> float:
    """Compute the reconstruction error.

    Given a tensor network TN and the target tensor X,
    it returns ||X - TN|| / ||X||.
    """
    target_free_indices = tensor.indices
    net_free_indices = net.free_indices()
    net_value = net.contract().value
    perm = [net_free_indices.index(i) for i in target_free_indices]
    net_value = net_value.transpose(perm)
    error = float(
        np.linalg.norm(net_value - tensor.value) / np.linalg.norm(tensor.value)
    )
    return error


def log_stats(
    search_stats: SearchStats,
    target_tensor: Tensor,
    ts: float,
    st: SearchState,
    bn: TreeNetwork,
):
    """Log statistics of a given state."""
    search_stats.ops.append((ts, len(st.past_actions)))
    search_stats.costs.append((ts, st.network.cost()))
    err = approx_error(target_tensor, st.network)
    search_stats.errors.append((ts, err))
    search_stats.best_cost.append((ts, bn.cost()))
    ukey = st.network.canonical_structure()
    search_stats.unique[ukey] = search_stats.unique.get(ukey, 0) + 1


def rtol(
    base_tensor: np.ndarray,
    approx_tensor: np.ndarray,
    norm: Literal["F", "M"] = "F",
) -> float:
    """
    Compute the relative error between two given tensors
    on the specified norms.
    """
    if norm == "F":
        return float(
            np.linalg.norm(base_tensor - approx_tensor)
            / np.linalg.norm(base_tensor)
        )

    if norm == "M":
        return float(
            np.max(abs(base_tensor - approx_tensor)) / np.max(abs(base_tensor))
        )

    raise ValueError("unsupported norm type")


def remove_temp_dir(temp_dir, temp_files):
    """Remove temporary npz files"""
    try:
        for temp_file in temp_files:
            os.remove(temp_file)

        if len(os.listdir(temp_dir)) == 0:
            os.rmdir(temp_dir)

    except FileNotFoundError:
        pass


# def reshape_indices(reshape_ops, indices, data):
#     """Get corresponding indices after splitting"""
#     indices = [[ind] for ind in indices]
#     for reshape_op in reshape_ops:
#         new_indices = []
#         new_sizes = []
#         if isinstance(reshape_op, IndexSplit):
#             for ind_group in indices:
#                 new_ind_group = []
#                 for ind in ind_group:
#                     if ind == reshape_op.index:
#                         assert reshape_op.result is not None
#                         new_ind_group.extend(reshape_op.result)
#                     else:
#                         new_ind_group.append(ind)

#                 new_indices.append(new_ind_group)
#                 new_sizes.extend([ind.size for ind in new_ind_group])

#         elif isinstance(reshape_op, IndexMerge):
#             for group_idx, ind_group in enumerate(indices):
#                 new_ind_group = []
#                 for ind in ind_group:
#                     # we should not assume that all merging indices are on the same node
#                     if ind in reshape_op.indices:
#                         unchanged = [
#                             ind
#                             for ind in ind_group
#                             if ind not in reshape_op.indices
#                         ]
#                         assert reshape_op.result is not None
#                         new_ind_group = [reshape_op.result] + unchanged
#                         # we want to permute these indices before comparison
#                         cnt_before = sum(len(g) for g in indices[:group_idx])
#                         cnt_after = sum(
#                             len(g) for g in indices[group_idx + 1 :]
#                         )
#                         curr_perm = [
#                             ind_group.index(ind) + cnt_before
#                             for ind in reshape_op.indices
#                         ] + [
#                             ind_group.index(ind) + cnt_before
#                             for ind in unchanged
#                         ]
#                         prev_perm = list(range(cnt_before))
#                         next_perm = [
#                             cnt_before + len(curr_perm) + i
#                             for i in range(cnt_after)
#                         ]
#                         data = data.transpose(
#                             *(prev_perm + curr_perm + next_perm)
#                         )
#                         break

#                     new_ind_group.append(ind)

#                 new_sizes.extend([ind.size for ind in new_ind_group])
#                 new_indices.append(new_ind_group)

#         data = data.reshape(*new_sizes)
#         indices = new_indices

#     return indices, data


def reshape_indices(reshape_ops, indices, data):
    """Reshape the data tensor according to the operations."""
    for reshape_op in reshape_ops:
        new_indices = []
        assert reshape_op.result is not None

        if isinstance(reshape_op, IndexSplit):
            for ind in indices:
                if ind == reshape_op.index:
                    new_indices.extend(reshape_op.result)
                else:
                    new_indices.append(ind)

        elif isinstance(reshape_op, IndexMerge):
            # find all indices and swap them to the front
            swap_pos = []
            other_pos = []
            for ind in reshape_op.indices:
                swap_pos.append(indices.index(ind))

            for i, ind in enumerate(indices):
                if ind not in reshape_op.indices:
                    new_indices.append(ind)
                    other_pos.append(i)

            new_indices = [reshape_op.result] + new_indices
            data = data.transpose(swap_pos + other_pos)

        else:
            raise TypeError("unknown reshape operation")

        data = data.reshape([ind.size for ind in new_indices])
        indices = new_indices

    return indices, data


def reshape_func(reshape_ops, func):
    """Reshape the function inputs according to the operations."""
    old_func = func
    indices = func.indices
    for reshape_op in reshape_ops:
        if isinstance(reshape_op, IndexSplit):
            # find the source index and replace with result indices
            split_indices = []
            ind_mapping = {}
            for i, ind in enumerate(indices):
                if ind == reshape_op.index:
                    assert reshape_op.result is not None
                    split_indices.extend(reshape_op.result)

                    result_start = len(split_indices) - len(reshape_op.result)
                    result_end = len(split_indices)
                    split_pos = range(result_start, result_end)
                    split_sizes = [x.size for x in reshape_op.result]
                    ind_mapping[i] = (split_pos, split_sizes)
                else:
                    split_indices.append(ind)

            old_func = SplitFunc(split_indices, old_func, ind_mapping)
            indices = split_indices

        elif isinstance(reshape_op, IndexMerge):
            assert reshape_op.result is not None
            merge_indices = [reshape_op.result]
            ind_mapping = {0: []}
            for ind in reshape_op.indices:
                ind_mapping[0].append((indices.index(ind), ind.size))
            for i, ind in enumerate(indices):
                if ind not in reshape_op.indices:
                    merge_indices.append(ind)

            ind_mapping[0] = list(zip(*ind_mapping[0]))
            old_func = MergeFunc(merge_indices, old_func, ind_mapping)
            indices = merge_indices

        elif isinstance(reshape_op, IndexPermute):
            indices = [indices[i] for i in reshape_op.perm]
            old_func = PermuteFunc(indices, old_func, reshape_op.unperm)

        else:
            raise TypeError("Unknown operation type")

    return old_func


def unravel_indices(reshape_ops, indices, data):
    """Get corresponding indices after splitting"""
    for reshape_op in reshape_ops:
        new_indices = []
        new_data = []
        if isinstance(reshape_op, IndexSplit):
            for ind_idx, ind in enumerate(indices):
                if ind == reshape_op.index:
                    assert reshape_op.result is not None
                    new_indices.extend(reshape_op.result)
                    new_sizes = [i.size for i in reshape_op.result]
                    new_data.extend(
                        np.unravel_index(data[:, ind_idx], new_sizes)
                    )
                else:
                    new_indices.append(ind)
                    new_data.append(data[:, ind_idx])

        elif isinstance(reshape_op, IndexMerge):
            idxs = []
            sizes = []
            for ind in reshape_op.indices:
                idxs.append(indices.index(ind))
                sizes.append(ind.size)

            assert reshape_op.result is not None
            new_indices.append(reshape_op.result)
            merged_data = [data[:, idx] for idx in idxs]
            new_data.append(np.ravel_multi_index(merged_data, sizes))

            for ind in indices:
                if ind in reshape_op.indices:
                    continue

                new_indices.append(ind)
                new_data.append(data[:, indices.index(ind)])

        else:
            continue

        data = np.stack(new_data, axis=-1)
        indices = new_indices

    # print(indices)
    # data = np.hstack([np.stack(g, axis=-1) for g in data])
    # return data[:, np.argsort([i for inds in indices for i in inds])]
    return indices, data


def ravel_indices(reshape_ops, indices, data):
    """Get corresponding indices before splitting"""
    indices = [[ind] for ind in indices]
    all_funcs = []
    for reshape_op in reshape_ops:
        new_indices = []
        funcs = []
        if isinstance(reshape_op, IndexSplit):
            for group_idx, ind_group in enumerate(indices):
                new_ind_group = []
                for ind in ind_group:
                    if ind == reshape_op.index:
                        assert reshape_op.result is not None
                        new_ind_group.extend(reshape_op.result)
                    else:
                        new_ind_group.append(ind)

                new_indices.append(new_ind_group)
                new_sizes = [ind.size for ind in new_ind_group]
                funcs.append(new_sizes)

        elif isinstance(reshape_op, IndexMerge):
            for group_idx, ind_group in enumerate(indices):
                new_ind_group = []
                for ind in ind_group:
                    if ind in reshape_op.indices:
                        unchanged = [
                            ind
                            for ind in ind_group
                            if ind not in reshape_op.indices
                        ]
                        assert reshape_op.result is not None
                        new_ind_group = [reshape_op.result] + unchanged
                        # we want to permute these indices before comparison
                        cnt_before = sum(len(g) for g in indices[:group_idx])
                        cnt_after = sum(
                            len(g) for g in indices[group_idx + 1 :]
                        )
                        curr_perm = [
                            ind_group.index(ind) + cnt_before
                            for ind in reshape_op.indices
                        ] + [
                            ind_group.index(ind) + cnt_before
                            for ind in unchanged
                        ]
                        prev_perm = list(range(cnt_before))
                        next_perm = [
                            cnt_before + len(curr_perm) + i
                            for i in range(cnt_after)
                        ]
                        data = data[:, *(prev_perm + curr_perm + next_perm)]
                        break

                    new_ind_group.append(ind)

                # new_sizes.extend([ind.size for ind in new_ind_group])
                new_indices.append(new_ind_group)

        indices = new_indices
        all_funcs.append(funcs)

    # print(indices)

    return data[:, np.argsort([i for inds in indices for i in inds])]


def init_state(data_tensor: DataTensor, delta) -> SearchState:
    """Create initial search state for the input data tensor."""
    # print(type(data_tensor))
    if isinstance(data_tensor, TreeNetwork):
        return SearchState(data_tensor, delta)

    if isinstance(data_tensor, TensorFunc):
        net = TreeNetwork()
        net.add_node(
            "G0",
            Tensor(
                np.empty([0 for _ in data_tensor.indices]), data_tensor.indices
            ),
        )
        return SearchState(net, delta)

    raise TypeError(
        f"Expect data tensors to have types TreeNetwork or TensorFunc, "
        f"but get {type(data_tensor)}"
    )


def index_partition(
    net: TreeNetwork, node1: NodeName, node2: NodeName
) -> Tuple[List[Index], List[Index]]:
    """Compute the partition of the index by the given edge."""

    def indices_of(start: NodeName, exclude: NodeName):
        visited = set()
        queue = [start]
        indices = []
        while len(queue) > 0:
            n = queue.pop(0)
            if n == exclude:
                continue

            visited.add(n)
            for ind in net.node_tensor(n).indices:
                if ind in net.free_indices():
                    indices.append(ind)

            for nbr in net.network.neighbors(n):
                if nbr not in visited:
                    queue.append(nbr)

        return indices

    return indices_of(node1, node2), indices_of(node2, node1)


def to_splits(net: TreeNetwork) -> List[OSplit]:
    """Convert a tree network into a list of OSplits."""
    free_indices = net.free_indices()
    tree = net.network
    nodelist = list(tree.nodes)

    # Step 1: Prepare data structures
    subtree_indices = {}
    parent = {}
    visited = set()

    # Step 2: Post-order DFS to compute subtree indices
    def dfs(node, p):
        indices = []
        visited.add(node)
        parent[node] = p
        # Add this node's indices
        for ind in net.node_tensor(node).indices:
            if ind in free_indices:
                indices.append(ind)
        # Visit children
        for nbr in tree.neighbors(node):
            if nbr == p:
                continue
            indices.extend(dfs(nbr, node))
        subtree_indices[node] = indices
        return indices

    root = nodelist[0]
    dfs(root, None)

    # All free indices are now the indices of the whole tree
    all_indices = subtree_indices[root]

    actions = []
    # Step 3: For each edge, produce splits
    for n1, n2 in tree.edges:
        # Ensure n1 is the parent and n2 is the child in rooted tree
        if parent[n2] == n1:
            child = n2
        elif parent[n1] == n2:
            child = n1
        else:
            raise ValueError("The tree structure is invalid.")

        inds1 = subtree_indices[child]
        inds2 = [
            ind for ind in all_indices if ind not in inds1
        ]  # The complement

        ac1 = OSplit(inds1, reversible=True, reverse_edge=(n1, n2))
        ac2 = OSplit(inds2, reversible=True, reverse_edge=(n1, n2))
        actions.append(min(ac1, ac2))

    return actions


def get_conflicts(ac: OSplit, past_acs: List[OSplit]) -> Optional[OSplit]:
    """Get the list of conflict actions."""
    ac_indices = set(ac.indices)
    for past_ac in past_acs:
        past_indices = set(past_ac.indices)
        if (
            len(ac_indices.intersection(past_indices)) > 0
            and not ac_indices.issubset(past_indices)
            and not ac_indices.issuperset(past_indices)
        ) or ac == past_ac:
            if past_ac.reversible:
                return past_ac
            else:
                print(
                    "Warning: the action",
                    past_ac,
                    "conflicts with",
                    ac,
                    "but it is not reversible",
                )

    return None


def seed_all(seed_value: int) -> None:
    """
    Sets the random seed for reproducibility across Python's random module,
    NumPy, and PyTorch (both CPU and CUDA).
    Also sets the PYTHONHASHSEED environment variable.
    """

    # Set Python's built-in random seed
    random.seed(seed_value)

    # Set NumPy's random seed
    np.random.seed(seed_value)

    # Set PyTorch's random seed for all devices (CPU and CUDA)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups

    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

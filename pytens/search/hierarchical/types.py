"""Type definitions for hierarchical search"""

from dataclasses import dataclass
from typing import List, Sequence, Self
import copy
from pytens.search.types import Action
from pytens.search.utils import SearchResult, SearchStats
from pytens.algs import TreeNetwork
from pytens.types import Index, IndexOp, IndexMerge, IndexSplit, IndexName


class HSearchState:
    """Hierarchical search state"""

    def __init__(
        self,
        free_indices: List[Index],
        reshape_history: List[IndexOp],
        network: TreeNetwork,
        unused_delta: float = 0,
    ):
        self.free_indices = free_indices
        self.reshape_history = reshape_history
        self.network = network
        self.unused_delta = unused_delta
        self.replay_traces = []
        self.level = 0

    # after the cross, we do the normal but the data tensor is a tensor network.
    def merge_index(self, merge_op: IndexMerge) -> Self:
        """Perform a merge operation on the given node."""
        new_st = copy.deepcopy(self)
        new_st.network = new_st.network.merge_index(merge_op)
        new_net = new_st.network
        assert new_net is not None, "merge operation failed"

        new_indices = []
        for i in new_net.free_indices():
            if i not in self.network.free_indices():
                new_indices.append(i)

        for mi in merge_op.indices:
            new_st.free_indices.remove(mi)

        merge_op.result = new_indices[0]
        new_st.reshape_history.append(merge_op)
        new_st.free_indices.extend(new_indices)

        return new_st

    def split_index(
        self, split_op: IndexSplit, compute_data: bool = True
    ) -> Self:
        """Perform a split operation on the given node."""
        # print("applying split", split_op)
        new_st = copy.deepcopy(self)
        new_net = new_st.network
        if len(split_op.shape) == 1:
            return new_st

        new_net.split_index(split_op, compute_data=compute_data)
        # print(node)

        old_indices = self.network.free_indices()
        new_indices = []
        for i in new_net.free_indices():
            if i not in old_indices:
                new_indices.append(i)

        ind = split_op.index
        new_st.free_indices.remove(ind)
        new_st.free_indices.extend(new_indices)
        new_st.reshape_history.append(split_op)

        return new_st


class TopDownSearchResult(SearchResult):
    def __init__(
        self,
        stats=SearchStats(),
        best_state=None,
        unused_delta=0.0,
        init_splits=0,
        valid_set=None,
        valid_indices=None,
        reshape_history=None,
    ):
        super().__init__(stats, best_state, unused_delta)
        self.init_splits = init_splits
        self.valid_set = valid_set
        self.valid_indices = valid_indices
        self.reshape_history = reshape_history


class SubnetResult:
    """Result for optimizing a subnet."""

    def __init__(
        self, network: TreeNetwork, subnet: TreeNetwork, state: HSearchState
    ):
        self.subnet_state = state
        self.subnet = subnet
        self.network = network


class IndexSplitResult:
    """Result for index splits of one node."""

    def __init__(self, state: HSearchState, splits: Sequence[IndexSplit]):
        self.state = state
        self.splits = splits


class SuperIndex(Index):
    """A symbolic grouping of indices."""

    def __init__(
        self,
        name: IndexName,
        size: int,
        constituents: Sequence[Index],
        values: Sequence[float] = tuple([]),
    ):
        super().__init__(name, size, values)
        self.constituents = constituents

@dataclass
class ReplayTrace:
    level: int
    splits: Sequence[IndexSplit]
    merge_ops: Sequence[IndexMerge]
    split_ops: Sequence[IndexSplit]
    actions: Sequence[Action]

@dataclass
class ReplaySweep:
    indices: Sequence[Index]
    traces: Sequence[ReplayTrace]
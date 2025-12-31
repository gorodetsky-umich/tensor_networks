"""Interfaces for all search decorators."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from pytens.algs import TreeNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.hierarchical.types import HSearchState
from pytens.search.stages.stage_runner import StageRunner
from pytens.search.utils import SearchResult, SearchStats
from pytens.types import Index, IndexMerge, NodeName

@dataclass
class StageContext:
    """Context information for graph processing"""
    nodes: Sequence[NodeName]
    indices: Sequence[Index]
    is_modified: bool

@dataclass
class StageRunParams:
    """Parameters used during stage running"""

    state: HSearchState
    delta: float
    merge_ops: Sequence[IndexMerge]
    ctx: Optional[StageContext] = None
    is_top: bool = False

    @property
    def is_modified(self):
        """Whether the current network has been modified by any optimizer"""
        return self.ctx is not None and self.ctx.is_modified


class SearchStage:
    """The base class for search decorators."""

    def __init__(self, config: SearchConfig):
        self._config = config
        self._stats = SearchStats()

    @abstractmethod
    def run(self, runner: StageRunner, params: StageRunParams) -> SearchResult:
        """Perform the search over the given state and error tolerance."""
        raise NotImplementedError

    def _add_merge_transform_time(self, mtime: float):
        self._stats.merge_transform_time += mtime

    def _add_merge_time(self, mtime: float):
        self._stats.merge_time += mtime

"""Interface for all search algorithms"""

from abc import abstractmethod
from typing import Sequence
from pytens.search.configuration import SearchConfig
from pytens.search.hierarchical.types import HSearchState
from pytens.search.utils import SearchResult, SearchStats
from pytens.types import Index, IndexMerge


class SearchAlgo:
    """The base search algorithm"""

    def __init__(self, config: SearchConfig):
        self.config = config

        self.stats = SearchStats()

    @abstractmethod
    def search(
        self,
        st: HSearchState,
        delta: float,
        merge_ops: Sequence[IndexMerge],
        is_top: bool = False,
    ) -> SearchResult:
        """Perform the search over the given state and error tolerance."""
        raise NotImplementedError

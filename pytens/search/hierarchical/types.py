from pytens.search.utils import SearchResult, SearchStats


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
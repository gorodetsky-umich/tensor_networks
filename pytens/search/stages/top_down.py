"""The top down search algorithm"""

import copy
from typing import List, Sequence
import math
import logging

import networkx as nx
import numpy as np

from pytens.algs import TreeNetwork
from pytens.search.algs.base import SearchAlgo
from pytens.search.configuration import SearchConfig
from pytens.search.stages.base import SearchStage, StageRunParams
from pytens.search.hierarchical.error_dist import AlphaErrorDist
from pytens.search.hierarchical.types import HSearchState, SubnetResult
from pytens.search.hierarchical.utils import DisjointSet
from pytens.search.stages.stage_runner import StageRunner
from pytens.search.utils import SearchResult
from pytens.types import IndexMerge

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TopDownDecorator(SearchStage):
    """The top down search implementation."""

    def __init__(self, config: SearchConfig):
        super().__init__(config)

        self._subnet_algo = SearchAlgo(config)
        self._error_dist = AlphaErrorDist(alpha=config.topdown.alpha)

    def run(self, runner: StageRunner, params: StageRunParams) -> SearchResult:
        """Search the optimized network structure with index merging."""
        # decrease the delta budget exponentially
        delta, remaining_delta = self._error_dist.split_delta(params.delta)

        st = params.state
        free_inds = st.network.free_indices()

        result = runner.run(st, delta, [], True)
        assert result.best_state is not None
        bn = result.best_state.network

        bn_free_inds = bn.free_indices()

        next_nets = self._get_next_nets(bn)

        


    

    

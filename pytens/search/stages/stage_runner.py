"""Running stages"""

from typing import List

from pytens.search.stages.base import SearchStage, StageRunParams
from pytens.search.utils import SearchResult


class StageRunner:
    """The runner process for a sequence of stages"""

    def __init__(self):
        self._stages: List[SearchStage] = []

    def add_stage(self, stage: SearchStage):
        """Add a new stage to the runner"""
        self._stages.append(stage)

    def run(
        self,
        params: StageRunParams
    ) -> SearchResult:
        """Run the stages in order"""

        result = SearchResult()
        for stage in self._stages:
            result = stage.run(self, params)
            assert result.best_state is not None
            params.state = result.best_state

        return result

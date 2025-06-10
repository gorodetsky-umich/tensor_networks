"""Configuration fields for the structure search process."""

from typing import Literal, Optional

import pydantic


class HeuristicConfig(pydantic.BaseModel):
    """Configuration for pruning heuristics"""

    prune_full_rank: bool = pydantic.Field(
        default=False,
        description="Prune away structures with full ranks after each split",
    )
    prune_duplicates: bool = pydantic.Field(
        default=False,
        description="Prune away seen topologies during search (ignore ranks)",
    )
    prune_by_ranks: bool = pydantic.Field(
        default=True,
        description=(
            "Prune away seen structures during search."
            "Used together with prune_duplicates."
        ),
    )


class RankSearchConfig(pydantic.BaseModel):
    """Configuration for the rank search phase"""

    error_split_stepsize: int = pydantic.Field(
        default=1,
        description="The number of different ranks considered for each split",
    )
    fit_mode: Literal["topk", "all"] = pydantic.Field(
        default="topk",
        description=(
            "The choice of rank search algorithm"
            "topk: choose the topk sketches by constraint solving"
            "all: try rank search for all and select the best"
        ),
    )
    k: int = pydantic.Field(
        default=1,
        description=(
            "The number of optimality selected from constraint solving"
            "(Used together with fit_mode==topk)"
        ),
    )


class ProgramSearchConfig(pydantic.BaseModel):
    """Configuration for search with program synthesis"""

    bin_size: float = pydantic.Field(
        default=0.1,
        description=(
            "The singular values will be grouped if "
            "their square sum is in the same bin_size * tensor norm"
        ),
    )
    action_type: Literal["isplit", "osplit"] = pydantic.Field(
        default="osplit",
        description=(
            "The choice of split actions"
            "isplit: input-directed split operations"
            "osplit: output-directed split operations"
        ),
    )
    replay_from: Optional[str] = pydantic.Field(
        default=None,
        description="Config to replay a series of splits from a pickle file",
    )


class SearchEngineConfig(pydantic.BaseModel):
    """Configuration for the search engine"""

    eps: float = pydantic.Field(
        default=0.1,
        description="The relative error bound for the tensor network repr",
    )
    max_ops: int = pydantic.Field(
        default=5,
        description="The maximum number of split operations",
    )
    timeout: Optional[float] = pydantic.Field(
        default=None,
        description="The maximum amount of time used for search",
    )
    verbose: bool = pydantic.Field(
        default=False,
        description="Enable verbose logging for intermediate search steps",
    )


class OutputConfig(pydantic.BaseModel):
    """Configuration for the output settings"""

    output_dir: str = pydantic.Field(
        default="./output",
        description="Directory for storing temp data, results, and logs",
    )
    remove_temp_after_run: bool = pydantic.Field(
        default=True,
        description="Configuration for removing temp data before termination",
    )


class PreprocessConfig(pydantic.BaseModel):
    """Configuration for the preprocess phase"""

    force_recompute: bool = pydantic.Field(
        default=True,
        description="Enable recomputation and ignore the stored SVD results",
    )


class TopDownConfig(pydantic.BaseModel):
    """Configuration for the top down structure search"""

    enabled: bool = pydantic.Field(
        default=False,
        description="Configure for enabling the top down search",
    )
    merge_mode: Literal["all", "not_first"] = pydantic.Field(
        default="not_first",
        description="Configure whether to merge indices at the first level",
    )
    search_algo: Literal["random", "enumerate", "correlation", "svd"] = (
        pydantic.Field(
            default="enumerate",
            description="Configure whether to use random algorithms",
        )
    )
    aggregation: Literal["mean", "det", "norm", "sval"] = pydantic.Field(
        default="mean",
        description="Configure the aggregation method for correlations",
    )
    random_algorithm: Literal["random", "anneal"] = pydantic.Field(
        default="random",
        description="Configure to use which random search algorithm",
    )
    group_threshold: int = pydantic.Field(
        default=4,
        description="Configure the number of indices allowed in one search",
    )
    annel_step: int = pydantic.Field(
        default=10,
        description="Configure the step number for SA",
    )
    init_temp: float = pydantic.Field(
        default=100,
        description="Configure the initial temperature for AS",
    )
    temp_schedule: Literal["linear", "exp", "log"] = pydantic.Field(
        default="linear",
        description="Configure the temperature schedule for SA",
    )
    alpha: float = pydantic.Field(
        default=0.01,
        description="Configure the error distribution between steps",
    )


class SearchConfig(pydantic.BaseModel):
    """Configuration for the entire search process"""

    engine: SearchEngineConfig = pydantic.Field(
        default_factory=SearchEngineConfig,
        description="Configurations for search engines",
    )
    heuristics: HeuristicConfig = pydantic.Field(
        default_factory=HeuristicConfig,
        description="Configurations for heuristics used in search",
    )
    rank_search: RankSearchConfig = pydantic.Field(
        default_factory=RankSearchConfig,
        description="Configurations for rank search algorithms",
    )
    synthesizer: ProgramSearchConfig = pydantic.Field(
        default_factory=ProgramSearchConfig,
        description="Configurations for constraint solving",
    )
    output: OutputConfig = pydantic.Field(
        default_factory=OutputConfig,
        description="Configurations for search outputs",
    )
    preprocess: PreprocessConfig = pydantic.Field(
        default_factory=PreprocessConfig,
        description="Configurations for the preprocessing phase",
    )
    topdown: TopDownConfig = pydantic.Field(
        default_factory=TopDownConfig,
        description="Configurations for top down hierarchical search",
    )

    @staticmethod
    def load(json_str: str) -> "SearchConfig":
        """Load configurations from JSON strings"""
        try:
            return SearchConfig.model_validate_json(json_str)
        except pydantic.ValidationError as e:
            print(e)

    @staticmethod
    def load_file(json_file: str) -> "SearchConfig":
        """Load configuration from JSON files"""
        with open(json_file, "r", encoding="utf-8") as json_file:
            return SearchConfig.load(json_file.read())

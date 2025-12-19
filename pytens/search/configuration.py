"""Configuration fields for the structure search process."""

from typing import Literal, Optional
from enum import Enum, auto

import pydantic


class InputFormat(Enum):
    """Types for different input formats"""

    WHITE_BOX = auto()
    BLACK_BOX = auto()


class ClusterMethod(Enum):
    """Different merge algorithms"""

    RAND = auto()
    CORR = auto()
    SVD = auto()
    NBR = auto()
    RAND_NBR = auto()


class InitStructType(Enum):
    """Different initial structures"""

    TUCKER = auto()
    HT = auto()
    TT = auto()
    FTT = auto()
    TT_CROSS = auto()


class ReshapeOption(Enum):
    """Different options to merge indices to produce a lower-dim data."""

    RANDOM = auto()
    ENUMERATE = auto()
    CLUSTER = auto()

class ReorderAlgo(Enum):
    """Different algorithms to reorder the indices in a TT."""

    CROSS = auto()
    SVD = auto()

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
    search_mode: Literal["topk", "all"] = pydantic.Field(
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
            "(Used together with search_mode==topk)"
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
    seed: int = pydantic.Field(
        default=0,
        description="Random seed used in random algorithms",
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
    max_rank: int = pydantic.Field(
        default=100,
        description="Config the maximum number of singular values in randomized SVD",
    )
    rand_svd: bool = pydantic.Field(
        default=True,
        description="Whether to use random SVD in the computation of preprocessing singular values",
    )
    reorder_algo: ReorderAlgo = pydantic.Field(
        default=ReorderAlgo.CROSS,
        description="Config the algorithm used to reorder the indices before preprocessing",
    )
    reorder_eps: float = pydantic.Field(
        default=0.5,
        description="Configure the error tolerance for cross during reordering",
    )


class CrossConfig(pydantic.BaseModel):
    """Configuration for cross approximation"""

    init_eps: float = pydantic.Field(
        default=0.1,
        description="Initial error setting for cross approximation",
    )
    init_struct: InitStructType = pydantic.Field(
        default=InitStructType.TT_CROSS,
        description="Choice of the initial network structure before cross",
    )
    init_dim: int = pydantic.Field(
        default=100,
        description="Configure the number of initial cross dimensions",
    )
    init_kickrank: int = pydantic.Field(
        default=2,
        description="Number of rank steps for initial cross approximation",
    )
    init_reshape: bool = pydantic.Field(
        default=False,
        description="Reshape the data into smaller factors before running cross",
    )


class TopDownConfig(pydantic.BaseModel):
    """Configuration for the top down structure search"""

    reshape_enabled: bool = pydantic.Field(
        default=False,
        description="Configure for enabling index reshaping during top down search",
    )
    merge_mode: Literal["all", "not_first"] = pydantic.Field(
        default="not_first",
        description="Configure whether to merge indices at the first level",
    )
    reshape_algo: ReshapeOption = pydantic.Field(
        default=ReshapeOption.CLUSTER,
        description="Configure whether to use random algorithms",
    )
    cluster_method: ClusterMethod = pydantic.Field(
        default=ClusterMethod.RAND,
        description="Configure what heuristic to use during index merge",
    )
    aggregation: Literal["mean", "det", "norm", "sval"] = pydantic.Field(
        default="mean",
        description="Configure the aggregation method for correlations",
    )
    random_algorithm: Literal["random"] = pydantic.Field(
        default="random",
        description="Configure to use which random search algorithm",
    )
    group_threshold: int = pydantic.Field(
        default=4,
        description="Configure the number of indices allowed in one search",
    )
    alpha: float = pydantic.Field(
        default=10,
        description="Configure the error distribution between steps",
    )


class InputConfig(pydantic.BaseModel):
    """Configuration for input-related fields"""

    input_format: InputFormat = pydantic.Field(
        default=InputFormat.WHITE_BOX,
        description="Choose the input data format",
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
    input: InputConfig = pydantic.Field(
        default_factory=InputConfig,
        description="Configuration for input formats",
    )
    preprocess: PreprocessConfig = pydantic.Field(
        default_factory=PreprocessConfig,
        description="Configurations for the preprocessing phase",
    )
    topdown: TopDownConfig = pydantic.Field(
        default_factory=TopDownConfig,
        description="Configurations for top down hierarchical search",
    )
    cross: CrossConfig = pydantic.Field(
        default_factory=CrossConfig,
        description="Configurations for cross approximation",
    )

    @staticmethod
    def load(json_str: str) -> "SearchConfig":
        """Load configurations from JSON strings"""
        return SearchConfig.model_validate_json(json_str)

    @staticmethod
    def load_file(json_file: str) -> "SearchConfig":
        """Load configuration from JSON files"""
        with open(json_file, "r", encoding="utf-8") as f:
            return SearchConfig.load(f.read())

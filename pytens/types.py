"""Type definitions."""

import logging
import itertools
import dataclasses
from typing import Union, Sequence, Self, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pydantic

logger = logging.getLogger(__name__)

IntOrStr = Union[str, int]
IndexName = IntOrStr
IndexChain = Union[List[int], Tuple[int]]
NodeName = IntOrStr


@dataclass(frozen=True, eq=True)
class Index:
    """An index labelling one dimension of a tensor.

    Attributes:
        name: Unique identifier for the index, used as the key when looking
            up nodes in a tensor network.
        size: Number of discrete grid points along this dimension.
        value_choices: The actual coordinate values corresponding to each
            integer position ``0 .. size-1``. Empty by default, meaning the
            index is purely symbolic with no associated coordinates.
            Equality and hashing intentionally ignore this field — two
            ``Index`` objects with the same ``name`` and ``size`` are
            considered equal regardless of their ``value_choices``.
    """

    name: Union[str, int]
    size: int
    value_choices: Sequence[float] = tuple([])

    def with_new_size(self, new_size: int) -> "Index":
        """Create a new index with same name but new size"""
        return Index(self.name, new_size)

    def with_new_name(self, name: IndexName) -> "Index":
        """Create a new index with same size but new name"""
        return Index(name, self.size)

    def with_new_rng(self, rng: Sequence[float]) -> "Index":
        """Create a new index with different value choices"""
        return Index(self.name, self.size, rng)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return False
        return self.name == other.name and self.size == other.size

    def __lt__(self, other: Self) -> bool:
        return str(self.name) < str(other.name)

    def __gt__(self, other: Self) -> bool:
        return str(self.name) > str(other.name)

    def __hash__(self) -> int:
        return hash((self.name, self.size))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data_dict: dict) -> "Index":
        """Reconstruct from dictionary."""
        return cls(**data_dict)


@dataclass
class SVDConfig:
    """Configuration fields for SVD in tensor networks.

    At most one of ``delta`` or ``rel_delta`` may be set.  When neither is
    given the dataclass defaults to ``rel_delta=1e-6``.

    Attributes:
        delta: Absolute truncation threshold.  Singular values are discarded
            from the smallest upward as long as their cumulative squared sum
            does not exceed ``delta ** 2``.  Mutually exclusive with
            ``rel_delta``.
        rel_delta: Relative truncation threshold.  Converted to an absolute
            threshold by multiplying by the Frobenius norm of the unfolded
            matrix.  Defaults to ``1e-6`` when neither argument is given.
            Mutually exclusive with ``delta``.
        compute_data: When ``True``, the truncated data tensor is
            reconstructed (``U @ S @ Vt``) after the SVD. Set to ``False``
            to skip the reconstruction and keep only the factored form.
        compute_uv: When ``True``, the left/right singular vectors (``U``
            and ``Vt``) are computed. Set to ``False`` to return only the
            singular values, which is cheaper when the vectors are not needed.
    """

    delta: Optional[float] = None
    rel_delta: Optional[float] = None
    compute_data: bool = True
    compute_uv: bool = True

    def __post_init__(self) -> None:
        if self.delta is not None and self.rel_delta is not None:
            raise ValueError(
                "specify exactly one of 'delta' (absolute) or"
                " 'rel_delta' (relative), not both"
            )
        if self.delta is None and self.rel_delta is None:
            self.rel_delta = 1e-6


class NodeInfo:
    """Cross-approximation bookkeeping attached to one side of a dim-tree edge.

    Each ``DimTreeNode`` owns two ``NodeInfo`` objects: one for the *up*
    direction (toward the root) and one for the *down* direction (toward the
    leaves).  Together they record the current cross-approximation rows/columns
    and the target rank for that edge.

    Attributes:
        nodes: Adjacent ``DimTreeNode`` objects on this side of the edge
            (typically the parent for ``up_info`` or the children for
            ``down_info``).
        indices: The tensor indices that belong to this side of the partition.
        vals: Integer sample matrix of shape ``(rank, len(indices))``.
            Each row is a set of discrete index positions selected by the
            cross algorithm as a representative row or column.
        rank: Target rank for this edge, updated incrementally by the cross
            sweeps.
    """

    def __init__(
        self,
        nodes: List["DimTreeNode"],
        indices: List[Index],
        vals: np.ndarray,
    ):
        self.nodes = nodes
        self.indices = indices
        self.vals = vals
        self.rank = 0


class DimTreeNode:
    """A node in the hierarchical dimension tree used by cross approximation.

    The dimension tree mirrors the tensor-network graph: each ``DimTreeNode``
    corresponds to one network node, and its two ``NodeInfo`` sides record
    the cross-approximation state for the edges connecting it to its parent
    (``up_info``) and its children (``down_info``).

    Attributes:
        node: The name of the corresponding tensor-network node.
        indices: All tensor indices (free and bond) attached to this node.
        free_indices: The physical/free indices carried by this node — the
            subset of ``indices`` that are not shared with other nodes.
        up_info: Cross-approximation state for the edge toward the tree root
            (parent side).
        down_info: Cross-approximation state for the edges toward the tree
            leaves (children side).
        perm: Permutation applied to align the tensor values with the
            canonical ordering ``[free_indices, down children, up parent]``
            during cross-approximation sweeps.
    """

    def __init__(  # pylint: disable=R0913,R0917
        self,
        node: NodeName,
        indices: List[Index],
        free_indices: List[Index],
        up_info: NodeInfo,
        down_info: NodeInfo,
    ):
        self.node = node
        self.indices = indices
        self.free_indices = free_indices
        self.up_info = up_info
        self.down_info = down_info
        self.perm = list(
            range(
                len(free_indices) + len(down_info.nodes) + len(up_info.nodes)
            )
        )

    def __lt__(self, other: Self) -> bool:
        return sorted(self.indices) < sorted(other.indices)

    def preorder(self) -> Sequence["DimTreeNode"]:
        """Get the list of tree nodes in the pre-order traversal."""
        results = [self]
        for c in self.down_info.nodes:
            results = list(itertools.chain(results, c.preorder()))

        return list(results)

    def increment_ranks(
        self, kickrank: int = 1, max_rank: Optional[int] = None
    ) -> None:
        """Increment the ranks without value modification"""
        self.up_info.rank += kickrank
        if max_rank is not None:
            self.up_info.rank = min(max_rank, self.up_info.rank)

        for c in self.down_info.nodes:
            c.increment_ranks(kickrank, max_rank)

    def ranks(self) -> List[int]:
        """Get all up ranks in the dimension tree."""
        res = [self.up_info.rank]
        for c in self.down_info.nodes:
            res.extend(c.ranks())

        return res

    def bound_ranks(self) -> None:
        """Adjust the ranks according to the ranks of neighbor edges"""
        # if we move leaves to root
        rank_up = 1
        for c in self.down_info.nodes:
            if c.up_info.rank != 0:
                rank_up *= c.up_info.rank

        for ind in self.free_indices:
            rank_up *= ind.size

        # if we move root to leaves
        rank_down = self.up_info.rank
        for p in self.up_info.nodes:
            rank_down = 1
            if p.up_info.rank != 0:
                rank_down *= p.up_info.rank

            for s in p.down_info.nodes:
                if s.node != self.node and s.up_info.rank != 0:
                    rank_down *= s.up_info.rank

            for ind in p.free_indices:
                rank_down *= ind.size

        # rank_up = max(1, rank_up)
        # rank_down = max(1, rank_down)
        logger.debug(
            "node: %s, indices: %s, rank_up: %i, rank_down: %i, curr_rankL %i",
            self.node,
            self.free_indices,
            rank_up,
            rank_down,
            self.up_info.rank,
        )
        self.up_info.rank = min([rank_up, rank_down, self.up_info.rank])

        for c in self.down_info.nodes:
            c.bound_ranks()

    def add_values(self, up_vals: np.ndarray) -> None:
        """Initialize the up and down values for the given dimension tree."""
        # if len(self.up_info.nodes) == 0:
        #     self.up_info.rank = 1
        # else:
        #     self.up_info.rank += len(up_vals)

        for c in self.down_info.nodes:
            cvals = up_vals[:, [self.indices.index(ind) for ind in c.indices]]
            c.up_info.vals = np.append(c.up_info.vals, cvals, axis=0)
            c.up_info.vals = c.up_info.vals[: c.up_info.rank]
            c.add_values(cvals)

    def locate(self, node: NodeName) -> Optional["DimTreeNode"]:
        """Locate a node by its name."""
        if node == self.node:
            return self

        for c in self.down_info.nodes:
            res = c.locate(node)
            if res is not None:
                return res

        return None

    def leaves(self) -> Sequence["DimTreeNode"]:
        """Get the leaf nodes in the current tree."""
        results = []
        if len(self.up_info.nodes) == 0:
            results.append(self)
            return results

        for c in self.up_info.nodes:
            results.extend(c.leaves())

        return results

    def height(self) -> int:
        """Get the height of the tree."""
        max_c = 0
        for c in self.up_info.nodes:
            max_c = max(max_c, c.height())

        return max_c + 1

    def path(self, node1: NodeName, node2: NodeName) -> List["DimTreeNode"]:
        """Get the list of nodes between the source and destination."""

        n1 = self.locate(node1)
        assert n1 is not None
        n2 = self.locate(node2)
        assert n2 is not None
        # find the common ancestor that subsumes both n1 and n2

        res = [n1]
        p = n1
        while p is not None:
            if all(ind in p.indices for ind in n1.indices + n2.indices):
                break

            p = p.up_info.nodes[0]
            res.append(p)

        if p is None:
            raise RuntimeError("not a valid tree")

        p2 = n2
        res2 = [p2]
        while p2 is not None and p2 != p:
            p2 = p2.up_info.nodes[0]
            res2.append(p2)

        if p2 is None:
            raise RuntimeError("not a valid tree")

        return res + list(reversed(res2[:-1]))

    def distance(self, node1: NodeName, node2: NodeName) -> int:
        """Get the distance between the two indices."""
        return len(self.path(node1, node2))

    def entries(self) -> np.ndarray:
        """Extract the up entries"""

        if len(self.up_info.vals) != 0:
            vals = self.up_info.vals
        else:
            vals = np.empty((0, len(self.up_info.indices)))

        return vals

    def known_entries(self) -> np.ndarray:
        """Extract the up and down entries"""

        vals = np.empty((0, len(self.indices)))
        if len(self.up_info.vals) != 0:
            vals = np.concat([self.down_info.vals, self.up_info.vals], axis=-1)

        self_inds = self.down_info.indices + self.up_info.indices
        for c in self.down_info.nodes:
            cvals = c.known_entries()
            # cvals follows the order of
            # c.down_info.indices + c.up_info.indices
            # reorder the values to match self
            cindices = c.down_info.indices + c.up_info.indices
            perm = [self_inds.index(ind) for ind in cindices]
            vals = np.concat([vals, cvals[:, perm]], axis=0)

        return vals

    def highest_frontier(
        self, indices: Sequence[Index]
    ) -> List["DimTreeNode"]:
        """Find the frontier of nodes that contain the given indices."""
        inds = self.indices
        if len(inds) > 0 and all(ind in indices for ind in inds):
            return [self]

        res = []
        for c in self.down_info.nodes:
            res.extend(c.highest_frontier(indices))

        return res

    def sibling(self, node: "DimTreeNode") -> "DimTreeNode":
        """Get one of the sibling node of the given node"""
        if len(node.up_info.nodes) != 1:
            raise ValueError("root node does not have a sibling")

        p = node.up_info.nodes[0]
        for c in p.down_info.nodes:
            if c.node == node.node:
                continue

            return c

        raise ValueError("No sibling for the given node")

    def is_ancestor(self, other: "DimTreeNode") -> bool:
        """Return true if the current node is an ancestor of other"""
        while len(other.up_info.nodes) > 0:
            other = other.up_info.nodes[0]
            if other.node == self.node:
                return True

        return False


class IndexMerge(pydantic.BaseModel):
    """An index merge request and response.

    Represents the operation of fusing several indices into a single combined
    index whose size is the product of the originals.

    Attributes:
        indices: The indices to be merged, in the order they will be combined.
        result: The single merged index produced by the operation. ``None``
            before the merge has been executed.
    """

    indices: Sequence[Index]
    result: Optional[Index] = None

    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.__dict__.values()))

    def __lt__(self, other: "IndexMerge") -> bool:
        if self.indices != other.indices:
            return tuple(self.indices) < tuple(other.indices)

        if self.result is None:
            return True

        if other.result is None:
            return False

        return self.result < other.result


class IndexSplit(pydantic.BaseModel):
    """An index split request and response.

    Represents the operation of reshaping a single index into several smaller
    indices whose sizes multiply to the original size.

    Attributes:
        index: The index to be split.
        shape: Target sizes for the new sub-indices, whose product must equal
            ``index.size``.
        result: The sequence of new indices produced by the split. ``None``
            before the split has been executed.
    """

    index: Index
    shape: Sequence[int]
    result: Optional[Sequence[Index]] = None

    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.__dict__.values()))

    def __lt__(self, other: "IndexSplit") -> bool:
        if self.index != other.index:
            return self.index < other.index

        return tuple(self.shape) < tuple(other.shape)


class IndexPermute(pydantic.BaseModel):
    """A permutation applied to the index ordering of a function.

    Attributes:
        perm: Permutation to apply — ``perm[i]`` is the original position of
            the index that should appear at position ``i`` after reordering.
        unperm: Inverse permutation of ``perm``, used to undo the reordering
            when mapping results back to the original index order.
    """

    perm: Sequence[int]
    unperm: Sequence[int]

    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.__dict__.values()))


class IndexSwap(pydantic.BaseModel):
    """Swap free indices between two neighbouring nodes in a tensor network.

    Attributes:
        node: The name of the network node whose index assignment changes.
        left_indices: The indices that should be placed on the *left* node
            after the swap (the complement goes to the right node).
    """

    node: NodeName
    left_indices: Sequence[Index]


IndexOp = Union[IndexMerge, IndexSplit, IndexPermute, IndexSwap]


def split_index(
    ind: Index, indices: List[Index], vals: np.ndarray, split_op: IndexSplit
) -> Tuple[List[Index], np.ndarray]:
    """Split the given index into multiple sub-indices."""
    assert split_op.result is not None

    pos = indices.index(ind)
    split_sizes = [i.size for i in split_op.result]
    new_vals = np.empty(
        (vals.shape[0], vals.shape[1] - 1 + len(split_sizes)), dtype=int
    )
    new_vals[:, : vals.shape[1] - 1] = np.hstack(
        [vals[:, :pos], vals[:, pos + 1 :]]
    )
    new_vals[:, -len(split_sizes) :] = np.vstack(
        np.unravel_index(vals[:, pos], split_sizes)
    ).T
    indices.remove(ind)
    indices.extend(split_op.result)
    return indices, new_vals


class PartitionStatus(Enum):
    """Status codes for partition."""

    OK = auto()
    FAIL = auto()
    EXIST = auto()


class PartitionResult:
    """Result of checking whether a set of indices forms a valid partition.

    Attributes:
        code: Outcome of the check (``OK``, ``FAIL``, or ``EXIST``).
        lca_node: Name of the lowest common ancestor node in the tensor
            network that subsumes all queried indices.
        lca_indices: The indices held by ``lca_node`` that correspond to the
            queried partition.
    """

    code: PartitionStatus
    lca_node: NodeName
    lca_indices: List[Index]


class SVDAlgorithm(Enum):
    """Strategy for computing singular values of a tensor partition.

    Attributes:
        MERGE: Merge the nodes that contain the target indices into a single
            node first, then perform SVD on the merged tensor.
        CROSS: Estimate singular values via cross approximation, which avoids
            explicit contraction and is cheaper for large networks.
    """

    MERGE = auto()
    CROSS = auto()


@dataclass
class NodeIndexPair:
    """A network node paired with an optional index it carries.

    Attributes:
        node: Name of the tensor-network node.
        ind: An index associated with ``node``, or ``None`` if the pairing
            is node-only (e.g. when referring to the node without specifying
            which of its indices is of interest).
    """

    node: NodeName
    ind: Optional[Index] = None


@dataclass
class AlgoParams:
    """Algorithm selection and tolerance for singular value computation.

    Attributes:
        algo: Which ``SVDAlgorithm`` strategy to use. Defaults to
            ``SVDAlgorithm.MERGE`` when not specified.
        eps: Error tolerance passed to the chosen algorithm (e.g. used as the
            cross-approximation accuracy when ``algo`` is ``CROSS``).
    """

    algo: "SVDAlgorithm" = SVDAlgorithm.MERGE
    eps: float = 0.0

    def __post_init__(self) -> None:
        if self.algo is None:
            self.algo = SVDAlgorithm.MERGE


@dataclass
class SVDParams:
    """Parameters controlling the SVD truncation and randomisation.

    Attributes:
        max_rank: Maximum number of singular values/vectors to retain.
        orthonormal: Name of the network node that is already orthonormalised
            before the SVD is performed. ``None`` means no orthonormalisation
            is assumed.
        random_seed: Seed for the randomised SVD. ``None`` selects a fixed
            default seed; pass an integer to make results reproducible or to
            vary the random draw.
    """

    max_rank: int = 100
    orthonormal: Optional[NodeName] = None
    random_seed: Optional[int] = None

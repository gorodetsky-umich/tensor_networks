"""Algorithms for tensor networks."""

import copy
import itertools
import logging
import math
import operator
import typing
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Self,
    Set,
    Tuple,
    Union,
    cast,
    Callable,
    FrozenSet,
)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import opt_einsum as oe
from sklearn.utils.extmath import randomized_svd

from pytens.utils import delta_svd
from pytens.types import (
    IndexName,
    NodeName,
    Index,
    SVDConfig,
    DimTreeNode,
    NodeInfo,
    IndexMerge,
    IndexOp,
    IndexSplit,
    PartitionStatus,
    SubtreeViews,
    PartitionResult,
    SValsParams,
)
from pytens.search.types import Action
from pytens.cross.func_impl import FuncTensorNetwork

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Tensor:
    """Base class for a tensor."""

    value: np.ndarray
    indices: List[Index]

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Tensor":
        # indices are immutable and shared; the list and the array are not
        # (order="K" keeps the memory layout, like a generic deepcopy would)
        return Tensor(self.value.copy(order="K"), list(self.indices))

    def to_dict(self) -> dict:
        """Convert to dictionary. Useful for serialization."""

        return {
            "value": np.ascontiguousarray(self.value),
            "indices": [index.to_dict() for index in self.indices],
        }

    @classmethod
    def from_dict(cls, data_dict: dict) -> "Tensor":
        """Reconstruct from dictionary."""

        indices_list = [
            Index.from_dict(idx_dict) for idx_dict in data_dict["indices"]
        ]
        return cls(value=data_dict["value"], indices=indices_list)

    def update_val_size(self, value: np.ndarray) -> Self:
        """Update the tensor with a new value."""
        assert value.ndim == len(self.indices), (
            f"{value.shape}, {self.indices}"
        )
        self.value = value
        for ii, index in enumerate(self.indices):
            self.indices[ii] = index.with_new_size(value.shape[ii])
        return self

    def rename_indices(self, rename_map: Dict[IndexName, IndexName]) -> Self:
        """Rename the indices of the tensor."""
        for ii, index in enumerate(self.indices):
            if index.name in rename_map:
                self.indices[ii] = index.with_new_name(rename_map[index.name])

        return self

    def relabel_indices(self, relabel_map: Dict[IndexName, int]) -> Self:
        """Relabel the index size."""
        for ii, index in enumerate(self.indices):
            if index.name in relabel_map:
                self.indices[ii] = index.with_new_size(relabel_map[index.name])
        return self

    def rerange_indices(
        self, rerange_map: Dict[IndexName, Sequence[float]]
    ) -> Self:
        """Rerange the index size."""
        for ii, index in enumerate(self.indices):
            if index.name in rerange_map:
                self.indices[ii] = index.with_new_rng(rerange_map[index.name])
        return self

    def concat_fill(
        self, other: Self, indices_common: List[Index]
    ) -> "Tensor":
        """Concatenate two arrays.

        keep dimensions corresponding to indices_common the same.
        pad zeros on all other dimensions. new dimensions retain
        index names currently used, but have updated size
        """
        shape_here = self.value.shape
        shape_other = other.value.shape

        assert len(shape_here) == len(shape_other)

        new_shape = []
        new_indices = []
        for index_here, index_other in zip(self.indices, other.indices):
            if index_here in indices_common:
                assert index_here.size == index_other.size
                new_indices.append(index_here)
                new_shape.append(index_here.size)
            else:
                new_shape.append(index_here.size + index_other.size)
                new_index = Index(
                    f"{index_here.name}", index_here.size + index_other.size
                )
                new_indices.append(new_index)

        new_val = np.zeros(new_shape)

        ix1 = []
        ix2 = []
        for index_here in self.indices:
            if index_here in indices_common:
                ix1.append(slice(None))
                ix2.append(slice(None))
            else:
                ix1.append(slice(0, index_here.size, 1))
                ix2.append(slice(index_here.size, None, 1))
        new_val[*ix1] = self.value
        new_val[*ix2] = other.value
        tens = Tensor(new_val, new_indices)
        return tens

    def mult(self, other: Self, indices_common: List[Index]) -> "Tensor":
        """Outer product of two tensors except at common indices

        retain naming of self
        """
        shape_here = self.value.shape
        shape_other = other.value.shape

        assert len(shape_here) == len(shape_other)

        new_shape = []
        new_indices = []
        str1 = ""
        str2 = ""
        output_str = ""
        on_index = 97
        for index_here, index_other in zip(self.indices, other.indices):
            if index_here in indices_common:
                assert index_here.size == index_other.size
                new_indices.append(index_here)
                new_shape.append(index_here.size)
                new_char = chr(on_index)
                str1 += new_char
                str2 += new_char
                output_str += new_char
                on_index += 1
            else:
                new_shape.append(index_here.size * index_other.size)
                new_index = Index(
                    f"{index_here.name}", index_here.size * index_other.size
                )
                new_indices.append(new_index)
                c1 = chr(on_index)
                str1 += c1
                output_str += c1
                on_index += 1

                c2 = chr(on_index)
                str2 += c2
                output_str += c2
                on_index += 1

        # print("shape here", shape_here)
        # print("shape there", shape_other)
        # print("new_shape = ", new_shape)

        # print("mult string 1 = ", str1)
        # print("mult string 2 = ", str2)
        # print("output_str = ", output_str)
        # print("new_indices = ", new_indices)
        estr = str1 + "," + str2 + "->" + output_str
        # print("estr", estr)

        new_val = oe.contract(estr, self.value, other.value)
        new_val = np.reshape(new_val, new_shape)
        tens = Tensor(new_val, new_indices)
        return tens

    def contract(self, other: Self) -> "Tensor":
        """Contract two tensors by their common indices."""
        new_shape = []
        new_indices = []
        str1 = ""
        str2 = ""
        output_str = ""
        on_index = 97
        for index_here in self.indices:
            new_char = chr(on_index)
            str1 += new_char
            on_index += 1
            if index_here not in other.indices:
                new_indices.append(index_here)
                new_shape.append(index_here.size)
                output_str += new_char

        for index_other in other.indices:
            if index_other in self.indices:
                idx = self.indices.index(index_other)
                str2 += str1[idx]
            else:
                new_char = chr(on_index)
                str2 += new_char
                output_str += new_char
                on_index += 1
                new_indices.append(index_other)
                new_shape.append(index_other.size)

        estr = str1 + "," + str2 + "->" + output_str
        # print("estr", estr)

        new_val = oe.contract(estr, self.value, other.value)
        new_val = np.reshape(new_val, new_shape)
        tens = Tensor(new_val, new_indices)
        return tens

    def svd(
        self,
        lefts: Sequence[int],
        *,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        compute_uv: bool = True,
    ) -> Tuple[List["Tensor"], float]:
        """Split a tensor into three factors by (optionally truncated) SVD.

        The tensor is unfolded into a matrix with the ``lefts`` dimensions on
        the left and all remaining dimensions on the right.  A truncated SVD
        is then computed and the result is reshaped back into three tensors
        ``[U, S, Vt]``, where ``S`` is diagonal.

        At most one of ``atol`` or ``rtol`` may be provided.  When
        neither is given, ``rtol=1e-6`` is used by default.

        Args:
            lefts: Positions of the indices that form the left (row) side of
                the unfolding matrix.  All other index positions become the
                right (column) side.
            atol: Absolute truncation threshold.  Singular values are
                discarded from the smallest upward as long as their cumulative
                squared sum does not exceed ``atol ** 2``.  Mutually
                exclusive with ``rtol``.
            rtol: Relative truncation threshold.  Converted to an
                absolute threshold by multiplying by the Frobenius norm of the
                unfolded matrix before applying the same rule as ``atol``.
                Defaults to ``1e-6`` when neither argument is given.  Mutually
                exclusive with ``atol``.
            compute_uv: When ``True`` (default), return all three factors
                ``[U, S, Vt]``.  When ``False``, return only ``[S]``, which
                is cheaper when the singular vectors are not needed.

        Returns:
            A tuple ``(tensors, remaining_atol)`` where ``tensors`` is
            ``[U, S, Vt]`` (or ``[S]`` when ``compute_uv=False``), and
            ``remaining_atol`` is the unused portion of the absolute error
            budget after truncation.

        Raises:
            ValueError: If both ``atol`` and ``rtol`` are provided.
        """
        if atol is not None and rtol is not None:
            raise ValueError(
                "specify exactly one of 'atol' (absolute) or"
                " 'rtol' (relative), not both"
            )
        if atol is None and rtol is None:
            rtol = 1e-6
        with_normalizing = atol is None
        effective_delta = atol or rtol or 0.0

        rights = [i for i in range(len(self.indices)) if i not in lefts]
        permute_indices = itertools.chain(lefts, rights)
        value = np.permute_dims(self.value, tuple(permute_indices))
        left_sz = int(np.prod([self.indices[i].size for i in lefts]))
        right_sz = int(np.prod([self.indices[j].size for j in rights]))
        value = value.reshape(left_sz, right_sz)

        result = delta_svd(
            value,
            effective_delta,
            with_normalizing=with_normalizing,
            compute_uv=compute_uv,
        )
        u = result.u
        s = result.s
        v = result.v
        d = result.remaining_delta

        s_indices = [
            Index("r_split_l", s.shape[0]),
            Index("r_split_r", s.shape[0]),
        ]
        s_tensor = Tensor(np.diag(s), s_indices)

        if compute_uv:
            assert u is not None
            u = u.reshape([self.indices[i].size for i in lefts] + [-1])
            u_indices = [self.indices[i] for i in lefts]
            u_indices.append(Index("r_split_l", u.shape[-1]))
            u_tensor = Tensor(u, u_indices)

            assert v is not None
            v = v.reshape([-1] + [self.indices[j].size for j in rights])
            v_indices = [self.indices[j] for j in rights]
            v_indices = [Index("r_split_r", v.shape[0])] + v_indices
            v_tensor = Tensor(v, v_indices)

            return [u_tensor, s_tensor, v_tensor], d

        return [s_tensor], d

    def qr(self, lefts: Sequence[int]) -> Tuple["Tensor", "Tensor"]:
        """Split a tensor into two by QR."""
        rights = [i for i in range(len(self.indices)) if i not in lefts]
        permute_indices = itertools.chain(lefts, rights)
        value = np.permute_dims(self.value, tuple(permute_indices))
        left_sz = int(np.prod([self.indices[i].size for i in lefts]))
        right_sz = int(np.prod([self.indices[j].size for j in rights]))
        value = value.reshape(left_sz, right_sz)

        q, r = np.linalg.qr(value)

        q = q.reshape([self.indices[i].size for i in lefts] + [-1])
        q_indices = [self.indices[i] for i in lefts]
        q_indices.append(Index("r_split", q.shape[-1]))
        q_tensor = Tensor(q, q_indices)

        r = r.reshape([-1] + [self.indices[j].size for j in rights])
        r_indices = [self.indices[j] for j in rights]
        r_indices = [Index("r_split", r.shape[0])] + r_indices
        r_tensor = Tensor(r, r_indices)

        return q_tensor, r_tensor

    def permute(self, target_indices: Sequence[int]) -> "Tensor":
        """Return a new tensor with indices permuted by the specified order."""
        value = np.permute_dims(self.value, tuple(target_indices))
        indices = [self.indices[i] for i in target_indices]
        return Tensor(value, indices)

    def permute_by_name(self, target_indices: Sequence[IndexName]) -> "Tensor":
        """Permute a tensor by its index names"""
        index_names = [ind.name for ind in self.indices]
        perm = [index_names.index(n) for n in target_indices]
        return self.permute(perm)

    def merge_indices(
        self, merged_indices: Sequence[Index], new_ind: IndexName
    ) -> "Tensor":
        """Merge the specified indices into one"""
        # print("merge_indices:", self.indices)
        if set(merged_indices).issubset(set(self.indices)):
            merged_sizes, unmerged_sizes = [], []
            merged_names, unmerged_names = [], []
            unmerged_indices = []
            for ind in merged_indices:
                merged_sizes.append(ind.size)
                merged_names.append(ind.name)

            for ind in self.indices:
                if ind not in merged_indices:
                    unmerged_indices.append(ind)
                    unmerged_names.append(ind.name)
                    unmerged_sizes.append(ind.size)

            perm_names = list(itertools.chain(merged_names, unmerged_names))
            new_tensor = self.permute_by_name(perm_names)
            new_data = new_tensor.value.reshape(-1, *unmerged_sizes)

            new_size = math.prod(merged_sizes)
            new_index = Index(new_ind, new_size, range(0, new_size))
            return Tensor(new_data, [new_index] + unmerged_indices)

        return self

    def split_indices(
        self, split_into: IndexSplit, compute_data: bool = True
    ) -> "Tensor":
        """Split specified indices into smaller ones"""
        new_indices = []
        next_index = 0
        for ind in self.indices:
            if ind == split_into.index:
                for sz in split_into.shape:
                    new_ind = f"_fresh_index_{next_index}"
                    new_indices.append(Index(new_ind, sz, range(0, sz)))
                    next_index += 1
            else:
                new_indices.append(ind)

        if compute_data:
            new_data = self.value.reshape([ind.size for ind in new_indices])
        else:
            new_data = np.empty(0)
        return Tensor(new_data, new_indices)

    def block_diagonal(
        self, other: "Tensor", free_inds: Sequence[Index]
    ) -> "Tensor":
        """
        Concat tensors along contract indices diagonally but keep free indices.
        When there is only one contract index, concat them directly.
        """
        sz = []
        start_offsets = {}
        for i, ind in enumerate(self.indices):
            if ind in free_inds:
                assert ind.size == other.indices[i].size
                sz.append(ind.size)
            else:
                sz.append(ind.size + other.indices[i].size)
                start_offsets[i] = 0

        large_array = np.zeros(sz, dtype=self.value.dtype)
        for arr in [self.value, other.value]:
            slices = []
            for i in range(len(sz)):
                if self.indices[i] in free_inds:
                    slices.append(slice(None))
                else:
                    start = start_offsets[i]
                    end = start + arr.shape[i]
                    slices.append(slice(start, end))
                    start_offsets[i] = end

            # Place the current array on the diagonal
            large_array[tuple(slices)] = arr

        large_indices: List[Index] = []
        for i, ind in enumerate(self.indices):
            large_indices.append(Index(ind.name, large_array.shape[i]))

        return Tensor(large_array, large_indices)


# @dataclass(frozen=True, eq=True)
@dataclass(eq=True)
class EinsumArgs:
    """Represent information about contraction in einsum list format"""

    input_str_map: Dict[NodeName, str]
    output_str: str
    output_str_index_map: Dict[str, Index]

    def replace_char(self, value: str, replacement: str) -> None:
        """Replace a character in the einsum string."""
        for _, vals in self.input_str_map.items():
            vals = vals.replace(value, replacement)
        self.output_str = self.output_str.replace(value, replacement)


class TensorNetwork:  # pylint: disable=R0904
    """Tensor Network Base Class."""

    _id_counter = itertools.count(start=1)

    def __init__(self) -> None:
        """Initialize the network."""
        self.network = nx.Graph()

    def __deepcopy__(self, memo: Dict[int, Any]) -> "TensorNetwork":
        # Rebuild the graph directly: the network is copied on nearly every
        # search step, and the generic copy of networkx's nested dicts is
        # far slower than adding the nodes and edges again.
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.network = nx.Graph()
        for name, data in self.network.nodes(data=True):
            new.network.add_node(name, tensor=copy.deepcopy(data["tensor"]))
        # copy the adjacency dicts as they are so that neighbours are
        # visited in the same order as in the original
        # pylint: disable-next=protected-access
        for name, nbrs in self.network._adj.items():
            new.network._adj[name].update((nbr, {}) for nbr in nbrs)
        return new

    def add_node(self, name: NodeName, tensor: Tensor) -> None:
        """Add a node to the network."""
        self.network.add_node(name, tensor=tensor)

    def node_tensor(self, node_name: NodeName) -> Tensor:
        """Get tensor (of type Tensor) at a particular node"""
        ret: Tensor = self.network.nodes[node_name]["tensor"]
        return ret

    def set_node_tensor(self, node_name: NodeName, value: Tensor) -> None:
        """Set the Tensor value at a particular node"""
        self.network.nodes[node_name]["tensor"] = value

    def add_edge(self, name1: NodeName, name2: NodeName) -> None:
        """Add an edget to the network."""
        self.network.add_edge(name1, name2)

    def value(self, node_name: NodeName) -> np.ndarray:
        """Get the value of a node."""
        val: np.ndarray = self.network.nodes[node_name]["tensor"].value
        return val

    def all_indices(self) -> Counter[Index]:
        """Get all indices in the network."""
        indices = []
        for _, data in self.network.nodes(data=True):
            indices += data["tensor"].indices
        cnt = Counter(indices)
        return cnt

    def rename_indices(self, rename_map: Dict[IndexName, IndexName]) -> Self:
        """Rename the indices in the network."""
        for n in self.network.nodes:
            self.node_tensor(n).rename_indices(rename_map)
        return self

    def relabel_indices(self, relabel_map: Dict[IndexName, int]) -> Self:
        """Relabel the indices in the network."""
        for n in self.network.nodes:
            self.node_tensor(n).relabel_indices(relabel_map)
        return self

    def rerange_indices(
        self, rerange_map: Dict[IndexName, Sequence[float]]
    ) -> Self:
        """Reassign ranges to the indices in the network."""
        for n in self.network.nodes:
            self.node_tensor(n).rerange_indices(rerange_map)
        return self

    def assign_index_range(
        self, rerange_map: Dict[IndexName, Sequence[float]]
    ) -> None:
        """Change the indices value choices in the network."""
        for n in self.network.nodes:
            self.node_tensor(n).rerange_indices(rerange_map)

    def free_indices(self) -> List[Index]:
        """Get the free indices."""
        icount = self.all_indices()
        free_indices = [i for i, v in icount.items() if v == 1]
        return free_indices

    def get_contraction_index(
        self, node1: NodeName, node2: NodeName
    ) -> List[Index]:
        """Get the contraction indices."""
        ind1 = self.node_tensor(node1).indices
        ind2 = self.node_tensor(node2).indices
        inds = list(ind1) + list(ind2)
        cnt = Counter(inds)
        indices = [i for i, v in cnt.items() if v > 1]
        return indices

    def nodes_by_contraction_index(self, ind: Index) -> List[NodeName]:
        """Get the two nodes connected by the given index."""
        nodes = []
        for n in self.network.nodes:
            if ind in self.node_tensor(n).indices:
                nodes.append(n)

        # an index on exactly one node is free, not a contraction index
        assert len(nodes) != 1, f"{ind} is a free index"
        assert len(nodes) <= 2

        return nodes

    def inner_indices(self) -> List[Index]:
        """Get hte interior indices."""
        icount = self.all_indices()
        free_indices = [i for i, v in icount.items() if v > 1]
        return free_indices

    def ranks(self) -> List[int]:
        """Get the ranks."""
        inner_indices = self.inner_indices()
        return [r.size for r in inner_indices]

    def shape(self) -> List[int]:
        """Get the shape of tensor represented \
            by the TensorNetwork."""
        free_indices = self.free_indices()
        return [i.size for i in free_indices]

    def einsum_args(self) -> EinsumArgs:
        """Compute einsum args.

        Need to respect the edges, currently not using edges
        """
        all_indices = self.all_indices()
        free_indices = self.free_indices()

        mapping = {
            name: chr(i + 97) for i, name in enumerate(all_indices.keys())
        }
        input_str_map = {}
        for node, data in self.network.nodes(data=True):
            input_str_map[node] = "".join(
                [mapping[ind] for ind in data["tensor"].indices]
            )
        output_str = "".join([mapping[ind] for ind in free_indices])
        output_str_index_map = {}
        for ind in free_indices:
            output_str_index_map[mapping[ind]] = ind

        return EinsumArgs(input_str_map, output_str, output_str_index_map)

    def contract(self, eargs: Optional[EinsumArgs] = None) -> Tensor:
        """Contract the tensor."""
        if eargs is None:
            eargs = self.einsum_args()

        estr_l = []
        arrs = []
        for key, val in eargs.input_str_map.items():
            arrs.append(self.value(key))
            estr_l.append(val)
        estr = ",".join(estr_l) + "->" + eargs.output_str  # explicit
        # estr = ','.join(estr)
        logger.debug("Contraction string = %s", estr)
        out = oe.contract(estr, *arrs, optimize="auto")
        logger.debug("finish contraction")
        indices = [eargs.output_str_index_map[s] for s in eargs.output_str]
        tens = Tensor(out, indices)
        return tens

    @typing.no_type_check
    def __getitem__(self, ind: slice) -> "Tensor":
        """Evaluate at some elements.

        Assumes indices are provided in the order retrieved by
        TensorNetwork.free_indices()
        """
        free_indices = self.free_indices()

        new_network = TensorNetwork()
        for node, data in self.network.nodes(data=True):
            tens = data["tensor"]
            ix = []
            new_indices = []
            for local_ind in tens.indices:
                try:
                    dim = free_indices.index(local_ind)
                    ix.append(ind[dim])
                    if not isinstance(ind[dim], int):
                        new_indices.append(local_ind)

                except ValueError:  # no dimension is in a free index
                    ix.append(slice(None))
                    new_indices.append(local_ind)

            new_arr = tens.value[*ix]
            new_tens = Tensor(new_arr, new_indices)
            new_network.add_node(node, new_tens)

        for u, v in self.network.edges():
            new_network.add_edge(u, v)

        return new_network.contract()

    def attach(
        self,
        other: "TensorNetwork",
        rename: Tuple[str, str] = ("G", "H"),
        indices: Optional[Sequence[Index]] = None,
    ) -> "TensorNetwork":
        """Attach two tensor networks together."""
        # U = nx.union(copy.deepcopy(self.network),
        #              copy.deepcopy(other.network),
        #              rename=rename)

        new_self = copy.deepcopy(self)
        new_other = copy.deepcopy(other)

        u = nx.union(new_self.network, new_other.network, rename=rename)

        all_indices = self.all_indices()
        free_indices = self.free_indices()
        rename_ix = {}
        for index in all_indices:
            if (indices is not None and index in indices) or (
                indices is None and index in free_indices
            ):
                rename_ix[index.name] = index.name
            else:
                rename_ix[index.name] = f"{rename[0]}{index.name}"

        # print("rename_ix = ", rename_ix)
        for n in self.network.nodes():
            u.nodes[f"{rename[0]}{n}"]["tensor"].rename_indices(rename_ix)

        all_indices = other.all_indices()
        free_indices = other.free_indices()
        rename_ix_o = {}
        for index in all_indices:
            if (indices is not None and index in indices) or (
                indices is None and index in free_indices
            ):
                rename_ix_o[index.name] = index.name
            else:
                rename_ix_o[index.name] = f"{rename[1]}{index.name}"

        for n in other.network.nodes():
            u.nodes[f"{rename[1]}{n}"]["tensor"].rename_indices(rename_ix_o)

        for n1 in self.network.nodes:
            for n2 in other.network.nodes:
                d1_indices = u.nodes[f"{rename[0]}{n1}"]["tensor"].indices
                d2_indices = u.nodes[f"{rename[1]}{n2}"]["tensor"].indices
                total_indices = d1_indices + d2_indices
                if len(total_indices) > len(set(total_indices)):
                    u.add_edge(f"{rename[0]}{n1}", f"{rename[1]}{n2}")

        tn = TensorNetwork()
        tn.network = u

        # print("ATTACH TN: ", tn)
        # print("self is: ", self)
        return tn

    def dim(self) -> int:
        """Number of dimensions in equivalent tensor"""
        return len(self.free_indices())

    def scale(self, scale_factor: float) -> Self:
        """Scale the tensor network."""
        for _, data in self.network.nodes(data=True):
            data["tensor"].value *= scale_factor
            break
        return self

    def inner(self, other: "TensorNetwork") -> np.ndarray:
        """Compute the inner product."""
        return self.attach(other).contract().value

    def norm(self) -> float:
        """Compute a norm of the tensor network"""
        # return np.sqrt(np.abs(self.inner(copy.deepcopy(self))))
        val = float(self.inner(self))
        out: float = np.sqrt(np.abs(val))
        return out

    def integrate(
        self,
        indices: Sequence[Index],
        weights: Union[Sequence[np.ndarray], Sequence[float], np.ndarray],
    ) -> "TensorNetwork":
        """Integrate over the chosen indices. So far just uses simpson rule."""

        out = self
        for weight, index in zip(weights, indices):
            if isinstance(weight, float):
                v = np.ones(index.size) * weight
            elif isinstance(weight, np.ndarray):
                v = weight
            else:
                raise TypeError(f"unexpected type: {type(weights)}")
            tens = vector(f"w_{index.name}", index, v)
            out = out.attach(tens, rename=("", ""))

        return out

    def fresh_index(
        self, used_indices: Optional[Sequence[IndexName]] = None
    ) -> str:
        """Generate an index that does not appear in the current network."""
        all_indices = [ind.name for ind in self.all_indices().keys()]
        i = next(self._id_counter)
        while f"s_{i}" in all_indices or (
            used_indices is not None and f"s_{i}" in used_indices
        ):
            i = next(self._id_counter)

        return f"s_{i}"

    def fresh_node(
        self, used_nodes: Optional[Sequence[NodeName]] = None
    ) -> NodeName:
        """Generate a node name that does not appear in the current network."""
        i = next(self._id_counter)
        node = f"n{i}"
        while node in self.network.nodes or (
            used_nodes is not None and node in used_nodes
        ):
            node = f"n{i}"
            i = next(self._id_counter)

        return node

    def svd(
        self,
        node_name: NodeName,
        lefts: Sequence[int],
        config: SVDConfig = SVDConfig(),
    ) -> Tuple[Tuple[NodeName, NodeName, NodeName], float]:
        """Split a node in the network into three nodes by SVD.

        The tensor at ``node_name`` is unfolded using ``lefts`` and subjected
        to a (optionally truncated) SVD.  The network is updated in-place:
        the original node becomes the left factor ``U``, and two new nodes are
        added for the singular-value diagonal ``S`` and the right factor
        ``Vt``.

        Args:
            node_name: The node to decompose.
            lefts: Index positions that form the left (row) side of the
                unfolding matrix; all other positions go to the right side.
            config: Truncation and output settings.  See ``SVDConfig`` for the
                ``atol``/``rtol`` and ``compute_data``/``compute_uv``
                options.

        Returns:
            A tuple ``((u_name, s_name, v_name), remaining_delta)`` where the
            three names identify the newly created network nodes and
            ``remaining_delta`` is the unused portion of the absolute error
            budget after truncation.
        """
        x = self.node_tensor(node_name)
        rights = [i for i in range(len(x.indices)) if i not in lefts]
        if not config.compute_data or not config.compute_uv:
            rl = Index("r_split_l", 1)
            rr = Index("r_split_r", 1)
            u_indices = [x.indices[i] for i in lefts] + [rl]
            u = Tensor(np.empty([0 for _ in u_indices]), u_indices)
            v_indices = [rr] + [x.indices[i] for i in rights]
            v = Tensor(np.empty([0 for _ in v_indices]), v_indices)
            d = config.atol or 0.0

            if config.compute_data:
                s_tuple, _ = x.svd(
                    lefts,
                    atol=config.atol,
                    rtol=config.rtol,
                    compute_uv=config.compute_uv,
                )
                s: Tensor = s_tuple[0]
            else:
                s = Tensor(np.empty(0), [rl, rr])
        else:
            x = self.node_tensor(node_name)
            # svd decompose the data into specified index partition
            [u, s, v], d = x.svd(lefts, atol=config.atol, rtol=config.rtol)

        v_name = self.fresh_node()
        new_index_r = self.fresh_index()
        self.add_node(v_name, v.rename_indices({"r_split_r": new_index_r}))

        u_name = node_name
        new_index_l = self.fresh_index()
        x_nbrs = list(self.network.neighbors(node_name))
        self.network.remove_node(node_name)
        self.add_node(u_name, u.rename_indices({"r_split_l": new_index_l}))

        s_name = self.fresh_node()
        self.add_node(
            s_name,
            s.rename_indices(
                {
                    "r_split_l": new_index_l,
                    "r_split_r": new_index_r,
                }
            ),
        )

        for y in x_nbrs:
            y_inds = self.node_tensor(y).indices
            if any(i in y_inds for i in u.indices):
                self.add_edge(u_name, y)
            elif any(i in y_inds for i in v.indices):
                self.add_edge(v_name, y)
            else:
                raise ValueError(
                    f"Indices {y_inds} does not exist in splits (",
                    u.indices,
                    ",",
                    v.indices,
                )

        self.add_edge(u_name, s_name)
        self.add_edge(s_name, v_name)

        return (u_name, s_name, v_name), d

    def qr(
        self, node_name: NodeName, lefts: Sequence[int]
    ) -> Tuple[NodeName, NodeName]:
        """Split a node by the specified index partition with QR decomposition
        and return the new node names.
        """
        # To ensure the error bound in SVD, we first orthornormalize its env.
        x = self.node_tensor(node_name)
        # svd decompose the data into specified index partition
        q, r = x.qr(lefts)

        new_index = self.fresh_index()
        x_nbrs = list(self.network.neighbors(node_name))
        self.network.remove_node(node_name)

        q_name = node_name
        self.add_node(q_name, q.rename_indices({"r_split": new_index}))
        r_name = self.fresh_node()
        self.add_node(r_name, r.rename_indices({"r_split": new_index}))

        for y in x_nbrs:
            y_inds = self.node_tensor(y).indices
            if any(i in y_inds for i in q.indices):
                self.add_edge(q_name, y)
            if any(i in y_inds for i in r.indices):
                self.add_edge(r_name, y)

        self.add_edge(q_name, r_name)

        return q_name, r_name

    def merge(
        self, name1: NodeName, name2: NodeName, compute_data: bool = True
    ) -> NodeName:
        """Merge two specified nodes into one."""
        # if not self.network.has_edge(name1, name2):
        #     raise RuntimeError(
        #         f"Cannot merge nodes that are not adjacent: {name1}, {name2}"
        #     )

        t1 = self.node_tensor(name1)
        t2 = self.node_tensor(name2)

        if compute_data:
            result = t1.contract(t2)
        else:
            l_inds = [ind for ind in t1.indices if ind not in t2.indices]
            r_inds = [ind for ind in t2.indices if ind not in t1.indices]
            inds = l_inds + r_inds
            result = Tensor(np.empty([0 for _ in inds]), inds)

        n2_nbrs = list(self.network.neighbors(name2))
        self.network.remove_node(name2)
        self.set_node_tensor(name1, result)
        for n in n2_nbrs:
            if n != name1:
                self.add_edge(name1, n)

        # check whether there are multiple contraction indices after merge
        # if there exists, we reshape the data to collapse them into one
        for n in self.network.neighbors(name1):
            contract_inds = self.get_contraction_index(n, name1)
            if len(contract_inds) > 1:
                new_ind = self.fresh_index()
                # reshape both n and name1 to collapse these inds
                n_tensor = self.node_tensor(n)
                new_n_tensor = n_tensor.merge_indices(contract_inds, new_ind)
                self.set_node_tensor(n, new_n_tensor)

                tensor = self.node_tensor(name1)
                new_tensor = tensor.merge_indices(contract_inds, new_ind)
                self.set_node_tensor(name1, new_tensor)

        return name1

    def cost(self) -> int:
        """Compute the cost for the tensor network.

        The cost is defined as sum of tensor core sizes.
        """
        cost = 0
        for n in self.network.nodes:
            indices = self.node_tensor(n).indices
            # print(indices)
            n_cost = int(np.prod([i.size for i in indices]))
            cost += n_cost

        return int(cost)

    def size(self) -> int:
        """Compute the size of the tensor network by multiplying the sizes
        of each free index."""
        indices = self.free_indices()
        return int(np.prod([ind.size for ind in indices]))

    def split_index(
        self, split_op: IndexSplit, compute_data: bool = True
    ) -> Optional[NodeName]:
        """Split free indices into smaller parts"""
        n: NodeName
        for n in self.network.nodes:
            n = typing.cast(NodeName, n)
            tensor = self.node_tensor(n)
            tensor = tensor.split_indices(split_op, compute_data)
            rename_map: Dict[IndexName, IndexName] = {}
            new_indices = []
            used_results_cnt = 0
            for ind in tensor.indices:
                if isinstance(ind.name, str) and ind.name.startswith(
                    "_fresh_index_"
                ):
                    if split_op.result is not None:
                        new_ind = split_op.result[used_results_cnt].name
                        used_results_cnt += 1
                    else:
                        new_ind = f"{split_op.index.name}_{ind.name[13:]}"
                    rename_map[ind.name] = new_ind
                    new_indices.append(new_ind)
            # print(rename_map)
            tensor = tensor.rename_indices(rename_map)
            if len(new_indices) > 0:
                split_op.result = [
                    ind for ind in tensor.indices if ind.name in new_indices
                ]
                self.set_node_tensor(n, tensor)
                return n

        return None

    def fresh_names(
        self, used_nodes: Sequence[NodeName], used_indices: Sequence[IndexName]
    ) -> Tuple[dict, dict]:
        """Create fresh node and index names"""
        node_subst = {}
        free_inds = self.free_indices()
        index_subst = {}

        for n in self.network.nodes:
            node_subst[n] = self.fresh_node(used_nodes)
            new_indices = []
            tensor = self.node_tensor(n)
            for ind in tensor.indices:
                if ind not in free_inds and ind.name not in index_subst:
                    index_subst[ind.name] = self.fresh_index(used_indices)

                new_name = index_subst.get(ind.name, ind.name)
                new_indices.append(ind.with_new_name(new_name))

            new_tensor = Tensor(tensor.value, new_indices)
            self.set_node_tensor(n, new_tensor)

        self.network = nx.relabel_nodes(self.network, node_subst, copy=True)
        return node_subst, index_subst

    def evaluate(
        self, indices: Sequence[Index], values: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the tensor network at the given indices.

        Assumes indices are provided in the order retrieved by
        TensorNetwork.free_indices()
        """
        free_indices = self.free_indices()
        assert values.shape[1] == len(indices), (
            f"Expected {len(free_indices)} indices, got {values.shape[1]}"
        )

        # Every node is sliced at the sampled indices, which become one
        # shared batch axis "a" in front, and the rest is contracted as
        # usual. The plan is the same for every chunk, so build it once.
        column = {ind: k for k, ind in enumerate(indices)}
        letters = {}
        for k, ind in enumerate(self.all_indices()):
            letters[ind] = chr(98 + k)

        plans = []
        node_strs = []
        for node in self.network.nodes:
            value, columns, node_str = self._batch_plan(
                self.node_tensor(node), column, letters
            )
            plans.append((value, columns))
            node_strs.append(node_str)

        out_inds = [ind for ind in free_indices if ind not in column]
        out_str = "a" + "".join(letters[ind] for ind in out_inds)
        estr = ",".join(node_strs) + "->" + out_str

        results = np.empty([len(values)] + [ind.size for ind in out_inds])
        chunk_size = 50000
        for start in range(0, len(values), chunk_size):
            chunk = values[start : start + chunk_size]
            node_vals = []
            for value, columns in plans:
                if columns:
                    value = value[tuple(chunk[:, c] for c in columns)]
                node_vals.append(value)

            logger.debug(
                "contraction args: %s, shapes: %s",
                estr,
                [n.shape for n in node_vals],
            )
            results[start : start + chunk_size] = oe.contract(
                estr, *node_vals, optimize="auto"
            )

        return results

    @staticmethod
    def _batch_plan(
        tensor: Tensor,
        column: Dict[Index, int],
        letters: Dict[Index, str],
    ) -> Tuple[np.ndarray, List[int], str]:
        """How one node enters the batched contraction of `evaluate`.

        Returns the node value with the sampled indices moved to the front,
        the sample column of each of those indices, and the node's einsum
        term ("a" for the batch axis followed by the remaining indices).
        """
        sampled = []
        rest = []
        for i, ind in enumerate(tensor.indices):
            if ind in column:
                sampled.append(i)
            else:
                rest.append(i)

        columns = [column[tensor.indices[i]] for i in sampled]
        node_str = "".join(letters[tensor.indices[i]] for i in rest)
        if sampled:
            node_str = "a" + node_str

        return tensor.value.transpose(sampled + rest), columns, node_str

    def node_size(self, node: NodeName) -> int:
        """Get the tensor size of the given node."""
        node_inds = self.node_tensor(node).indices
        return int(np.prod([ind.size for ind in node_inds]))

    def __lt__(self, other: Self) -> bool:
        return self.cost() < other.cost()

    def __str__(self) -> str:
        """Convert to string."""
        out = "TensorNetwork\n==========\nNodes:\n"
        out += "------\n"
        for node, data in self.network.nodes(data=True):
            out += (
                f"\t{node}: shape = {data['tensor'].value.shape},"
                f"indices = {[i.name for i in data['tensor'].indices]}\n"
            )

        out += "Edges:\n"
        out += "------\n"
        for node1, node2, data in self.network.edges(data=True):
            out += f"\t{node1} -> {node2}\n"

        return out

    def ranks_along_path(self, path: Sequence[NodeName]) -> Sequence[int]:
        """Get the ranks between the nodes on the given path."""
        ranks = []
        for i, ni in enumerate(path[:-1]):
            ranks.append(self.get_contraction_index(ni, path[i + 1])[0].size)

        return ranks

    def merge_along_path(self, path: Sequence[NodeName]) -> NodeName:
        """Merge the nodes on the given path."""
        # node = path[0]
        # for n in path[1:]:
        #     self.merge(node, n)
        all_nbrs = []
        for n in path:
            for nbr in self.network.neighbors(n):
                if nbr not in path:
                    all_nbrs.append(nbr)

        subnet = TensorNetwork()
        subnet.network = nx.subgraph(self.network, path).copy()
        subnode = subnet.contract()

        for n in set(path):
            self.network.remove_node(n)

        self.add_node(path[0], subnode)
        for nbr in all_nbrs:
            self.add_edge(path[0], nbr)

        return path[0]

    def as_func(self, indices: List[Index]) -> FuncTensorNetwork:
        """Convert the tensor network to a function call representation."""
        return FuncTensorNetwork(indices, self)

    @typing.no_type_check
    def draw(self, ax=None, node_label=False):
        """Draw a networkx representation of the network."""

        _ = plt.figure(1, figsize=(15, 10), dpi=100)

        # Define color and shape maps
        shape_map = {"A": "o", "B": "o", "C": "o"}
        size_map = {"A": 800, "B": 500, "C": 100}
        node_groups = {"A": [], "B": [], "C": []}

        # with_label = {'A': True, 'B': False}

        free_indices = sorted(self.free_indices())

        free_graph = nx.Graph()
        for index in free_indices:
            if index.size == 1:
                continue

            free_graph.add_node(f"{index.name}-{index.size}")

        new_graph = nx.compose(self.network, free_graph)
        for index in free_indices:
            if index.size == 1:
                continue

            name1 = f"{index.name}-{index.size}"
            # name1 = f"I{i}-{index.size}"
            for node, data in self.network.nodes(data=True):
                if index in data["tensor"].indices:
                    new_graph.add_edge(node, name1)

        # To use graphviz layout,
        # you need to install both graphviz and pygraphviz.
        pos = nx.drawing.nx_agraph.graphviz_layout(
            new_graph,
            prog="neato",
            args="-Gnodesep=1.5 -Granksep=1.5 -Goverlap=false",
        )
        # pos = nx.planar_layout(new_graph)

        for node, data in self.network.nodes(data=True):
            node_groups["A"].append(node)

        for node in free_graph.nodes():
            if str(node).endswith("_node"):
                node_groups["C"].append(node)
            else:
                node_groups["B"].append(node)

        # node_info maps each group to (nodes_list, shape, size)
        node_info = {
            g: (nodes, shape_map[g], size_map[g])
            for g, nodes in node_groups.items()
        }
        render_ctx = {"pos": pos, "ax": ax}
        self._draw_nodes(new_graph, render_ctx, node_info, node_label)
        self._draw_edges(new_graph, render_ctx)

    def _draw_nodes(
        self, graph: Any, render_ctx: Any, node_info: Any, node_label: Any
    ) -> None:
        """Draw all node groups with their labels."""
        pos, ax = render_ctx["pos"], render_ctx["ax"]
        for group, (nodes, shape, size) in node_info.items():
            color = "tab:blue" if group == "A" else "w"
            nx.draw_networkx_nodes(
                graph,
                pos,
                ax=ax,
                nodelist=nodes,
                node_color=color,
                node_shape=shape,
                node_size=size,
                **(
                    {"linewidths": 2.0, "edgecolors": "k"}
                    if group == "A"
                    else {}
                ),
            )
            if group == "A" and node_label:
                node_labels = {node: node for node in nodes}
                nx.draw_networkx_labels(
                    graph, pos, ax=ax, labels=node_labels, font_size=12
                )
            if group == "B":
                node_labels = {node: node for node in nodes}
                nx.draw_networkx_labels(
                    graph, pos, ax=ax, labels=node_labels, font_size=12
                )

    def _draw_edges(self, graph: Any, render_ctx: Any) -> None:
        """Draw visible and invisible edges plus edge labels."""
        pos, ax = render_ctx["pos"], render_ctx["ax"]
        edge_labels = {}
        for u, v in self.network.edges():
            indices = self.get_contraction_index(u, v)
            edge_labels[(u, v)] = "-".join(f"{i.size}" for i in indices)

        visible_edges, invisible_edges = [], []
        for u, v in graph.edges:
            is_ghost = (
                str(u).endswith("_node") and str(u).startswith(str(v))
            ) or (str(v).endswith("_node") and str(v).startswith(str(u)))
            if is_ghost:
                invisible_edges.append((u, v))
            else:
                visible_edges.append((u, v))

        nx.draw_networkx_edges(
            graph, pos, edgelist=visible_edges, ax=ax, width=2.0
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=invisible_edges,
            ax=ax,
            width=0.0,
            edge_color="white",
        )
        nx.draw_networkx_edge_labels(
            graph, pos, ax=ax, edge_labels=edge_labels, font_size=10
        )

    def to_dict(self) -> dict:
        """Convert Tensor Network to dictionary."""
        # temporary graph
        plain_graph = nx.Graph()
        plain_graph.add_nodes_from(self.network.nodes)
        plain_graph.add_edges_from(self.network.edges)

        for node_name, node_data in self.network.nodes(data=True):
            if "tensor" in node_data:
                plain_graph.nodes[node_name]["tensor_dict"] = node_data[
                    "tensor"
                ].to_dict()

        # nx built in convert to dict
        out: dict = nx.node_link_data(plain_graph)
        return out

    @classmethod
    def from_dict(cls, data_dict: dict) -> "TensorNetwork":
        """Build a Tensor Network from a dictionary."""

        reconstructed_graph = nx.node_link_graph(data_dict)

        # Create a new, empty TensorNetwork instance
        new_tn = cls()

        # Add nodes and edges to the new instance's network
        new_tn.network.add_nodes_from(reconstructed_graph.nodes)
        new_tn.network.add_edges_from(reconstructed_graph.edges)

        # Iterate through the reconstructed nodes and
        # convert the dictionaries back to Tensor objects
        for node_name, node_data in reconstructed_graph.nodes(data=True):
            if "tensor_dict" in node_data:
                tensor_obj = Tensor.from_dict(node_data["tensor_dict"])
                new_tn.set_node_tensor(node_name, tensor_obj)

        return new_tn

    def to_separated_dict(self) -> tuple[dict, dict[int, np.ndarray]]:
        """Separates the network into metadata and a dict of NumPy arrays."""

        plain_graph = nx.Graph()
        plain_graph.add_nodes_from(
            self.network.nodes(data=False)
        )  # Add nodes without attributes first
        plain_graph.add_edges_from(self.network.edges)

        for node_name, node_data in self.network.nodes(data=True):
            if "tensor" in node_data:
                tensor_as_dict = node_data["tensor"].to_dict()
                plain_graph.nodes[node_name]["tensor_dict"] = tensor_as_dict

        metadata = nx.node_link_data(plain_graph)
        numpy_arrays = {}

        metadata["numpy_arrays_info"] = {}

        for node_metadata in metadata.get("nodes", []):
            tensor_dict = node_metadata.pop("tensor_dict")
            node_id = node_metadata["id"]

            array_value = np.ascontiguousarray(tensor_dict["value"])
            numpy_arrays[node_id] = array_value

            metadata["numpy_arrays_info"][node_id] = {
                "shape": [int(dim) for dim in array_value.shape],
                "dtype": array_value.dtype.name,
            }
            node_metadata["tensor_indices"] = tensor_dict["indices"]
            for elem in node_metadata["tensor_indices"]:
                if not isinstance(elem["size"], int):
                    try:
                        elem["size"] = [int(dim) for dim in elem["size"]]
                    except TypeError:
                        elem["size"] = int(elem["size"])

        return metadata, numpy_arrays

    @classmethod
    def from_separated_dict(
        cls, metadata: dict, numpy_arrays: dict[int, np.ndarray]
    ) -> "TensorNetwork":
        """Reconstructs a TensorNetwork from separated metadata and arrays."""

        for node_data in metadata["nodes"]:
            node_id = node_data["id"]
            if node_id in numpy_arrays:
                node_data["tensor_dict"] = {
                    "value": numpy_arrays[node_id],
                    "indices": node_data.pop("tensor_indices"),
                }
        return cls.from_dict(metadata)

    def round(
        self,
        node_name: NodeName,
        *,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> Tuple[NodeName, float]:
        """
        Truncate bond dimensions across the tree using a shared error budget.

        Orthonormalises the tree at ``node_name``, then performs a depth-first
        sweep discarding singular values whose cumulative squared sum fits
        within the budget.  Because the tree is orthonormal, errors at each
        bond are independent and additive, so the total squared error stays
        within the original tolerance.

        Args:
            node_name: The node at which the sweep begins.
            atol: Absolute error budget.  Mutually exclusive with ``rtol``.
            rtol: Relative error tolerance.  Converted to an absolute budget
                by multiplying by the network's Frobenius norm before the
                sweep begins.  Mutually exclusive with ``atol``.

        Returns:
            A tuple ``(root_node, remaining_delta)`` where ``root_node`` is
            the name of the node representing the root of the processed
            sub-tree, and ``remaining_delta`` is the unused portion of the
            absolute error budget after all truncations.

        Raises:
            ValueError: If both or neither of ``atol`` and ``rtol`` are given.
        """
        if atol is not None and rtol is not None:
            raise ValueError(
                "specify exactly one of 'atol' or 'rtol', not both"
            )
        if atol is None and rtol is None:
            raise ValueError("specify one of 'atol' or 'rtol'")
        delta = atol if atol is not None else (rtol or 0.0) * self.norm()
        self.orthonormalize(node_name)
        return self._round_impl(node_name, delta, set(), initial=True)

    def _round_impl(
        self,
        node_name: NodeName,
        delta: float,
        visited: set,
        initial: bool,
    ) -> Tuple[NodeName, float]:
        """Recursive depth-first truncation sweep with an absolute budget."""
        node_indices = self.node_tensor(node_name).indices
        # which neighbour each bond index leads to; the bonds handled later
        # in the loop are untouched by the merges of earlier ones
        nbr_by_bond: Dict[Index, NodeName] = {}
        for nbr in self.network.neighbors(node_name):
            for bond in self.get_contraction_index(node_name, nbr):
                nbr_by_bond[bond] = nbr

        kept_indices = []
        free_indices = []
        r = node_name
        for idx in node_indices:
            if idx in visited:
                kept_indices.append(idx)
                continue

            nbr = nbr_by_bond.get(idx)
            if nbr is None:
                free_indices.append(idx)
                continue

            curr_indices = self.node_tensor(node_name).indices
            left_indices = [
                curr_indices.index(i) for i in curr_indices if i != idx
            ]
            [node_name, s, v], delta = self.svd(
                node_name,
                left_indices,
                SVDConfig(atol=delta),
            )
            self.merge(v, s)
            self.merge(nbr, v)
            visited_index = self.get_contraction_index(node_name, nbr)
            for idx in visited_index:
                visited.add(idx)

            r, delta = self._round_impl(nbr, delta, visited, initial=False)
            self.merge(node_name, r)

        if not initial:
            node_indices = self.node_tensor(node_name).indices
            left_indices, right_indices = [], []
            for i, idx in enumerate(node_indices):
                if idx in free_indices or idx not in kept_indices:
                    left_indices.append(i)
                else:
                    right_indices.append(i)
            _, r = self.qr(node_name, left_indices)

        return r, delta

    def collect_node_index_merges(self, node: NodeName) -> List[IndexMerge]:
        """Collect index merge operations for a single node."""
        merges = []
        all_indices = self.free_indices()
        node_indices = self.node_tensor(node).indices
        ind_groups: Dict[str, List[Index]] = {}
        for ind in node_indices:
            if ind not in all_indices:
                continue

            ind_parent = str(ind.name).split("_", maxsplit=1)[0]
            if ind_parent not in ind_groups:
                ind_groups[ind_parent] = []

            ind_groups[ind_parent].append(ind)

        # print(ind_groups)

        for parent, children in ind_groups.items():
            children.sort()
            all_children = []
            for ind in all_indices:
                if str(ind.name).startswith(parent):
                    all_children.append(ind)

            parent_size = math.prod(c.size for c in children)
            parent_name: IndexName
            if all(c in children for c in all_children):
                parent_name = parent
            else:
                parent_name = children[0].name

            parent_ind = Index(parent_name, parent_size)
            merge_op = IndexMerge(indices=children, result=parent_ind)
            merges.append(merge_op)

        return merges

    def _compress_indices(self) -> List[IndexMerge]:
        """Compress consectutive indices that are decomposed from one index"""
        tree = TensorNetwork()
        tree.network = self.network
        merges = []

        # collect all the indices in the current network by the original names
        for node in tree.network.nodes:
            node_merges = tree.collect_node_index_merges(node)
            merges.extend(node_merges)

        for merge_op in merges:
            tree.merge_index(merge_op)

        return merges

    def compress(self) -> "TensorNetwork":
        """Compress the network by removing nodes
        where one index equals to the product of other indices.
        """
        for n, nd in list(self.network.nodes(data=True)):
            indices = nd["tensor"].indices
            deleted = False
            for ind in indices:
                if ind.size == np.prod([j.size for j in indices if j != ind]):
                    # we can merge the nodes on the two ends of ind
                    nbrs = list(self.network.neighbors(n))
                    for nbr in nbrs:
                        nbr_indices = self.node_tensor(nbr).indices
                        if ind in nbr_indices:
                            self.merge(nbr, n)
                            deleted = True
                            break

                    if deleted:
                        break

        tree = TensorNetwork()
        tree.network = self.network
        return tree

    def postorder_orthonormal(
        self,
        visited: Dict[NodeName, int],
        pname: Optional[NodeName],
        name: NodeName,
    ) -> NodeName:
        """Postorder traversal the network from a given node name."""
        visited[name] = 1
        nbrs = list(self.network.neighbors(name))
        permute_indices = []
        merged = name
        for n in nbrs:
            if n not in visited:
                # Process children before the current node.
                c = self.postorder_orthonormal(visited, name, n)

                # Since split relying on ordered indices,
                # we should restore the index order here.
                indices = self.node_tensor(merged).indices
                permute_index = indices.index(
                    self.get_contraction_index(merged, c)[0]
                )
                permute_indices = list(range(permute_index))
                permute_indices.append(len(indices) - 1)
                permute_indices.extend(
                    list(range(permute_index, len(indices) - 1))
                )

                merged = self.merge(merged, c)

                # restore the last index into the permute_index position
                self.set_node_tensor(
                    merged,
                    self.node_tensor(merged).permute(permute_indices),
                )

        if pname is None:
            return merged

        left_indices, right_indices = [], []
        merged_indices = self.node_tensor(merged).indices
        # print(merged_indices)
        # print(visited)
        for i, index in enumerate(merged_indices):
            common_index = None
            for n in self.network.neighbors(merged):
                n_indices = self.node_tensor(n).indices
                if index in n_indices:
                    common_index = i

                    # The edge direction is determined by
                    # whether a neighbor node has been processed.
                    # In post-order traversal, if a neighbor has been
                    # processed before the current node, it is view as
                    # a child of the current node.
                    # Otherwise, it is viewed as the parent.
                    # The edge direction matters in orthonormalization
                    # because the q part should include indices
                    # shared with its children and the r part should
                    # include indices shared with its parent.
                    # We use the left_indices to keep track of indices
                    # shared with children, and right_indices to keep
                    # track of indices shared with the parent.
                    if n not in visited or visited[n] == 2:
                        left_indices.append(common_index)
                    else:
                        right_indices.append(common_index)

                    break
                # print(left_indices, right_indices)

            if common_index is None:
                left_indices.append(i)

        # if len(right_indices) == 0:
        #     print(self)
        visited[name] = 2
        visited[merged] = 2

        # right_sz = np.prod([merged_indices[i].size for i in right_indices])
        # optimization: this step creates redundant nodes,
        # so to avoid them we directly eliminate the node with a merge.
        # if (
        #     len(left_indices) == 1
        #     and merged_indices[left_indices[0]].size <= right_sz
        # ):
        #     return merged

        q, r = self.qr(merged, left_indices)
        # this split changes the index orders,
        # which affects the outer split result.
        # q has the indices r_split x right_indices
        # but we want r_split to replace the original left_indices
        # so we need to permute this tensor
        permute_indices = list(range(right_indices[0]))
        permute_indices.append(len(left_indices))
        permute_indices.extend(
            list(range(right_indices[0], len(left_indices)))
        )
        self.set_node_tensor(q, self.node_tensor(q).permute(permute_indices))

        return r

    def orthonormalize(self, name: NodeName) -> NodeName:
        """Orthonormalize the environment network for the specified node.

        Note that this method changes all node names in the network.
        It returns the new name for the given node after orthonormalization.
        """
        # traverse the tree rooted at the given node in the post order
        # 1 for visited and 2 for processed
        result: NodeName = self.postorder_orthonormal({}, None, name)
        return result

    def canonical_structure(self, consider_ranks: bool = False) -> int:
        """Compute the canonical structure of the tensor network.

        This method ignores all values, keeps all free indices and edge labels.
        If the resulted topology is the same, we consider
        """
        # find the node with first free index and use it as the tree root
        free_indices = sorted(self.free_indices())
        root = ""
        for n, d in self.network.nodes(data=True):
            if free_indices[0] in d["tensor"].indices:
                root = n
                break

        visited = {}

        def _postorder(name: NodeName) -> int:
            """Hash the nodes by their postorder"""
            visited[name] = 1
            children_rs = []
            # node names may mix ints and strs, so sort by their text
            nbrs = sorted(self.network.neighbors(name), key=str)
            for n in nbrs:
                if n not in visited:
                    # Process children before the current node.
                    children_rs.append(_postorder(n))

            sorted_children_rs = tuple(sorted(children_rs))
            indices = self.node_tensor(name).indices
            all_free_indices = self.free_indices()
            ranks = tuple(sorted([i.size for i in indices]))
            self_free_indices = tuple(
                sorted([i for i in indices if i in all_free_indices])
            )

            visited[name] = 2
            if consider_ranks:
                return hash((self_free_indices, ranks, sorted_children_rs))

            return hash((self_free_indices, sorted_children_rs))

        return _postorder(root)

    def leaf_indices(
        self,
        visited: Set[NodeName],
        node_name: NodeName,
        cut: Optional[Set[IndexName]] = None,
    ) -> List:
        """Get all leaf indices for the subtree rooted at the given node."""
        free = set(self.free_indices())
        return self._leaf_indices(free, visited, node_name, cut or set())

    def _leaf_indices(
        self,
        free: Set[Index],
        visited: Set[NodeName],
        node_name: NodeName,
        cut: Set[IndexName],
    ) -> List:
        indices = self.node_tensor(node_name).indices
        perm = []
        leaves = []
        visited.add(node_name)

        # free indices are added first
        if len(visited) != 1:
            for i, ind in enumerate(indices):
                if ind in free:
                    leaves.append([ind])
                    perm.append(i)

        for n in self.network.neighbors(node_name):
            if n in visited:
                continue

            common_index = self.get_contraction_index(n, node_name)
            assert len(common_index) == 1
            if common_index[0].name in cut:
                continue

            # the cut only applies to the edges of the root
            leaves.append(self._leaf_indices(free, visited, n, set()))
            perm.append(indices.index(common_index[0]))

        # reorder the leaves according to the order of the indices
        return [leaves[i] for i in np.argsort(perm)]

    def node_by_free_index(self, index: IndexName) -> NodeName:
        """Identify the node in the network containing the given free index"""
        node: NodeName
        for node in self.network.nodes:
            if any(
                ind.name == index for ind in self.node_tensor(node).indices
            ):
                return node

        raise KeyError(f"Cannot find index {index} in the network")

    def _canonicalize_indices(self, tree: DimTreeNode) -> None:
        """sort the children by free indices
        and get the corresponding children nodes
        """
        indices: List[Index] = []
        node_indices = self.node_tensor(tree.node).indices
        for ind in tree.free_indices:
            indices.append(ind)

        # children indices
        for n in sorted(tree.down_info.nodes):
            self._canonicalize_indices(n)
            ind = self.get_contraction_index(n.node, tree.node)[0]
            indices.append(ind)

        # parent indices, should be one
        p_indices = [ind for ind in node_indices if ind not in indices]
        assert len(p_indices) <= 1, (
            f"should have at most one parent index, but get {p_indices}"
        )

        indices.extend(p_indices)
        perm = [node_indices.index(ind) for ind in indices]
        tree.perm = perm

    def dimension_tree(self, root: NodeName) -> DimTreeNode:
        """Create a mapping from set of indices to node names.
        Assume that the tree is rooted at the give node.
        """
        free_indices = self.free_indices()

        # do the dfs traversal starting from the root
        def construct(visited: Set[NodeName], node: NodeName) -> DimTreeNode:
            visited.add(node)

            children: List[DimTreeNode] = []
            for nbr in self.network.neighbors(node):
                if nbr not in visited:
                    nbr_tree = construct(visited, nbr)
                    children.append(nbr_tree)

            indices, node_free_indices = [], []
            up_indices = []
            for ind in self.node_tensor(node).indices:
                if ind in free_indices:
                    indices.append(ind)
                    node_free_indices.append(ind)
                    up_indices.append(ind)

            sorted_children = sorted(children, key=lambda x: x.indices)
            for c in sorted_children:
                up_indices.extend(c.indices)
                indices.extend(c.indices)

            res = DimTreeNode(
                node=node,
                indices=indices,
                free_indices=sorted(node_free_indices),
                down_info=NodeInfo(sorted_children, [], np.empty(0)),
                up_info=NodeInfo(
                    [], up_indices, np.empty((0, len(up_indices)))
                ),
            )

            for c in sorted_children:
                c.up_info.nodes = [res]

            return res

        def assign_indices(tree: DimTreeNode) -> None:
            if len(tree.up_info.nodes) > 0:
                p = tree.up_info.nodes[0]
                tree.down_info.indices = p.free_indices[:]
                tree.down_info.indices.extend(p.down_info.indices)
                for c in p.down_info.nodes:
                    if c.node != tree.node:
                        tree.down_info.indices.extend(c.up_info.indices)

                tree.down_info.vals = np.empty(
                    (0, len(tree.down_info.indices))
                )

            for c in tree.down_info.nodes:
                assign_indices(c)

        tree = construct(set(), root)
        assign_indices(tree)
        self._canonicalize_indices(tree)
        return tree

    def merge_index(self, merge_op: IndexMerge) -> Self:
        """Merge specified free indices"""
        for n in self.network.nodes:
            tensor = self.node_tensor(n)

            new_ind: IndexName = ""
            if merge_op.result is None:
                new_ind = "_".join(str(ind.name) for ind in merge_op.indices)
            else:
                new_ind = merge_op.result.name
            tensor = tensor.merge_indices(merge_op.indices, new_ind)

            self.set_node_tensor(n, tensor)

        return self

    # ========================================
    # Swap and its helper functions
    # ========================================

    def swap(
        self,
        ind_nodes: Sequence[NodeName],
        _delta: float = 0,
        anchor: Optional[NodeName] = None,
    ) -> Tuple[NodeName, NodeName]:
        """Swap the indices so that the target indices are adjacent."""
        ind_nodes = list(set(ind_nodes))
        # if they are already neighbors
        subnet = nx.subgraph(self.network, ind_nodes)
        if len(subnet.edges) == len(subnet.nodes) - 1:
            return self._max_dist_nodes(ind_nodes, ind_nodes[0])

        if anchor is None:
            anchor, _ = self._best_anchor(ind_nodes)
        logger.debug(
            "swapping nodes %s to be neighbors in the network %s",
            ind_nodes,
            self,
        )
        for node in ind_nodes:
            path = nx.shortest_path(self.network, node, anchor)
            self._swap_along_path(path, node)

        return self._max_dist_nodes(ind_nodes, anchor)

    def swap_nbr(
        self,
        moving: NodeName,
        other: NodeName,
        ahead: Optional[NodeName] = None,
    ) -> None:
        """Move `moving` one step past its neighbor `other`.

        Afterwards `moving` occupies the position `other` had, still
        carrying its own free indices, and `other` occupies the position
        `moving` had, keeping its free indices and its other subtrees:

            before:  [subtrees of moving] - moving - other - ahead
            after:   [subtrees of moving] - other - moving - ahead

        `ahead` is the neighbor of `other` that `moving` should be
        connected to next (the next node on the path it travels along);
        with `ahead=None` the swap is a plain exchange of positions.
        """
        free = self.free_indices()

        # decide what travels with `moving`. Only its free indices
        # do, plus the bond towards `ahead` so that it lands next to it.
        # Everything else -- the bonds to moving's old subtrees, all of
        # other's free indices and other's remaining bonds -- stays with
        # `other`, which is what puts `other` into moving's old position.
        carried = [
            ind for ind in self.node_tensor(moving).indices if ind in free
        ]
        if ahead is not None:
            carried.append(self.get_contraction_index(other, ahead)[0])

        # contract the two neighbors into one node, then split it
        # again by QR. Q takes the indices that stay behind and is
        # orthonormal; R takes the carried indices along with the norm.
        merged = self.merge(moving, other)
        merged_inds = self.node_tensor(merged).indices
        lefts = [i for i, ind in enumerate(merged_inds) if ind not in carried]
        q, r = self.qr(merged, lefts)

        # name the factors after the positions they now occupy.
        self.network = nx.relabel_nodes(self.network, {q: other, r: moving})

    def _anchor_distance(
        self,
        visited: Set[NodeName],
        nodes: Sequence[NodeName],
        node: NodeName,
        dist_state: Tuple[int, int] = (0, 0),
    ) -> int:
        num_confirmed, curr_dist = dist_state
        visited.add(node)

        total_dist = curr_dist - num_confirmed
        if node in nodes:
            num_confirmed += 1

        for nbr in self.network.neighbors(node):
            if nbr not in visited:
                total_dist += self._anchor_distance(
                    visited, nodes, nbr, (num_confirmed, curr_dist + 1)
                )

        return total_dist

    def _best_anchor(
        self, candidates: Sequence[NodeName]
    ) -> Tuple[NodeName, float]:
        """Find the best anchor node such that all other nodes are closest
        to it."""
        # first, we find the best anchor node
        best_anchor = candidates[0]
        best_dist = float("inf")

        for anchor_candidate in candidates:
            total_dist = self._anchor_distance(
                set(), candidates, anchor_candidate
            )

            if total_dist < best_dist:
                best_anchor = anchor_candidate
                best_dist = total_dist

        return best_anchor, best_dist

    def _swap_along_path(
        self,
        path: Sequence[NodeName],
        moving_node: NodeName,
    ) -> None:
        """Move `moving_node` to the end of `path`, one neighbor at a time.

        `path` starts at `moving_node`; each step swaps it past the next
        node while telling the swap which node comes after that, so the
        moved node keeps heading down the path.
        """
        logger.debug("moving %s along the path %s", moving_node, path)
        for i, other in enumerate(path[1:], start=1):
            ahead = path[i + 1] if i + 1 < len(path) else None
            self.swap_nbr(moving_node, other, ahead)

    def _max_dist_nodes(
        self, ind_nodes: Sequence[NodeName], anchor: NodeName
    ) -> Tuple[NodeName, NodeName]:
        """Find the two nodes with the maximum distance among the given
        nodes."""
        if len(ind_nodes) < 2:
            return anchor, anchor

        max_distance = -1
        left_anchor = right_anchor = anchor

        for u, v in itertools.combinations(ind_nodes, 2):
            distance = self.distance(u, v)
            if distance > max_distance:
                max_distance = distance
                left_anchor, right_anchor = u, v

        return left_anchor, right_anchor

    def __add__(self, other: Any) -> Self:
        """Add two tree networks."""
        if not isinstance(other, TensorNetwork):
            raise NotImplementedError

        assert nx.is_isomorphic(self.network, other.network)

        # assign the root at the same index name
        root_ind = self.free_indices()[0]
        self_root = self.node_by_free_index(root_ind.name)
        self_tree = self.dimension_tree(self_root)
        other_root = other.node_by_free_index(root_ind.name)
        other_tree = other.dimension_tree(other_root)

        result_net = copy.deepcopy(self)
        self._binary_op(other, "add", (self_tree, other_tree), result_net)

        return result_net

    def __sub__(self, other: Any) -> Self:
        """Subtract two tree networks."""
        if not isinstance(other, TensorNetwork):
            raise NotImplementedError

        assert nx.is_isomorphic(self.network, other.network)

        neg_net = copy.deepcopy(other)
        a_node = list(neg_net.network.nodes)[0]
        a_tensor = neg_net.node_tensor(a_node)
        neg_net.set_node_tensor(
            a_node, a_tensor.update_val_size(a_tensor.value * -1)
        )
        return self + neg_net

    def __mul__(self, other: Any) -> Self:
        """Elementwise multiplication of two tree networks."""
        if not isinstance(other, TensorNetwork):
            raise NotImplementedError

        assert nx.is_isomorphic(self.network, other.network)

        # assign the root at the same index name
        root_ind = self.free_indices()[0]
        self_root = self.node_by_free_index(root_ind.name)
        self_tree = self.dimension_tree(self_root)
        other_root = other.node_by_free_index(root_ind.name)
        other_tree = other.dimension_tree(other_root)

        result_net = copy.deepcopy(self)
        self._binary_op(other, "mul", (self_tree, other_tree), result_net)

        return result_net

    def _binary_op(
        self,
        other: "TensorNetwork",
        op: Literal["add", "mul"],
        trees: Tuple[DimTreeNode, DimTreeNode],
        result_net: Self,
    ) -> None:
        tree1, tree2 = trees
        tensor1 = self.node_tensor(tree1.node)
        tensor2 = other.node_tensor(tree2.node)
        assert len(tensor1.indices) == len(tensor2.indices)

        if op == "add":
            res = tensor1.block_diagonal(tensor2, tree1.free_indices)
        elif op == "mul":
            res = tensor1.mult(tensor2, self.free_indices())
        else:
            raise ValueError(f"Unknown operation {op}")

        result_net.set_node_tensor(tree1.node, res)

        for c1, c2 in zip(tree1.down_info.nodes, tree2.down_info.nodes):
            self._binary_op(other, op, (c1, c2), result_net)

    def distance(self, node1: NodeName, node2: NodeName) -> int:
        """Compute the distance between two nodes without creating dimension
        trees."""
        if node1 == node2:
            return 0

        return int(nx.shortest_path_length(self.network, node1, node2))

    def _svd_lefts(
        self, indices: Sequence[Index], svd_node: NodeName
    ) -> List[int]:
        """Get the positions such that the given indices are on the left."""
        res = self._partition_at(svd_node, self._subtree_views(indices))
        assert res is not None, (
            f"{svd_node} is not the correct partition point"
        )

        svd_node_inds = self.node_tensor(svd_node).indices
        return [svd_node_inds.index(ind) for ind in res.lca_indices]

    def _subtree_views(self, indices: Sequence[Index]) -> SubtreeViews:
        """Precompute what lies behind every directed edge.

        One pass from the leaves up gives the target indices and the number
        of free indices below every node; the view across an edge in the
        other direction is the complement.
        """
        desired = frozenset(indices)
        free = set(self.free_indices())

        root = next(iter(self.network.nodes))
        order = list(nx.bfs_tree(self.network, root).nodes)
        parent: Dict[NodeName, Optional[NodeName]] = {root: None}
        parent.update(nx.bfs_predecessors(self.network, root))

        down_desired: Dict[NodeName, FrozenSet[Index]] = {}
        down_free: Dict[NodeName, int] = {}
        for node in reversed(order):
            own = [i for i in self.node_tensor(node).indices if i in free]
            got = frozenset(i for i in own if i in desired)
            n_free = len(own)
            for child in self.network.neighbors(node):
                if parent[child] == node:
                    got |= down_desired[child]
                    n_free += down_free[child]

            down_desired[node] = got
            down_free[node] = n_free

        views = SubtreeViews(desired, root, parent, down_desired, down_free)
        views.existing = self._existing_split(views)
        return views

    def _existing_split(
        self, views: SubtreeViews
    ) -> Optional[Tuple[NodeName, Index, FrozenSet[NodeName]]]:
        """Find an edge whose one side holds exactly the target indices.

        If several edges qualify (nodes without free indices between them),
        the one with the smallest target side is returned: it is the one a
        search from outside reaches first.
        """
        best = None
        best_size = len(self.network.nodes) + 1
        for a, b in self.network.edges:
            for near, far in ((a, b), (b, a)):
                got, n_other = views.behind(near, far)
                if got != views.desired or n_other != 0:
                    continue

                cut = nx.restricted_view(self.network, [], [(a, b)])
                side = frozenset(nx.node_connected_component(cut, far))
                if len(side) < best_size:
                    bond = self.get_contraction_index(near, far)[0]
                    best = (far, bond, side)
                    best_size = len(side)

        return best

    def _partition_at(
        self, node: NodeName, views: SubtreeViews
    ) -> Optional[PartitionResult]:
        """Check whether `node` separates the target indices from the rest.

        If an edge already separates exactly the targets, a search from any
        node outside its target side finds it (``EXIST``); the target-side
        endpoint is itself a valid split point (``OK``); nodes further
        inside the target side are not. Otherwise every subtree hanging off
        `node` must be pure -- only targets or none of them -- and `node` is
        the split point with its target-carrying indices as the left side
        (``OK``). Returns ``None`` when some subtree mixes both.
        """
        if views.existing is not None:
            far, bond, side = views.existing
            if node not in side:
                return PartitionResult(PartitionStatus.EXIST, far, [bond])
            if node != far:
                return None

        lefts: List[Index] = []
        for nbr in self.network.neighbors(node):
            got, n_other = views.behind(node, nbr)
            if got and n_other:
                return None

            if got:
                lefts.append(self.get_contraction_index(node, nbr)[0])

        for ind in self.node_tensor(node).indices:
            if ind in views.desired:
                lefts.append(ind)

        return PartitionResult(PartitionStatus.OK, node, lefts)

    def partition_node(self, indices: Sequence[Index]) -> PartitionResult:
        """Find a proper node that partitions the free indices as specified.

        Nodes are tried in insertion order and the first one that separates
        the target indices from the rest wins.
        """
        views = self._subtree_views(indices)
        for node in self.network.nodes:
            res = self._partition_at(node, views)
            if res is not None:
                return res

        raise ValueError(
            "Cannot find a node that realizes the partition", indices
        )

    def random_svals(
        self,
        node: NodeName,
        indices: Sequence[Index],
        params: SValsParams,
    ) -> np.ndarray:
        """Compute singular values at a specific node using randomized SVD."""
        tensor = self.node_tensor(node)
        svd_node_inds = tensor.indices
        svd_ls = self._svd_lefts(indices, node)
        svd_rs = [i for i in range(len(tensor.indices)) if i not in svd_ls]
        lsize = np.prod([svd_node_inds[lidx].size for lidx in svd_ls])
        perm = list(svd_ls) + svd_rs
        tensor_val = tensor.value.transpose(perm).reshape(int(lsize), -1)

        s: np.ndarray
        if params.random_seed is not None:
            _, s, _ = randomized_svd(
                tensor_val, params.max_rank, random_state=params.random_seed
            )
        else:
            s = np.linalg.svdvals(tensor_val)

        return s

    def svals_at(
        self,
        node: NodeName,
        indices: Sequence[Index],
        max_rank: int = 100,
        with_orthonormal: bool = True,
    ) -> np.ndarray:
        """Compute the singular values for a tensor train at a given node."""
        random_seed = 42
        logger.debug(
            "computing singular values at node %s with indices %s",
            node,
            indices,
        )
        if with_orthonormal:
            self.orthonormalize(node)

        logger.debug("after orthonormalize: %s", self)

        tensor = self.node_tensor(node)
        lefts = []
        for i, ind in enumerate(tensor.indices):
            if ind in indices:
                lefts.append(i)

        logger.debug("get left indices %s", lefts)

        perm = lefts + [
            i for i in range(len(tensor.indices)) if i not in lefts
        ]
        left_size = np.prod([ind.size for ind in indices])
        tensor_val = tensor.value.transpose(perm).reshape(left_size, -1)

        if max_rank < 10:
            return delta_svd(tensor_val, delta=0, compute_uv=False).s

        s: np.ndarray
        _, s, _ = randomized_svd(
            tensor_val, max_rank, random_state=random_seed
        )
        return s

    def replace_with(
        self,
        old_subnet: "TensorNetwork",
        new_subnet: "TensorNetwork",
        _split_info: Optional[List[IndexOp]] = None,
    ) -> "TensorNetwork":
        """Replace a node with a sub-tensor network."""
        for n in old_subnet.network.nodes:
            if n not in self.network.nodes:
                raise RuntimeError("Cannot replace nodes that doesn't exist")

        # rename the new_subnet to unique node names and index names
        curr_inds = [ind.name for ind in self.all_indices()]
        new_subnet.fresh_names(list(self.network.nodes), curr_inds)

        # all free indices from the old subnet should be maintained
        old_free_indices = set(old_subnet.free_indices())
        new_free_indices = set(new_subnet.free_indices())

        for n in new_subnet.network.nodes:
            tensor = new_subnet.node_tensor(n)
            self.add_node(n, tensor)
            for ind in tensor.indices:
                if ind not in new_free_indices or ind not in old_free_indices:
                    continue

                m = old_subnet.node_by_free_index(ind.name)
                for nbr in self.network.neighbors(m):
                    common_inds = self.get_contraction_index(nbr, m)
                    if ind in common_inds:
                        self.add_edge(n, nbr)
                        break

        for n in old_subnet.network.nodes:
            self.network.remove_node(n)

        for u, v in new_subnet.network.edges:
            self.add_edge(u, v)

        tree = TensorNetwork()
        tree.network = self.network
        return tree

    def replay_preprocess(self, actions: Sequence[Action]) -> None:
        """Apply the given actions around the given ranks."""

    def are_adjacent(self, indices: Sequence[Index]) -> bool:
        """Check whether the nodes containing the given free indices form a
        connected subgraph in the tree.
        """
        ind_nodes = list(
            {self.node_by_free_index(ind.name) for ind in indices}
        )
        if len(ind_nodes) <= 1:
            return True
        return bool(nx.is_connected(self.network.subgraph(ind_nodes)))

    def end_nodes(self) -> List[NodeName]:
        """Get the leaf nodes."""
        # end node is the one with one neighbor or whose value is 2D
        end_nodes: List[NodeName] = []
        for node in self.network.nodes:
            if len(list(self.network.neighbors(node))) <= 1:
                end_nodes.append(node)

        return end_nodes

    def is_tensor_train(self) -> bool:
        """Check whether the network is a tensor train: a connected, acyclic
        chain where every node has at most two neighbors and carries exactly
        one free index.
        """
        graph = self.network
        if graph.number_of_nodes() == 0:
            return False
        if not nx.is_connected(graph):
            return False
        if graph.number_of_edges() != graph.number_of_nodes() - 1:
            return False
        if any(deg > 2 for _, deg in graph.degree()):
            return False

        free_inds = self.free_indices()
        for node in graph.nodes:
            node_inds = self.node_tensor(node).indices
            num_free = sum(1 for ind in node_inds if ind in free_inds)
            if num_free != 1:
                return False

        return True

    def svals_by_merge(
        self,
        indices: Sequence[Index],
        max_rank: int = 100,
        rand: bool = True,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Compute the singular values for a tensor train."""
        tree = self.compress()
        ind_nodes = [tree.node_by_free_index(ind.name) for ind in indices]
        nodes = tree.swap(ind_nodes)
        if len(nodes) > 1:
            ind_nodes = [tree.node_by_free_index(ind.name) for ind in indices]
            tree.orthonormalize(ind_nodes[0])
            n = tree.merge_along_path(ind_nodes)
        else:
            n = tree.node_by_free_index(indices[0].name)
            tree.orthonormalize(n)

        return _svals_at_node(
            tree.node_tensor(n), indices, max_rank, rand, random_seed
        )


def _svals_at_node(
    node_tensor: "Tensor",
    indices: Sequence[Index],
    max_rank: int,
    rand: bool,
    random_seed: int,
) -> np.ndarray:
    """
    Compute singular values by partitioning a node's tensor at given indices.
    """
    lefts = [i for i, ind in enumerate(node_tensor.indices) if ind in indices]
    perm = lefts + [
        i for i in range(len(node_tensor.indices)) if i not in lefts
    ]
    left_size = np.prod([ind.size for ind in indices])
    tensor_val = node_tensor.value.transpose(perm).reshape(left_size, -1)
    if rand:
        _, s, _ = randomized_svd(
            tensor_val, max_rank, random_state=random_seed
        )
    else:
        s = np.linalg.svdvals(tensor_val)
    return cast(np.ndarray, s)


def vector(
    name: Union[str, int], index: Index, value: np.ndarray
) -> "TensorNetwork":
    """Convert a vector to a tensor network."""
    vec = TensorNetwork()
    vec.add_node(name, Tensor(value, [index]))
    return vec


def rand_tt(indices: List[Index], ranks: List[int]) -> TensorNetwork:
    """Return a random tt."""

    dim = len(indices)
    assert len(ranks) + 1 == len(indices)

    tt = TensorNetwork()

    r = [Index("r1", ranks[0])]
    tt.add_node(
        0,
        Tensor(np.random.randn(indices[0].size, ranks[0]), [indices[0], r[0]]),
    )

    core = 1
    for ii, index in enumerate(indices[1:-1]):
        r.append(Index(f"r{ii + 2}", ranks[ii + 1]))
        tt.add_node(
            core,
            Tensor(
                np.random.randn(ranks[ii], index.size, ranks[ii + 1]),
                [r[ii], index, r[ii + 1]],
            ),
        )
        core += 1
        tt.add_edge(ii, ii + 1)

    tt.add_node(
        dim - 1,
        Tensor(
            np.random.randn(ranks[-1], indices[-1].size), [r[-1], indices[-1]]
        ),
    )
    tt.add_edge(dim - 2, dim - 1)

    return tt


def tt_rank1(indices: List[Index], vals: List[np.ndarray]) -> TensorNetwork:
    """Return a random rank 1 TT tensor."""

    dim = len(indices)

    tt = TensorNetwork()

    r = [Index("r1", 1)]
    # print("vals[0] ", vals[0][:, np.newaxis])
    new_tens = Tensor(vals[0][:, np.newaxis], [indices[0], r[0]])
    tt.add_node(0, new_tens)
    # print("new_tens = ", new_tens.indices)

    core = 1
    for ii, index in enumerate(indices[1:-1]):
        r.append(Index(f"r{ii + 2}", 1))
        new_tens = Tensor(
            vals[ii + 1][np.newaxis, :, np.newaxis], [r[ii], index, r[ii + 1]]
        )
        tt.add_node(core, new_tens)
        tt.add_edge(core - 1, core)
        core += 1

    tt.add_node(dim - 1, Tensor(vals[-1][np.newaxis, :], [r[-1], indices[-1]]))
    tt.add_edge(dim - 2, dim - 1)
    # print("tt_rank1 = ", tt)
    return tt


def tt_separable(
    indices: List[Index], funcs: List[np.ndarray]
) -> TensorNetwork:
    """Rank 2 function formed by sums of functions of individual dimensions."""

    dim = len(indices)

    tt = TensorNetwork()
    ranks = []
    for ii, index in enumerate(indices):
        ranks.append(Index(f"r_{ii + 1}", 2))
        if ii == 0:
            val = np.ones((index.size, 2))
            val[:, 0] = funcs[ii]

            tt.add_node(ii, Tensor(val, [index, ranks[-1]]))
        elif ii < dim - 1:
            val = np.zeros((2, index.size, 2))
            val[0, :, 0] = 1.0
            val[1, :, 0] = funcs[ii]
            val[1, :, 1] = 1.0
            tt.add_node(ii, Tensor(val, [ranks[-2], index, ranks[-1]]))
        else:
            val = np.ones((2, index.size))
            val[1, :] = funcs[ii]
            tt.add_node(ii, Tensor(val, [ranks[-2], index]))

        if ii > 0:
            tt.add_edge(ii - 1, ii)

    return tt


def tt_right_orth(tn: TensorNetwork, node: int) -> TensorNetwork:
    """Right orthogonalize all but first core.

    Tree tensor network as a TT and right orthogonalize

    A right orthogonal core has the r_{k-1} x nk r_k matrix
    R(Gk(ik)) = ( Gk(1) Gk(2) · · · Gk(nk) )
    having orthonormal rows so that

    sum G_k(i) G_k(i)^T  = I

    Modifies the input tensor network

    Assumes nodes have integer names
    """
    # pylint: disable=C0103
    # above disables the snake case complaints for variables like R
    val1 = tn.value(node)
    if val1.ndim == 3:
        # print("val1.shape = ", val1.shape)
        r, n, b = val1.shape
        # val1 = np.reshape(val1, (r, n * b), order="F")
        val1 = np.reshape(val1, (r, n * b))
        # print("val1.T.shape = ", val1.T.shape)
        q, R = np.linalg.qr(val1.T, mode="reduced")
        if q.shape[1] < r:
            newq = np.zeros((q.shape[0], r))
            newq[:, : q.shape[1]] = q
            q = newq
            newr = np.zeros((r, R.shape[1]))
            newr[: R.shape[0], :] = R
            R = newr

        # print("q.shape = ", q.shape)
        # print("r.shape = ", R.shape)
        # print("r = ", r)
        # print("q shape = ", q.shape)
        # new_val = np.reshape(q.T, (r, n, b), order="F")
        new_val = np.reshape(q.T, (r, n, b))
        tn.node_tensor(node).update_val_size(new_val)
    else:
        q, R = np.linalg.qr(val1.T)
        new_val = q.T
        tn.node_tensor(node).update_val_size(new_val)

    val2 = tn.value(node - 1)
    # new_val2 = np.einsum("...i,ij->...j", val2, R.T)
    new_val2 = np.dot(val2, R.T)
    tn.node_tensor(node - 1).update_val_size(new_val2)

    return tn


def eps_to_rank(s: np.ndarray, eps: float) -> int:
    """Tranlates the matrix approximation error \
    to rank in a truncated SVD"""
    tmp = (np.sqrt(np.cumsum(np.square(s[::-1])))[::-1]) <= eps
    res: int = int(np.argmax(tmp))
    if res == 0 and not tmp[0]:
        return int(s.shape[0])
    if res == 0 and tmp[0]:
        return 1
    return res


def gram_eig_and_svd(
    gl: np.ndarray, gr: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """ Implements eigenvalue decomposition + svd to \
        the gram matrices of a TT-core and returns the \
        low-rank factors """
    pos_tol = 1e-15

    eigl, vl = np.linalg.eigh(gl)
    eigr, vr = np.linalg.eigh(gr)
    eigl = np.abs(eigl)
    eigr = np.abs(eigr)

    eigl12 = np.sqrt(eigl)
    eigr12 = np.sqrt(eigr)

    threshold = np.ceil(np.log10(np.max(eigl12) * 1e-8 + pos_tol))
    eigl12 = np.round(eigl12, min(-int(threshold), 16))
    threshold = np.ceil(np.log10(np.max(eigr12) * 1e-8 + pos_tol))
    eigr12 = np.round(eigr12, min(-int(threshold), 16))

    maskl = eigl12 == 0
    maskr = eigr12 == 0

    eiglm12 = np.zeros_like(eigl12)
    eigrm12 = np.zeros_like(eigr12)
    eiglm12[~maskl] = 1 / eigl12[~maskl]
    eigrm12[~maskr] = 1 / eigr12[~maskr]

    # eiglm12 = np.nan_to_num(eiglm12, nan=0, posinf=0, neginf=0)
    # eigrm12 = np.nan_to_num(eigrm12, nan=0, posinf=0, neginf=0)

    tmp = (eigl12[:, np.newaxis] * vl.T) @ (vr * eigr12[np.newaxis, :])

    u, s, v = np.linalg.svd(tmp)
    rk = min(tmp.shape[0], tmp.shape[1], eps_to_rank(s, delta))

    u = u[:, :rk]
    s = s[:rk]
    v = v[:rk, :]

    curr_val = vl @ (eiglm12[:, np.newaxis] * u)

    next_val = (s[:, np.newaxis] * v * eigrm12[np.newaxis, :]) @ vr.T
    return curr_val, next_val


def tt_gramsvd_round(tn: TensorNetwork, eps: float) -> TensorNetwork:
    """
    Description: Modifies the input tensor network and returns the
    rounded version by implementing the Gram-SVD based rounding
    approach [1].

    [1] - H. Al Daas, G. Ballard and L. Manning, "Parallel Tensor-
    Train Rounding using Gram SVD," 2022 IEEE International Parallel
    and Distributed Processing Symposium (IPDPS), Lyon, France, 2022,
    pp. 930-940, doi: 10.1109/IPDPS53621.2022.00095."""

    def next_gram(
        gram_now: np.ndarray, core_next: np.ndarray, order: str = "lr"
    ) -> np.ndarray:
        """ Calculates the Gram matrix corresponding to the next \
        TT-core using the Gram matrix corresponding to the \
        current core. For example, if order == "lr" (left to \
        right sweep), the function outputs a matrix of size \
        R_{k} x R{k} from a matrix of size R_{k-1} x R_{k-1}"""
        snext = core_next.shape
        if order == "lr":
            tmp = (gram_now.T @ core_next.reshape((snext[0], -1))).reshape(
                (-1, snext[-1])
            )
            return np.asarray(tmp.T @ core_next.reshape((-1, snext[-1])))

        if order == "rl":
            tmp = (core_next.reshape((-1, snext[-1])) @ gram_now).reshape(
                (-1, snext[-2] * snext[-1])
            )
            return np.asarray(
                np.dot(tmp, core_next.reshape((-1, snext[-2] * snext[-1])).T)
            )

        raise ValueError(f"Invalid order: {order}. Use 'lr' or 'rl'.")

    dim = tn.dim()
    gr_list = [tn.value(dim - 1) @ tn.value(dim - 1).T]
    # print("pre backward sweep 1")
    # print("tn" , tn)
    # Collect gram matrices from right to left
    for i in range(dim - 2, -1, -1):
        # print("i = ", i)
        # print(tn.value(i))
        gr_list.append(next_gram(gr_list[-1], tn.value(i), "rl"))
    # print("end")

    norm = np.sqrt(gr_list[-1])[0, 0]
    delta = eps * norm / (dim - 1) ** 0.5
    gr_list = gr_list[::-1]

    # print("pre backward sweep 2")
    for i in range(dim - 1):
        sh = list(tn.value(i).shape)
        shp1 = list(tn.value(i + 1).shape)
        gl = tn.value(i).reshape((-1, sh[-1])).T @ tn.value(i).reshape(
            (-1, sh[-1])
        )

        curr_val, next_val = gram_eig_and_svd(gl, gr_list[i + 1], delta)

        curr_val = tn.value(i).reshape((-1, sh[-1])) @ curr_val
        next_val = next_val @ tn.value(i + 1).reshape((shp1[0], -1))

        rk = curr_val.shape[-1]
        sh[-1] = rk
        shp1[0] = rk
        curr_val = curr_val.reshape(sh)
        next_val = next_val.reshape(shp1)
        tn.node_tensor(i).update_val_size(curr_val)
        tn.node_tensor(i + 1).update_val_size(next_val)

    return tn


def tt_svd_round(tn: TensorNetwork, eps: float) -> TensorNetwork:
    """Round a tensor train.

    Nodes should be integers 0,1,2,...,dim-1

    orthogonalize determines of QR is used to orthogonalize the
    cores. If orthogonalize=False, the Gram-SVD rounding algo-
    rithm is used.
    """
    # pylint: disable=C0103
    # above disables the snake case complaints for variables like R
    # norm2 = tn.norm()
    dim = tn.dim()
    delta = None
    out = tt_right_orth(tn, dim - 1)
    for jj in range(dim - 2, 0, -1):
        out = tt_right_orth(out, jj)

    core_list = list(out.network.nodes(data=True))
    node = core_list[0][0]
    data = core_list[0][1]
    value = out.value(node)
    trunc_svd = delta_svd(value, eps / np.sqrt(dim - 1), with_normalizing=True)
    delta = trunc_svd.delta
    assert delta is not None
    assert trunc_svd.v is not None
    assert trunc_svd.u is not None

    v = np.dot(np.diag(trunc_svd.s), trunc_svd.v)
    r2 = trunc_svd.u.shape[1]
    new_core = np.reshape(trunc_svd.u, (value.shape[0], r2))

    data["tensor"].update_val_size(new_core)

    # print("In here")
    val_old = out.node_tensor(node + 1).value
    next_val = np.einsum("ij,jk...->ik...", v, val_old)
    out.node_tensor(node + 1).update_val_size(next_val)

    for node, data in core_list[1:-1]:
        value = data["tensor"].value
        r1, n, r2a = value.shape
        val = np.reshape(value, (r1 * n, r2a))
        trunc_svd = delta_svd(val, delta)
        assert trunc_svd.v is not None
        assert trunc_svd.u is not None
        v = np.dot(np.diag(trunc_svd.s), trunc_svd.v)
        r2 = trunc_svd.u.shape[1]
        new_core = np.reshape(trunc_svd.u, (r1, n, r2))
        data["tensor"].update_val_size(new_core)

        val_old = out.node_tensor(node + 1).value
        next_val = np.einsum("ij,jk...->ik...", v, val_old)
        out.node_tensor(node + 1).update_val_size(next_val)

    return out


# Rounding sum of TT cores
def get_indices(
    maximum: int, periodicity: int, consecutive: int, start: int
) -> np.ndarray:
    """
    Gets the column indices of a matrix when right multiplied
    by the horizontal unfolding (or its transpose) of a TT-sum
    (H(X) or H(X).T). The indices correspond to the non-zero
    parts of H and helps to avoid unnecessary computation.
    """
    indices = np.asarray(
        np.concatenate(
            [
                np.arange(i, i + consecutive)
                for i in range(start, maximum, periodicity)
            ]
        )
    )
    return indices


def multiply_core_unfolding(  # pylint: disable=R0912
    mat: np.ndarray,
    cores_list: list,
    v_unfolding: bool,
    left_multiply: bool,
    transpose: bool,
) -> np.ndarray:
    """
    Multiplies a dense matrix 'mat' with a sparse block-diagonal
    core in a TT-sum.
    -   the summands are given as a Python list of tensor trains
        ('cores_list' in the arguments).
    -   'v_unfolding' indicates Vertical unfolding of a TT-core.
        i.e., a Rk-1 cross nk cross Rk core will be reshaped as
        Rk-1 * nk cross Rk matrix if v_unfolding is True. If
        False, the core will be reshaped as Rk-1 cross nk * Rk.
    -   'left_multiply' decides if 'cores_list' is left multiplied
        or right multiplied to 'mat'
    -   'transpose' denotes if we take a transpose of 'cores_list'
        before multiplication.
    """
    rows, cols = mat.shape
    n_cores = len(cores_list)
    if left_multiply:
        rk = [s.shape[-1] for s in cores_list]
        rk_cumsum = np.cumsum([0] + rk)
        rk_sum = np.sum(rk)
        if cores_list[0].ndim == 2:
            rk1 = [1 for s in cores_list]
        else:
            rk1 = [s.shape[0] for s in cores_list]
        rk1_sum = np.sum(rk1)
        rk1_cumsum = np.cumsum([0] + rk1)
        n = cores_list[0].shape[1]

        if v_unfolding and (not transpose):
            assert rows == rk_sum, f"Dimension mismatch {rows} != {rk_sum}"
            res = np.zeros((rk1_sum * n, cols))
            for i in range(n_cores):
                res[rk1_cumsum[i] * n : rk1_cumsum[i + 1] * n, :] = (
                    cores_list[i].reshape((-1, rk[i]))
                    @ mat[rk_cumsum[i] : rk_cumsum[i + 1], :]
                )
            return res

    else:
        rk = [s.shape[0] for s in cores_list]
        rk_cumsum = np.cumsum([0] + rk)
        rk_sum = np.sum(rk)
        if cores_list[0].ndim == 2:
            rk1 = [1 for s in cores_list]
        else:
            rk1 = [s.shape[-1] for s in cores_list]
        rk1_sum = np.sum(rk1)
        rk1_cumsum = np.cumsum([0] + rk1)
        n = cores_list[0].shape[1]

        if v_unfolding and (not transpose):
            assert cols == rk_sum * n, (
                f"Dimension mismatch {cols} != {rk_sum * n}"
            )
            res = np.zeros((rows, rk1_sum))
            for i in range(n_cores):
                res[:, rk1_cumsum[i] : rk1_cumsum[i + 1]] = mat[
                    :, rk_cumsum[i] * n : rk_cumsum[i + 1] * n
                ] @ cores_list[i].reshape((-1, rk1[i]))
            return res

        if (not v_unfolding) and (transpose):
            assert cols == rk1_sum * n, (
                f"Dimension mismatch {cols} != {rk1_sum * n}"
            )
            res = np.zeros((rows, rk_sum))
            for i in range(n_cores):
                ind = get_indices(cols, rk1_sum, rk1[i], rk1_cumsum[i])
                res[:, rk_cumsum[i] : rk_cumsum[i + 1]] = (
                    mat[:, ind] @ (cores_list[i].reshape((rk[i], -1))).T
                )
            return res

        if (not v_unfolding) and (not transpose):
            assert cols == rk_sum, f"Dimension mismatch {cols} != {rk_sum}"
            res = np.zeros((rows, n * rk1_sum))
            for i in range(n_cores):
                ind = get_indices(rk1_sum * n, rk1_sum, rk1[i], rk1_cumsum[i])
                res[:, ind] = mat[
                    :, rk_cumsum[i] : rk_cumsum[i + 1]
                ] @ cores_list[i].reshape((rk[i], -1))
            return res

    raise ValueError("Invalid options")


def next_gram_sum(
    gram_now: np.ndarray, core_next: list[np.ndarray], order: str = "rl"
) -> np.ndarray:
    """
    Let's say that we are dealing with 's' summands in our TT sum.

    gram_now is a sigma r_i times sigma r_i matrix (i from 1 to s)
    where r_i represents the sum of rank of a particular TT core of
    the summands. For example, it could be the sum for the last TT
    core of every summand.

    core_next is the list (of size s) of adjacent TT-core of all
    summands. For example, if gram_now corresponds to the last TT core
    of all the summands, then assuming order = rl, core_next will be a
    list of the penultimate cores of all the summands.

    order: 'lr' means left to right and 'rl' means right to left.
    """

    # shnext = [s.shape for s in core_next]
    if order == "rl":
        rk1_sum, _, rk_sum = np.sum([list(s.shape) for s in core_next], axis=0)
        n = core_next[0].shape[1]
        tmp = multiply_core_unfolding(gram_now, core_next, True, True, False)
        tmp = tmp.reshape((rk1_sum, n * rk_sum))
        return multiply_core_unfolding(tmp, core_next, False, False, True)

    if order == "lr":
        rk_sum, _, rk1_sum = np.sum([list(s.shape) for s in core_next], axis=0)
        n = core_next[0].shape[1]
        tmp = multiply_core_unfolding(gram_now, core_next, False, False, False)
        tmp = tmp.reshape((rk_sum * n, rk1_sum)).T
        return multiply_core_unfolding(tmp, core_next, True, False, False)

    raise ValueError(
        "Invalid argument for order. order should either be lr or rl"
    )


def tt_sum_gramsvd_round(
    factors_list: list[TensorNetwork],
    eps: float = 1e-14,
) -> TensorNetwork:
    """Gram-rounding of sum of tensor trains."""

    def core_info(k: int) -> tuple[list, list]:
        cores = [f.value(k) for f in factors_list]
        rk = [s.shape[0] for s in cores]
        rk1 = [s.shape[-1] for s in cores]
        n = cores[0].shape[1]
        if cores[0].ndim == 3:
            return cores, [np.sum(rk), n, np.sum(rk1)]
        return cores, [np.sum(rk), n]

    dim = factors_list[0].dim()

    ttsum = copy.deepcopy(factors_list[0])

    gr_list = [
        np.concatenate([f.value(dim - 1) for f in factors_list], axis=0)
    ]

    ttsum.node_tensor(dim - 1).update_val_size(gr_list[-1])
    gr_list = [gr_list[-1] @ gr_list[-1].T]

    gl = np.concatenate([f.value(0) for f in factors_list], axis=1)
    ttsum.node_tensor(0).update_val_size(gl)

    # Collect gram matrices from right to left
    for i in range(dim - 2, 0, -1):
        gr_list.append(
            next_gram_sum(
                gr_list[-1], [f.value(i) for f in factors_list], "rl"
            )
        )

    gr_list.append(np.sum((ttsum.value(0) @ gr_list[-1]) * ttsum.value(0)))
    norm = np.sqrt(gr_list[-1])
    delta = eps * norm / (dim - 1) ** 0.5

    gr_list = gr_list[::-1]

    for i in range(dim - 1):
        sh = list(ttsum.value(i).shape)
        core_next, shp1 = core_info(i + 1)

        gl = ttsum.value(i).reshape((-1, sh[-1])).T @ ttsum.value(i).reshape(
            (-1, sh[-1])
        )

        curr_val, next_val = gram_eig_and_svd(gl, gr_list[i + 1], delta)

        curr_val = ttsum.value(i).reshape((-1, sh[-1])) @ curr_val
        if i == (dim - 2):
            next_val = next_val @ ttsum.value(dim - 1)
        else:
            next_val = multiply_core_unfolding(
                next_val, core_next, False, False, False
            )

        rk = curr_val.shape[-1]
        sh[-1] = rk
        shp1[0] = rk

        curr_val = curr_val.reshape(sh)
        next_val = next_val.reshape(shp1)

        ttsum.node_tensor(i).update_val_size(curr_val)
        ttsum.node_tensor(i + 1).update_val_size(next_val)

    return ttsum


class TTRandRound:
    """
    Implementation of randomized rounding algorithms for Tensor Trains.

    Reference:
    [1] - Daas et. al, "Randomized algorithms for rounding in the Tensor-
    Train format." arxiv preprint arxiv:2110.04393 (2021). Available at:
    https://arxiv.org/abs/2110.04393.
    """

    def __init__(
        self, y: Union[TensorNetwork, List[TensorNetwork]], target_ranks: List
    ):
        self.y = y
        self.target_ranks = target_ranks

        if isinstance(y, List) and isinstance(y[0], TensorNetwork):
            self.ns = len(y)
            self.d = y[0].network.number_of_nodes()

        elif isinstance(y, TensorNetwork):
            self.ns = 1
            self.d = y.network.number_of_nodes()

        else:
            raise ValueError(
                f"Invalid type for y ({type(y)}). \
                             Argument y only accepts a list of \
                             TensorNetworks or a TensorNetwork"
            )

    def init_rand_mat(self, ranks: Optional[List] = None) -> List[np.ndarray]:
        """Generates a list of random TT-cores. Individual entries
        of the cores are Gaussian RVs and are normalized according
        to the size of the core"""
        if ranks is None:
            ranks = self.target_ranks

        sh = self.y[0].shape() if isinstance(self.y, list) else self.y.shape()
        r = []
        # Initialize random TT-tensor with specified variance
        for i in range(self.d):
            if i == 0:
                curr_shp = [sh[i], ranks[i]]
            elif i == self.d - 1:
                curr_shp = [ranks[i - 1], sh[i]]
            else:
                curr_shp = [ranks[i - 1], sh[i], ranks[i]]
            r.append(np.random.randn(*curr_shp) / np.sqrt(np.prod(curr_shp)))
        return r

    def partial_contraction(
        self, tt: TensorNetwork, y: List[np.ndarray], direction: str = "rl"
    ) -> List[np.ndarray]:
        """
        Partial contraction of TT cores. Returns a list of contracted cores
        w_i (by combining corresponding cores of, x[:i] and y[:i] for lr and
        x[i:] and y[i:] for rl contraction)
        """
        w = []
        if direction == "rl":
            for i in range(self.d - 1, 0, -1):
                x = tt.value(i)
                sx = x.shape
                sy = y[i].shape
                # tmp = np.einsum('ijk,ljm->ilkm', x[i], y[i])
                if i == self.d - 1:
                    w.append(x @ y[i].T)
                    continue
                tmp = (x.reshape((-1, sx[-1])) @ w[-1]).reshape((sx[0], -1))
                tmp = tmp @ y[i].reshape((sy[0], -1)).T
                w.append(tmp)

            w = w[::-1]
            return w

        raise ValueError("Invalid option")

    def rand_then_orth(self) -> TensorNetwork:
        """Implements Algorithm 3.2 in reference [1]"""

        if isinstance(self.y, TensorNetwork):
            r = self.init_rand_mat()
            w = self.partial_contraction(self.y, r, "rl")
            x_approx: np.ndarray = self.y.value(0)
            res = copy.deepcopy(self.y)

            for i in range(self.d - 1):
                sx = list(x_approx.shape)
                zn = x_approx.reshape((-1, x_approx.shape[-1]))
                yn = zn @ w[i]
                q, _ = np.linalg.qr(yn)
                x_approx = q.reshape(sx[:-1] + [q.shape[-1]])
                res.node_tensor(i).update_val_size(x_approx)
                sy = list(self.y.value(i + 1).shape)
                x_approx = (
                    q.T @ zn @ self.y.value(i + 1).reshape((sy[0], -1))
                ).reshape([q.shape[-1]] + sy[1:])

            res.node_tensor(self.d - 1).update_val_size(x_approx)
            return res

        raise ValueError(
            "It seems that this function is \
                        being used to round a TT-sum"
        )

    def rto_rounding_ttsum(self) -> TensorNetwork:
        """Implements Algorithm 3.4 in reference [1]"""

        if isinstance(self.y, List):
            r = self.init_rand_mat()
            tmp0 = []
            w = []
            res = copy.deepcopy(self.y[0])

            for y in self.y:
                tmp0.append(y.value(0))
                w.append(self.partial_contraction(y, r))
            x_approx = np.concatenate(tmp0, axis=1)

            del tmp0

            for i in range(self.d - 1):
                sx = list(x_approx.shape)
                rk = []
                rkp1 = []
                w_curr = []

                # Setup
                for j in range(self.ns):
                    sh = self.y[j].value(i).shape
                    rk.append(sh[-1])
                    rkp1.append(self.y[j].value(i + 1).shape[-1])
                    w_curr.append(w[j][i])

                # rksum = np.sum(rk)
                rkp1sum = np.sum(rkp1)
                rkcumsum = np.cumsum([0] + rk)

                # Start
                zn = x_approx.reshape((-1, sx[-1]))
                yn = zn @ np.concatenate(w_curr, axis=0)
                q, _ = np.linalg.qr(yn)
                self.target_ranks[i] = min(self.target_ranks[i], q.shape[-1])
                mn = q.T @ zn
                x_approx = q.reshape((sx[:-1] + [self.target_ranks[i]]))
                res.node_tensor(i).update_val_size(x_approx)
                xnp1 = []
                shp1 = []
                for j in range(self.ns):
                    shp1 = self.y[j].value(i + 1).shape
                    tmp = mn[:, rkcumsum[j] : rkcumsum[j + 1]] @ self.y[
                        j
                    ].value(i + 1).reshape((shp1[0], -1))
                    xnp1.append(tmp.reshape((-1, rkp1[j])))

                if i < self.d - 2:
                    x_approx = np.concatenate(xnp1, axis=1).reshape(
                        (self.target_ranks[i], shp1[1], rkp1sum)
                    )
                else:
                    x_approx = np.sum(xnp1, axis=0).reshape(
                        (self.target_ranks[i], shp1[1])
                    )
                    res.node_tensor(self.d - 1).update_val_size(x_approx)

            return res

        raise ValueError(
            "It seems that this function is being used \
                             to round a single TT"
        )

    def round(self) -> TensorNetwork:
        """Executes rounding"""
        if isinstance(self.y, List):
            res = self.rto_rounding_ttsum()
        else:
            res = self.rand_then_orth()
        return res


def tt_randomized_round(y: TensorNetwork, target_ranks: List) -> TensorNetwork:
    """Executes randomized rounding for a TT TensorNetwork"""

    rand_setup = TTRandRound(y, target_ranks)
    return rand_setup.rand_then_orth()


def tt_sum_randomized_round(
    y: List[TensorNetwork], target_ranks: List
) -> TensorNetwork:
    """Executes randomized rounding for a TT TensorNetwork"""

    rand_setup = TTRandRound(y, target_ranks)
    return rand_setup.rto_rounding_ttsum()


def tt_rand_precond_svd_round(
    tn: Union[TensorNetwork, List[TensorNetwork]],
    eps: float,
    rank_bound: list[int],
) -> TensorNetwork:
    """
    Uses randomized rounding as a preconditioner to lower the ranks of
    the Tensor Train to the specified target rank before truncating the
    ranks further to a specified tolerance (eps) using SVD.

    Issues right now:
        - Total error accumulated post rounding is unknown due to initi-
        al rank-based truncation.
        - Need to adjust the eps in SVD-based truncation so that total
        error stays consistent with the global prespecified tolerance.
    """

    rand_rounded_tn = TTRandRound(y=tn, target_ranks=rank_bound)
    res = rand_rounded_tn.round()
    dim = rand_rounded_tn.d

    for i in range(dim - 1, 0, -1):
        tens_curr = res.value(i)
        sh = list(tens_curr.shape)
        tens_next = res.value(i - 1)

        delta = eps / (dim - 1) ** 0.5

        trunc_svd = delta_svd(tens_curr.reshape((sh[0], -1)), delta, True)
        assert trunc_svd.v is not None

        tens_curr = trunc_svd.v.reshape([-1] + sh[1:])
        if i == 1:
            tens_next = np.einsum(
                "jk,kl->jl",
                tens_next,
                trunc_svd.u * trunc_svd.s[np.newaxis, :],
            )
        else:
            tens_next = np.einsum(
                "ijk,kl->ijl",
                tens_next,
                trunc_svd.u * trunc_svd.s[np.newaxis, :],
            )

        res.node_tensor(i).update_val_size(tens_curr)
        res.node_tensor(i - 1).update_val_size(tens_next)

    return res


def ttop_rank1(
    indices_in: List[Index],
    indices_out: List[Index],
    cores: List[np.ndarray],
    rank_name_prefix: str,
) -> TensorNetwork:
    """Rank 1 TT-op with op in the first dimension."""
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    tt_op = TensorNetwork()

    rank_indices = [Index(f"{rank_name_prefix}_r1", 1)]
    a1_tens = Tensor(
        cores[0][:, :, np.newaxis],
        [indices_out[0], indices_in[0], rank_indices[0]],
    )
    tt_op.add_node(0, a1_tens)
    for ii in range(1, dim):
        rank_indices.append(Index(f"{rank_name_prefix}_r{ii + 1}", 1))
        if ii < dim - 1:
            eye = cores[ii][np.newaxis, :, :, np.newaxis]
            eye_tens = Tensor(
                eye,
                [
                    rank_indices[ii - 1],
                    indices_out[ii],
                    indices_in[ii],
                    rank_indices[ii],
                ],
            )
            tt_op.add_node(ii, eye_tens)
        else:
            eye = cores[ii][np.newaxis, :, :]
            eye_tens = Tensor(
                eye, [rank_indices[ii - 1], indices_out[ii], indices_in[ii]]
            )
            tt_op.add_node(ii, eye_tens)
        if ii == 1:
            tt_op.add_edge(ii - 1, ii)
        else:
            tt_op.add_edge(ii - 1, ii)

    return tt_op


def ttop_rank2(
    indices_in: List[Index],
    indices_out: List[Index],
    cores_r1: List[np.ndarray],
    cores_r2: List[np.ndarray],
    rank_name_prefix: str,
) -> TensorNetwork:
    """Rank 2 Sum of two ttops"""
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    tt_op = TensorNetwork()

    rank_indices = [Index(f"{rank_name_prefix}_r1", 2)]

    core = np.zeros((indices_out[0].size, indices_in[0].size, 2))
    core[:, :, 0] = cores_r1[0]
    core[:, :, 1] = cores_r2[0]

    a1_tens = Tensor(core, [indices_out[0], indices_in[0], rank_indices[0]])

    tt_op.add_node(0, a1_tens)
    for ii in range(1, dim):
        rank_indices.append(Index(f"{rank_name_prefix}_r{ii + 1}", 2))
        if ii < dim - 1:
            core = np.zeros((2, indices_out[ii].size, indices_in[ii].size, 2))
            core[0, :, :, 0] = cores_r1[ii]
            core[1, :, :, 1] = cores_r2[ii]

            ai_tens = Tensor(
                core,
                [
                    rank_indices[ii - 1],
                    indices_out[ii],
                    indices_in[ii],
                    rank_indices[ii],
                ],
            )
            tt_op.add_node(ii, ai_tens)
        else:
            core = np.zeros((2, indices_out[ii].size, indices_in[ii].size))
            core[0, :, :] = cores_r1[ii]
            core[1, :, :] = cores_r2[ii]
            ai_tens = Tensor(
                core, [rank_indices[ii - 1], indices_out[ii], indices_in[ii]]
            )
            tt_op.add_node(ii, ai_tens)
        tt_op.add_edge(ii - 1, ii)

    return tt_op


def ttop_sum(
    indices_in: List[Index],
    indices_out: List[Index],
    cores: List[List[np.ndarray]],
    rank_name_prefix: str,
) -> TensorNetwork:
    """Sum of ttops"""
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    tt_op = TensorNetwork()

    num_sum = len(cores)
    rank_indices = [Index(f"{rank_name_prefix}_r1", num_sum)]

    core = np.zeros((indices_out[0].size, indices_in[0].size, num_sum))
    for ii in range(num_sum):
        core[:, :, ii] = cores[ii][0]

    a1_tens = Tensor(core, [indices_out[0], indices_in[0], rank_indices[0]])

    tt_op.add_node(0, a1_tens)
    for ii in range(1, dim):
        rank_indices.append(Index(f"{rank_name_prefix}_r{ii + 1}", num_sum))
        if ii < dim - 1:
            core = np.zeros(
                (num_sum, indices_out[ii].size, indices_in[ii].size, num_sum)
            )
            for jj in range(num_sum):
                core[jj, :, :, jj] = cores[jj][ii]

            ai_tens = Tensor(
                core,
                [
                    rank_indices[ii - 1],
                    indices_out[ii],
                    indices_in[ii],
                    rank_indices[ii],
                ],
            )
            tt_op.add_node(ii, ai_tens)
        else:
            core = np.zeros(
                (num_sum, indices_out[ii].size, indices_in[ii].size)
            )
            for jj in range(num_sum):
                core[jj, :, :] = cores[jj][ii]

            ai_tens = Tensor(
                core, [rank_indices[ii - 1], indices_out[ii], indices_in[ii]]
            )
            tt_op.add_node(ii, ai_tens)
        tt_op.add_edge(ii - 1, ii)

    return tt_op


def tt_sum(
    tt_in: List[TensorNetwork],
) -> TensorNetwork:
    """Sum a set of tensor trains."""

    tt_out = TensorNetwork()
    dim = tt_in[0].dim()
    for ii, node in enumerate(tt_in[0].network.nodes):
        inds = tt_in[0].node_tensor(node).indices
        core_values = [tt.value(node) for tt in tt_in]

        if ii == 0:
            new_value = np.hstack(core_values)
            index_left = Index(inds[0].name, inds[0].size)
            index_right = Index("rank_0", new_value.shape[1])
            new_inds = [index_left, index_right]

        elif ii == dim - 1:
            new_value = np.vstack(core_values)
            index_left = Index(f"rank_{ii - 1}", new_value.shape[0])
            index_right = Index(inds[1].name, inds[1].size)
            new_inds = [index_left, index_right]

        else:
            rank_left = np.sum([v.shape[0] for v in core_values])
            rank_right = np.sum([v.shape[2] for v in core_values])
            new_shape = (rank_left, core_values[0].shape[1], rank_right)
            new_value = np.zeros(new_shape)
            on_rank_left = 0
            on_rank_right = 0
            for core_value in core_values:
                increment_left = core_value.shape[0]
                increment_right = core_value.shape[2]
                new_value[
                    on_rank_left : on_rank_left + increment_left,
                    :,
                    on_rank_right : on_rank_right + increment_right,
                ] = core_value
                on_rank_left += increment_left
                on_rank_right += increment_right

            index_left = Index(f"rank_{ii - 1}", new_value.shape[0])
            index_middle = Index(inds[1].name, inds[1].size)
            index_right = Index(f"rank_{ii}", new_value.shape[2])
            new_inds = [index_left, index_middle, index_right]

        tt_out.add_node(ii, Tensor(new_value, new_inds))
        if ii > 0:
            tt_out.add_edge(ii - 1, ii)

    return tt_out


def ttop_sum_apply(
    tt_in: TensorNetwork,
    indices_in: List[Index],
    indices_out: List[Index],
    cores: List[List[Callable[[np.ndarray], np.ndarray]]],
    rank_name_prefix: str,
) -> TensorNetwork:
    """Apply sum of rank1 tt ops to a tt."""

    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    tt_out = TensorNetwork()
    num_sum = len(cores)

    node_list = list(tt_in.network.nodes())
    ii = 0
    v = tt_in.value(node_list[ii])
    rank_indices = [Index(f"{rank_name_prefix}_r1", num_sum * v.shape[1])]
    core = np.zeros((indices_out[ii].size, v.shape[1] * num_sum))
    indices = [indices_out[ii], rank_indices[ii]]
    on_ind = 0
    for jj in range(num_sum):
        new_core = cores[jj][ii](v)
        new_core = np.reshape(new_core, (core.shape[0], -1))
        core[:, on_ind : on_ind + new_core.shape[1]] = new_core
        on_ind += new_core.shape[1]
    tt_out.add_node(ii, Tensor(core, indices))

    for ii, node_tt in enumerate(node_list[1:], start=1):
        v = tt_in.value(node_tt)

        if ii < dim - 1:
            rank_indices.append(
                Index(f"{rank_name_prefix}_r{ii + 1}", v.shape[2] * num_sum)
            )

            core = np.zeros(
                (
                    num_sum * v.shape[0],
                    indices_out[ii].size,
                    num_sum * v.shape[2],
                )
            )

            indices = [rank_indices[ii - 1], indices_out[ii], rank_indices[ii]]
            on_ind1 = 0
            on_ind2 = 0
            for jj in range(num_sum):
                # new_core = np.einsum('jk,mkp->mjp', cores[jj][ii], v)
                new_core = cores[jj][ii](v)
                shape = new_core.shape
                new_core = np.reshape(new_core, (shape[0], shape[1], shape[2]))
                n1 = new_core.shape[0]
                n2 = new_core.shape[2]
                core[on_ind1 : on_ind1 + n1, :, on_ind2 : on_ind2 + n2] = (
                    new_core
                )
                on_ind1 += n1
                on_ind2 += n2
        else:
            core = np.zeros((num_sum * v.shape[0], indices_out[ii].size))
            indices = [rank_indices[ii - 1], indices_out[ii]]
            on_ind = 0
            for jj in range(num_sum):
                new_core = cores[jj][ii](v)
                core[on_ind : on_ind + new_core.shape[0], :] = new_core
                on_ind += new_core.shape[0]

        tt_out.add_node(ii, Tensor(core, indices))
        tt_out.add_edge(ii - 1, ii)

    return tt_out


def ttop_apply(ttop: TensorNetwork, tt_in: TensorNetwork) -> TensorNetwork:
    """Apply a ttop to a tt tensor.

    # tt overwritten, same free_indices as before
    """
    tt = copy.deepcopy(tt_in)
    dim = tt.dim()
    for ii, (node_op, node_tt) in enumerate(
        zip(ttop.network.nodes(), tt.network.nodes())
    ):
        op = ttop.value(node_op)
        v = tt.value(node_tt)
        # print(f"op shape: {node_op}", op.shape)
        # print(f"v shape: {node_tt}", v.shape)
        if ii == 0:
            new_core = np.einsum("ijk,jl->ilk", op, v)
            n = v.shape[0]
            new_core = np.reshape(new_core, (n, -1))
        elif ii < dim - 1:
            new_core = np.einsum("ijkl,mkp->mijpl", op, v)
            shape = new_core.shape
            new_core = np.reshape(
                new_core, (shape[0] * shape[1], shape[2], shape[3] * shape[4])
            )
        else:
            new_core = np.einsum("ijk,mk->mij", op, v)
            shape = new_core.shape
            new_core = np.reshape(new_core, (shape[0] * shape[1], -1))

        tt.set_node_tensor(
            node_tt, tt.node_tensor(node_tt).update_val_size(new_core)
        )

    # print("After op = ")
    # print(tt)
    return tt


@typing.no_type_check
def gmres(  # pylint: disable=R0913,R0917
    op,  # function from in to out
    rhs: TensorNetwork,
    x0: TensorNetwork,
    eps: float = 1e-5,
    round_eps: float = 1e-10,
    maxiter: int = 100,
) -> Tuple[TensorNetwork, float]:
    """Perform GMRES.
    VERY HACKY
    """
    # pylint: disable=C0103
    # above disables the snake case complaints for variables like R
    r0 = rhs + op(x0).scale(-1.0)
    r0 = tt_svd_round(r0, round_eps)
    beta = r0.norm()

    r0.scale(1.0 / beta)
    # print("r0 norm = ", r0.norm())

    v = [r0]

    # print("beta = ", beta)
    # print("v0 norm = ", v[0].norm())

    y = []
    H = None
    for jj in range(maxiter):
        # print(f"jj = {jj}")
        delta = round_eps

        w = op(v[-1])
        w = tt_svd_round(w, delta)

        if H is None:
            H = np.zeros((jj + 2, jj + 1))
        else:
            m, n = H.shape
            newH = np.zeros((m + 1, n + 1))
            newH[:m, :n] = H
            H = newH
        # print(f"H shape = {H.shape}")
        # print("inner w = ", w.inner(v[0]))
        # # print("w = ", w)
        # warr = w.contract().value.flatten()
        # varr = v[0].contract().value.flatten()
        # warr_next = warr - np.dot(warr, varr) * varr
        # print("check inner =", np.dot(warr, warr), w.inner(w))
        # print("H should be = ", np.dot(warr, varr), w.inner(v[0]))
        # # exit(1)
        # print("in arrays = ", np.dot(varr, varr), np.dot(warr_next, varr))
        for ii in range(jj + 1):
            # print("ii = ", ii)
            H[ii, jj] = w.inner(v[ii])
            vv = copy.deepcopy(v[ii])
            vv.scale(-H[ii, jj])
            w = w + vv
        # print("inner w = ", w.inner(v[0]), w.inner(w))
        # print("H = ", H)
        # exit(1)
        w = tt_svd_round(w, round_eps)
        H[jj + 1, jj] = w.norm()
        v.append(w.scale(1.0 / H[jj + 1, jj]))
        # for ii in range(jj+2):
        #     print(f"inner {-1, ii} = ", v[-1].inner(v[ii]))

        # exit(1)
        # + 1e-14

        # print(H)

        e = np.zeros((H.shape[0]))
        e[0] = beta
        yy, resid, _, _ = np.linalg.lstsq(H, e)
        y.append(yy)
        # print(f"Iteration {jj}: resid = {resid}")
        if np.abs(resid) < eps:
            break

        # if resid < eps:
        #     break
    # exit(1)
    x = copy.deepcopy(x0)
    # print("len y = ", len(y[-1]))
    # print("len v = ", len(v))
    for ii, (vv, yy) in enumerate(zip(v, y[-1])):
        x = x + vv.scale(yy)
    x = tt_svd_round(x, round_eps)
    r0 = rhs + op(x).scale(-1.0)
    resid = r0.norm()
    # print("resid = ", resid)
    # exit(1);
    return x, resid


def tt_to_ht(net: TensorNetwork) -> TensorNetwork:
    """Convert a tensor train to a hierarchical tucker"""
    ht = TensorNetwork()
    ht.network = copy.deepcopy(net.network)

    if len(net.network.nodes) == 1:
        return ht

    ends = net.end_nodes()
    assert len(ends) == 2
    path = nx.shortest_path(net.network, ends[0], ends[1])

    def _merge_pair(
        n1: NodeName, n2: NodeName, connecting_inds: List[Index]
    ) -> NodeName:
        contract_edges = ht.get_contraction_index(n1, n2)
        rs = []
        for n in (n1, n2):
            indices = ht.node_tensor(n).indices
            split_pos = [
                indices.index(ind)
                for ind in indices
                if ind not in connecting_inds and ind not in contract_edges
            ]
            _, r = ht.qr(n, split_pos)
            rs.append(r)
        return ht.merge(rs[0], rs[1])

    def _to_ht(
        nodes: Sequence[NodeName], connecting_inds: List[Index]
    ) -> NodeName:
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:
            return _merge_pair(nodes[0], nodes[1], connecting_inds)

        mid = len(nodes) // 2
        contract_ind = ht.get_contraction_index(nodes[mid - 1], nodes[mid])[0]
        connecting_inds.append(contract_ind)
        r1 = _to_ht(nodes[:mid], connecting_inds)
        r2 = _to_ht(nodes[mid:], connecting_inds)
        return _merge_pair(r1, r2, connecting_inds)

    _to_ht(path, [])
    return ht


# -------------------------
# Hierarchical Tucker (HT)
# -------------------------


def rand_ht(
    indices: List[Index], rank: int, child_each_level: int = 2
) -> TensorNetwork:
    """Return a random hierarchical tucker."""
    ht = TensorNetwork()

    def build_child(
        pid: int, node_id: int, sub_indices: List[Index], rank: int = 1
    ) -> int:
        if len(sub_indices) == 1:
            ind = sub_indices[0]
            val = np.random.random((rank, ind.size))
            node = Tensor(
                val,
                [
                    Index(f"R_{pid}_{node_id}", rank),
                    ind,
                ],
            )
            ht.add_node(f"G{node_id}", node)
            return node_id + 1

        # partition the indices into groups hierarchically,
        # the leftovers are always in the last group
        ind_group_num = child_each_level
        ind_group_size = len(sub_indices) // ind_group_num
        last_group_size = (
            len(sub_indices) - (ind_group_num - 1) * ind_group_size
        )
        next_node_id = node_id + 1

        if pid == -1:
            val = np.random.random([rank] * child_each_level)
            indices = []
        else:
            val = np.random.random([rank] * (child_each_level + 1))
            indices = [Index(f"R_{pid}_{node_id}", rank)]

        for i in range(ind_group_num - 1):
            child_id = next_node_id
            indices.append(Index(f"R_{node_id}_{child_id}", rank))
            next_node_id = build_child(
                node_id,
                next_node_id,
                sub_indices[i * ind_group_size : (i + 1) * ind_group_size],
                rank,
            )
            ht.add_edge(f"G{child_id}", f"G{node_id}")

        child_id = next_node_id
        indices.append(Index(f"R_{node_id}_{child_id}", rank))
        next_node_id = build_child(
            node_id, next_node_id, sub_indices[-last_group_size:], rank
        )
        ht.add_edge(f"G{child_id}", f"G{node_id}")

        ht.set_node_tensor(f"G{node_id}", Tensor(val, indices))

        return next_node_id

    build_child(-1, 0, indices, rank)
    return ht


def ht_root(ht: TensorNetwork) -> NodeName:
    """Find the root of a hierarchical tucker: the node with exactly two
    indices, neither of them free."""
    free_inds = ht.free_indices()

    n: NodeName
    for n in ht.network.nodes:
        node_inds = ht.node_tensor(n).indices
        if len(node_inds) != 2:
            continue

        if any(ind in free_inds for ind in node_inds):
            continue

        return n

    raise ValueError("Invalid hierarchical tucker, cannot find the root.")


def _build_ht_layers(
    free_inds: Sequence[Index],
) -> List[List[Sequence[Index]]]:
    """Build the binary dimension-tree layers from root to leaves."""
    layer: List[Sequence[Index]] = [free_inds]
    layers = [layer]
    while not all(len(ind_group) <= 1 for ind_group in layers[-1]):
        next_layer: List[Sequence[Index]] = []
        for ind_group in layers[-1]:
            if len(ind_group) > 1:
                mid = len(ind_group) // 2
                next_layer.append(ind_group[:mid])
                next_layer.append(ind_group[mid:])
            else:
                next_layer.append([])
        layers.append(next_layer)
    return layers


def ht_svd(
    data: np.ndarray, free_inds: Sequence[Index], eps: float
) -> TensorNetwork:
    """Create a hierarchical tucker for the given data using SVD."""
    ht = TensorNetwork()
    delta = np.linalg.norm(data) * eps / np.sqrt(2 * len(data.shape) - 3)

    ind_sizes = [ind.size for ind in free_inds]
    assert data.shape == tuple(ind_sizes)

    layers = _build_ht_layers(free_inds)

    # from leaves to root
    leaf_inds = list(free_inds)[:]
    for level, layer in enumerate(layers[1:][::-1]):
        i = 0
        ind_cnt = 0
        leaf_inds_copy = leaf_inds[:]
        for i, inds in enumerate(layer):
            if len(inds) == 0:
                ind_cnt += 1
                continue

            if len(inds) == 1:
                curr_data = np.moveaxis(data, ind_cnt, 0)
                res = delta_svd(
                    curr_data.reshape(inds[0].size, -1), delta=delta
                )
                assert res.u is not None
                leaf_ind = Index(f"s_{level}_{i}", res.u.shape[1])
                ht.add_node(
                    f"n_{level}_{i}",
                    Tensor(res.u, [inds[0], leaf_ind]),
                )
                data = np.einsum("a...,ab->...b", curr_data, res.u)
                data = np.moveaxis(data, -1, i)

                ind_pos = leaf_inds.index(inds[0])
                leaf_inds.pop(ind_pos)
                leaf_inds.insert(ind_pos, leaf_ind)
                ind_cnt += 1
            else:
                # build transition nodes
                # take two leaf inds and make the reshape
                left_size = (
                    leaf_inds[ind_cnt].size * leaf_inds[ind_cnt + 1].size
                )
                curr_data = np.moveaxis(data, [ind_cnt, ind_cnt + 1], [0, 1])
                res = delta_svd(curr_data.reshape(left_size, -1), delta=delta)
                assert res.u is not None
                leaf_ind = Index(f"s_{level}_{i}", res.u.shape[1])
                ht.add_node(
                    f"n_{level}_{i}",
                    Tensor(
                        res.u.reshape(
                            leaf_inds[ind_cnt].size,
                            leaf_inds[ind_cnt + 1].size,
                            res.u.shape[1],
                        ),
                        [
                            leaf_inds[ind_cnt],
                            leaf_inds[ind_cnt + 1],
                            leaf_ind,
                        ],
                    ),
                )
                data = np.einsum(
                    "a...,ab->...b",
                    curr_data.reshape(left_size, *curr_data.shape[2:]),
                    res.u,
                )
                data = np.moveaxis(data, -1, i)

                idx0 = leaf_inds_copy.index(leaf_inds[ind_cnt])
                idx1 = leaf_inds_copy.index(leaf_inds[ind_cnt + 1])
                ht.add_edge(f"n_{level}_{i}", f"n_{level - 1}_{idx0}")
                ht.add_edge(f"n_{level}_{i}", f"n_{level - 1}_{idx1}")

                leaf_inds.pop(ind_cnt)
                leaf_inds.pop(ind_cnt)
                leaf_inds.insert(ind_cnt, leaf_ind)

                ind_cnt += 1

    ht.add_node("root", Tensor(data, [leaf_inds[0], leaf_inds[1]]))

    if len(layers) > 1:
        ht.add_edge("root", f"n_{len(layers) - 2}_0")
        ht.add_edge("root", f"n_{len(layers) - 2}_1")

    return ht


# --------------------------
# Random Trees
# --------------------------


def rand_tree(indices: List[Index], ranks: List[int]) -> TensorNetwork:
    """Return a random tensor tree."""

    ndims = len(indices)
    num_of_nodes = len(ranks) + 1
    assert ndims <= num_of_nodes  # In a tree, #edges = #nodes - 1

    # sample a topology from given ranks
    np.random.shuffle(ranks)
    # sample nodes for free indices
    nodes_with_free = np.random.choice(
        num_of_nodes, len(indices), replace=False
    )
    # assign edges between nodes
    parent: Dict[int, Tuple[NodeName, int]] = {}
    nodes = list(range(num_of_nodes))
    while len(nodes) > 1:
        node = np.random.choice(nodes, 1)[0]
        nodes.remove(node)

        p = np.random.choice(num_of_nodes, 1)[0]
        while p == node:
            p = np.random.choice(num_of_nodes, 1)[0]
        # print("suggesting parent of", node, "as", p)
        # check for cycles
        ancestor = p
        while ancestor in parent:
            # print("ancestor of", ancestor)
            ancestor, _ = parent[ancestor]
            if ancestor == node:
                p = np.random.choice(num_of_nodes, 1)[0]
                while p == node:
                    p = np.random.choice(num_of_nodes, 1)[0]
                ancestor = p

        # print("finalizing parent of", node, "as", p)
        parent[node] = (p, len(nodes) - 1)

    tree = TensorNetwork()

    for i in range(num_of_nodes):
        i_ranks = []
        i_dims = []
        if i in nodes_with_free:
            idx = list(nodes_with_free).index(i)
            dim = indices[idx].size
            i_ranks.append(indices[idx])
            i_dims.append(dim)

        if i in parent:
            _, ridx = parent[i]
            dim = ranks[ridx]
            i_ranks.append(Index(f"r_{ridx}", dim))
            i_dims.append(dim)

        for p, ridx in parent.values():
            if p == i:
                dim = ranks[ridx]
                i_ranks.append(Index(f"r_{ridx}", dim))
                i_dims.append(dim)

        value = np.random.randn(*i_dims)
        tensor = Tensor(value, i_ranks)
        tree.add_node(i, tensor)

    for i, (p, _) in parent.items():
        # print("edge between", i, "and", p)
        tree.add_edge(i, p)

    return tree


def rand_tucker(indices: List[Index], rank: int = 1) -> "TensorNetwork":
    """Return a random tucker with the given indices."""

    tucker = TensorNetwork()
    root_val = np.random.random([rank] * len(indices))
    root_inds = [Index(f"s_{i}", rank) for i in range(len(indices))]
    tucker.add_node("root", Tensor(root_val, root_inds))
    for i, ind in enumerate(indices):
        tensor_val = np.random.random((ind.size, rank))
        tensor_inds = [ind, root_inds[i]]
        tucker.add_node(f"G{i}", Tensor(tensor_val, tensor_inds))
        tucker.add_edge(f"G{i}", "root")

    return tucker


# ==========================================================================
# Structure-agnostic operations (used by the transport solver)
#
# The functions below work on arbitrary tree networks, including networks
# whose free indices were reshaped by the structure search (IndexSplit /
# IndexMerge).  `IndexLayout` records how the "original" indices of a
# problem map onto the current free indices; the operations use it to
# express dense per-index data (factors of a separable term, integration
# weights) on the reshaped indices.
# ==========================================================================


@dataclass
class IndexLayout:
    """Mapping between the original indices of a tensor and the free
    indices of a network whose indices were reshaped by IndexSplit and
    IndexMerge operations.

    Every original index and every current free index is a C-ordered
    product of *atoms* (the finest sub-indices appearing in the history).

    Attributes:
        originals: The original indices.
        atoms: The finest sub-indices.
        orig_to_atoms: Positions in `atoms` of each original index.
        free_to_atoms: Positions in `atoms` of each current free index.
    """

    originals: List[Index]
    atoms: List[Index]
    orig_to_atoms: Dict[IndexName, List[int]]
    free_to_atoms: Dict[IndexName, List[int]]

    @staticmethod
    def identity(originals: Sequence[Index]) -> "IndexLayout":
        """Layout where the current free indices are the originals."""
        atoms = list(originals)
        mapping = {ind.name: [i] for i, ind in enumerate(atoms)}
        return IndexLayout(list(originals), atoms, mapping, dict(mapping))

    @staticmethod
    def from_history(
        originals: Sequence[Index], history: Sequence[IndexOp]
    ) -> "IndexLayout":
        """Replay a reshape history (e.g., HSearchState.reshape_history)."""
        layout = IndexLayout.identity(originals)
        for op in history:
            if isinstance(op, IndexSplit):
                layout.apply_split(op)
            elif isinstance(op, IndexMerge):
                layout.apply_merge(op)
            # permutations and swaps do not change the index identities

        # name the atoms after the free index they belong to
        for name, atom_ids in layout.free_to_atoms.items():
            for k, aid in enumerate(atom_ids):
                atom_name = name if len(atom_ids) == 1 else f"{name}__{k}"
                layout.atoms[aid] = Index(atom_name, layout.atoms[aid].size)

        return layout

    def apply_split(self, op: IndexSplit) -> None:
        """Replay one (executed) IndexSplit operation."""
        assert op.result is not None, "split op must be executed first"
        name = op.index.name
        if name not in self.free_to_atoms:
            raise ValueError(f"{name} is not a free index of the layout")

        old_ids = self.free_to_atoms.pop(name)
        # prefix products of the requested split shape define the cuts
        req_cuts = set(itertools.accumulate(op.shape, operator.mul))
        new_ids: List[int] = []
        offset = 1
        replaced: Dict[int, List[int]] = {}
        for aid in old_ids:
            size = self.atoms[aid].size
            inner = sorted(c for c in req_cuts if offset < c < offset * size)
            bounds = [offset] + inner + [offset * size]
            parts = []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                if hi % lo != 0:
                    raise ValueError(
                        f"split {op.shape} of {name} is not aligned with"
                        f" the existing sub-indices"
                    )
                parts.append(hi // lo)

            offset *= size
            if len(parts) == 1:
                replaced[aid] = [aid]
                new_ids.append(aid)
                continue

            ids = []
            for part in parts:
                ids.append(len(self.atoms))
                self.atoms.append(Index(f"{name}_{len(self.atoms)}", part))
            replaced[aid] = ids
            new_ids.extend(ids)

        for mapping in (self.orig_to_atoms, self.free_to_atoms):
            for key, ids in mapping.items():
                mapping[key] = list(
                    itertools.chain.from_iterable(
                        replaced.get(i, [i]) for i in ids
                    )
                )

        # assign the new atoms to the resulting indices
        pos = 0
        for res in op.result:
            size, ids = 1, []
            while size < res.size:
                ids.append(new_ids[pos])
                size *= self.atoms[new_ids[pos]].size
                pos += 1
            if size != res.size:
                raise ValueError(f"cannot form {res} from atoms of {name}")
            self.free_to_atoms[res.name] = ids

    def apply_merge(self, op: IndexMerge) -> None:
        """Replay one (executed) IndexMerge operation."""
        assert op.result is not None, "merge op must be executed first"
        ids: List[int] = []
        for ind in op.indices:
            if ind.name not in self.free_to_atoms:
                raise ValueError(f"{ind} is not a free index of the layout")
            ids.extend(self.free_to_atoms.pop(ind.name))
        self.free_to_atoms[op.result.name] = ids

    def is_identity(self) -> bool:
        """True if the free indices are exactly the originals."""
        return all(
            self.free_to_atoms.get(ind.name) == self.orig_to_atoms[ind.name]
            for ind in self.originals
        ) and len(self.free_to_atoms) == len(self.originals)

    def free_indices(self) -> List[Index]:
        """Current free indices."""
        return [
            Index(name, math.prod(self.atoms[i].size for i in ids))
            for name, ids in self.free_to_atoms.items()
        ]

    def atom_indices(self, name: IndexName) -> List[Index]:
        """Atoms of a current free index (or of an original index)."""
        ids = self.free_to_atoms.get(name)
        if ids is None:
            ids = self.orig_to_atoms[name]
        return [self.atoms[i] for i in ids]

    def validate(self, tn: TensorNetwork) -> None:
        """Check that the network has the free indices of the layout."""
        expected = set(self.free_indices())
        actual = set(tn.free_indices())
        if expected != actual:
            raise ValueError(
                f"free indices {actual} do not match layout {expected}"
            )

    def reshape_vector(self, name: IndexName, vals: np.ndarray) -> Tensor:
        """Express dense data over an original index on its atoms."""
        atoms = self.atom_indices(name)
        vals = np.asarray(vals)
        if vals.size != math.prod(a.size for a in atoms):
            raise ValueError(
                f"{name}: expected size {atoms}, got {vals.shape}"
            )
        return Tensor(vals.reshape([a.size for a in atoms]), atoms)

    def atomize(self, tn: TensorNetwork) -> TensorNetwork:
        """Copy of the network with every free index split into atoms."""
        out = copy.deepcopy(tn)
        for ind in self.free_indices():
            atoms = self.atom_indices(ind.name)
            if len(atoms) > 1:
                shape = [a.size for a in atoms]
                out.split_index(
                    IndexSplit(index=ind, shape=shape, result=atoms)
                )
        return out

    def atomize_tensor(self, tensor: Tensor) -> Tensor:
        """Split the (free) indices of a tensor into atoms."""
        for ind in self.free_indices():
            atoms = self.atom_indices(ind.name)
            if len(atoms) > 1 and ind in tensor.indices:
                shape = [a.size for a in atoms]
                op = IndexSplit(index=ind, shape=shape, result=atoms)
                tensor = tensor.split_indices(op)
                tensor = tensor.rename_indices(
                    {f"_fresh_index_{k}": a.name for k, a in enumerate(atoms)}
                )
        return tensor

    def deatomize_tensor(self, tensor: Tensor) -> Tensor:
        """Merge the atoms in a tensor back into the current free indices."""
        for ind in self.free_indices():
            atoms = self.atom_indices(ind.name)
            if len(atoms) > 1 and all(a in tensor.indices for a in atoms):
                tensor = tensor.merge_indices(atoms, ind.name)
        return tensor

    def original_order(
        self, tensor: Tensor, order: Optional[Sequence[IndexName]] = None
    ) -> np.ndarray:
        """Dense values of a (contracted) tensor in the original indices.

        Args:
            tensor: A tensor over (some of) the current free indices.
            order: Names of the original indices in the wanted order.
                Defaults to the originals present in the tensor.
        """
        tensor = self.atomize_tensor(tensor)
        if order is None:
            order = [
                ind.name
                for ind in self.originals
                if all(
                    a in tensor.indices for a in self.atom_indices(ind.name)
                )
            ]
        names = [a.name for o in order for a in self.atom_indices(o)]
        tensor = tensor.permute_by_name(names)
        sizes = [
            math.prod(a.size for a in self.atom_indices(o)) for o in order
        ]
        return tensor.value.reshape(sizes)


def _steiner_tree(graph: nx.Graph, nodes: Sequence[NodeName]) -> nx.Graph:
    """Minimal subtree of a tree connecting the given nodes."""
    keep = set(nodes)
    for n1, n2 in itertools.combinations(nodes, 2):
        keep.update(nx.shortest_path(graph, n1, n2))
    return graph.subgraph(keep)


def _expand_to_layout(
    tensor: Optional[Tensor],
    layout: Sequence[Index],
    free_inds: Sequence[Index],
) -> Tensor:
    """Broadcast a (partial) core to the full index layout of a node.

    Missing free indices become ones (outer product), missing bond indices
    become size-1 bonds.  The result has the indices of `layout` in order.
    """
    if tensor is None:
        tensor = Tensor(np.ones(()), [])

    value = tensor.value
    indices = list(tensor.indices)
    present = {ind.name for ind in indices}
    for ind in layout:
        if ind.name in present:
            continue
        size = ind.size if ind in free_inds else 1
        value = np.broadcast_to(value[..., np.newaxis], value.shape + (size,))
        indices.append(Index(ind.name, size))

    tensor = Tensor(np.ascontiguousarray(value), indices)
    return tensor.permute_by_name([ind.name for ind in layout])


def separable_like(
    ref: TensorNetwork,
    factors: Dict[IndexName, np.ndarray],
    layout: Optional[IndexLayout] = None,
    tol: float = 0.0,
) -> TensorNetwork:
    """Represent the separable tensor `prod_i factors[i]` on the structure
    of a reference network.

    The result has the same topology, node names, bond names and per-node
    index order as `ref`, so `ref + result` and `ref * result` are valid.
    Bonds inside the subtree spanned by the atoms of one original index get
    the ranks of the (truncated) SVDs of that factor; all other bonds have
    size one.

    Args:
        ref: The reference network.
        factors: Dense values over each original index, keyed by name.
        layout: Mapping from the original indices to the free indices of
            `ref`.  Defaults to the identity on `ref.free_indices()`.
        tol: Absolute SVD truncation tolerance for a factor spread over
            several nodes.  Zero (default) keeps the factor exact.
    """
    if layout is None:
        layout = IndexLayout.identity(ref.free_indices())
    layout.validate(ref)
    missing = {ind.name for ind in layout.originals} - set(factors)
    if missing:
        raise KeyError(f"missing factors for {missing}")

    atomized = layout.atomize(ref)
    graph = atomized.network
    free_inds = atomized.free_indices()
    node_layout = {n: list(atomized.node_tensor(n).indices) for n in graph}
    leaf_of = {
        ind.name: atomized.node_by_free_index(ind.name) for ind in free_inds
    }

    def decompose(
        subtree: nx.Graph,
        node: NodeName,
        parent: Optional[NodeName],
        cur: Tensor,
    ) -> Dict[NodeName, Tensor]:
        """Recursively SVD `cur` along the edges of the subtree."""
        cores = {}
        for child in subtree.neighbors(node):
            if child == parent:
                continue
            # atoms of the factor that live in the child's branch
            branch = nx.descendants(nx.bfs_tree(subtree, node), child)
            branch.add(child)
            below = [
                i
                for i, ind in enumerate(cur.indices)
                if ind in free_inds and leaf_of[ind.name] in branch
            ]
            [u, s, v], _ = cur.svd(below, atol=tol)
            bond = atomized.get_contraction_index(node, child)[0]
            u = u.rename_indices({"r_split_l": bond.name})
            cur = s.contract(v).rename_indices({"r_split_l": bond.name})
            cores.update(decompose(subtree, child, node, u))
        cores[node] = cur
        return cores

    result = copy.deepcopy(ref)
    combined: Dict[NodeName, Tensor] = {}
    for orig in layout.originals:
        data = layout.reshape_vector(orig.name, factors[orig.name])
        leaves = sorted({leaf_of[ind.name] for ind in data.indices}, key=str)
        if len(leaves) == 1:
            cores = {leaves[0]: data}
        else:
            subtree = _steiner_tree(graph, leaves)
            cores = decompose(subtree, leaves[0], None, data)

        for n in graph:
            core = _expand_to_layout(cores.get(n), node_layout[n], free_inds)
            if n in combined:
                core = combined[n].mult(core, free_inds)
            combined[n] = core

    for n, core in combined.items():
        core = layout.deatomize_tensor(core)
        names = [ind.name for ind in ref.node_tensor(n).indices]
        result.set_node_tensor(n, core.permute_by_name(names))

    return result


def constant_like(
    ref: TensorNetwork, value: float, layout: Optional[IndexLayout] = None
) -> TensorNetwork:
    """Represent a constant tensor on the structure of a reference network."""
    if layout is None:
        layout = IndexLayout.identity(ref.free_indices())
    factors = {ind.name: np.ones(ind.size) for ind in layout.originals}
    factors[layout.originals[0].name] *= value
    return separable_like(ref, factors, layout)


def separable_factors(tn: TensorNetwork) -> Dict[IndexName, np.ndarray]:
    """Inverse of `separable_like` for a rank-one network whose nodes each
    carry one free index: returns the factor of every free index."""
    free_inds = tn.free_indices()
    factors: Dict[IndexName, np.ndarray] = {}
    scalar = 1.0
    for n in tn.network.nodes:
        tensor = tn.node_tensor(n)
        node_free = [ind for ind in tensor.indices if ind in free_inds]
        bonds = [ind for ind in tensor.indices if ind not in node_free]
        if any(ind.size != 1 for ind in bonds):
            raise ValueError(f"node {n} is not rank one: {tensor.indices}")
        if len(node_free) > 1:
            raise ValueError(f"node {n} holds several free indices")
        if not node_free:
            scalar *= float(tensor.value.reshape(-1)[0])
            continue
        factors[node_free[0].name] = tensor.value.reshape(-1).copy()

    factors[free_inds[0].name] *= scalar
    return factors


def apply_index_ops(
    tn: TensorNetwork,
    ops: Dict[IndexName, Callable[[np.ndarray], np.ndarray]],
) -> TensorNetwork:
    """Apply linear operators acting on single free indices.

    Every operator receives the unfolding `(index size, rest)` of the node
    holding that index and must return an array `(new size, rest)`.  This
    is the tree analogue of applying one rank-one TT operator.
    """
    out = copy.deepcopy(tn)
    for name, func in ops.items():
        node = out.node_by_free_index(name)
        tensor = out.node_tensor(node)
        axis = [ind.name for ind in tensor.indices].index(name)
        value = np.moveaxis(tensor.value, axis, 0)
        rest = value.shape[1:]
        value = np.asarray(func(value.reshape(value.shape[0], -1)))
        value = np.moveaxis(value.reshape((value.shape[0],) + rest), 0, axis)
        tensor.update_val_size(value)
    return out


def apply_index_ops_sum(
    tn: TensorNetwork,
    terms: Sequence[Dict[IndexName, Callable[[np.ndarray], np.ndarray]]],
    scale: Optional[float] = None,
) -> TensorNetwork:
    """Apply a sum of rank-one index operators (see `apply_index_ops`)."""
    out = apply_index_ops(tn, terms[0])
    for term in terms[1:]:
        out = out + apply_index_ops(tn, term)
    if scale is not None:
        out.scale(scale)
    return out


def integrate_to_dense(
    tn: TensorNetwork,
    weights: Dict[IndexName, np.ndarray],
    layout: Optional[IndexLayout] = None,
    order: Optional[Sequence[IndexName]] = None,
) -> np.ndarray:
    """Integrate over original indices and contract to a dense array.

    Args:
        tn: The network to integrate.
        weights: Quadrature weights over each integrated original index.
        layout: Mapping from the original indices to the free indices.
        order: Order of the remaining original indices in the output.
    """
    if layout is None:
        layout = IndexLayout.identity(tn.free_indices())
    out = layout.atomize(tn)
    for name, weight in weights.items():
        wnet = TensorNetwork()
        wnet.add_node(f"w_{name}", layout.reshape_vector(name, weight))
        out = out.attach(wnet, rename=("", ""))

    if order is None:
        order = [
            ind.name for ind in layout.originals if ind.name not in weights
        ]
    return layout.original_order(out.contract(), order)

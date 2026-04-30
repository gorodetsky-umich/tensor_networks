"""Algorithms for tensor networks."""

import copy
import itertools
import logging
import math
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
)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import opt_einsum as oe
from sklearn.utils.extmath import randomized_svd  # type: ignore

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
    NodeIndexPair,
    PartitionStatus,
    PartitionResult,
    SVDParams,
)
from pytens.search.types import Action
from pytens.cross.func_impl import FuncTensorNetwork

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass  # (frozen=True, eq=True)
class Tensor:
    """Base class for a tensor."""

    value: np.ndarray
    indices: List[Index]

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

        # print("new shape = ", new_shape)
        # print("new_indices = ", new_indices)
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
        # print(slice(1,2,4))
        # exit(1)
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
        delta: float = 1e-5,
        compute_uv: bool = True,
    ) -> Tuple[List["Tensor"], float]:
        """Split a tensor into three by SVD.

        If delta > 0, the truncated SVD is performed.
        """
        rights = [i for i in range(len(self.indices)) if i not in lefts]
        permute_indices = itertools.chain(lefts, rights)
        value = np.permute_dims(self.value, tuple(permute_indices))
        left_sz = int(np.prod([self.indices[i].size for i in lefts]))
        right_sz = int(np.prod([self.indices[j].size for j in rights]))
        value = value.reshape(left_sz, right_sz)

        result = delta_svd(value, delta, compute_uv=compute_uv)
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
        # for i in icount:
        #     if "_batch" in i.name:
        #         free_indices.append(i)

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
        assert ind not in self.free_indices(), f"{ind} is a free index"

        nodes = []
        for n in self.network.nodes:
            n_indices = self.node_tensor(n).indices
            if ind in n_indices:
                nodes.append(n)

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
        """Perform the SVD split and returns u, s, v.

        with_orthonormal: create orthogonality centers with QR before splitting
        compute_data: update the tensor values for the nodes created by split
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
            d = config.delta

            if config.compute_data:
                s_tuple, _ = x.svd(
                    lefts, delta=config.delta, compute_uv=config.compute_uv
                )
                s: Tensor = s_tuple[0]
            else:
                s = Tensor(np.empty(0), [rl, rr])
        else:
            x = self.node_tensor(node_name)
            # svd decompose the data into specified index partition
            [u, s, v], d = x.svd(lefts, delta=config.delta)

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
        results_shape = [values.shape[0]]
        results_shape.extend(
            [ind.size for ind in free_indices if ind not in indices]
        )
        results = np.empty(results_shape)
        chunk_size = 50000
        chunk_start = 0
        while chunk_start < values.shape[0]:
            batch_size = min(chunk_size, values.shape[0] - chunk_start)
            batch_ind = Index("_batch", batch_size)
            ind_mapping = {batch_ind: "a"}
            node_vals = []
            node_strs = []
            for node in self.network.nodes:
                tensor = self.node_tensor(node)
                tslices = []
                node_str = ""

                for ii, ind in enumerate(tensor.indices):
                    ind_letter = chr(97 + len(ind_mapping))
                    if ind in indices:
                        tslices.append(
                            (
                                ii,
                                values[
                                    chunk_start : chunk_start + batch_size,
                                    indices.index(ind),
                                ],
                            )
                        )
                    else:
                        if ind not in ind_mapping:
                            ind_mapping[ind] = ind_letter
                        node_str += ind_mapping[ind]

                    # print(ind, node_str)

                # swap batch to the front
                if len(tslices) > 0:
                    perm, pslices = zip(*tslices)
                    perm = list(perm)
                    # add other indices to the end of perm
                    for i in range(len(tensor.indices)):
                        if i not in perm:
                            perm.append(i)
                    node_str = ind_mapping[batch_ind] + node_str
                    batch_val = tensor.value.transpose(perm)[tuple(pslices)]
                else:
                    batch_val = tensor.value

                node_vals.append(batch_val)
                node_strs.append(node_str)

            estr = ",".join(node_strs) + "->" + ind_mapping[batch_ind]
            for ind in free_indices:
                if ind not in indices:
                    estr += ind_mapping[ind]

            logger.debug(
                "contraction args: %s, shapes: %s",
                estr,
                [n.shape for n in node_vals],
            )

            results[chunk_start : chunk_start + batch_size] = oe.contract(
                estr, *node_vals, optimize="random-greedy-128"
            )
            chunk_start += batch_size

        return results

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


class TreeNetwork(TensorNetwork):  # pylint: disable=R0904
    """Class for arbitrary tree-structured networks"""

    def round(
        self, node_name: NodeName, delta: float, visited: Optional[set] = None
    ) -> Tuple[NodeName, float]:
        """Optimize the tree rooted at the given node."""
        # print("optimize", node_name)
        # import matplotlib.pyplot as plt
        if visited is None:
            initial_optimize = True
            visited = set()
            self.orthonormalize(node_name)
        else:
            initial_optimize = False

        node_indices = self.node_tensor(node_name).indices
        kept_indices = []
        free_indices = []
        r = node_name
        for idx in node_indices:
            if idx in visited:
                kept_indices.append(idx)
                continue

            shared_index = None
            nbr = node_name
            for nbr in self.network.neighbors(node_name):
                nbr_indices = self.node_tensor(nbr).indices
                if idx in nbr_indices:
                    shared_index = idx
                    break

            if shared_index is None:
                free_indices.append(idx)
                continue

            curr_indices = self.node_tensor(node_name).indices
            left_indices = [
                curr_indices.index(i) for i in curr_indices if i != idx
            ]
            right_indices = [curr_indices.index(idx)]
            [node_name, s, v], delta = self.svd(
                node_name,
                left_indices,
                SVDConfig(delta=delta),
            )
            self.merge(v, s)
            self.merge(nbr, v)
            visited_index = self.get_contraction_index(node_name, nbr)
            for idx in visited_index:
                visited.add(idx)

            r, delta = self.round(nbr, delta, visited)
            self.merge(node_name, r)

        if not initial_optimize:
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
        tree = TreeNetwork()
        tree.network = self.network
        merges = []

        # collect all the indices in the current network by the original names
        for node in tree.network.nodes:
            node_merges = tree.collect_node_index_merges(node)
            merges.extend(node_merges)

        for merge_op in merges:
            tree.merge_index(merge_op)

        return merges

    def compress(self) -> "TreeNetwork":
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

        tree = TreeNetwork()
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
            nbrs = sorted(list(self.network.neighbors(name)))
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
        indices = self.node_tensor(node_name).indices
        perm = []
        leaves = []
        visited.add(node_name)

        if cut is None:
            cut = set()

        # free indices are added first
        if len(visited) != 1:
            for i, ind in enumerate(indices):
                if ind in self.free_indices():
                    leaves.append([ind])
                    perm.append(i)

        for n in self.network.neighbors(node_name):
            if n in visited:
                continue

            if self.get_contraction_index(n, node_name)[0].name in cut:
                continue

            leaves.append(self.leaf_indices(visited, n))
            common_index = self.get_contraction_index(n, node_name)
            assert len(common_index) == 1
            perm.append(indices.index(common_index[0]))

        # reorder the leaves according to the order of the indices
        return [leaves[i] for i in np.argsort(perm)]

    @staticmethod
    def rand_tucker(indices: List[Index], rank: int = 1) -> "TreeNetwork":
        """Return a random tucker with the given indices."""

        tucker = TreeNetwork()
        root_val = np.random.random([rank] * len(indices))
        root_inds = [Index(f"s_{i}", rank) for i in range(len(indices))]
        tucker.add_node("root", Tensor(root_val, root_inds))
        for i, ind in enumerate(indices):
            tensor_val = np.random.random((ind.size, rank))
            tensor_inds = [ind, root_inds[i]]
            tucker.add_node(f"G{i}", Tensor(tensor_val, tensor_inds))
            tucker.add_edge(f"G{i}", "root")

        return tucker

    def node_by_free_index(self, index: IndexName) -> NodeName:
        """Identify the node in the network containing the given free index"""
        node: NodeName
        for node in self.network.nodes:
            tensor = self.node_tensor(node)
            if index in [ind.name for ind in tensor.indices]:
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

    @staticmethod
    def _tucker(indices: Sequence[Index]) -> "TreeNetwork":
        """Create a Tucker with the given indices."""
        net = TreeNetwork()
        core_indices = [Index(f"s{i}", 1) for i in range(len(indices))]
        core_size = [ind.size for ind in core_indices]
        core = Tensor(np.random.random(core_size), core_indices)
        net.add_node("G", core)
        for i, ind in enumerate(indices):
            t_indices = [Index(f"s{i}", 1), ind]
            t_size = [1, ind.size]
            net.add_node(f"n{i}", Tensor(np.empty(t_size), t_indices))
            net.add_edge("G", f"n{i}")

        return net

    def _corrcoef(
        self, indices: Sequence[Index], sample_size: int = 50000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Pearson's correlation coefficient along the given
        indices."""
        # print(indices)
        # 1) sample points along the two indices
        ind_sizes = [ind.size for ind in indices]
        raw_samples: List[np.ndarray]
        if sample_size < int(np.prod(ind_sizes)):
            raw_samples = []
            for ind in indices:
                raw_samples.append(
                    np.random.randint(0, ind.size, size=(sample_size, 1))
                )
        else:
            raw_samples = list(
                np.meshgrid(*[np.arange(0, ind.size) for ind in indices])
            )

        samples: np.ndarray = np.stack(raw_samples, axis=-1).reshape(
            -1, len(raw_samples)
        )

        # create a sample network
        # net = self.evaluate(indices, samples)

        others = [ind for ind in self.free_indices() if ind not in indices]
        other_size = np.prod([ind.size for ind in others])
        # 2) compute the sum value, i.e. integration over selected indices
        weights = np.ones(len(others))
        sums_net = self.integrate(others, weights)
        # print(sums_net.free_indices())
        # print(sums_net)
        sums = sums_net.evaluate(indices, samples)
        # print(sums.shape)

        # 3) compute the inner product over the selected indices
        inner_net = self.attach(self, indices=others)
        # print(inner_net.free_indices())
        # print(inner_net)
        inner_indices = [
            ind.with_new_name(f"G{ind.name}") for ind in indices
        ] + [ind.with_new_name(f"H{ind.name}") for ind in indices]
        samples_i, samples_j = np.triu_indices(len(samples))
        pairs = np.hstack((samples[samples_i], samples[samples_j]))
        half_inner = inner_net.evaluate(inner_indices, pairs)
        inner: np.ndarray = np.empty((len(samples), len(samples)))
        inner[samples_i, samples_j] = half_inner
        inner[samples_j, samples_i] = half_inner
        # print(inner.shape)

        # 4) compute the covariance matrix
        mu = sums / other_size
        cov = 1.0 / (other_size - 1) * (inner - other_size * np.outer(mu, mu))
        # print(cov.shape)
        # print(cov)

        # 5) compute the correlation coefficient
        stddev = np.sqrt(np.diag(cov))
        denom = np.outer(stddev, stddev)
        corr = cov / denom
        corr[denom == 0] = 0
        return samples, corr

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

    def _move_index(self, ind1: Index, ind2: Index) -> None:
        """Move ind1 to the neighborhood of ind2"""
        logger.debug("moving index %s to index %s", ind1, ind2)
        node1 = self.node_by_free_index(ind1.name)
        node2 = self.node_by_free_index(ind2.name)
        logger.debug("moving node %s to node %s", node1, node2)
        if node1 == node2:
            return

        path = nx.shortest_path(self.network, node1, node2)
        for n in path[1:]:
            swapping_ind = None if n != node2 else ind2
            self.swap_nbr(
                path,
                NodeIndexPair(node1, ind1),
                NodeIndexPair(n, swapping_ind),
            )
            logger.debug("after swapping %s and %s, get %s", node1, n, self)

    def swap_nbr(
        self,
        path: Sequence[NodeName],
        node1_pair: NodeIndexPair,
        node2_pair: NodeIndexPair,
    ) -> None:
        """Swap two neighbor nodes."""
        node1, ind1 = node1_pair.node, node1_pair.ind
        node2, ind2 = node2_pair.node, node2_pair.ind
        logger.debug("swapping the neighbors %s and %s", node1, node2)
        common_ind = self.get_contraction_index(node1, node2)[0]
        node_indices = []

        # Collect indices from node1 that should be kept
        for ind in self.node_tensor(node1).indices:
            if self._should_keep_index_from_left(
                path, common_ind, NodeIndexPair(node1, ind), ind1
            ):
                node_indices.append(ind)

        # Collect indices from node2 that should be kept
        for ind in self.node_tensor(node2).indices:
            if self._should_keep_index_from_right(
                path, common_ind, NodeIndexPair(node2, ind), ind2
            ):
                node_indices.append(ind)

        name = self.merge(node1, node2)
        new_indices = self.node_tensor(name).indices
        lefts = [new_indices.index(ind) for ind in node_indices]
        u, v = self.qr(name, lefts)
        self.network = nx.relabel_nodes(self.network, {u: node2, v: node1})

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
        """Swap a node along the given path."""
        logger.debug("moving %s along the path %s", moving_node, path)
        for other in path[1:]:
            # Swap the nodes along the path
            self.swap_nbr(
                path, NodeIndexPair(moving_node), NodeIndexPair(other)
            )

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

    def get_subtree(self, u: NodeName, v: NodeName) -> "TreeNetwork":
        """create a subgraph by breaking the edge"""
        net = self.network.copy()
        net.remove_edge(u, v)
        subnet_nodes = nx.node_connected_component(net, v)
        sub_tree = TreeNetwork()
        sub_tree.network = net.subgraph(subnet_nodes)
        return sub_tree

    def _subtree_has_free_indices(
        self, fixed_nodes: Sequence[NodeName], node: NodeName, edge: Index
    ) -> bool:
        if edge in self.free_indices():
            return True

        for m in self.network.neighbors(node):
            m_inds = self.node_tensor(m).indices
            if edge in m_inds:
                if node in fixed_nodes:
                    return False

                sub_tree = self.get_subtree(node, m)
                free_inds = sub_tree.free_indices()
                return len(free_inds) > 1

        return False

    def _index_appears_on_path(
        self, ind: Index, path: Sequence[NodeName]
    ) -> bool:
        for i, n in enumerate(path[:-1]):
            if ind in self.get_contraction_index(n, path[i + 1]):
                return True

        return False

    def _should_keep_index_from_left(
        self,
        path: Sequence[NodeName],
        common_ind: Index,
        ind_node: NodeIndexPair,
        swapping_ind: Optional[Index] = None,
    ) -> bool:
        """Determine if an index from the left should be kept during swap."""
        ind, node = ind_node.ind, ind_node.node
        if (
            ind == common_ind
            or (swapping_ind is None and ind in self.free_indices())
            or (swapping_ind is not None and ind == swapping_ind)
        ):
            return False

        assert ind is not None
        if self._index_appears_on_path(ind, path):
            return True

        for nbr in self.network.neighbors(node):
            if ind in self.node_tensor(nbr).indices:
                return True

        return False

    def _should_keep_index_from_right(
        self,
        path: Sequence[NodeName],
        common_ind: Index,
        ind_node: NodeIndexPair,
        swapping_ind: Optional[Index] = None,
    ) -> bool:
        """Determine if an index from the right should be kept during swap."""
        ind, node = ind_node.ind, ind_node.node
        if (swapping_ind is None and ind in self.free_indices()) or (
            swapping_ind is not None and ind == swapping_ind
        ):
            return True

        for nbr in self.network.neighbors(node):
            if ind is not None and ind in self.node_tensor(nbr).indices:
                # found the correct neighbor
                return ind != common_ind and not self._index_appears_on_path(
                    ind, path
                )

        return False

    @staticmethod
    def rand_tree(indices: List[Index], ranks: List[int]) -> "TreeNetwork":
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

        tree = TreeNetwork()

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

    def __add__(self, other: Any) -> Self:
        """Add two tree networks."""
        if not isinstance(other, TreeNetwork):
            raise NotImplementedError

        assert nx.is_isomorphic(self.network, other.network)

        # assign the root at the same index name
        root_ind = self.free_indices()[0]
        self_root = self.node_by_free_index(root_ind.name)
        self_tree = self.dimension_tree(self_root)
        other_root = other.node_by_free_index(root_ind.name)
        other_tree = other.dimension_tree(other_root)

        result_net = copy.deepcopy(self)
        self._binary_op(other, "add", self_tree, other_tree, result_net)

        return result_net

    def __sub__(self, other: Any) -> Self:
        """Subtract two tree networks."""
        if not isinstance(other, TreeNetwork):
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
        if not isinstance(other, TreeNetwork):
            raise NotImplementedError

        assert nx.is_isomorphic(self.network, other.network)

        # assign the root at the same index name
        root_ind = self.free_indices()[0]
        self_root = self.node_by_free_index(root_ind.name)
        self_tree = self.dimension_tree(self_root)
        other_root = other.node_by_free_index(root_ind.name)
        other_tree = other.dimension_tree(other_root)

        result_net = copy.deepcopy(self)
        self._binary_op(other, "mul", self_tree, other_tree, result_net)

        return result_net

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def _binary_op(
        self,
        other: "TreeNetwork",
        op: Literal["add", "mul"],
        tree1: DimTreeNode,
        tree2: DimTreeNode,
        result_net: Self,
    ) -> None:
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
            self._binary_op(other, op, c1, c2, result_net)

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
        ok, inds = self._check_indices(indices, set(), svd_node)
        assert ok.code in (PartitionStatus.OK, PartitionStatus.EXIST), (
            f"{svd_node} is not the correct partition point"
        )

        tensor = self.node_tensor(svd_node)
        svd_node_inds = tensor.indices
        svd_ls = set()
        for ind in indices:
            for svd_ind, ind_group in inds.items():
                if ind in ind_group:
                    svd_ls.add(svd_node_inds.index(svd_ind))
                    break

        return list(svd_ls)

    def _check_indices(
        self, indices: Sequence[Index], visited: Set[NodeName], node: NodeName
    ) -> Tuple[PartitionResult, Dict[Index, List[Index]]]:
        """Check whether a node is a partition of the given indices."""
        visited.add(node)
        results = {}
        for m in self.network.neighbors(node):
            if m not in visited:
                res, finds = self._check_indices(indices, visited, m)
                if res.code != PartitionStatus.OK:
                    return res, finds

                # print("get", finds, "for", m, "with parent", node)
                inds = [v for vs in finds.values() for v in vs]
                # if finds include both desired and undesired, skip
                desired = set(indices).intersection(set(inds))
                undesired = set(inds).difference(set(indices))
                # print(desired, undesired)

                if len(desired) > 0 and len(undesired) > 0:
                    res.code = PartitionStatus.FAIL
                    return res, {}

                results[self.get_contraction_index(m, node)[0]] = inds
                if len(undesired) == 0 and len(desired) == len(indices):
                    res.code = PartitionStatus.EXIST
                    res.lca_node = m
                    return res, results

        free_indices = self.free_indices()
        node_indices = self.node_tensor(node).indices
        for i in node_indices:
            if i in free_indices:
                results[i] = [i]

        res = PartitionResult()
        res.code = PartitionStatus.OK
        return res, results

    def partition_node(self, indices: Sequence[Index]) -> PartitionResult:
        """Find a proper node that partitions the free indices as specified."""
        # we should find a node where the expected indices and
        # the unexpected indices are on different indices

        lca_indices = []
        for n in self.network.nodes:
            # postorder traversal from each node and
            # if we find each index
            visited: Set[NodeName] = set()
            # print("postordering", n)
            res, results = self._check_indices(indices, visited, n)

            if res.code in (PartitionStatus.EXIST, PartitionStatus.OK):
                for i in indices:
                    for e, inds in results.items():
                        if i in inds:
                            lca_indices.append(e)
                            break

                assert len(lca_indices) == len(indices), (
                    "each index should correspond to one of the edges, "
                    f"but get {lca_indices}, {indices}"
                )

                if res.code == PartitionStatus.OK:
                    res.lca_node = n

                res.lca_indices = list(set(lca_indices))
                return res

        raise ValueError(
            "Cannot find a node that realizes the partition", indices
        )

    def random_svals(
        self,
        node: NodeName,
        indices: Sequence[Index],
        params: SVDParams,
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
        old_subnet: "TreeNetwork",
        new_subnet: "TreeNetwork",
        _split_info: Optional[List[IndexOp]] = None,
    ) -> "TreeNetwork":
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

        tree = TreeNetwork()
        tree.network = self.network
        return tree

    def _longest_path(self) -> List[NodeName]:
        """Get the longest path in the current tree."""
        u = list(self.network.nodes())[0]
        dist = nx.single_source_shortest_path_length(self.network, u)
        a = max(dist, key=dist.get)

        dist = nx.single_source_shortest_path_length(self.network, a)
        b = max(dist, key=dist.get)

        path: List[NodeName] = nx.shortest_path(self.network, a, b)
        return path

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

"""Algorithms for tensor networks."""

import copy
import itertools
import logging
import math
import typing
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Self,
    Set,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import opt_einsum as oe
from line_profiler import profile
from sklearn.utils.extmath import randomized_svd

from pytens.cross.cross import CrossResult, cross
from pytens.cross.funcs import FuncTensorNetwork, TensorFunc
from pytens.search.types import Action
from pytens.types import (
    DimTreeNode,
    FoldDir,
    Index,
    IndexMerge,
    IndexName,
    IndexOp,
    IndexSplit,
    NodeInfo,
    NodeName,
    NodeStatus,
    PartitionResult,
    PartitionStatus,
    SVDConfig,
)
from pytens.utils import delta_svd

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass  # (frozen=True, eq=True)
class Tensor:
    """Base class for a tensor."""

    value: np.ndarray
    indices: List[Index]

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

    def block_diagonal(self, other: "Tensor", block_start: int) -> "Tensor":
        """
        Concat tensors along contract indices diagonally but keep free indices.
        When there is only one contract index, concat them directly.
        """
        sz = []
        for i, ind in enumerate(self.indices):
            if i < block_start:
                assert ind.size == other.indices[i].size
                sz.append(ind.size)
            else:
                sz.append(ind.size + other.indices[i].size)

        large_array = np.zeros(sz, dtype=self.value.dtype)
        start = np.zeros(len(sz) - block_start, dtype=int)
        slice_prefix = tuple([slice(None)] * block_start)

        for arr in [self.value, other.value]:
            end = start + arr.shape[block_start:]
            # Create a slice object for each dimension
            slice_suffix = tuple(
                slice(start[i], end[i]) for i in range(len(end))
            )
            slices = slice_prefix + slice_suffix
            # Place the current array on the diagonal
            large_array[slices] = arr
            # Update start positions
            start = end

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

    next_index_id = 0

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

    def inner_indices(self) -> List[Index]:
        """Get hte interior indices."""
        icount = self.all_indices()
        free_indices = sorted([i for i, v in icount.items() if v > 1])
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
                    if (
                        not isinstance(ind[dim], int)
                        and not ind[dim].is_integer()
                    ):
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

    @classmethod
    def get_next_id(cls) -> int:
        """Return the next available index id"""
        i = cls.next_index_id
        cls.next_index_id += 1
        return i

    def fresh_index(
        self, used_indices: Optional[Sequence[IndexName]] = None
    ) -> str:
        """Generate an index that does not appear in the current network."""
        all_indices = [ind.name for ind in self.all_indices().keys()]
        i = self.get_next_id()
        while f"s_{i}" in all_indices or (
            used_indices is not None and f"s_{i}" in used_indices
        ):
            i = self.get_next_id()

        return f"s_{i}"

    def fresh_node(
        self, used_nodes: Optional[Sequence[NodeName]] = None
    ) -> NodeName:
        """Generate a node name that does not appear in the current network."""
        i = self.get_next_id()
        node = f"n{i}"
        while node in self.network.nodes or (
            used_nodes is not None and node in used_nodes
        ):
            node = f"n{i}"
            i = self.get_next_id()

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
                s, _ = x.svd(
                    lefts, delta=config.delta, compute_uv=config.compute_uv
                )
                s = s[0]
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
            n_cost = np.prod([i.size for i in indices])
            cost += n_cost

        return int(cost)
    
    def size(self) -> int:
        """Compute the size of the tensor network by multiplying the sizes of each free index."""
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
    ):
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

    @profile
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
        # perm = [indices.index(i) for i in free_indices]
        # values = values[:, perm]

        # not_selected = []
        # for n in self.network.nodes:
        #     tensor = self.node_tensor(n)
        #     if len(set(indices).intersection(set(tensor.indices))) == 0:
        #         not_selected.append(n)

        # logger.debug("evaluating on %s, not selected are %s", indices, not_selected)

        # # get the subgraphs with only the not selected nodes
        # new_net = copy.deepcopy(self)
        # subnets = nx.subgraph(self.network, not_selected)
        # for i, subnet_nodes in enumerate(nx.connected_components(subnets)):
        #     tmp_net = TensorNetwork()
        #     tmp_net.network = new_net.network.subgraph(subnet_nodes).copy()
        #     tmp_tensor = tmp_net.contract()
        #     tmp_name = f"subnet_{i}"
        #     new_net.add_node(tmp_name, tmp_tensor)
        #     for n in subnet_nodes:
        #         for nbr in self.network.neighbors(n):
        #             if nbr not in not_selected:
        #                 new_net.add_edge(tmp_name, nbr)

        #         logger.debug("removing %s", n)
        #         new_net.network.remove_node(n)
        #         logger.debug(new_net)

        results = np.empty(values.shape[0])
        # for i, v in enumerate(values):
        #     results[i] = new_net[tuple(v)].value

        # # divide the values into small chunks
        # print(values.shape[0])
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

                # print(node, tensor.indices, tslices, tensor.value.shape, batch_val.shape)
                node_vals.append(batch_val)
                node_strs.append(node_str)

            estr = ",".join(node_strs) + "->" + ind_mapping[batch_ind]
            logger.debug(
                "contraction args: %s, shapes: %s",
                estr,
                [n.shape for n in node_vals],
            )
            # _, path_info = oe.contract_path(estr, *node_vals)
            # logger.debug("The cost of %s is %s", estr, path_info.opt_cost)
            results[chunk_start : chunk_start + batch_size] = oe.contract(
                estr, *node_vals, optimize="auto"
            )
            chunk_start += batch_size

        return results

    def __lt__(self, other: Self) -> bool:
        return self.cost() < other.cost()

    def __str__(self) -> str:
        """Convert to string."""
        out = "Nodes:\n"
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
        for i, index in enumerate(free_indices):
            if index.size == 1:
                continue

            free_graph.add_node(f"{index.name}-{index.size}")

        new_graph = nx.compose(self.network, free_graph)
        for i, index in enumerate(free_indices):
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

        for group, nodes in node_groups.items():
            if group == "A":
                nx.draw_networkx_nodes(
                    new_graph,
                    pos,
                    ax=ax,
                    nodelist=nodes,
                    node_color="tab:blue",
                    node_shape=shape_map[group],
                    node_size=size_map[group],
                    linewidths=2.0,
                    edgecolors="k",
                )

                if node_label:
                    node_labels = {node: node for node in node_groups["A"]}
                    nx.draw_networkx_labels(
                        new_graph, pos, ax=ax, labels=node_labels, font_size=12
                    )
            elif group == "C":
                nx.draw_networkx_nodes(
                    new_graph,
                    pos,
                    ax=ax,
                    nodelist=nodes,
                    node_color="w",
                    # node_color=range(1, len(nodes) + 1),
                    node_shape=shape_map[group],
                    node_size=size_map[group],
                    # cmap=plt.get_cmap("tab20"),
                    # with_label=with_label[group]
                )
            else:
                nx.draw_networkx_nodes(
                    new_graph,
                    pos,
                    ax=ax,
                    nodelist=nodes,
                    node_color="w",
                    # node_color=range(1, len(nodes) + 1),
                    node_shape=shape_map[group],
                    node_size=size_map[group],
                    # cmap=plt.get_cmap("tab20"),
                    # with_label=with_label[group]
                )

                node_labels = {node: node for node in nodes}
                nx.draw_networkx_labels(
                    new_graph,
                    pos,
                    ax=ax,
                    labels=node_labels,
                    font_size=12,
                    # verticalalignment="top",
                )

        edge_labels = {}
        for u, v in self.network.edges():
            indices = self.get_contraction_index(u, v)
            labels = [f"{i.size}" for i in indices]
            label = "-".join(labels)
            edge_labels[(u, v)] = label

        visible_edges, invisible_edges = [], []
        for u, v in new_graph.edges:
            if str(u).endswith("_node") and str(u).startswith(str(v)):
                invisible_edges.append((u, v))
                continue

            if str(v).endswith("_node") and str(v).startswith(str(u)):
                invisible_edges.append((u, v))
                continue

            visible_edges.append((u, v))

        nx.draw_networkx_edges(
            new_graph, pos, edgelist=visible_edges, ax=ax, width=2.0
        )
        nx.draw_networkx_edges(
            new_graph,
            pos,
            edgelist=invisible_edges,
            ax=ax,
            width=0.0,
            edge_color="white",
        )
        # nx.draw_networkx_edges(new_graph, pos, ax=ax, width=2.0, min_source_margin=5, min_target_margin=5)
        nx.draw_networkx_edge_labels(
            new_graph, pos, ax=ax, edge_labels=edge_labels, font_size=10
        )

    def node_size(self, node: NodeName) -> int:
        """Get the tensor size of the given node."""
        node_inds = self.node_tensor(node).indices
        return int(np.prod([ind.size for ind in node_inds]))

    def ranks_along_path(self, path: Sequence[NodeName]) -> Sequence[int]:
        """Get the ranks between the nodes on the given path."""
        ranks = []
        for i, ni in enumerate(path[:-1]):
            ranks.append(self.get_contraction_index(ni, path[i + 1])[0].size)

        return ranks

    def merge_along_path(self, path: Sequence[NodeName]) -> NodeName:
        """Merge the nodes on the given path."""
        node = path[0]
        for n in path[1:]:
            self.merge(node, n)

        return node


class TreeNetwork(TensorNetwork):
    """Class for arbitrary tree-structured networks"""

    def __init__(self):
        super().__init__()
        self.node_status = {}

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

    def _compress_index(self, node: NodeName):
        """Compress consectutive indices that are decomposed from one index"""
        #TODO: implement this

    def compress(self) -> None:
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

    @profile
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
        return self.postorder_orthonormal({}, None, name)

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
        cut: Set[IndexName] = set(),
    ) -> List:
        """Get all leaf indices for the subtree rooted at the given node."""
        indices = self.node_tensor(node_name).indices
        perm = []
        leaves = []
        visited.add(node_name)

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

    def cross(
        self,
        tensor_func: TensorFunc,
        eps: Optional[float] = None,
        kickrank: int = 2,
    ) -> CrossResult:
        """Run cross approximation over the current network structure
        until the relative error goes below the prescribed epsilon."""
        root = None
        free_indices = self.free_indices()
        for node in self.network.nodes:
            if root is None:
                root = node
                continue

            root_free = 1
            for ind in self.node_tensor(root).indices:
                if ind in free_indices:
                    root_free *= ind.size

            node_free = 1
            for ind in self.node_tensor(node).indices:
                if ind in free_indices:
                    node_free *= ind.size

            if node_free > root_free:
                root = node

        return cross(tensor_func, self, root, eps, kickrank=kickrank)

    def node_by_free_index(self, index: IndexName) -> NodeName:
        """Identify the node in the network containing the given free index"""
        for n in self.network.nodes:
            tensor = self.node_tensor(n)
            if index in [ind.name for ind in tensor.indices]:
                return n

        raise KeyError(f"Cannot find index {index} in the network")

    def canonicalize_indices(self, tree: DimTreeNode):
        """sort the children by free indices
        and get the corresponding children nodes
        """
        indices: List[Index] = []
        node_indices = self.node_tensor(tree.node).indices
        for ind in tree.free_indices:
            indices.append(ind)

        # children indices
        for n in sorted(tree.down_info.nodes):
            self.canonicalize_indices(n)
            ind = self.get_contraction_index(n.node, tree.node)[0]
            indices.append(ind)

        # parent indices, should be one
        p_indices = [ind for ind in node_indices if ind not in indices]
        assert len(p_indices) <= 1, (
            f"should have at most one parent index, but get {p_indices}"
        )

        indices.extend(p_indices)
        perm = [node_indices.index(ind) for ind in indices]
        new_tensor = self.node_tensor(tree.node).permute(perm)
        self.set_node_tensor(tree.node, new_tensor)

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
        self.canonicalize_indices(tree)
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
    def tucker(indices: Sequence[Index]) -> "TreeNetwork":
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

    @profile
    def corrcoef(
        self, indices: Sequence[Index], sample_size: int = 50000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Pearson's correlation coefficient along the given indices."""
        # print(indices)
        # 1) sample points along the two indices
        ind_sizes = [ind.size for ind in indices]
        if sample_size < int(np.prod(ind_sizes)):
            samples = []
            for ind in indices:
                samples.append(
                    np.random.randint(0, ind.size, size=(sample_size, 1))
                )
        else:
            samples = np.meshgrid(*[np.arange(0, ind.size) for ind in indices])

        samples = np.stack(samples, axis=-1).reshape(-1, len(samples))

        # create a sample network
        # net = self.evaluate(indices, samples)

        others = [ind for ind in self.free_indices() if ind not in indices]
        other_size = np.prod([ind.size for ind in others])
        # 2) compute the sum value, i.e. integration over selected indices
        weights = np.ones(len(others))
        sums = self.integrate(others, weights)
        # print(sums.free_indices())
        # print(sums)
        sums = sums.evaluate(indices, samples)
        # print(sums.shape)

        # 3) compute the inner product over the selected indices
        inner = self.attach(self, indices=others)
        # print(inner.free_indices())
        # print(inner)
        inner_indices = [
            ind.with_new_name(f"G{ind.name}") for ind in indices
        ] + [ind.with_new_name(f"H{ind.name}") for ind in indices]
        samples_i, samples_j = np.triu_indices(len(samples))
        pairs = np.hstack((samples[samples_i], samples[samples_j]))
        half_inner = inner.evaluate(inner_indices, pairs)
        inner = np.empty((len(samples), len(samples)))
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

    ##########################################
    ##### Swap and its helper functions  #####
    ##########################################

    @profile
    def swap(
        self, ind_nodes: Sequence[NodeName], delta: float = 0
    ) -> Tuple[NodeName, NodeName]:
        """Swap the indices so that the target indices are adjacent."""
        ind_nodes = list(set(ind_nodes))
        anchor, _ = self.best_anchor(ind_nodes)
        self.node_status = {}
        self.node_status[anchor] = NodeStatus.CONFIRMED
        for node in ind_nodes:
            path = nx.shortest_path(self.network, node, anchor)
            self.swap_along_path(path, node)
            self.node_status[node] = NodeStatus.CONFIRMED

        return self._max_dist_nodes(ind_nodes, anchor)

    @profile
    def swap_nbr(
        self,
        path: Sequence[NodeName],
        node1: NodeName,
        node2: NodeName,
    ):
        """Swap two neighbor nodes."""
        common_ind = self.get_contraction_index(node1, node2)[0]
        node_indices = []

        # Collect indices from node1 that should be kept
        for ind in self.node_tensor(node1).indices:
            if self._should_keep_index_from_left(path, common_ind, ind, node1):
                node_indices.append(ind)

        # Collect indices from node2 that should be kept
        for ind in self.node_tensor(node2).indices:
            if self._should_keep_index_from_right(
                path, common_ind, ind, node2
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
        num_confirmed: int,
        curr_dist: int,
        node: NodeName,
    ) -> int:
        visited.add(node)

        total_dist = curr_dist - num_confirmed
        if node in nodes:
            num_confirmed += 1

        for nbr in self.network.neighbors(node):
            if nbr not in visited:
                total_dist += self._anchor_distance(
                    visited, nodes, num_confirmed, curr_dist + 1, nbr
                )

        return total_dist

    def best_anchor(
        self, candidates: Sequence[NodeName]
    ) -> Tuple[NodeName, float]:
        """Find the best anchor node such that all other nodes are closest to it."""
        # first, we find the best anchor node
        best_anchor = candidates[0]
        best_dist = float("inf")

        for anchor_candidate in candidates:
            total_dist = self._anchor_distance(
                set(), candidates, 0, 0, anchor_candidate
            )

            if total_dist < best_dist:
                best_anchor = anchor_candidate
                best_dist = total_dist

        return best_anchor, best_dist

    def swap_along_path(
        self,
        path: Sequence[NodeName],
        moving_node: NodeName,
    ):
        """Swap a node along the given path."""
        for other in path[1:]:
            # if we encounter a confirmed node,
            # we should have seen all following nodes.
            status = self.node_status.get(other, NodeStatus.UNKNOWN)
            if status == NodeStatus.CONFIRMED:
                break

            # Swap the nodes along the path
            self.swap_nbr(path, moving_node, other)

        self.node_status[moving_node] = NodeStatus.CONFIRMED

    def _max_dist_nodes(
        self, ind_nodes: Sequence[NodeName], anchor: NodeName
    ) -> Tuple[NodeName, NodeName]:
        """Find the two nodes with the maximum distance among the given nodes."""
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
        ind: Index,
        node: NodeName,
    ) -> bool:
        """Determine if an index from the left should be kept during swap."""
        if ind == common_ind or ind in self.free_indices():
            return False

        if self._index_appears_on_path(ind, path):
            return True

        for nbr in self.network.neighbors(node):
            if ind in self.node_tensor(nbr).indices:
                return (
                    self.node_status.get(nbr, NodeStatus.UNKNOWN)
                    != NodeStatus.ATTACHED
                )

        return False

    def _should_keep_index_from_right(
        self,
        path: Sequence[NodeName],
        common_ind: Index,
        ind: Index,
        node: NodeName,
    ) -> bool:
        """Determine if an index from the right should be kept during swap."""
        if ind in self.free_indices():
            return True

        for nbr in self.network.neighbors(node):
            if ind in self.node_tensor(nbr).indices:
                # found the correct neighbor
                return (
                    ind != common_ind
                    and not self._index_appears_on_path(ind, path)
                    and (
                        self.node_status.get(nbr, NodeStatus.UNKNOWN)
                        == NodeStatus.ATTACHED
                    )
                )

        return False

    ##########################################
    ##### Fold and its helper functions  #####
    ##########################################

    def fold(
        self, nodes: Sequence[NodeName]
    ) -> Tuple["TreeNetwork", NodeName]:
        """Transform the network such that the target indices are merged to
        a single node while the nodes between them are folded.
        """
        if len(nodes) <= 1:
            return self, nodes[0]

        # we skip the non-root nodes during traversal
        # find a path to fold the specified nodes
        left_anchor, right_anchor = self.swap(nodes)
        return self.fold_nodes(left_anchor, right_anchor, FoldDir.IN_BOUND)

    def fold_nodes(
        self, node1: NodeName, node2: NodeName, fold_dir: FoldDir
    ) -> Tuple["TreeNetwork", NodeName]:
        """Merge the indices on two nodes"""
        net = TreeNetwork()
        net.network = copy.deepcopy(self.network)
        # take the path between nodes[0] and nodes[1]
        path = nx.shortest_path(net.network, source=node1, target=node2)
        # fold the path by merging the pairs of nodes
        # for i in range(len(path) // 2):
        #     self.merge(path[i], path[-i - 1])

        # to avoid generating nodes of very large sizes, we split one of them
        first_size = net.node_size(path[0])
        last_size = net.node_size(path[-1])
        res = path[0]
        # we need to set a criteria when to
        if first_size < last_size:
            _, r = net.qr_decompose_anchor(path[-1], path[-2], fold_dir)
            net.merge(path[0], r)
        else:
            _, r = net.qr_decompose_anchor(path[0], path[1], fold_dir)
            net.merge(path[-1], r)
            res = path[-1]
        # res = path[0]
        # self.merge(path[0], path[-1])

        net.network.remove_edge(path[len(path) // 2 - 1], path[len(path) // 2])
        return net, res

    def _check_neighbor_status(self, node: NodeName, ind: Index):
        for nbr in self.network.neighbors(node):
            if ind in self.node_tensor(nbr).indices:
                return self.node_status.get(nbr, NodeStatus.UNKNOWN)

        return NodeStatus.UNKNOWN

    def qr_decompose_anchor(
        self,
        node: NodeName,
        adjacent_node: Optional[NodeName],
        fold_dir: FoldDir,
    ) -> Tuple[NodeName, NodeName]:
        """Helper method to perform QR decomposition on a node
        with respect to an adjacent node.
        """
        if adjacent_node is not None:
            right_ind = self.get_contraction_index(node, adjacent_node)[0]
        else:
            right_ind = None

        free_inds = self.free_indices()
        lefts = []
        for i, ind in enumerate(self.node_tensor(node).indices):
            if fold_dir == FoldDir.IN_BOUND and (
                ind == right_ind or ind in free_inds
            ):
                lefts.append(i)

            if fold_dir == FoldDir.OUT_BOUND and (
                ind != right_ind or ind in free_inds
            ):
                lefts.append(i)

        return self.qr(node, lefts)

    def _group_by_proximity(
        self,
        ind_nodes: Sequence[NodeName],
        max_dist: int = 5,
    ) -> Sequence[Sequence[NodeName]]:
        groups = []
        anchor, _ = self.best_anchor(ind_nodes)
        remaining_nodes = list(
            sorted(ind_nodes, key=lambda x: self.distance(x, anchor))
        )

        while remaining_nodes:
            curr_group = [remaining_nodes[0]]
            for node in remaining_nodes[1:]:
                if self.distance(curr_group[0], node) <= max_dist:
                    curr_group.append(node)

            for node in curr_group:
                if node in remaining_nodes:
                    remaining_nodes.remove(node)

            groups.append(curr_group)

        return groups

    def fold_hierarchical(
        self, ind_nodes: Sequence[NodeName], max_dist: int = 5
    ) -> Tuple["TreeNetwork", NodeName]:
        """Seperate the indices into groups and fold groups hierarchically."""
        # when the nodes are scattered very far from each other
        # we first group them in the neighhood and then swap and then fold
        if len(ind_nodes) == 1:
            return self, ind_nodes[0]

        groups = self._group_by_proximity(ind_nodes, max_dist)
        # if all(len(group) == 1 for group in groups):
        #     groups = [[n for group in groups for n in group]]

        anchors = []
        net = self
        for group in groups:
            net, anchor_node = net.fold(group)
            anchors.append(anchor_node)

        # we need to temporarily detach other nodes and only keep the anchors
        # before proceeding to the next hierarchy
        connections = []
        for gi, group in enumerate(groups):
            for u in group:
                e = (u, anchors[gi])
                if e in net.network.edges:
                    connections.append((e, net.get_subtree(anchors[gi], u)))

        include_nodes = anchors[:]
        for n in net.network.nodes:
            if n not in ind_nodes:
                include_nodes.append(n)

        subnet = TreeNetwork()
        subnet.network = nx.subgraph(net.network, include_nodes).copy()
        subnet, anchor = subnet.fold_hierarchical(anchors, max_dist + 1)

        subnet = subnet.network
        for e, subgraph in connections:
            subnet = nx.compose(subnet, subgraph.network)
            subnet.add_edge(*e)

        result_net = TreeNetwork()
        result_net.network = subnet
        return result_net, anchor

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
        result_net.network.clear()
        self._binary_op(other, "add", self_tree, other_tree, result_net)
        # round the result to compress the ranks
        result_net.round(self_root, delta=1e-6)

        return result_net

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
        result_net.network.clear()
        self._binary_op(other, "mul", self_tree, other_tree, result_net)
        # round the result to compress the ranks
        result_net.round(self_root, delta=1e-6)

        return result_net

    def _binary_op(
        self,
        other: "TreeNetwork",
        op: Literal["add", "mul"],
        tree1: DimTreeNode,
        tree2: DimTreeNode,
        result_net: Self,
    ) -> None:
        tensor1 = self.node_tensor(tree1.info.node)
        tensor2 = other.node_tensor(tree2.info.node)
        assert len(tensor1.indices) == len(tensor2.indices)

        if op == "add":
            res = tensor1.block_diagonal(tensor2, len(tree1.info.free_indices))
        elif op == "mul":
            res = tensor1.mult(tensor2, self.free_indices())
        else:
            raise ValueError(f"Unknown operation {op}")

        result_net.add_node(tree1.info.node, res)

        for c1, c2 in zip(tree1.conn.children, tree2.conn.children):
            self._binary_op(other, op, c1, c2, result_net)
            result_net.add_edge(tree1.info.node, c1.info.node)

    def distance(self, node1: NodeName, node2: NodeName) -> int:
        """Compute the distance between two nodes without creating dimension trees."""
        if node1 == node2:
            return 0

        return nx.shortest_path_length(self.network, node1, node2)

    def svd_lefts(
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

    @profile
    def svals(
        self,
        indices: Sequence[Index],
        max_rank: int = 100,
        orthonormal: Optional[NodeName] = None,
        delta: float = 0,
    ) -> np.ndarray:
        """Compute the singular values for a tree network structure."""
        assert len(indices) == 2
        # print(indices)
        # net = self.move_to(indices, delta=delta)
        net = copy.deepcopy(self)
        nodes = [net.node_by_free_index(ind.name) for ind in indices]
        unique_nodes = list(set(nodes))
        if len(unique_nodes) > 1:
            net.fold(unique_nodes)

        nodes = [net.node_by_free_index(ind.name) for ind in indices]
        unique_nodes = list(set(nodes))
        assert len(unique_nodes) == 1, "more than one node after folding"

        svd_node = unique_nodes[0]

        if orthonormal is None:
            net.orthonormalize(svd_node)

        tensor = net.node_tensor(svd_node)
        svd_node_inds = tensor.indices
        svd_ls = [svd_node_inds.index(ind) for ind in indices]
        svd_rs = [i for i in range(len(tensor.indices)) if i not in svd_ls]
        lsize = np.prod([ind.size for ind in indices])
        perm = svd_ls + svd_rs
        # print(svd_ls, svd_rs, perm)
        # print(tensor.value.shape)
        tensor_val = tensor.value.transpose(perm).reshape(int(lsize), -1)
        _, s, _ = randomized_svd(tensor_val, max_rank)
        return s

    def _check_indices(
        self, indices: Sequence[Index], visited: Set[NodeName], node: NodeName
    ) -> Tuple[PartitionResult, Dict[Index, Sequence[Index]]]:
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
            visited = set()
            # print("postordering", n)
            res, results = self._check_indices(indices, visited, n)

            if res.code in (PartitionStatus.EXIST, PartitionStatus.OK):
                for i in indices:
                    for e, inds in results.items():
                        if i in inds:
                            lca_indices.append(e)
                            break

                assert len(lca_indices) == len(indices), (
                    "each index should correspond to one of the edges"
                )

                if res.code == PartitionStatus.OK:
                    res.lca_node = n

                res.lca_indices = list(set(lca_indices))
                return res

        raise ValueError(
            "Cannot find a node that realizes the partition", indices
        )

    def qr_along_path(self, path: Sequence[NodeName]) -> List[NodeName]:
        """Perform QR decomposition along a path."""
        nodes = []
        for i in range(len(path) - 1):
            q, r = self.qr_decompose_anchor(
                path[i], path[i + 1], FoldDir.IN_BOUND
            )
            nodes.extend([r, q])

        q, r = self.qr_decompose_anchor(path[-1], path[-2], FoldDir.OUT_BOUND)
        nodes.extend([r, q])

        return nodes

    def random_svals(
        self, node: NodeName, indices: Sequence[Index], max_rank: int = 100, rand: bool = True
    ) -> np.ndarray:
        tensor = self.node_tensor(node)
        svd_node_inds = tensor.indices
        svd_ls = self.svd_lefts(indices, node)
        svd_rs = [i for i in range(len(tensor.indices)) if i not in svd_ls]
        lsize = np.prod([svd_node_inds[lidx].size for lidx in svd_ls])
        perm = list(svd_ls) + svd_rs
        # print(svd_ls, svd_rs, perm)
        # print(tensor.value.shape)
        tensor_val = tensor.value.transpose(perm).reshape(int(lsize), -1)
        if rand:
            _, s, _ = randomized_svd(tensor_val, max_rank)
        else:
            s = np.linalg.svdvals(tensor_val)

        return s

    @profile
    def evaluate_cross(
        self, indices: Sequence[Index], values: np.ndarray
    ) -> np.ndarray:
        # contract the whole parts without the free indices
        # include_nodes, exclude_nodes = [], []
        # free_indices = self.free_indices()
        # for n in self.network.nodes:
        #     n_indices = self.node_tensor(n).indices
        #     if any(ind in free_indices for ind in n_indices):
        #         exclude_nodes.append(n)
        #     else:
        #         include_nodes.append(n)

        # subnet = nx.subgraph(self.network, include_nodes).copy()
        # top_net = HierarchicalTucker()
        # top_net.network = subnet
        # import time
        # top_start = time.time()
        # top_core = top_net.contract()
        # print("contract top time:", time.time() - top_start)

        # whole_net = TreeNetwork()
        # top_root = top_net.root()
        # whole_net.add_node(top_root, top_core)
        # for n in exclude_nodes:
        #     whole_net.add_node(n, self.node_tensor(n))
        #     whole_net.add_edge(n, top_root)

        # whole_start = time.time()
        # results = np.empty(values.shape[0])
        # whole_inds = whole_net.free_indices()
        # perm = [whole_inds.index(ind) for ind in indices]
        # values = values[:, perm]
        # whole_core = whole_net.contract().value
        whole_core = self.contract().value
        results = whole_core[*values.T]
        # for i, v in enumerate(values):
        #     results[i] = whole_net[v].value
        # print("contract whole time:", time.time() - whole_start)

        return results

    def replace_with(
        self,
        old_subnet: "TreeNetwork",
        new_subnet: "TreeNetwork",
        split_info: Optional[List[IndexOp]] = None,
    ):
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
        assert old_free_indices == new_free_indices, (
            "the old and new subnets should have the same set of free indices"
        )

        for n in new_subnet.network.nodes:
            tensor = new_subnet.node_tensor(n)
            self.add_node(n, tensor)
            for ind in tensor.indices:
                if ind not in new_free_indices:
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

    def longest_path(self) -> Sequence[NodeName]:
        """Get the longest path in the current tree."""
        u = list(self.network.nodes())[0]
        dist = nx.single_source_shortest_path_length(self.network, u)
        a = max(dist, key=dist.get)

        dist = nx.single_source_shortest_path_length(self.network, a)
        b = max(dist, key=dist.get)

        return nx.shortest_path(self.network, a, b)

    # def _nbr_to_tt(self, path: Sequence[NodeName], node: NodeName, )

    # def to_tt(self, delta = 0) -> "TensorTrain":
    #     """Convert an arbitrary tree to a tensor train format."""
    #     tt = TensorTrain()
    #     tt.network = copy.deepcopy(self.network)

    #     # choose the longest diameter in the tree throught two DFS
    #     path = tt.longest_path()
    #     # an over-approximation of number of split operations
    #     svd_cnts = len(tt.network.nodes) - len(path)
    #     visited = {}
    #     for n in path:
    #         while True:
    #             nbrs = list(tt.network.neighbors(n))
    #             if all(nbr in path for nbr in nbrs):
    #                 break

    #             tt.postorder_orthonormal(visited, None, n)
    #             for nbr in nbrs:
    #                 # if a neighbor is not on the path, merge and split
    #                 if nbr not in path:
    #                     tt.merge(n, nbr)
    #                     # split the 

    #         visited[n] = 2


    def replay_preprocess(
        self, actions: Sequence[Action]
    ):
        """Apply the given actions around the given ranks."""
        pass
        

class HierarchicalTucker(TreeNetwork):
    """Class for hierarchical tuckers."""

    @staticmethod
    def rand_ht(
        indices: List[Index], rank: int, child_each_level: int = 2
    ) -> "HierarchicalTucker":
        """Return a random hierarchical tucker."""
        ht = HierarchicalTucker()

        def build_child(
            pid: int, node_id: int, sub_indices: List[Index], rank: int = 1
        ) -> int:
            # print(node_id, sub_indices)
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

    def root(self):
        """Find the root node for a hierarchical tucker."""

        free_inds = self.free_indices()

        for n in self.network.nodes:
            node_inds = self.node_tensor(n).indices
            if len(node_inds) != 2:
                continue

            if any(ind in free_inds for ind in node_inds):
                continue

            return n

        # impossible path
        raise ValueError("Invalid hierarchical tucker, cannot find the root.")

    @profile
    def move_to(
        self, indices: Sequence[Index], delta: float = 0
    ) -> "HierarchicalTucker":
        """Swap the target indices to the same subtree."""
        # find the LCA and a path to move the indices
        root = self.root()
        dim_tree = self.dimension_tree(root)

        # starting from the root, find all nodes that
        # only contain the target indices
        frontier_nodes = dim_tree.highest_frontier(indices)

        if len(frontier_nodes) == 1:
            return self

        subnet = self
        while len(frontier_nodes) >= 2:
            # find a plan to move these nodes to one single subtree
            # but maintain the general HT structure
            # Basically we need to swap one of the node to be the sibling of the other
            # assert len(frontier_nodes) == 2, "only two nodes is supported"

            siblings = [dim_tree.sibling(n) for n in frontier_nodes]
            # check validity of siblings, make sure one is not the ancestor of the other
            if not siblings[0].is_ancestor(frontier_nodes[1]):
                # extract the list of nodes on the swapping path
                pnodes = dim_tree.path(
                    frontier_nodes[1].node,
                    siblings[0].node,
                )
            elif not siblings[1].is_ancestor(frontier_nodes[0]):
                # extract the list of nodes on the swapping path
                pnodes = dim_tree.path(
                    frontier_nodes[0].node,
                    siblings[1].node,
                )
            else:
                raise ValueError("no suitable way to move indices", indices)

            # print("moving path", [n.node for n in pnodes])

            # we temporary cut everything in a subtree and keep only the root
            include_nodes = []
            exclude_nodes = []
            for n in subnet.network.nodes:
                node = dim_tree.locate(n)
                assert node is not None, f"{n} does not exists"
                if pnodes[0].is_ancestor(node) or pnodes[-1].is_ancestor(node):
                    exclude_nodes.append(n)
                    continue

                include_nodes.append(n)

            new_net = HierarchicalTucker()
            new_net.network = nx.subgraph(subnet.network, include_nodes).copy()

            # go up to the node that is the parent of the other node
            for i in range(len(pnodes) - 1):
                # print("before swap")
                # print(self)
                # print("swapping", pnodes[i].node, pnodes[i+1].node)
                new_net.swap_nbr(pnodes[i].node, pnodes[i + 1].node)
                # print("after swap")
                # print(self)
                pnodes[i], pnodes[i + 1] = pnodes[i + 1], pnodes[i]

            for i in range(len(pnodes) - 2, 0, -1):
                # print("before swap")
                # print(self)
                # print("swapping", pnodes[i].node, pnodes[i - 1].node)
                new_net.swap_nbr(pnodes[i].node, pnodes[i - 1].node)
                # print("after swap")
                # print(self)
                pnodes[i - 1], pnodes[i] = pnodes[i], pnodes[i - 1]

            for n in exclude_nodes:
                new_net.add_node(n, subnet.node_tensor(n))

            for u, v in subnet.network.edges:
                if u in exclude_nodes or v in exclude_nodes:
                    new_net.add_edge(u, v)

            root = new_net.root()
            dim_tree = new_net.dimension_tree(root)

            # starting from the root, find all nodes that
            # only contain the target indices
            frontier_nodes = dim_tree.highest_frontier(indices)
            subnet = new_net

        return subnet

    # @profile
    # def svals(
    #     self,
    #     indices: Sequence[Index],
    #     max_rank: int = 100,
    #     orthonormal: Optional[NodeName] = None,
    #     delta: float = 0,
    # ) -> np.ndarray:
    #     """Compute the singular values for a hierarchical tucker."""
    #     # print(indices)
    #     net = self.move_to(indices, delta=delta)
    #     dim_tree = net.dimension_tree(net.root())
    #     frontier_nodes = dim_tree.highest_frontier(indices)
    #     # print("after move")
    #     # print(self)
    #     # print(frontier_nodes)
    #     assert len(frontier_nodes) == 1, "more than one node after swapping"

    #     svd_node = frontier_nodes[0]

    #     if orthonormal is None:
    #         net.orthonormalize(svd_node.node)

    #     tensor = net.node_tensor(svd_node.node)
    #     svd_node_inds = tensor.indices

    #     svd_inds = []
    #     for n in svd_node.down_info.nodes:
    #         svd_inds.extend(net.get_contraction_index(svd_node.node, n.node))

    #     free_inds = net.free_indices()
    #     for ind in svd_node_inds:
    #         if ind in free_inds:
    #             svd_inds.append(ind)

    #     tensor = net.node_tensor(svd_node.node)
    #     svd_node_inds = tensor.indices
    #     svd_ls = [svd_node_inds.index(ind) for ind in svd_inds]
    #     svd_rs = [i for i in range(len(tensor.indices)) if i not in svd_ls]
    #     lsize = np.prod([ind.size for ind in svd_inds])
    #     perm = svd_ls + svd_rs
    #     # print(svd_ls, svd_rs, perm)
    #     # print(tensor.value.shape)
    #     tensor_val = tensor.value.transpose(perm).reshape(int(lsize), -1)
    #     _, s, _ = randomized_svd(tensor_val, max_rank)
    #     return s


class FoldedTensorTrain(TreeNetwork):
    """Class for tensor trains with folded nodes"""

    def __init__(self, backbone_nodes: Optional[List[NodeName]] = None):
        super().__init__()
        if backbone_nodes is not None:
            self.backbone_nodes = backbone_nodes
        else:
            self.backbone_nodes = []

    @staticmethod
    def rand_ftt(index_groups: Sequence[Sequence[Index]]):
        """Construct a random structure for the specified groups of indices"""
        ftt = FoldedTensorTrain()

        # construct the backbone tensors
        prev = None
        for gid, ind_group in enumerate(index_groups):
            for iid, ind in enumerate(ind_group):
                indices = [ind]
                if iid == 0 and gid > 0:
                    # we pick the first one to be the backbone node
                    indices.append(Index(f"s_{gid - 1}_0_{gid}_0", 1))

                if iid == 0 and gid < len(index_groups) - 1:
                    indices.append(Index(f"s_{gid}_0_{gid + 1}_0", 1))

                if iid > 0:
                    indices.append(Index(f"s_{gid}_{iid - 1}_{iid}", 1))

                if iid < len(ind_group) - 1:
                    indices.append(Index(f"s_{gid}_{iid}_{iid + 1}", 1))

                size = [i.size for i in indices]
                val = np.random.random(size)
                node = f"G_{gid}_{iid}"
                ftt.add_node(node, Tensor(val, indices))

                if iid == 0:
                    ftt.backbone_nodes.append(node)

                if prev is not None:
                    ftt.add_edge(node, prev)

                prev = node

                if iid == len(ind_group) - 1:
                    prev = f"G_{gid}_0"

        return ftt

    def _should_keep_index_from_left(
        self,
        path: Sequence[NodeName],
        common_ind: Index,
        ind: Index,
        node: NodeName,
    ) -> bool:
        # an index should be kept if it is a connection between backbone nodes
        # and it is not the common index

        if ind == common_ind or ind in self.free_indices():
            return False

        for nbr in self.network.neighbors(node):
            nbr_inds = self.node_tensor(nbr).indices
            if ind in nbr_inds:
                return nbr in self.backbone_nodes

        return True

    def _should_keep_index_from_right(
        self,
        path: Sequence[NodeName],
        common_ind: Index,
        ind: Index,
        node: NodeName,
    ) -> bool:
        # an index should be kept if it is a connection between backbone nodes
        # and it is not the common index

        if ind == common_ind:
            return False

        for nbr in self.network.neighbors(node):
            nbr_inds = self.node_tensor(nbr).indices
            if ind in nbr_inds:
                return nbr not in self.backbone_nodes

        return True

    def move_to_end(self, backbone_targets: Sequence[NodeName]):
        """Move the targets to one of the ends of the FTT"""
        # swap the backbone targets to make them adjacent and at the end
        ends = []
        for n in self.backbone_nodes:
            num_edges = 0
            for m in self.backbone_nodes:
                if n != m and self.network.has_edge(n, m):
                    num_edges += 1

            if num_edges == 1:
                ends.append(n)

        assert len(ends) == 2, "invalid backbone structure"

        if len(backbone_targets) > 1:
            anchor = ends[0]
            best_dist = float("inf")
            for end in ends:
                if end in backbone_targets:
                    anchor = end
                    break

                total_dist = 0
                for n in backbone_targets:
                    total_dist += self.distance(n, anchor)

                if total_dist < best_dist:
                    best_dist = total_dist
                    anchor = end

            backbone_targets = sorted(
                backbone_targets, key=lambda x: self.distance(x, anchor)
            )
            for n in backbone_targets:
                if n != anchor:
                    path = nx.shortest_path(
                        self.network, source=n, target=anchor
                    )
                    self.swap_along_path(path, n)

    def svals(
        self, indices: Sequence[Index], max_rank: int = 100, rand: bool = True
    ) -> np.ndarray:
        """Compute the singular values for a folded tensor train.

        A folded tensor train has a backbone tensor train structure,
        with each node representing the merged nodes in the original tensor train.
        """
        # For a given set of indices,
        # they should all belong to a set of backbone nodes.
        # We might need to swap the backbone nodes to make them adjacent
        net = copy.deepcopy(self)

        # permute the backbone nodes to make the target indices adjacent
        backbone_targets = net.target_backbone_nodes(indices)
        assert len(backbone_targets) >= 1, (
            "no backbone nodes contain the target indices"
        )

        net.move_to_end(backbone_targets)

        # orthonormalize the backbone nodes
        visited = {}
        for n in net.network.nodes:
            if n not in net.backbone_nodes:
                visited[n] = 2

        res = net.partition_node(indices)
        net.postorder_orthonormal(visited, None, res.lca_node)
        return net.random_svals(res.lca_node, indices, max_rank, rand=rand)

    def target_backbone_nodes(
        self, indices: Sequence[Index]
    ) -> List[NodeName]:
        """Find the backbone nodes that contain the target indices."""
        target_backbone_nodes = set()
        # # cut the edges between backbone nodes and get the subtrees
        # tmp_net = self.network.copy()
        # for u in self.backbone_nodes:
        #     for v in self.backbone_nodes:
        #         if tmp_net.has_edge(u, v):
        #             tmp_net.remove_edge(u, v)

        # free_inds = self.free_indices()
        # for comps in nx.connected_components(tmp_net):
        #     subtree = nx.subgraph(self.network, comps)
        #     tree = TreeNetwork()
        #     tree.network = subtree
        #     tree_inds = set(tree.free_indices()) & set(free_inds)
        #     # check the relation with target indices
        #     target_inds = set(indices)
        #     overlap = tree_inds & target_inds
        #     all_contained = len(overlap) == len(tree_inds)
        #     none_contained = len(overlap) == 0
        #     assert all_contained or none_contained, (
        #         "Each subtree must either contain all or none of the target indices"
        #     )
        #     if all_contained:
        #         for n in self.backbone_nodes:
        #             if n in comps:
        #                 target_backbone_nodes.append(n)

        for ind in indices:
            ind_node = self.node_by_free_index(ind.name)
            if ind_node in self.backbone_nodes:
                target_backbone_nodes.add(ind_node)

        return list(target_backbone_nodes)

    def fold_nodes(
        self, node1: NodeName, node2: NodeName, fold_dir: FoldDir
    ) -> Tuple["FoldedTensorTrain", NodeName]:
        """Merge the indices on two nodes"""
        net = copy.deepcopy(self)
        # take the path between nodes[0] and nodes[1]
        path = nx.shortest_path(net.network, source=node1, target=node2)
        # fold the path by merging the pairs of nodes
        # for i in range(len(path) // 2):
        #     self.merge(path[i], path[-i - 1])
        assert any(n not in net.backbone_nodes for n in path), (
            "backbone nodes must not be folded"
        )

        # if the ranks on the path is very large, we merge these nodes
        ranks = net.ranks_along_path(path)
        if max(ranks) > 100:
            node = net.merge_along_path(path)
            net.backbone_nodes.append(node)
            return net, node

        # to avoid generating nodes of very large sizes, we split and then fold
        qrs = net.qr_along_path(path)
        # fold the nodes along the QR decomposition results
        for i in range(0, len(qrs) // 2):
            if qrs[i] in self.network.nodes:
                net.merge(qrs[i], qrs[-i - 1])
            else:
                net.merge(qrs[-i - 1], qrs[i])

        net.backbone_nodes.append(qrs[-1])
        return net, qrs[-1]

    def is_end(self, node):
        """Returns True if the given node is one of the ends"""
        inds = self.node_tensor(node).indices
        return len(inds) <= 3
    
    def _need_swap(self, operand_nodes: Sequence[NodeName]) -> bool:
        """Whether these operand nodes need to be moved to ends"""
        assert len(operand_nodes) == 2, (
            "current implementation supports only two operands"
        )

        # if the two nodes are not adjacent
        common_inds = self.get_contraction_index(
            operand_nodes[0], operand_nodes[1]
        )
        if not common_inds:
            return True

        # if none of the nodes on the end
        for operand in operand_nodes:
            if self.is_end(operand):
                return False

        return True

    def replay_preprocess(self, actions: Sequence[Action]):
        """Apply the given actions around the given ranks."""
        for ac in actions:
            operand_nodes = self.target_backbone_nodes(ac.indices)
            operand_nodes = list(set(operand_nodes))
            # check whether the operands are adjacent
            if len(operand_nodes) > 1 and self._need_swap(operand_nodes):
                self.move_to_end(operand_nodes)


class TensorTrain(TreeNetwork):
    """Class for tensor trains"""

    @property
    def backbone_nodes(self):
        return list(self.network.nodes)

    @staticmethod
    def rand_tt(
        indices: Sequence[Index], ranks: Optional[Sequence[int]] = None
    ) -> "TensorTrain":
        """Create a tensor train with the given indices."""
        net = TensorTrain()
        if ranks is None:
            ranks = [1] * (len(indices) - 1)
        for i, ind in enumerate(indices):
            if i == 0:
                core_indices = [ind, Index(f"s{i + 1}", ranks[i])]
            elif i == len(indices) - 1:
                core_indices = [Index(f"s{i}", ranks[i - 1]), ind]
            else:
                core_indices = [
                    Index(f"s{i}", ranks[i - 1]),
                    ind,
                    Index(f"s{i + 1}", ranks[i]),
                ]
            core_size = [ind.size for ind in core_indices]
            core_tensor = Tensor(np.random.random(core_size), core_indices)
            net.add_node(f"G{i}", core_tensor)

            if i != 0:
                net.add_edge(f"G{i}", f"G{i - 1}")

        return net

    @profile
    def merge_index(self, merge_op: IndexMerge) -> "HTensorTrain":
        """After index merge, we create a hierarchical tensor train to
        represent such virtual index merging.
        """

        # (1) take the merging indices and swap the nodes
        tt, _ = self.swap(merge_op.indices)
        # print("after swap")
        # print(tt)
        # print("========")

        # (2) create an abstract layers to mark several nodes are merged
        new_node = ""
        node_mapping = {}
        reverse_mapping = {}
        # (2.1) extract the swapped nodes and their corresponding indices
        for node in tt.network.nodes:
            tensor = tt.node_tensor(node)
            merge_indices = set(merge_op.indices)
            tensor_indices = set(tensor.indices)

            if merge_indices & tensor_indices:
                tmp_res = node_mapping.pop(new_node, [])
                new_node += "_" + node
                for n in tmp_res:
                    reverse_mapping.pop(n)
                node_mapping[new_node] = tmp_res + [node]
                for n in node_mapping[new_node]:
                    reverse_mapping[n] = new_node
            else:
                node_mapping[node] = [node]
                reverse_mapping[node] = node

        htt = HTensorTrain(tt, node_mapping, [merge_op])
        all_indices = Counter()
        # (2.2) create the fake nodes for the hierarchical tensor train
        for new_node, old_nodes in node_mapping.items():
            counter = Counter()
            for n in old_nodes:
                # extract the indices but remove the contraction indices
                counter.update(tt.node_tensor(n).indices)

            indices = [ind for ind, count in counter.items() if count == 1]
            all_indices.update(indices)
            if set(indices).issuperset(set(merge_op.indices)):
                # create a merged index
                new_name = "_".join(str(ind.name) for ind in merge_op.indices)
                new_size = int(np.prod([ind.size for ind in merge_op.indices]))
                new_ind = Index(new_name, new_size)
                remaining_indices = set(indices) - set(merge_op.indices)
                indices = [new_ind] + list(remaining_indices)

            shape = [0 for _ in indices]
            htt.add_node(new_node, Tensor(np.empty(shape), indices))

        for u, v in tt.network.edges:
            ind = tt.get_contraction_index(u, v)[0]
            if ind in all_indices:
                htt.add_edge(reverse_mapping[u], reverse_mapping[v])

        return htt

    def reorder(self, indices: Sequence[Index]) -> Self:
        """Swap the indices so that indices in all merge ops are adjacent"""
        net = copy.deepcopy(self)
        free_indices = self.free_indices()
        assert len(indices) == len(free_indices)

        left_end, right_end = net.end_nodes()
        ordered_nodes = nx.shortest_path(net.network, left_end, right_end)

        # align the indices from left to right or the reverse?

        # reorder the nodes according to the permuted merges
        for target_pos, ind in enumerate(indices):
            # swap i to current available position
            curr_node = net.node_by_free_index(ind.name)
            curr_pos = ordered_nodes.index(curr_node)
            path = nx.shortest_path(net.network, curr_node, ordered_nodes[target_pos])
            while curr_pos > target_pos:
                prev_pos = curr_pos - 1
                net.swap_nbr(path, ordered_nodes[curr_pos], ordered_nodes[prev_pos])
                (ordered_nodes[curr_pos], ordered_nodes[prev_pos]) = (
                    ordered_nodes[prev_pos],
                    ordered_nodes[curr_pos],
                )
                curr_pos -= 1

            target_pos += 1

        return net

    # @profile
    # def swap(
    #     self, ind_nodes: Sequence[NodeName], delta: float = 0
    # ) -> Tuple[NodeName, NodeName]:
    #     """Swap the indices so that the target indices are adjacent."""
    #     ind_nodes = list(set(ind_nodes))
    #     best_anchor, _ = self.best_anchor(ind_nodes)
    #     end_nodes = self.end_nodes()
    #     all_nodes = nx.shortest_path(self.network, end_nodes[0], end_nodes[1])
    #     # print("all nodes:", all_nodes)
    #     anchor_pos = all_nodes.index(best_anchor)
    #     # print("anchor node:", best_anchor, "at position", anchor_pos)
    #     left_range, right_range = 1, 1
    #     for node in ind_nodes:
    #         if node == best_anchor:
    #             continue

    #         # swap until the node within the range of the anchor node
    #         node_pos = all_nodes.index(node)
    #         # swap to the right
    #         while node_pos < anchor_pos - left_range:
    #             # print("left swapping", all_nodes[node_pos], all_nodes[node_pos + 1])
    #             self.swap_nbr(
    #                 all_nodes[node_pos], all_nodes[node_pos + 1], delta
    #             )
    #             all_nodes[node_pos], all_nodes[node_pos + 1] = (
    #                 all_nodes[node_pos + 1],
    #                 all_nodes[node_pos],
    #             )
    #             node_pos += 1

    #         while node_pos > anchor_pos + right_range:
    #             # print("right swapping", all_nodes[node_pos], all_nodes[node_pos - 1])
    #             self.swap_nbr(
    #                 all_nodes[node_pos], all_nodes[node_pos - 1], delta
    #             )
    #             all_nodes[node_pos], all_nodes[node_pos - 1] = (
    #                 all_nodes[node_pos - 1],
    #                 all_nodes[node_pos],
    #             )
    #             node_pos -= 1

    #         if node_pos < anchor_pos:
    #             left_range += 1
    #         else:
    #             right_range += 1

    #     return all_nodes[anchor_pos - left_range + 1], all_nodes[
    #         anchor_pos + right_range - 1
    #     ]

    def end_nodes(self) -> List[NodeName]:
        """Get the nodes at the two ends of a tensor train."""
        # end node is the one with one neighbor or whose value is 2D
        end_nodes: List[NodeName] = []
        for node in self.network.nodes:
            if len(list(self.network.neighbors(node))) == 1:
                end_nodes.append(node)

        return end_nodes

    @profile
    def swap_to_end(
        self,
        indices: Sequence[Index],
        orthonormal: Optional[NodeName] = None,
        delta: float = 0,
    ) -> Tuple[Self, NodeName]:
        """Swap the indices to the end of the tensor train.

        If the input tensor train is orthonormalized, we maintain the
        orthonormalization after the swap operation.
        """
        assert all(ind in self.free_indices() for ind in indices), (
            "all indices should be free"
        )
        # first, we need to find the end of the current tensor train
        end_nodes = self.end_nodes()
        if len(end_nodes) < 2:
            raise ValueError(
                "Cannot find enough end nodes, please check the tensor train structure."
            )

        if orthonormal is not None:
            best_end_node = orthonormal
        else:
            # check the distance between the given indices and the end nodes
            # pick the end node that is closest to all given indices
            min_dist = float("inf")
            best_end_node = None
            for end_node in end_nodes:
                all_dist = 0
                for node in self.network.nodes:
                    tensor = self.node_tensor(node)
                    if any(ind in tensor.indices for ind in indices):
                        # TODO: we can use a different score function here to evaluate the cost of swaps
                        dist = self.distance(end_node, node)
                        all_dist += dist

                if all_dist < min_dist:
                    best_end_node = end_node
                    min_dist = all_dist

            assert best_end_node is not None
            # print("Best end node:", best_end_node)

        # swap the given indices to the best end node
        net = copy.deepcopy(self)
        # collect the list of nodes start from the best_end_node
        nodes = net.linear_nodes(best_end_node)
        # print("list of nodes", nodes)
        index_nodes = [net.node_by_free_index(ind.name) for ind in indices]
        index_nodes = sorted(
            index_nodes, key=lambda x: self.distance(best_end_node, x)
        )
        for ind_idx, node in enumerate(index_nodes):
            if node == best_end_node:
                continue

            # print("moving", node)
            # swap the node with neighbors until it reaches the end node
            node_idx = nodes.index(node)
            # print("node index", node_idx, "ind_idx", ind_idx)
            while node_idx > ind_idx:
                # print("swapping", nodes[node_idx], nodes[node_idx - 1])
                net.swap_nbr(nodes[node_idx], nodes[node_idx - 1], delta)
                # print("after swap")
                # print(net)
                nodes[node_idx], nodes[node_idx - 1] = (
                    nodes[node_idx - 1],
                    nodes[node_idx],
                )
                node_idx -= 1

        # print("After swapping indices to the end node:")
        # print(net)
        if orthonormal is not None:
            # do the orthonormalization for nodes from 0 to len(indices)-1
            for nidx, n in enumerate(nodes[: len(indices) - 1]):
                lefts = []
                node_indices = net.node_tensor(n).indices
                if nidx == 0:
                    contract_indices = []
                else:
                    contract_indices = net.get_contraction_index(
                        n, nodes[nidx - 1]
                    )
                for ind in node_indices:
                    if ind in net.free_indices() or ind in contract_indices:
                        lefts.append(node_indices.index(ind))

                q, r = net.qr(n, lefts)
                net.merge(nodes[nidx + 1], r)
                nx.relabel_nodes(net.network, {q: n}, copy=False)

        return net, nodes[0]

    def __add__(self, other: Any) -> Self:
        """Add two tensor trains.

        New tensor has same names as self
        """
        if isinstance(other, TensorTrain):
            pass
        elif isinstance(other, TreeNetwork):
            return super().__add__(other)
        else:
            raise NotImplementedError

        assert nx.is_isomorphic(self.network, other.network)

        new_tens = copy.deepcopy(self)
        free_indices = self.free_indices()
        for _, (node1, node2) in enumerate(
            zip(self.network.nodes, other.network.nodes)
        ):
            logger.debug("Adding: Node %r and Node %r", node1, node2)

            tens1 = self.node_tensor(node1)
            tens2 = other.node_tensor(node2)
            new_tens.set_node_tensor(
                node1, tens1.concat_fill(tens2, free_indices)
            )

        return new_tens

    def __mul__(self, other: Self) -> Self:
        """Multiply two tensor trains.

        New tensor has same names as self
        """
        if isinstance(other, TensorTrain):
            pass
        elif isinstance(other, TreeNetwork):
            return super().__add__(other)
        else:
            raise NotImplementedError

        assert nx.is_isomorphic(self.network, other.network)

        new_tens = copy.deepcopy(self)
        free_indices = self.free_indices()
        for _, (node1, node2) in enumerate(
            zip(self.network.nodes, other.network.nodes)
        ):
            logger.debug("Multiplying: Node %r and Node %r", node1, node2)

            tens1 = self.node_tensor(node1)
            tens2 = other.node_tensor(node2)
            new_tens.set_node_tensor(node1, tens1.mult(tens2, free_indices))

        return new_tens

    def get_divisors(
        self, end_node: NodeName, node1: NodeName, node2: NodeName
    ) -> Tuple[NodeName, NodeName]:
        """Get the nodes on the boundary."""
        return node1, node2

    @profile
    def _scatter_to_ends(
        self, node_move_dsts: Dict[NodeName, NodeName]
    ) -> Tuple[Optional[NodeName], Optional[NodeName]]:
        """Scatter the nodes to target ends."""
        left_anchor = right_anchor = None
        left_dist = right_dist = -1
        left_end, right_end = self.end_nodes()
        self.node_status = {}

        # we should sort the nodes to avoid unnecessary moves
        for node, target_end in node_move_dsts.items():
            path = nx.shortest_path(self.network, node, target_end)
            self.swap_along_path(path, node)
            self.node_status[node] = NodeStatus.CONFIRMED

        for node, target_end in node_move_dsts.items():
            distance = min(self.distance(node, e) for e in self.end_nodes())

            if left_end == target_end and distance > left_dist:
                left_dist = distance
                left_anchor = node

            if right_end == target_end and distance > right_dist:
                right_dist = distance
                right_anchor = node

            self.node_status[node] = NodeStatus.ATTACHED

        # after scatter, we know the nodes now attached to the anchors
        if left_anchor is not None:
            self.node_status[left_anchor] = NodeStatus.CONFIRMED

        if right_anchor is not None:
            self.node_status[right_anchor] = NodeStatus.CONFIRMED

        return left_anchor, right_anchor

    def _fold_at_ends(
        self, node_move_dsts: Dict[NodeName, NodeName]
    ) -> Tuple[TreeNetwork, NodeName]:
        left_anchor, right_anchor = self._scatter_to_ends(node_move_dsts)

        if left_anchor is None:
            return self, right_anchor

        if right_anchor is None:
            return self, left_anchor

        return self.fold_nodes(left_anchor, right_anchor, FoldDir.OUT_BOUND)

    def _fold_at_middle(
        self, ind_nodes: Sequence[NodeName]
    ) -> Tuple[TreeNetwork, NodeName]:
        left_anchor, right_anchor = self.swap(ind_nodes)
        end_nodes = self.end_nodes()

        if left_anchor in end_nodes:
            return self, right_anchor

        if right_anchor in end_nodes:
            return self, left_anchor

        return self.fold_nodes(left_anchor, right_anchor, FoldDir.IN_BOUND)

    def _dist_to_ends(
        self, ind_nodes: Sequence[NodeName]
    ) -> Tuple[int, Dict[NodeName, NodeName]]:
        dist_to_ends = 0
        end_nodes = self.end_nodes()
        node_move_dsts = {}
        l_cnt, r_cnt = 0, 0
        for ind_node in ind_nodes:
            l_dist = self.distance(ind_node, end_nodes[0])
            r_dist = self.distance(ind_node, end_nodes[1])
            if l_dist < r_dist:
                dist_to_ends += l_dist - l_cnt
                node_move_dsts[ind_node] = end_nodes[0]
                l_cnt += 1
            else:
                dist_to_ends += r_dist - r_cnt
                node_move_dsts[ind_node] = end_nodes[1]
                r_cnt += 1

        return dist_to_ends, node_move_dsts

    @profile
    def fold(self, nodes: Sequence[NodeName]) -> Tuple[TreeNetwork, NodeName]:
        """Fold the given nodes into one subtree of the results."""
        # depending on the location of the indices
        # we adopt different folding strategies
        # case I: if the indices are closer to two ends,
        # we move them to adjacent to two ends and fold the boundaries
        # case II: if the indices are closer to the center,
        # we move them to the center and fold the center nodes

        # let's fix this for trees later, first implement the idea only for TT
        # anchor_node = self.node_by_free_index(indices[0].name)
        # for ind in indices[1:]:
        #     node = self.node_by_free_index(ind.name)
        #     if node != anchor_node:
        #         anchor_node = self.fold_node_pair([anchor_node, node], FoldDir.IN_BOUND)

        # now we find the left and right anchor positions
        # ind_nodes = [self.node_by_free_index(ind.name) for ind in indices]
        # for src, dst in itertools.combinations(ind_nodes, 2):
        #     path = nx.shortest_path(self.network, src, dst)
        #     if len(path) == len(ind_nodes):
        #         break

        # left_anchor, right_anchor = path[0], path[-1]

        # alternatively we may scatter the indices to both ends
        _, min_dist = self.best_anchor(nodes)
        dist_to_ends, node_move_dsts = self._dist_to_ends(nodes)

        if len(nodes) == 1:
            return self, nodes[0]

        if dist_to_ends <= min_dist:
            # case I: move indices to both ends
            return self._fold_at_ends(node_move_dsts)

        return self._fold_at_middle(nodes)

    def fold_nodes(
        self,
        node1: NodeName,
        node2: NodeName,
        fold_dir: Optional[FoldDir] = None,
    ) -> Tuple[FoldedTensorTrain, NodeName]:
        """Merge the indices on two nodes"""
        # take the path between nodes[0] and nodes[1]
        path = nx.shortest_path(self.network, source=node1, target=node2)
        # if fold_dir is None:
        #     # fold the path by merging the pairs of nodes
        #     for i in range(len(path) // 2):
        #         self.merge(path[i], path[-i - 1])

        #     return self, path[0]

        net = FoldedTensorTrain()
        net.network = copy.deepcopy(self.network)

        # if the ranks on the path is very large, we merge these nodes
        ranks = net.ranks_along_path(path)
        if max(ranks) > 100:
            node = net.merge_along_path(path)
            net.backbone_nodes.append(node)
            return net, node

        # to avoid generating nodes of very large sizes, we split and then fold
        qrs = net.qr_along_path(path)
        # fold the nodes along the QR decomposition results
        for i in range(0, len(qrs) // 2):
            if qrs[i] in self.network.nodes:
                net.merge(qrs[i], qrs[-i - 1])
            else:
                net.merge(qrs[-i - 1], qrs[i])

        net.backbone_nodes.append(qrs[-1])
        return net, qrs[-1]

    @profile
    def svals(
        self,
        indices: Sequence[Index],
        max_rank: int = 100,
        orthonormal: Optional[NodeName] = None,
        delta: float = 0,
    ) -> np.ndarray:
        """Compute the singular values for a hierarchical tucker."""
        ind_nodes = [self.node_by_free_index(ind.name) for ind in indices]
        # svd_node = self.fold_hierarchical(ind_nodes)
        net, svd_node = self.fold_hierarchical(ind_nodes)
        # temporarily break one of the edges in the loop
        if orthonormal is None:
            net.orthonormalize(svd_node)

        return net.random_svals(svd_node, indices, max_rank)

    # @profile
    # def svals(
    #     self,
    #     indices: Sequence[Index],
    #     max_rank: int = 100,
    #     orthonormal: Optional[NodeName] = None,
    #     delta: float = 0,
    # ) -> np.ndarray:
    #     """Compute the singular values for a tensor train."""
    #     # move the indices to one side of the tensor train
    #     net, end_node = self.swap_to_end(
    #         indices, orthonormal=orthonormal, delta=delta
    #     )
    #     # print(net)
    #     # print("best end node", best_end_node)
    #     # print("target neighbor", target_nbr)
    #     nodes = net.linear_nodes(end_node)
    #     node1, node2 = net.get_divisors(
    #         end_node, nodes[len(indices) - 1], nodes[len(indices)]
    #     )
    #     return net.svals_nbr(
    #         node1,
    #         node2,
    #         max_rank=max_rank,
    #         orthonormal=orthonormal is not None,
    #     )

    @profile
    def svals_at(
        self,
        node: NodeName,
        indices: Sequence[Index],
        max_rank: int = 100,
        with_orthonormal: bool = True,
    ) -> np.ndarray:
        """Compute the singular values for a tensor train at a given node."""
        if with_orthonormal:
            self.orthonormalize(node)

        tensor = self.node_tensor(node)
        lefts = []
        for i, ind in enumerate(tensor.indices):
            if ind in indices:
                lefts.append(i)

        perm = lefts + [
            i for i in range(len(tensor.indices)) if i not in lefts
        ]
        left_size = np.prod([ind.size for ind in indices])
        tensor_val = tensor.value.transpose(perm).reshape(left_size, -1)

        if max_rank < 10:
            return delta_svd(tensor_val, delta=0, compute_uv=False).s

        _, s, _ = randomized_svd(tensor_val, max_rank)
        return s

    @profile
    def svals_by_merge(
        self, indices: Sequence[Index], max_rank: int = 100, rand: bool = True
    ) -> np.ndarray:
        """Compute the singular values for a tensor train."""
        ind_nodes = [self.node_by_free_index(ind.name) for ind in indices]
        nodes = self.swap(ind_nodes)
        if len(nodes) > 1:
            path = nx.shortest_path(self.network, nodes[0], nodes[1])
            for n in path:
                if n != nodes[0]:
                    self.merge(nodes[0], n)

            n = nodes[0]
        else:
            n = self.node_by_free_index(indices[0].name)

        self.orthonormalize(n)

        node_indices = self.node_tensor(n).indices
        lefts = []
        for i, ind in enumerate(node_indices):
            if ind in indices:
                lefts.append(i)

        perm = lefts + [i for i in range(len(node_indices)) if i not in lefts]
        left_size = np.prod([ind.size for ind in indices])
        tensor_val = self.node_tensor(n).value
        tensor_val = tensor_val.transpose(perm).reshape(left_size, -1)
        if rand:
            _, s, _ = randomized_svd(tensor_val, max_rank)
        else:
            s = np.linalg.svdvals(tensor_val)
        return s
        # (_, s, _), _ = net.svd(n, lefts, SVDConfig(delta=0, compute_uv=False))
        # return np.diag(net.node_tensor(s).value)

    @profile
    def svals_by_cross(
        self, indices: Sequence[Index], max_rank: int = 100, eps: float = 0.1
    ) -> np.ndarray:
        """Compute the singular values for a tensor train by cross approximation."""
        # permute the indices so that the target indices are at the beginning
        free_inds = self.free_indices()
        target_inds = list(indices)[:]
        for ind in free_inds:
            if ind not in target_inds:
                target_inds.append(ind)

        inds = [ind.with_new_rng(range(ind.size)) for ind in target_inds]
        tt = self.rand_tt(inds)
        func = FuncTensorNetwork(inds, self)
        cross(
            func,
            tt,
            tt.end_nodes()[0],
            eps=eps,
            max_iters=max_rank - 1,
            kickrank=1,
        )

        # find the correct node to split
        res = tt.partition_node(indices)
        return tt.svals_at(res.lca_node, res.lca_indices, max_rank=max_rank)

    # def svals_by_fold(
    #     self, indices: Sequence[Index], max_rank: int = 100
    # ) -> np.ndarray:
    #     """Compute the singular values for a tensor train."""
    #     ind_nodes = [self.node_by_free_index(ind.name) for ind in indices]
    #     ind_nodes = list(set(ind_nodes))
    #     assert len(ind_nodes) == 2, "svals_by_fold only suppport two nodes"
    #     net, _ = self.fold_nodes(ind_nodes[0], ind_nodes[1])
    #     res = net.partition_node(indices)
    #     return net.random_svals(res.lca_node, indices, max_rank)

    @profile
    def reorder_by_cross(
        self, indices: Sequence[Index], eps: float = 0.1, max_iters: int=50
    ) -> "TensorTrain":
        """Reorder the tensor train so that the given indices are adjacent using cross approximation."""
        assert all(ind in self.free_indices() for ind in indices), (
            "all indices should be free"
        )
        # data = self.contract()
        indices = [ind.with_new_rng(range(ind.size)) for ind in indices]
        tt = self.rand_tt(indices)
        tmp_net = copy.deepcopy(self)
        tmp_net.compress()
        func = FuncTensorNetwork(list(indices), tmp_net)

        # perm = [data.indices.index(ind) for ind in indices]
        # func = FuncData(indices, data.permute(perm).value)
        # max_rank = max(ind.size for ind in self.inner_indices())
        # adaptive_kickrank = max(5, max_rank)
        # print("adaptive kickrank:", adaptive_kickrank)
        res = cross(
            func, tt, tt.end_nodes()[0], eps=eps, max_iters=max_iters, kickrank=10
        )
        if res.ranks_and_errors[-1][-1] < eps:
            return tt
        
        return self.reorder(indices)

    @staticmethod
    @profile
    def tt_svd(
        data: np.ndarray, indices: Sequence[Index], eps: float = 0.1
    ) -> "TensorTrain":
        tt = TensorTrain()
        norm = np.linalg.norm(data)
        tt.add_node("G0", Tensor(data, list(indices)))
        l, r = None, "G0"

        for ind in indices[:-1]:
            left_inds = tt.node_tensor(r).indices
            lefts = [left_inds.index(ind)]
            if l is not None:
                lefts.append(
                    left_inds.index(tt.get_contraction_index(l, r)[0])
                )
            [l, s, r], _ = tt.svd(
                r,
                lefts,
                SVDConfig(delta=norm * eps / ((len(indices) - 1) ** 0.5)),
            )
            tt.merge(r, s)

        return tt

    @profile
    def reorder_by_svd(
        self, indices: Sequence[Index], eps: float = 0
    ) -> "TensorTrain":
        """Reorder the indices into the target indices through SVD"""
        assert all(ind in self.free_indices() for ind in indices), (
            "all indices should be free"
        )
        data = self.contract()
        indices = [ind.with_new_rng(range(ind.size)) for ind in indices]

        perm = [data.indices.index(ind) for ind in indices]

        return TensorTrain.tt_svd(data.permute(perm).value, indices, eps)

    @profile
    def svals_nbr(
        self,
        node1: NodeName,
        node2: NodeName,
        max_rank: int = 100,
        orthonormal: bool = False,
    ) -> np.ndarray:
        """Compute the singular values for two neighbor nodes."""
        if not orthonormal:
            self.orthonormalize(node1)
        tensor = self.node_tensor(node1)
        left_ind = self.get_contraction_index(node1, node2)[0]
        left = tensor.indices.index(left_ind)
        perm = [left] + [i for i in range(len(tensor.indices)) if i != left]
        tensor_val = tensor.value.transpose(perm).reshape(left_ind.size, -1)
        _, s, _ = randomized_svd(tensor_val, max_rank)
        return s
        # (_, s, _), _ = self.svd(node1, [left], SVDConfig(delta=0, compute_uv=False))
        # return np.diag(self.node_tensor(s).value)

    def flatten(self):
        return self

    def ends(self):
        """Compute the end nodes for the current tensor train."""
        res = []
        for n in self.network.nodes:
            if len(self.node_tensor(n).indices) == 2:
                res.append(n)

        return res

    @profile
    def svals_all(self) -> Dict[Sequence[Index], np.ndarray]:
        """Compute singular values for the given index combos."""
        assert len(self.network.nodes) <= 4, "expecting only four indices"

        ends = self.ends()

        # right orthogonalize
        self.orthonormalize(ends[0])

        nodes = self.linear_nodes(ends[0])
        # now we can compute all nodes from one end to the other
        result = {}
        prior_frees = []
        for ni, n in enumerate(nodes):
            # find the free index
            lefts = []
            frees = []
            node_indices = self.node_tensor(n).indices
            for indi, ind in enumerate(node_indices):
                if ind in self.free_indices():
                    frees.append(indi)
                    prior_frees.append(ind)

                if ni != 0 and ind in self.get_contraction_index(
                    n, nodes[ni - 1]
                ):
                    lefts.append(indi)

            # singular values for the current index
            s, _ = self.node_tensor(n).svd(frees, delta=0, compute_uv=False)
            node_frees = [node_indices[i] for i in frees]
            result[tuple(node_frees)] = np.diag(s[0].value)

            if ni > 0:
                # singular values for binary partitions
                indices = sorted([prior_frees[ni - 1], prior_frees[ni]])
                tmp_net = copy.deepcopy(self)
                tmp_net.merge(n, nodes[ni - 1])
                tmp_indices = tmp_net.node_tensor(n).indices
                tmp_lefts = [tmp_indices.index(ind) for ind in indices]
                (_, s, _), _ = tmp_net.svd(n, tmp_lefts, SVDConfig(delta=0))
                result[tuple(indices)] = np.diag(tmp_net.node_tensor(s).value)

            if ni < len(nodes) - 1:
                _, r = self.qr(n, list(set(lefts + frees)))
                self.merge(nodes[ni + 1], r)

        return result


class HTensorTrain(TensorTrain):
    """Hierarchical tensor trains."""

    def __init__(
        self,
        tt: TensorTrain,
        node_mapping: Dict[NodeName, List[NodeName]],
        merge_ops: List[IndexMerge],
    ):
        super().__init__()
        self.tt = tt
        self.node_mapping = node_mapping
        self.merge_ops = merge_ops

    @profile
    def swap_nbr(self, node1: NodeName, node2: NodeName):
        """Swap two neighbor nodes in a hierarchical tensor train."""

        # For hierarchical tensor trains, we don't perform svd but
        # delegate this operation until we reach the lowest hierarchy.
        common_ind = self.get_contraction_index(node1, node2)[0]
        node_indices = []
        for ind in self.node_tensor(node1).indices:
            if ind == common_ind or ind in self.free_indices():
                continue

            node_indices.append(ind)

        for ind in self.node_tensor(node2).indices:
            if ind in self.free_indices():
                node_indices.append(ind)

        name = self.merge(node1, node2, compute_data=False)
        new_indices = self.node_tensor(name).indices
        lefts = [new_indices.index(ind) for ind in node_indices]
        # we need to record the singular values in tensor train
        (u, s, v), _ = self.svd(name, lefts, SVDConfig(compute_data=False))
        v = self.merge(v, s, compute_data=False)
        nx.relabel_nodes(self.network, {u: node2, v: node1}, copy=False)

        # We need to recursively swap the nodes in the underlying TTs
        # These nodes should be moved in a chunk
        # So before moving the nodes, we need to sort the underlying nodes.
        tt_tree = self.tt.dimension_tree(list(self.tt.network.nodes)[0])
        n1_nodes = self.node_mapping[node1]
        n2_nodes = self.node_mapping[node2]
        n1_nodes_sorted = sorted(
            n1_nodes, key=lambda n: tt_tree.distance(n, n2_nodes[0])
        )
        n2_nodes_sorted = sorted(
            n2_nodes, key=lambda n: tt_tree.distance(n, n1_nodes[0])
        )

        # print("sorted n1", n1_nodes_sorted)
        # print("sorted n2", n2_nodes_sorted)
        for n2 in n2_nodes_sorted:
            for n1 in n1_nodes_sorted:
                # print("swapping", n2, n1, "in the subnet")
                # print(self.tt)
                self.tt.swap_nbr(n2, n1)

    def get_divisors(
        self, end_node: NodeName, node1: NodeName, node2: NodeName
    ) -> Tuple[NodeName, NodeName]:
        """Get the nodes on the boundary of the hierarchical tensor train."""
        tt_end = None
        for n in self.node_mapping[end_node]:
            if self.tt.node_tensor(n).value.ndim == 2:
                tt_end = n
                break

        if tt_end is None:
            raise ValueError("Cannot find the end node in the underlying tt.")

        tt_nodes = self.tt.linear_nodes(tt_end)

        tt_node1, tt_node2 = None, None
        idx1, idx2 = -1, float("inf")
        for n in self.node_mapping[node1]:
            if idx1 < tt_nodes.index(n):
                tt_node1 = n
                idx1 = tt_nodes.index(n)

        for n in self.node_mapping[node2]:
            if idx2 > tt_nodes.index(n):
                tt_node2 = n
                idx2 = tt_nodes.index(n)

        if tt_node1 is None or tt_node2 is None:
            raise ValueError(
                "Cannot find the nodes in the hierarchical tensor train."
            )

        return self.tt.get_divisors(tt_end, tt_node1, tt_node2)

    @profile
    def svals_nbr(
        self, node1: NodeName, node2: NodeName, max_rank: int = 100
    ) -> np.ndarray:
        """Compute the singular values for two neighbor nodes in a hierarchical tensor train."""
        return self.tt.svals_nbr(node1, node2, max_rank=max_rank)

    # def merge(self, name1: NodeName, name2: NodeName, compute_data:bool = False) -> NodeName:
    #     """Merge the group sets corresponding to these nodes."""
    #     if name1 not in self.node_mapping:
    #         self.node_mapping[name1] = [name1]

    #     self.node_mapping[name1] = self.node_mapping.get(name1, [name1]) + self.node_mapping.get(name2, [name2])
    #     self.node_mapping.pop(name2, None)

    #     super().merge(name1, name2, compute_data=False)
    #     return name1

    def flatten(self):
        """Flatten the hierarchical structure into a normal tensor train
        by merging the grouped nodes.
        """
        node_mapping = {}
        merge_ops = self.merge_ops
        parent_htt = self
        htt = self.tt
        while isinstance(htt, HTensorTrain):
            merge_ops = htt.merge_ops + merge_ops
            # pass the node mapping down
            for k, ns in parent_htt.node_mapping.items():
                new_ns = []
                for n in ns:
                    new_ns.extend(htt.node_mapping.get(n, [n]))

                node_mapping[k] = new_ns

            parent_htt = htt
            htt = htt.tt

        net = TreeNetwork()
        net.network = copy.deepcopy(htt.network)
        nodes = htt.linear_nodes()
        # merge the underlying node groups
        for k, ns in node_mapping.items():
            # we need to keep the merged index also merged
            # sort the nodes by node positions
            ns = sorted(ns, key=lambda x: nodes.index(x))
            for n in ns[1:]:
                # print("merging", net.node_tensor(ns[0]).indices, net.node_tensor(n).indices)
                net.merge(ns[0], n)

            # how do we build the connection between the nodes and indices?

        for merge_op in merge_ops:
            net = net.merge_index(merge_op)
            # k_tensor = res_tt.node_tensor(k)
            # shape = [ind.size for ind in k_tensor.indices]
            # k_val = htt.node_tensor(ns[0]).value.reshape(shape)
            # res_tt.node_tensor(k).update_val_size(k_val)

        res_net = TensorTrain()
        res_net.network = net.network
        return res_net

    @profile
    def svals(self, indices: Sequence[Index], delta: float = 0) -> np.ndarray:
        """Compute the singular values for a tensor train."""
        # trigger the svals for underlying tensor train
        # convert indices to the underlying indices
        underlying_indices = []
        for merge_op in self.merge_ops:
            for ind in indices:
                if ind == merge_op.result:
                    underlying_indices.extend(merge_op.indices)
                else:
                    underlying_indices.append(ind)

        return self.tt.svals(underlying_indices, delta)


def vector(
    name: Union[str, int], index: Index, value: np.ndarray
) -> "TensorNetwork":
    """Convert a vector to a tensor network."""
    vec = TensorNetwork()
    vec.add_node(name, Tensor(value, [index]))
    return vec


def rand_tt(indices: List[Index], ranks: List[int]) -> TensorTrain:
    """Return a random tt."""

    dim = len(indices)
    assert len(ranks) + 1 == len(indices)

    tt = TensorTrain()

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


def tt_rank1(indices: List[Index], vals: List[np.ndarray]) -> TensorTrain:
    """Return a random rank 1 TT tensor."""

    dim = len(indices)

    tt = TensorTrain()

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


def tt_separable(indices: List[Index], funcs: List[np.ndarray]) -> TensorTrain:
    """Rank 2 function formed by sums of functions of individual dimensions."""

    dim = len(indices)

    tt = TensorTrain()
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


def tt_right_orth(tn: TensorTrain, node: int) -> TensorTrain:
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
    eigl, vl = np.linalg.eigh(gl)
    eigr, vr = np.linalg.eigh(gr)
    eigl = np.abs(eigl)
    eigr = np.abs(eigr)

    eigl12 = np.sqrt(eigl)
    eigr12 = np.sqrt(eigr)

    threshold = np.ceil(np.log10(np.max(eigl12) * 1e-8))
    eigl12 = np.round(eigl12, min(-int(threshold), 16))
    threshold = np.ceil(np.log10(np.max(eigr12) * 1e-8))
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


def tt_gramsvd_round(tn: TensorTrain, eps: float) -> TensorTrain:
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


def tt_svd_round(tn: TensorTrain, eps: float) -> TensorTrain:
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
    # delta = eps / np.sqrt(dim - 1) * norm2
    # cores = []
    # for node, data in tn.network.nodes(data=True):
    #     cores.append(node)

    # print("DIM = ", dim)
    out = tt_right_orth(tn, dim - 1)
    for jj in range(dim - 2, 0, -1):
        # print(f"orthogonalizing core {cores[jj]}")
        out = tt_right_orth(out, jj)

    # print("ON FORWARD SWEEP")
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
    factors_list: list[TensorTrain],
    eps: float = 1e-14,
) -> TensorTrain:
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
        self, y: Union[TensorTrain, List[TensorTrain]], target_ranks: List
    ):
        self.y = y
        self.target_ranks = target_ranks

        if isinstance(y, List) and isinstance(y[0], TensorTrain):
            self.ns = len(y)
            self.d = y[0].network.number_of_nodes()

        elif isinstance(y, TensorTrain):
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

    def rand_then_orth(self) -> TensorTrain:
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

    def rto_rounding_ttsum(self) -> TensorTrain:
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


def tt_randomized_round(y: TensorTrain, target_ranks: List) -> TensorNetwork:
    """Executes randomized rounding for a TT TensorNetwork"""

    rand_setup = TTRandRound(y, target_ranks)
    return rand_setup.rand_then_orth()


def tt_sum_randomized_round(
    y: List[TensorTrain], target_ranks: List
) -> TensorTrain:
    """Executes randomized rounding for a TT TensorNetwork"""

    rand_setup = TTRandRound(y, target_ranks)
    return rand_setup.rto_rounding_ttsum()


def tt_rand_precond_svd_round(
    tn: Union[TensorTrain, List[TensorTrain]],
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
    tt_in: List[TensorTrain],
) -> TensorTrain:
    """Sum a set of tensor trains."""

    tt_out = TensorTrain()
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

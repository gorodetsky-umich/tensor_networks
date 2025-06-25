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
    Callable,
    Dict,
    List,
    Optional,
    Self,
    Set,
    Tuple,
    Union,
    Literal,
)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import opt_einsum as oe

from .cross.cross import *
from .types import (
    Index,
    IndexMerge,
    IndexName,
    IndexSplit,
    NodeName,
    SVDConfig,
    DimTreeNode,
)
from .utils import delta_svd

logging.basicConfig(level=logging.DEBUG)
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

        new_val = np.einsum(estr, self.value, other.value)
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

        new_val = np.einsum(estr, self.value, other.value)
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

    def _index_to_args(
        self,
        selected_indices: Sequence[Index],
        index_map: Dict[Index, List[Index]],
        restrictions: Dict[Index, np.ndarray],
    ) -> Tuple[np.ndarray, List[Index]]:
        """
        index_map: from local indices to global free indices
        restrictions: from local indices to their restricted global choices
        """
        sizes = [i.size for i in selected_indices]
        # print(sizes, restrictions)
        local_indices = np.arange(np.prod(sizes))
        args = np.unravel_index(local_indices, sizes)
        assert len(args) == len(sizes), (
            f"unraveled index does not match the given dimensions, "
            f"expected {len(sizes)}, but get {len(args)}"
        )
        global_args: List[np.ndarray] = []
        order: List[Index] = []

        for iarg, arg in enumerate(args):
            # print(arg.shape)
            ind = selected_indices[iarg]
            global_indices = index_map.get(ind, [ind])
            order.extend(global_indices)

            # restrict choices of indices if they have been chosen
            if ind in restrictions:
                # print("looking for restrictions inside", restrictions[ind])
                restricted_args = restrictions[ind][arg]
                # print("expanding", arg, "into", restricted_args)
                assert len(global_indices) == restricted_args.shape[1], (
                    f"expected {global_indices}, "
                    f"but get {restricted_args.shape}"
                )
                global_args.append(restricted_args.squeeze())
            else:
                # map internal indices to global indices
                global_sizes = [idx.size for idx in global_indices]
                unravel_arg = np.unravel_index(arg, global_sizes)
                global_args.extend(unravel_arg)

        # print([args.shape for args in global_args])
        return np.stack(global_args, axis=-1), order

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
            for ind in self.indices:
                if ind in merged_indices:
                    merged_sizes.append(ind.size)
                    merged_names.append(ind.name)
                else:
                    unmerged_indices.append(ind)
                    unmerged_names.append(ind.name)
                    unmerged_sizes.append(ind.size)

            perm_names = list(itertools.chain(merged_names, unmerged_names))
            new_tensor = self.permute_by_name(perm_names)
            new_data = new_tensor.value.reshape(-1, *unmerged_sizes)

            new_size = math.prod(merged_sizes)
            new_index = Index(new_ind, new_size)
            return Tensor(new_data, [new_index] + unmerged_indices)

        return self

    def split_indices(self, split_into: IndexSplit) -> "Tensor":
        """Split specified indices into smaller ones"""
        new_indices = []
        next_index = 0
        for ind in self.indices:
            if ind == split_into.index:
                for sz in split_into.shape:
                    new_ind = f"_fresh_index_{next_index}"
                    new_indices.append(Index(new_ind, sz))
                    next_index += 1
            else:
                new_indices.append(ind)

        new_data = self.value.reshape([ind.size for ind in new_indices])
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
        free_indices = sorted([i for i, v in icount.items() if v == 1])
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
        free_indices = sorted([i for i, v in all_indices.items() if v == 1])

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
        self, other: "TensorNetwork", rename: Tuple[str, str] = ("G", "H")
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
            if index in free_indices:
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
            if index in free_indices:
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

    def fresh_index(self) -> str:
        """Generate an index that does not appear in the current network."""
        all_indices = [ind.name for ind in self.all_indices().keys()]
        i = self.get_next_id()
        while f"s_{i}" in all_indices:
            i = self.get_next_id()

        return f"s_{i}"

    def fresh_node(self) -> NodeName:
        """Generate a node name that does not appear in the current network."""
        i = 0
        node = f"n{i}"
        while node in self.network.nodes:
            i += 1
            node = f"n{i}"

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
            rl = Index("r_split_l", -1)
            rr = Index("r_split_r", -1)
            u = Tensor(np.empty(0), [x.indices[i] for i in lefts] + [rl])
            v = Tensor(np.empty(0), [rr] + [x.indices[i] for i in rights])
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
        if not self.network.has_edge(name1, name2):
            raise RuntimeError(
                f"Cannot merge nodes that are not adjacent: {name1}, {name2}"
            )

        t1 = self.node_tensor(name1)
        t2 = self.node_tensor(name2)

        if compute_data:
            result = t1.contract(t2)
        else:
            l_inds = [ind for ind in t1.indices if ind not in t2.indices]
            r_inds = [ind for ind in t2.indices if ind not in t1.indices]
            result = Tensor(np.array([]), l_inds + r_inds)

        n2_nbrs = list(self.network.neighbors(name2))
        self.network.remove_node(name2)
        self.set_node_tensor(name1, result)
        for n in n2_nbrs:
            if n != name1:
                self.add_edge(name1, n)

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

    def split_index(self, split_op: IndexSplit) -> Optional[NodeName]:
        """Split free indices into smaller parts"""
        for n in self.network.nodes:
            n = typing.cast(NodeName, n)
            tensor = self.node_tensor(n)
            tensor = tensor.split_indices(split_op)
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
                        new_ind = self.fresh_index()
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

    def merge_index(self, merge_op: IndexMerge) -> None:
        """Merge specified free indices"""
        for n in self.network.nodes:
            tensor = self.node_tensor(n)

            new_ind: IndexName = ""
            if merge_op.result is None:
                new_ind = self.fresh_index()
            else:
                new_ind = merge_op.result.name
            tensor = tensor.merge_indices(merge_op.indices, new_ind)

            self.set_node_tensor(n, tensor)

    def replace_with(
        self,
        name: NodeName,
        subnet: "TensorNetwork",
        split_info: Optional[List[Union[IndexSplit, IndexMerge]]] = None,
    ) -> None:
        """Replace a node with a sub-tensor network."""
        if name not in self.network.nodes:
            return

        ind_names = [ind.name for ind in self.all_indices()]
        subnet_free_indices = subnet.free_indices()
        index_subst: Dict[IndexName, IndexName] = {}
        node_subst = {}
        for m in subnet.network.nodes:
            m_tensor = subnet.node_tensor(m)
            if m in self.network.nodes:
                # rename the node
                new_m = self.fresh_node()
                node_subst[m] = new_m
                m = new_m
            else:
                node_subst[m] = m

            m_indices = []
            for mind in m_tensor.indices:
                mind_name = mind.name
                if mind in subnet_free_indices or mind_name not in ind_names:
                    m_indices.append(mind)
                elif mind.name in index_subst:
                    mind_name = index_subst[mind.name]
                    m_indices.append(Index(mind_name, mind.size))
                else:
                    ind_name = self.fresh_index()
                    index_subst[mind_name] = ind_name
                    ind_names.append(ind_name)
                    m_indices.append(Index(ind_name, mind.size))

            self.add_node(m, Tensor(m_tensor.value, m_indices))
            m_inds_set = set(m_indices)
            for nbr in self.network.neighbors(name):
                nbr_indices = set(self.node_tensor(nbr).indices)
                if len(nbr_indices.intersection(m_inds_set)) != 0:
                    self.add_edge(m, nbr)
                elif split_info is not None:
                    for split_op in split_info:
                        if (
                            isinstance(split_op, IndexSplit)
                            and split_op.index in nbr_indices
                        ):
                            split_ind_names = split_op.result
                            if (
                                split_ind_names is not None
                                and len(
                                    set(split_ind_names).intersection(
                                        m_inds_set
                                    )
                                )
                                != 0
                            ):
                                self.add_edge(m, nbr)

        self.network.remove_node(name)

        for u, v in subnet.network.edges:
            self.add_edge(node_subst[u], node_subst[v])

    def evaluate(self, indices: np.ndarray) -> np.ndarray:
        """
        Evaluate the tensor network at the given indices.

        Assumes indices are provided in the order retrieved by
        TensorNetwork.free_indices()
        """
        free_indices = self.free_indices()
        assert indices.shape[1] == len(free_indices), (
            f"the nbr of slices should be equal to the nbr of free indices"
            f"but get {indices.shape}, free indices {len(free_indices)}"
        )

        batch_ind = "_batch"
        ind_mapping = {batch_ind: "a"}
        node_vals = []
        node_strs = []
        for node in self.network.nodes:
            tensor = self.node_tensor(node)
            tslices = []
            has_batch = False
            node_str = ""
            for ind in tensor.indices:
                ind_letter = chr(97 + len(ind_mapping))
                if ind in free_indices:
                    tslices.append(indices[:, free_indices.index(ind)])
                    if not has_batch:
                        has_batch = True
                        node_str += ind_mapping[batch_ind]
                else:
                    tslices.append(slice(None))
                    if ind.name not in ind_mapping:
                        ind_mapping[ind.name] = ind_letter
                    node_str += ind_mapping[ind.name]

            batch_val = tensor.value[*tslices]
            node_vals.append(batch_val)
            node_strs.append(node_str)

        estr = ",".join(node_strs) + "->" + ind_mapping[batch_ind]
        logger.debug(
            "contraction args: %s, shapes: %s",
            estr,
            [n.shape for n in node_vals],
        )
        return oe.contract(estr, *node_vals, optimize="auto")

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
    def draw(self, ax=None):
        """Draw a networkx representation of the network."""

        # Define color and shape maps
        shape_map = {"A": "o", "B": "s"}
        size_map = {"A": 300, "B": 100}
        node_groups = {"A": [], "B": []}

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
            for node, data in self.network.nodes(data=True):
                if index in data["tensor"].indices:
                    new_graph.add_edge(node, name1)

        # To use graphviz layout,
        # you need to install both graphviz and pygraphviz.
        pos = nx.drawing.nx_agraph.graphviz_layout(
            new_graph,
            prog="neato",
            args="-Gsplines=true -Gnodesep=0.6 -Goverlap=scalexy",
        )
        # pos = nx.planar_layout(new_graph)

        for node, data in self.network.nodes(data=True):
            node_groups["A"].append(node)

        for node in free_graph.nodes():
            node_groups["B"].append(node)

        for group, nodes in node_groups.items():
            if group == "A":
                nx.draw_networkx_nodes(
                    new_graph,
                    pos,
                    ax=ax,
                    nodelist=nodes,
                    node_color="lightblue",
                    node_shape=shape_map[group],
                    node_size=size_map[group],
                )
                node_labels = {node: node for node in node_groups["A"]}
                nx.draw_networkx_labels(
                    new_graph, pos, ax=ax, labels=node_labels, font_size=12
                )
            else:
                nx.draw_networkx_nodes(
                    new_graph,
                    pos,
                    ax=ax,
                    nodelist=nodes,
                    node_color=range(1, len(nodes) + 1),
                    node_shape=shape_map[group],
                    node_size=size_map[group],
                    cmap=plt.get_cmap("Accent"),
                    # with_label=with_label[group]
                )
                node_labels = {node: node for node in nodes}
                nx.draw_networkx_labels(
                    new_graph, pos, ax=ax, labels=node_labels, font_size=12
                )

        edge_labels = {}
        for u, v in self.network.edges():
            indices = self.get_contraction_index(u, v)
            labels = [f"{i.size}" for i in indices]
            label = "-".join(labels)
            edge_labels[(u, v)] = label
        nx.draw_networkx_edges(new_graph, pos, ax=ax)
        nx.draw_networkx_edge_labels(
            new_graph, pos, ax=ax, edge_labels=edge_labels, font_size=10
        )


class TreeNetwork(TensorNetwork):
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

    def orthonormalize(self, name: NodeName) -> NodeName:
        """Orthonormalize the environment network for the specified node.

        Note that this method changes all node names in the network.
        It returns the new name for the given node after orthonormalization.
        """
        # traverse the tree rooted at the given node in the post order
        # 1 for visited and 2 for processed
        visited = {}

        def _postorder(pname: Optional[NodeName], name: NodeName) -> NodeName:
            """Postorder traversal the network from a given node name."""
            visited[name] = 1
            nbrs = list(self.network.neighbors(name))
            permute_indices = []
            merged = name
            for n in nbrs:
                if n not in visited:
                    # Process children before the current node.
                    c = _postorder(name, n)

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

            right_sz = np.prod([merged_indices[i].size for i in right_indices])
            # optimization: this step creates redundant nodes,
            # so to avoid them we directly eliminate the node with a merge.
            if (
                len(left_indices) == 1
                and merged_indices[left_indices[0]].size <= right_sz
            ):
                return merged

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
            self.set_node_tensor(
                q, self.node_tensor(q).permute(permute_indices)
            )

            return r

        return _postorder(None, name)

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

    # def cross(
    #     self,
    #     tensor_func: TensorFunc,
    #     delta: Optional[float] = None,
    #     max_k: Optional[int] = None,
    #     compute_data: bool = True,
    # ) -> Tuple[NodeName, NodeName, CrossApproxState]:
    #     """Split a node by cross approximation."""
    #     logger.debug("running cross")

    #     # reconstruct the dimension tree for the current network
    #     root = self.node_by_free_index(self.free_indices()[0].name)
    #     dim_tree = self.dimension_tree(root)
    #     logger.debug("dimension tree created")

    #     index_values = {} # mapping from index name to its position and values

    #     # walk and update the indices from root to leaves

    #             for ind in indices:
    #                 if ind in free_indices:
    #                     ranges.append(range(ind.size))
    #                     orders.append(free_indices.index(ind))

    #             for cidx, c in enumerate(tree.children):
    #                 ranges.append(tree.cvals[cidx])
    #                 orders.extend([free_indices.index(i) for i in c.indices])

    #             args = np.array(list(itertools.product(*ranges)))
    #             # reorder the args according to the indices
    #             args = args[:, np.argsort(orders)]
    #             # evaluate the function
    #             values = tensor_func(args)
    #             # update the index values by maxvol
    #             # first, reshape the values to the proper shape
    #             # the updating child on one side, and all others on the other

    #             values = values.reshape()

    def node_by_free_index(self, index: IndexName) -> NodeName:
        """Identify the node in the network containing the given free index"""
        for n in self.network.nodes:
            n = typing.cast(NodeName, n)
            tensor = self.node_tensor(n)
            if index in [ind.name for ind in tensor.indices]:
                return n

        raise KeyError(f"Cannot find index {index} in the network")

    def canonicalize_indices(self, tree: DimTreeNode):
        """sort the children by free indices
        and get the corresponding children nodes
        """
        indices: List[Index] = []
        node_indices = self.node_tensor(tree.info.node).indices
        for ind in tree.info.free_indices:
            indices.append(ind)

        # children indices
        for n in sorted(tree.conn.children):
            self.canonicalize_indices(n)
            ind = self.get_contraction_index(n.info.node, tree.info.node)[0]
            indices.append(ind)

        # parent indices, should be one
        p_indices = [ind for ind in node_indices if ind not in indices]
        assert len(p_indices) <= 1, (
            f"should have at most one parent index, but get {p_indices}"
        )

        indices.extend(p_indices)
        perm = [node_indices.index(ind) for ind in indices]
        new_tensor = self.node_tensor(tree.info.node).permute(perm)
        self.set_node_tensor(tree.info.node, new_tensor)

    def dimension_tree(self, root: NodeName) -> DimTreeNode:
        """Create a mapping from set of indices to node names.
        Assume that the tree is rooted at the give node.
        """
        free_indices = self.free_indices()

        # do the dfs traversal starting from the root
        def dfs(visited: Set[NodeName], node: NodeName) -> DimTreeNode:
            visited.add(node)
            children: List[DimTreeNode] = []
            up_vals, down_vals = [], []
            for nbr in self.network.neighbors(node):
                if nbr not in visited:
                    nbr_tree = dfs(visited, nbr)
                    children.append(nbr_tree)
                    up_vals.extend([0] * len(nbr_tree.info.indices))

            indices = set(ind for c in children for ind in c.info.indices)
            node_free_indices = []
            for ind in self.node_tensor(node).indices:
                if ind in free_indices:
                    indices.add(ind)
                    node_free_indices.append(ind)
                    if len(children) == 0:
                        up_vals.append(0)

            for ind in free_indices:
                if ind not in indices:
                    down_vals.append(0)

            # both up vals and down vals should include the free indices
            res = DimTreeNode(
                node=node,
                indices=list(indices),
                free_indices=sorted(node_free_indices),
                children=sorted(children, key=lambda x: x.info.indices),
                up_vals=[up_vals],
                down_vals=[down_vals],
            )
            for c in res.conn.children:
                c.conn.parent = res

            return res

        tree = dfs(set(), root)
        self.canonicalize_indices(tree)
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

        for c1, c2 in zip(
            tree1.conn.children, tree2.conn.children
        ):
            self._binary_op(other, op, c1, c2, result_net)
            result_net.add_edge(tree1.info.node, c1.info.node)


class TensorTrain(TreeNetwork):
    """Class for tensor trains"""

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


def rand_tree(indices: List[Index], ranks: List[int]) -> TreeNetwork:
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

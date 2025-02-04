"""Algorithms for tensor networks."""

import copy
import logging
from collections.abc import Sequence
from collections import Counter
from dataclasses import dataclass
import typing
from typing import Dict, Optional, Union, List, Self, Tuple, Callable, Any
import itertools

import numpy as np
import opt_einsum as oe
import networkx as nx
import matplotlib.pyplot as plt

from .utils import delta_svd

IntOrStr = Union[str, int]
IndexChain = Union[List[int], Tuple[int]]

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

NodeName = IntOrStr


@dataclass(frozen=True, eq=True)
class Index:
    """Class for denoting an index."""

    name: Union[str, int]
    size: int

    def with_new_size(self, new_size: int) -> "Index":
        """Create a new index with same name but new size"""
        return Index(self.name, new_size)

    def with_new_name(self, name: IntOrStr) -> "Index":
        """Create a new index with same size but new name"""
        return Index(name, self.size)

    def __lt__(self, other: Self) -> bool:
        return str(self.name) < str(other.name)


@dataclass
class IsoShape:
    """Isomorphic shape representation. Used for deduplicate networks."""

    ordered: List[Union[Index, int]]
    unordered: Dict[int, int]

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __str__(self) -> str:
        sorted_ordered = sorted([str(x) for x in self.ordered])
        sorted_unordered = sorted(list(self.unordered))
        return f"({sorted_ordered}, {sorted_unordered})"

    def __lt__(self, other: Self) -> bool:
        return self.__str__() < other.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IsoShape):
            return NotImplemented

        return self.__str__() == other.__str__()


@dataclass  # (frozen=True, eq=True)
class Tensor:
    """Base class for a tensor."""

    value: np.ndarray
    indices: List[Index]

    def update_val_size(self, value: np.ndarray) -> Self:
        """Update the tensor with a new value."""
        assert value.ndim == len(self.indices), f"{value.shape}, {self.indices}"
        self.value = value
        for ii, index in enumerate(self.indices):
            self.indices[ii] = index.with_new_size(value.shape[ii])
        return self

    def rename_indices(self, rename_map: Dict[IntOrStr, str]) -> Self:
        """Rename the indices of the tensor."""
        for ii, index in enumerate(self.indices):
            if index.name in rename_map:
                self.indices[ii] = index.with_new_name(rename_map[index.name])

        return self

    def relabel_indices(self, relabel_map: Dict[IntOrStr, Any]) -> Self:
        """Relabel the index size."""
        for ii, index in enumerate(self.indices):
            if index.name in relabel_map:
                self.indices[ii] = index.with_new_size(relabel_map[index.name])
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

    def split(
        self,
        left_indices: Sequence[int],
        right_indices: Sequence[int]
    ) -> List["Tensor"]:
        """Split a tensor into three by SVD."""
        permute_indices = itertools.chain(left_indices, right_indices)
        value = np.permute_dims(self.value, tuple(permute_indices))
        left_sz = int(np.prod([self.indices[i].size for i in left_indices]))
        right_sz = int(np.prod([self.indices[j].size for j in right_indices]))
        value = value.reshape(left_sz, right_sz)

        result = delta_svd(value, 1e-5) # we pass a small delta value here to exclude very small eigen values
        u = result.u
        s = result.s
        v = result.v
        # d = result.remaining_delta

        u = u.reshape([self.indices[i].size for i in left_indices] + [-1])
        u_indices = [self.indices[i] for i in left_indices]
        u_indices.append(Index("r_split_l", u.shape[-1]))
        u_tensor = Tensor(u, u_indices)

        s_indices = [Index("r_split_l", u.shape[-1]), Index("r_split_r", u.shape[-1])]
        s_tensor = Tensor(np.diag(s), s_indices)

        v = v.reshape([-1] + [self.indices[j].size for j in right_indices])
        v_indices = [self.indices[j] for j in right_indices]
        v_indices = [Index("r_split_r", v.shape[0])] + v_indices
        v_tensor = Tensor(v, v_indices)

        return [u_tensor, s_tensor, v_tensor]

    def delta_split(
        self,
        left_indices: Sequence[int],
        right_indices: Sequence[int],
        mode: str = "svd",
        delta: float = 0.1,
    ) -> Tuple[List["Tensor"], float]:
        """Split a tensor into two by SVD or QR."""
        permute_indices = itertools.chain(left_indices, right_indices)
        value = np.permute_dims(self.value, tuple(permute_indices))
        left_sz = int(np.prod([self.indices[i].size for i in left_indices]))
        right_sz = int(np.prod([self.indices[j].size for j in right_indices]))
        value = value.reshape(left_sz, right_sz)

        if mode == "svd":
            result = delta_svd(value, delta)
            u = result.u
            v = np.diag(result.s) @ result.v
            d = result.remaining_delta
        elif mode == "qr":
            u, v = np.linalg.qr(value)
            d = None
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        u = u.reshape([self.indices[i].size for i in left_indices] + [-1])
        u_indices = [self.indices[i] for i in left_indices]
        u_indices.append(Index("r_split", u.shape[-1]))
        u_tensor = Tensor(u, u_indices)

        v = v.reshape([-1] + [self.indices[j].size for j in right_indices])
        v_indices = [self.indices[j] for j in right_indices]
        v_indices = [Index("r_split", v.shape[0])] + v_indices
        v_tensor = Tensor(v, v_indices)

        return [u_tensor, v_tensor], d

    def permute(self, target_indices):
        """Return a new tensor with indices permuted by the specified order."""
        if not target_indices:
            return self

        value = np.permute_dims(self.value, tuple(target_indices))
        indices = [self.indices[i] for i in target_indices]
        return Tensor(value, indices)

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


class TensorNetwork:
    """Tensor Network Base Class."""

    def __init__(self) -> None:
        """Initialize the network."""
        self.network = nx.Graph()

    def add_node(self, name: NodeName, tensor: Tensor) -> None:
        """Add a node to the network."""
        self.network.add_node(name, tensor=tensor)

    def add_edge(self, name1: NodeName, name2: NodeName) -> None:
        """Add an edget to the network."""
        self.network.add_edge(name1, name2)

    def value(self, node_name: NodeName) -> np.ndarray:
        """Get the value of a node."""
        val: np.ndarray = self.network.nodes[node_name]["tensor"].value
        return val

    def all_indices(self) -> Counter:
        """Get all indices in the network."""
        indices = []
        for _, data in self.network.nodes(data=True):
            indices += data["tensor"].indices
        cnt = Counter(indices)
        return cnt

    def rename_indices(self, rename_map: Dict[IntOrStr, IntOrStr]) -> Self:
        """Rename the indices in the network."""
        for _, data in self.network.nodes(data=True):
            data["tensor"].rename_indices(rename_map)
        return self

    def relabel_indices(self, relabel_map: Dict[IntOrStr, Any]) -> Self:
        """Relabel the indices in the network."""
        for _, data in self.network.nodes(data=True):
            data["tensor"].relabel_indices(relabel_map)
        return self

    def free_indices(self) -> List[Index]:
        """Get the free indices."""
        icount = self.all_indices()
        free_indices = sorted([i for i, v in icount.items() if v == 1])
        return free_indices

    def get_contraction_index(
        self, node1: NodeName, node2: NodeName
    ) -> List[Index]:
        """Get the contraction indices."""
        ind1 = self.network.nodes[node1]["tensor"].indices
        ind2 = self.network.nodes[node2]["tensor"].indices
        inds = list(ind1) + list(ind2)
        cnt = Counter(inds)
        indices = [i for i, v in cnt.items() if v > 1]
        return indices

    def inner_indices(self) -> List[Index]:
        """Get hte interior indices."""
        icount = self.all_indices()
        free_indices = [i for i, v in icount.items() if v > 1]
        return free_indices

    def ranks(self) -> List[int]:
        """Get the ranks."""
        inner_indices = self.inner_indices()
        return [r.size for r in inner_indices]

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
            arrs.append(self.network.nodes[key]["tensor"].value)
            estr_l.append(val)
        estr = ",".join(estr_l) + "->" + eargs.output_str  # explicit
        # estr = ','.join(estr)
        logger.debug("Contraction string = %s", estr)
        out = oe.contract(estr, *arrs, optimize="auto")
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
            for _, local_ind in enumerate(tens.indices):
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
        weights: Sequence[Union[np.ndarray, float]],
    ) -> "TensorNetwork":
        """Integrate over the chosen indices. So far just uses simpson rule."""

        out = self
        for weight, index in zip(weights, indices):
            if isinstance(weight, float):
                v = np.ones(index.size) * weight
            else:
                v = weight
            tens = vector(f"w_{index.name}", index, v)
            out = out.attach(tens, rename=("", ""))

        return out

    def fresh_index(self) -> str:
        """Generate an index that does not appear in the current network."""
        all_indices = [i.name for i in self.all_indices().keys()]
        i = 0
        while f"s_{i}" in all_indices:
            i += 1

        return f"s_{i}"

    def fresh_node(self) -> NodeName:
        """Generate a node name that does not appear in the current network."""
        i = 0
        node = f"n{i}"
        while node in self.network.nodes:
            i += 1
            node = f"n{i}"

        return node

    def split(self, node_name: NodeName,
              left_indices: Sequence[int],
              right_indices: Sequence[int],
              with_orthonormalize: bool = True,
              preview: bool = False,) -> Tuple[NodeName, NodeName, NodeName]:
        """Perform the svd split and returns u, s, v"""
        x = self.network.nodes[node_name]["tensor"]

        if preview:
            u = Tensor(None, [x.indices[i] for i in left_indices] + [Index("r_split_l", -1)])
            v = Tensor(None, [Index("r_split_r", -1)] + [x.indices[i] for i in right_indices])
            s = Tensor(None, [Index("r_split_l", -1), Index("r_split_r", -1)])
        else:
            if with_orthonormalize:
                node_name = self.orthonormalize(node_name)
    
            x = self.network.nodes[node_name]["tensor"]
            # svd decompose the data into specified index partition
            [u, s, v] = x.split(left_indices, right_indices)

        v_name = self.fresh_node()
        new_index_r = self.fresh_index()
        self.add_node(v_name, v.rename_indices({"r_split_r": new_index_r}))

        u_name = node_name
        new_index_l = self.fresh_index()
        x_nbrs = list(self.network.neighbors(node_name))
        self.network.remove_node(node_name)
        self.add_node(u_name, u.rename_indices({"r_split_l": new_index_l}))

        s_name = self.fresh_node()
        self.add_node(s_name, s.rename_indices({"r_split_l": new_index_l, "r_split_r": new_index_r}))

        for y in x_nbrs:
            y_inds = self.network.nodes[y]["tensor"].indices
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

        # self.draw()
        # plt.show()
        return (u_name, s_name, v_name)

    def delta_split(  # pylint: disable=R0913
        self,
        node_name: NodeName,
        left_indices: Sequence[int],
        right_indices: Sequence[int],
        mode: str = "svd",
        delta: float = 0.1,
        with_orthonormal: bool = True,
        preview = False,
    ) -> Tuple[Tuple[NodeName, NodeName], float]:
        """Split a node by the specified index partition.

        Parameters
        ----------
        mode: "svd" or "qr" (Default: "svd")
        delta: threshold used in truncated SVD
        """
        # To ensure the error bound in SVD, we first orthornormalize its env.
        if not preview:
            if mode == "svd" and with_orthonormal:
                node_name = self.orthonormalize(node_name)
                # self.draw()
                # plt.show()

            x = self.network.nodes[node_name]["tensor"]
            # svd decompose the data into specified index partition
            [u, v], remaining_delta = x.delta_split(left_indices, right_indices, mode, delta)
        else:
            x = self.network.nodes[node_name]["tensor"]
            u = Tensor(None, [x.indices[i] for i in left_indices] + [Index("r_split", -1)])
            v = Tensor(None, [Index("r_split", -1)] + [x.indices[i] for i in right_indices])
            remaining_delta = delta

        new_index = self.fresh_index()
        x_nbrs = list(self.network.neighbors(node_name))
        self.network.remove_node(node_name)

        u_name = node_name
        self.add_node(u_name, u.rename_indices({"r_split": new_index}))
        v_name = self.fresh_node()
        self.add_node(v_name, v.rename_indices({"r_split": new_index}))

        for y in x_nbrs:
            y_inds = self.network.nodes[y]["tensor"].indices
            if any(i in y_inds for i in u.indices):
                self.add_edge(u_name, y)
            if any(i in y_inds for i in v.indices):
                self.add_edge(v_name, y)

        self.add_edge(u_name, v_name)

        # self.draw()
        # plt.show()
        return (u_name, v_name), remaining_delta

    def merge(self, name1: NodeName, name2: NodeName, preview=False) -> NodeName:
        """Merge two specified nodes into one."""
        if not self.network.has_edge(name1, name2):
            raise RuntimeError(
                f"Cannot merge nodes that are not adjacent: {name1}, {name2}"
            )

        t1 = self.network.nodes[name1]["tensor"]
        t2 = self.network.nodes[name2]["tensor"]
        if not preview:
            result = t1.contract(t2)
        else:
            result = Tensor(None, [ind for ind in t1.indices if ind not in t2.indices] + [ind for ind in t2.indices if ind not in t1.indices])

        n2_nbrs = list(self.network.neighbors(name2))
        self.network.remove_node(name2)
        self.network.nodes[name1]["tensor"] = result
        for n in n2_nbrs:
            if n != name1:
                self.add_edge(name1, n)

        # self.draw()
        # plt.show()
        return name1

    def round(self, node_name: NodeName, delta: float, visited: set = None) -> Tuple[NodeName, float]:
        """Optimize the tree rooted at the given node."""
        # print("optimize", node_name)
        # import matplotlib.pyplot as plt
        if visited is None:
            initial_optimize = True
            visited = set()
            self.orthonormalize(node_name)
            # print("start round")
        else:
            initial_optimize = False

        node_indices = self.network.nodes[node_name]["tensor"].indices
        kept_indices = []
        free_indices = []
        r = node_name
        for idx in node_indices:
            if idx in visited:
                kept_indices.append(idx)
                # print("visited idx", idx)
                continue

            shared_index = None
            nbr = None
            for nbr in self.network.neighbors(node_name):
                nbr_indices = self.network.nodes[nbr]["tensor"].indices
                if idx in nbr_indices:
                    shared_index = idx
                    # print("shared", idx, "with", nbr)
                    break

            if shared_index is None:
                free_indices.append(idx)
                # print("free index", idx)
                continue

            curr_indices = self.network.nodes[node_name]["tensor"].indices
            left_indices = [curr_indices.index(i) for i in curr_indices if i != idx]
            right_indices = [curr_indices.index(idx)]
            [node_name, v], delta = self.delta_split(node_name, left_indices, right_indices, delta=delta, with_orthonormal=False)
            self.merge(nbr, v)
            visited_index = self.get_contraction_index(node_name, nbr)
            for idx in visited_index:
                visited.add(idx)
            r, delta = self.round(nbr, delta, visited)
            self.merge(node_name, r)

        if not initial_optimize:
            node_indices = self.network.nodes[node_name]["tensor"].indices
            left_indices, right_indices = [], []
            for i, idx in enumerate(node_indices):
                if idx in free_indices or idx not in kept_indices:
                    left_indices.append(i)
                else:
                    right_indices.append(i)
            [_, r], _ = self.delta_split(node_name, left_indices, right_indices, mode="qr")

        # self.draw()
        # plt.show()
        return r, delta

    def compress(self):
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
                        nbr_indices = self.network.nodes[nbr]["tensor"].indices
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

                    # since split relying on ordered indices, we should restore the index order here.
                    indices = self.network.nodes[merged]["tensor"].indices
                    permute_index = indices.index(self.get_contraction_index(merged, c)[0])
                    permute_indices = list(range(permute_index))
                    permute_indices.append(len(indices) - 1)
                    permute_indices.extend(list(range(permute_index, len(indices) - 1)))

                    # print("before merge", self.network.nodes[merged]["tensor"].indices)
                    merged = self.merge(merged, c)

                    # restore the last index into the permute_index position
                    # print("[merge] permuting", self.network.nodes[merged]["tensor"].indices, "into", permute_indices)
                    self.network.nodes[merged]["tensor"] = self.network.nodes[merged]["tensor"].permute(permute_indices)

            if pname is None:
                return merged

            left_indices, right_indices = [], []
            merged_indices = self.network.nodes[merged]["tensor"].indices
            # print(merged_indices)
            for i, index in enumerate(merged_indices):
                common_index = None
                for n in self.network.neighbors(merged):
                    n_indices = self.network.nodes[n]["tensor"].indices
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

                if common_index is None:
                    left_indices.append(i)

            visited[name] = 2
            visited[merged] = 2

            # print("before split", self.network.nodes[merged]["tensor"].indices, left_indices, right_indices)
            right_sz = np.prod([merged_indices[i].size for i in right_indices])
            # optimization: this step creates redundant nodes, so to avoid them we directly eliminate the node with a merge
            if len(left_indices) == 1 and merged_indices[left_indices[0]].size <= right_sz:
                return merged

            (q, r), _ = self.delta_split(
                merged, left_indices, right_indices, mode="qr"
            )
            # this split changes the index orders, which affects the outer split result.
            # q has the indices r_split x right_indices
            # but we want r_split to replace the original left_indices
            # so we need to permute this tensor
            permute_indices = list(range(right_indices[0]))
            permute_indices.append(len(left_indices))
            permute_indices.extend(list(range(right_indices[0], len(left_indices))))
            # print("[qr] permuting", self.network.nodes[q]["tensor"].indices, "into", permute_indices)
            self.network.nodes[q]["tensor"] = self.network.nodes[q]["tensor"].permute(permute_indices)

            # print("returning r for", name, "with indices", self.network.nodes[r]["tensor"].indices)
            return r

        return _postorder(None, name)

    def cost(self) -> int:
        """Compute the cost for the tensor network.

        The cost is defined as sum of tensor core sizes.
        """
        cost = 0
        for n in self.network.nodes:
            indices = self.network.nodes[n]["tensor"].indices
            n_cost = np.prod([i.size for i in indices])
            cost += n_cost

        return int(cost)

    def canonical_structure(self, consider_ranks:bool=False):
        """Compute the canonical structure of the tensor network.

        This method ignores all values, keeps all free indices and edge labels.
        If the resulted topology is the same, we consider
        """
        # find the node with first free index and use it as the tree root
        free_indices = sorted(self.free_indices())
        root = None
        for n, d in self.network.nodes(data=True):
            if free_indices[0] in d["tensor"].indices:
                root = n
                break

        visited = {}
        def _postorder(name: NodeName):
            """Hash the nodes by their postorder"""
            visited[name] = 1
            children_rs = []
            nbrs = sorted(list(self.network.neighbors(name)))
            for n in nbrs:
                if n not in visited:
                    # Process children before the current node.
                    children_rs.append(_postorder(n))

            sorted_children_rs = tuple(sorted(children_rs))
            indices = self.network.nodes[name]["tensor"].indices
            all_free_indices = self.free_indices()
            ranks = tuple(sorted([i.size for i in indices]))
            self_free_indices = tuple(sorted([i for i in indices if i in all_free_indices]))

            visited[name] = 2
            if consider_ranks:
                features = (self_free_indices, ranks, sorted_children_rs)
            else:
                features = (self_free_indices, sorted_children_rs)

            return hash(features)

        return _postorder(root)

    def __lt__(self, other: Self) -> bool:
        return self.cost() < other.cost()

    def __add__(self, other: Self) -> Self:
        """Add two tensor trains.

        New tensor has same names as self
        """
        assert nx.is_isomorphic(self.network, other.network)

        new_tens = copy.deepcopy(self)
        free_indices = self.free_indices()
        for _, (node1, node2) in enumerate(
            zip(self.network.nodes, other.network.nodes)
        ):
            logger.debug("Adding: Node %r and Node %r", node1, node2)

            tens1 = self.network.nodes[node1]["tensor"]
            tens2 = other.network.nodes[node2]["tensor"]
            new_tens.network.nodes[node1]["tensor"] = tens1.concat_fill(
                tens2, free_indices
            )

        return new_tens

    def __mul__(self, other: Self) -> Self:
        """Multiply two tensor trains.

        New tensor has same names as self
        """
        assert nx.is_isomorphic(self.network, other.network)

        new_tens = copy.deepcopy(self)
        free_indices = self.free_indices()
        for _, (node1, node2) in enumerate(
            zip(self.network.nodes, other.network.nodes)
        ):
            logger.debug("Multiplying: Node %r and Node %r", node1, node2)

            tens1 = self.network.nodes[node1]["tensor"]
            tens2 = other.network.nodes[node2]["tensor"]
            new_tens.network.nodes[node1]["tensor"] = tens1.mult(
                tens2, free_indices
            )

        return new_tens

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

    def to_canonical_tt(self):
        # find the first node with two modes
        node_idx = 0
        node = None
        for n in self.network.nodes:
            if len(self.network.nodes[n]["tensor"].indices) == 2:
                node = n
                break

        if node is None:
            print("No tensor train start node found. Please double check")

        while node_idx != len(self.network.nodes):
            tensor = self.network.nodes[node]["tensor"]

            for n in self.network.neighbors(node):
                if n != node_idx - 1:
                    next_node = n
                    break

            self.add_node(node_idx, tensor)
            self.network.remove_node(node)
            node = next_node

            if node_idx != 0:
                self.network.add_edge(node_idx, node_idx - 1)
                # swap the tensor dimensions if needed
                prev = self.network.nodes[node_idx-1]["tensor"]
                prev_indices = prev.indices
                curr_indices = tensor.indices
                print(node_idx-1, prev_indices)
                print(curr_indices)
                if curr_indices[0] not in prev_indices:
                    the_other_dim = 2 if len(curr_indices) > 2 else 1
                    self.network.nodes[node_idx]["tensor"].value = tensor.value.swapaxes(the_other_dim, 0)
                    curr_indices[0], curr_indices[the_other_dim] = curr_indices[the_other_dim], curr_indices[0]
                    # self.network.nodes[node_idx]["tensor"].indices[0] = curr_indices[the_other_dim]
                    # self.network.nodes[node_idx]["tensor"].indices[the_other_dim] = curr_indices[0]

            node_idx += 1

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

        pos = nx.drawing.nx_agraph.graphviz_layout(new_graph, prog="neato", args='-Gsplines=true -Gnodesep=0.6 -Goverlap=scalexy')
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
                    node_color=range(1, len(nodes)+1),
                    node_shape=shape_map[group],
                    node_size=size_map[group],
                    cmap=plt.get_cmap("Accent"),
                    # with_label=with_label[group]
                )
                node_labels = {node: node for node in node_groups[group]}
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
        r.append(Index(f"r{ii+2}", ranks[ii + 1]))
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
        r.append(Index(f"r{ii+2}", 1))
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
        ranks.append(Index(f"r_{ii+1}", 2))
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
    val1 = tn.network.nodes[node]["tensor"].value
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
        tn.network.nodes[node]["tensor"].update_val_size(new_val)
    else:
        q, R = np.linalg.qr(val1.T)
        new_val = q.T
        tn.network.nodes[node]["tensor"].update_val_size(new_val)

    val2 = tn.network.nodes[node - 1]["tensor"].value
    # new_val2 = np.einsum("...i,ij->...j", val2, R.T)
    new_val2 = np.dot(val2, R.T)
    tn.network.nodes[node - 1]["tensor"].update_val_size(new_val2)

    return tn


def tt_round(tn: TensorNetwork, eps: float) -> TensorNetwork:
    """Round a tensor train.

    Nodes should be integers 0,1,2,...,dim-1
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

    v = np.dot(np.diag(trunc_svd.s), trunc_svd.v)
    r2 = trunc_svd.u.shape[1]
    new_core = np.reshape(trunc_svd.u, (value.shape[0], r2))

    data["tensor"].update_val_size(new_core)

    # print("In here")
    val_old = out.network.nodes[node + 1]["tensor"].value
    next_val = np.einsum("ij,jk...->ik...", v, val_old)
    out.network.nodes[node + 1]["tensor"].update_val_size(next_val)

    for node, data in core_list[1:-1]:
        value = data["tensor"].value
        r1, n, r2a = value.shape
        val = np.reshape(value, (r1 * n, r2a))
        trunc_svd = delta_svd(val, delta)
        v = np.dot(np.diag(trunc_svd.s), trunc_svd.v)
        r2 = trunc_svd.u.shape[1]
        new_core = np.reshape(trunc_svd.u, (r1, n, r2))

        data["tensor"].update_val_size(new_core)

        val_old = out.network.nodes[node + 1]["tensor"].value
        next_val = np.einsum("ij,jk...->ik...", v, val_old)
        out.network.nodes[node + 1]["tensor"].update_val_size(next_val)

    return out


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
        rank_indices.append(Index(f"{rank_name_prefix}_r{ii+1}", 1))
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
        rank_indices.append(Index(f"{rank_name_prefix}_r{ii+1}", 2))
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
        rank_indices.append(Index(f"{rank_name_prefix}_r{ii+1}", num_sum))
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
        new_core = np.reshape(new_core, (v.shape[0], -1))
        core[:, on_ind : on_ind + new_core.shape[1]] = new_core
        on_ind += new_core.shape[1]
    tt_out.add_node(ii, Tensor(core, indices))

    for ii, node_tt in enumerate(node_list[1:], start=1):
        v = tt_in.value(node_tt)

        if ii < dim - 1:
            rank_indices.append(
                Index(f"{rank_name_prefix}_r{ii+1}", v.shape[2] * num_sum)
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
        op = ttop.network.nodes[node_op]["tensor"].value
        v = tt.network.nodes[node_tt]["tensor"].value
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

        tt.network.nodes[node_tt]["tensor"] = tt.network.nodes[node_tt][
            "tensor"
        ].update_val_size(new_core)

    # print("After op = ")
    # print(tt)
    return tt


@typing.no_type_check
def gmres(  # pylint: disable=R0913
    op,  # function from in to out
    rhs: TensorNetwork,
    x0: TensorNetwork,
    eps: float = 1e-5,
    round_eps: float = 1e-10,
    maxiter: int = 100,
) -> TensorNetwork:
    """Perform GMRES.
    VERY HACKY
    """
    # pylint: disable=C0103
    # above disables the snake case complaints for variables like R
    r0 = rhs + op(x0).scale(-1.0)
    r0 = tt_round(r0, round_eps)
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
        w = tt_round(w, delta)

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
        w = tt_round(w, round_eps)
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
    x = tt_round(x, round_eps)
    r0 = rhs + op(x).scale(-1.0)
    resid = r0.norm()
    # print("resid = ", resid)
    # exit(1);
    return x, resid


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

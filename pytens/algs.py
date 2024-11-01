"""Algorithms for tensor networks."""

import copy
import logging
from collections.abc import Sequence
from collections import Counter
from dataclasses import dataclass
import typing
from typing import Dict, Optional, Union, List, Self, Tuple, Callable
import itertools

import numpy as np
import opt_einsum as oe
import networkx as nx

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


@dataclass  # (frozen=True, eq=True)
class Tensor:
    """Base class for a tensor."""

    value: np.ndarray
    indices: List[Index]

    def update_val_size(self, value: np.ndarray) -> Self:
        """Update the tensor with a new value."""
        assert value.ndim == len(self.indices)
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
        right_indices: Sequence[int],
        mode: str = "svd",
        delta: float = 0.1,
    ) -> List["Tensor"]:
        """Split a tensor into two by SVD or QR."""
        permute_indices = itertools.chain(left_indices, right_indices)
        value = np.permute_dims(self.value, tuple(permute_indices))
        left_sz = int(np.prod([self.indices[i].size for i in left_indices]))
        right_sz = int(np.prod([self.indices[j].size for j in right_indices]))
        value = value.reshape(left_sz, right_sz)

        if mode == "svd":
            result = delta_svd(value, delta)
            u = result.u @ np.diag(np.sqrt(result.s))
            v = np.diag(np.sqrt(result.s)) @ result.v
        elif mode == "qr":
            u, v = np.linalg.qr(value)
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

        return [u_tensor, v_tensor]


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

    def node_tensor(self, node_name: NodeName) -> None:
        return self.network.nodes[node_name]["tensor"]

    def set_node_tensor(self, node_name: NodeName, value: Tensor):
        self.network.nodes[node_name]["tensor"] = value
        
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

    def free_indices(self) -> List[Index]:
        """Get the free indices."""
        icount = self.all_indices()
        free_indices = [i for i, v in icount.items() if v == 1]
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
        free_indices = [i for i, v in all_indices.items() if v == 1]

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
        # print(estr)
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

        for n1, d1 in self.network.nodes(data=True):
            for n2, d2 in other.network.nodes(data=True):
                total_dim = len(d1["tensor"].indices) + len(
                    d2["tensor"].indices
                )
                if (
                    len(
                        set(
                            list(d1["tensor"].indices)
                            + list(d2["tensor"].indices)
                        )
                    )
                    < total_dim
                ):
                    u.add_edge(f"{rename[0]}{n1}", f"{rename[1]}{n2}")

        tn = TensorNetwork()
        tn.network = u

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
        value = self.attach(other).contract().value
        return value

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

    def split(  # pylint: disable=R0913
        self,
        node_name: NodeName,
        left_indices: Sequence[int],
        right_indices: Sequence[int],
        mode: str = "svd",
        delta: float = 0.1,
    ) -> Tuple[NodeName, NodeName]:
        """Split a node by the specified index partition.

        Parameters
        ----------
        mode: "svd" or "qr" (Default: "svd")
        delta: threshold used in truncated SVD
        """
        # To ensure the error bound in SVD, we first orthornormalize its env.
        if mode == "svd":
            node_name = self.orthonormalize(node_name)

        x = self.network.nodes[node_name]["tensor"]
        # svd decompose the data into specified index partition
        u, v = x.split(left_indices, right_indices, mode, delta)

        new_index = self.fresh_index()
        u_name = self.fresh_node()
        self.add_node(u_name, u.rename_indices({"r_split": new_index}))
        v_name = self.fresh_node()
        self.add_node(v_name, v.rename_indices({"r_split": new_index}))
        self.add_edge(u_name, v_name)

        x_nbrs = self.network.neighbors(node_name)
        for y in list(x_nbrs):
            y_inds = self.network.nodes[y]["tensor"].indices
            if any(i in y_inds for i in u.indices):
                self.add_edge(u_name, y)
            elif any(i in y_inds for i in v.indices):
                self.add_edge(v_name, y)
            else:
                raise ValueError(
                    f"Indices {y_inds} does not exist in splits (",
                    left_indices,
                    ",",
                    right_indices,
                )

        self.network.remove_node(node_name)

        return u_name, v_name

    def merge(self, name1: NodeName, name2: NodeName) -> NodeName:
        """Merge two specified nodes into one."""
        if not self.network.has_edge(name1, name2):
            raise RuntimeError(
                f"Cannot merge nodes that are not adjacent: {name1}, {name2}"
            )

        t1 = self.network.nodes[name1]["tensor"]
        t2 = self.network.nodes[name2]["tensor"]
        result = t1.contract(t2)
        new_node = self.fresh_node()
        self.add_node(new_node, result)
        for n in self.network.neighbors(name1):
            if n != name2:
                self.add_edge(new_node, n)

        for m in self.network.neighbors(name2):
            if m != name1:
                self.add_edge(new_node, m)

        self.network.remove_node(name1)
        self.network.remove_node(name2)

        return new_node

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
            children_rs = []
            nbrs = list(self.network.neighbors(name))
            for n in nbrs:
                if n not in visited:
                    # Process children before the current node.
                    children_rs.append(_postorder(name, n))

            # Subsume r values from its children.
            merged = name
            for c in children_rs:
                merged = self.merge(merged, c)

            r = merged
            if pname is not None:
                left_indices, right_indices = [], []
                merged_indices = self.network.nodes[merged]["tensor"].indices
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

                _, r = self.split(
                    merged, left_indices, right_indices, mode="qr"
                )

            visited[name] = 2
            visited[merged] = 2
            return r

        return _postorder(None, name)

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

    @typing.no_type_check
    def draw(self, ax=None):
        """Draw a networkx representation of the network."""

        # Define color and shape maps
        color_map = {"A": "lightblue", "B": "lightgreen"}
        shape_map = {"A": "o", "B": "s"}
        size_map = {"A": 300, "B": 100}
        node_groups = {"A": [], "B": []}

        # with_label = {'A': True, 'B': False}

        free_indices = self.free_indices()

        free_graph = nx.Graph()
        for index in free_indices:
            free_graph.add_node(f"{index.name}-f")

        new_graph = nx.compose(self.network, free_graph)
        for index in free_indices:
            name1 = f"{index.name}-f"
            for node, data in self.network.nodes(data=True):
                if index in data["tensor"].indices:
                    new_graph.add_edge(node, name1)

        pos = nx.planar_layout(new_graph)

        for node, data in self.network.nodes(data=True):
            node_groups["A"].append(node)

        for node in free_graph.nodes():
            node_groups["B"].append(node)

        for group, nodes in node_groups.items():
            nx.draw_networkx_nodes(
                new_graph,
                pos,
                ax=ax,
                nodelist=nodes,
                node_color=color_map[group],
                node_shape=shape_map[group],
                node_size=size_map[group],
                # with_label=with_label[group]
            )
            if group == "A":
                node_labels = {node: node for node in node_groups["A"]}
                nx.draw_networkx_labels(
                    new_graph, pos, ax=ax, labels=node_labels, font_size=12
                )
            if group == "B":
                node_labels = {node: node for node in node_groups["B"]}
                nx.draw_networkx_labels(
                    new_graph, pos, ax=ax, labels=node_labels, font_size=12
                )

        edge_labels = {}
        for u, v in self.network.edges():
            indices = self.get_contraction_index(u, v)
            labels = [f"{i.name}:{i.size}" for i in indices]
            label = "-".join(labels)
            edge_labels[(u, v)] = label
        nx.draw_networkx_edges(new_graph, pos, ax=ax)
        nx.draw_networkx_edge_labels(
            new_graph, pos, ax=ax, edge_labels=edge_labels
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


def gram_svd_round(
    tn: TensorNetwork, eps: float, threshold: float = 1e-14
) -> TensorNetwork:
    """
    Description: Modifies the input tensor network and returns the
    rounded version by implementing the Gram-SVD based rounding
    approach [1].

    [1] - H. Al Daas, G. Ballard and L. Manning, "Parallel Tensor-
    Train Rounding using Gram SVD," 2022 IEEE International Parallel
    and Distributed Processing Symposium (IPDPS), Lyon, France, 2022,
    pp. 930-940, doi: 10.1109/IPDPS53621.2022.00095.
    """

    def eps_to_rank(s: np.ndarray, eps: float) -> int:
        tmp = (np.sqrt(np.cumsum(np.square(s[::-1])))[::-1]) <= eps
        res: int = int(np.argmax(tmp))
        if res == 0 and not tmp[0]:
            return int(s.shape[0])
        if res == 0 and tmp[0]:
            return 1
        return res

    def next_gram(
        gram_now: np.ndarray, core_next: np.ndarray, order: str = "lr"
    ) -> np.ndarray:
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

        eigl, vl = np.linalg.eigh(gl)
        eigr, vr = np.linalg.eigh(gr_list[i + 1])
        eigl = np.abs(eigl)
        eigr = np.abs(eigr)

        maskl = eigl < threshold
        maskr = eigr < threshold
        eigl[maskl] = 0
        eigr[maskr] = 0

        eigl12 = np.sqrt(eigl)
        eigr12 = np.sqrt(eigr)
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

        curr_val = (
            tn.value(i).reshape((-1, sh[-1]))
            @ vl
            @ (eiglm12[:, np.newaxis] * u)
        )
        next_val = (
            (s[:, np.newaxis] * v * eigrm12[np.newaxis, :])
            @ vr.T
            @ tn.value(i + 1).reshape((shp1[0], -1))
        )

        sh[-1] = rk
        shp1[0] = rk
        curr_val = curr_val.reshape(sh)
        next_val = next_val.reshape(shp1)
        tn.network.nodes[i]["tensor"].update_val_size(curr_val)
        tn.network.nodes[i + 1]["tensor"].update_val_size(next_val)

    return tn


def tt_round(
    tn: TensorNetwork,
    eps: float,
    orthogonalize: bool = True,
    threshold: float = 1e-14,
) -> TensorNetwork:
    """Round a tensor train.

    Nodes should be integers 0,1,2,...,dim-1

    orthogonalize determines of QR is used to orthogonalize the
    cores. If orthogonalize=False, the Gram-SVD rounding algo-
    rithm is used.
    """
    # pylint: disable=C0103
    # above disables the snake case complaints for variables like R
    # norm2 = tn.norm()
    if orthogonalize:
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
        trunc_svd = delta_svd(
            value, eps / np.sqrt(dim - 1), with_normalizing=True
        )
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

    # elif orthogonalize == False
    return gram_svd_round(tn, eps, threshold)


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


def multiply_core_unfolding(
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
            assert (
                cols == rk_sum * n
            ), f"Dimension mismatch {cols} != {rk_sum*n}"
            res = np.zeros((rows, rk1_sum))
            for i in range(n_cores):
                res[:, rk1_cumsum[i] : rk1_cumsum[i + 1]] = mat[
                    :, rk_cumsum[i] * n : rk_cumsum[i + 1] * n
                ] @ cores_list[i].reshape((-1, rk1[i]))
            return res

        if (not v_unfolding) and (transpose):
            assert (
                cols == rk1_sum * n
            ), f"Dimension mismatch {cols} != {rk1_sum*n}"
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

    raise ValueError()


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


def round_ttsum(
    factors_list: list[TensorNetwork],
    eps: float = 1e-14,
    threshold: float = 1e-10,
) -> TensorNetwork:
    """Round a list of tensor networks that should be summed."""

    def eps_to_rank(s: np.ndarray, eps: float) -> int:
        tmp = (np.sqrt(np.cumsum(np.square(s[::-1])))[::-1]) <= eps
        res: int = int(np.argmax(tmp))
        if res == 0 and not tmp[0]:
            return int(s.shape[0])
        if res == 0 and tmp[0]:
            return 1
        return res

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

    ttsum.network.nodes[dim - 1]["tensor"].update_val_size(gr_list[-1])
    gr_list = [gr_list[-1] @ gr_list[-1].T]

    gl = np.concatenate([f.value(0) for f in factors_list], axis=1)
    ttsum.network.nodes[0]["tensor"].update_val_size(gl)

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

        eigl, vl = np.linalg.eigh(gl)
        eigr, vr = np.linalg.eigh(gr_list[i + 1])

        eigl = np.abs(eigl)
        eigr = np.abs(eigr)
        maskl = eigl < threshold
        maskr = eigr < threshold

        eigl[maskl] = 0
        eigr[maskr] = 0

        eigl12 = np.sqrt(eigl)
        eigr12 = np.sqrt(eigr)
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

        curr_val = (
            ttsum.value(i).reshape((-1, sh[-1]))
            @ vl
            @ (eiglm12[:, np.newaxis] * u)
        )

        next_val = (s[:, np.newaxis] * v * eigrm12[np.newaxis, :]) @ vr.T
        if i == (dim - 2):
            next_val = next_val @ ttsum.value(dim - 1)
        else:
            next_val = multiply_core_unfolding(
                next_val, core_next, False, False, False
            )

        sh[-1] = rk
        shp1[0] = rk

        curr_val = curr_val.reshape(sh)
        next_val = next_val.reshape(shp1)

        ttsum.network.nodes[i]["tensor"].update_val_size(curr_val)
        ttsum.network.nodes[i + 1]["tensor"].update_val_size(next_val)

    return ttsum


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
        new_core = np.reshape(new_core, (core.shape[0], -1))
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

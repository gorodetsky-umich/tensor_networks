"""Algorithms for tensor networks."""

import copy
import logging
from collections.abc import Sequence
from collections import Counter
from dataclasses import dataclass
import typing
from typing import Dict, Optional, Union, List, Self, Tuple
import numpy as np
import opt_einsum as oe
import networkx as nx
from .utils import delta_svd

IntOrStr = Union[str, int]

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
            self.indices[ii] = index.with_new_name(rename_map[index.name])

        return self

    def concat_fill(self, other: Self, indices_common: List[Index]) -> "Tensor":
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

    def free_indices(self) -> List[Index]:
        """Get the free indices."""
        icount = self.all_indices()
        free_indices = [i for i, v in icount.items() if v == 1]
        return free_indices

    def get_contraction_index(self, node1: NodeName, node2: NodeName) -> List[Index]:
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

        mapping = {name: chr(i + 97) for i, name in enumerate(all_indices.keys())}
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
                total_dim = len(d1["tensor"].indices) + len(d2["tensor"].indices)
                if (
                    len(set(list(d1["tensor"].indices) + list(d2["tensor"].indices)))
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
        self, indices: Sequence[Index], weights: Sequence[Union[np.ndarray, float]]
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
            new_tens.network.nodes[node1]["tensor"] = tens1.mult(tens2, free_indices)

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
            labels = [i.name for i in indices]
            label = "-".join(labels)
            edge_labels[(u, v)] = label
        nx.draw_networkx_edges(new_graph, pos, ax=ax)
        nx.draw_networkx_edge_labels(new_graph, pos, ax=ax, edge_labels=edge_labels)


def vector(name: Union[str, int], index: Index, value: np.ndarray) -> "TensorNetwork":
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
        0, Tensor(np.random.randn(indices[0].size, ranks[0]), [indices[0], r[0]])
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
        Tensor(np.random.randn(ranks[-1], indices[-1].size), [r[-1], indices[-1]]),
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


def tt_separable(indices: List[Index], funcs: List[np.ndarray]) -> TensorNetwork:
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


def tt_round(tn: TensorNetwork, eps: float, orthogonalize=True) -> TensorNetwork:
    """Round a tensor train.

    Nodes should be integers 0,1,2,...,dim-1
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
        for ii, (node, data) in enumerate(out.network.nodes(data=True)):
            value = data["tensor"].value
            if value.ndim == 3:
                r1, n, r2a = value.shape
                val = np.reshape(value, (r1 * n, r2a))
                u, s, v = delta_svd(val, delta)
                # print("u shape = ", u.shape)
                # print("s shape = ", s.shape)
                # print("v shape = ", v.shape)
                v = np.dot(np.diag(s), v)
                r2 = u.shape[1]
                new_core = np.reshape(u, (r1, n, r2))
            else:
                n, r2a = value.shape
                if ii == 0:
                    u, s, v, delta = delta_svd(
                        value, eps / np.sqrt(dim - 1), with_normalizing=True
                    )
                else:
                    u, s, v = delta_svd(value, delta)
                v = np.dot(np.diag(s), v)
                r2 = u.shape[1]
                new_core = np.reshape(u, (n, r2))

            data["tensor"].update_val_size(new_core)

            # print("In here")
            val_old = out.network.nodes[node + 1]["tensor"].value
            next_val = np.einsum("ij,jk...->ik...", v, val_old)
            out.network.nodes[node + 1]["tensor"].update_val_size(next_val)

            if node == dim - 2:
                break
        
        return out
    
    else:
        def eps_to_rank(s, eps):
            l = (np.sqrt(np.cumsum(np.square(s[::-1])))[::-1]) <= eps
            res = np.argmax(l)
            if res == 0 and l[0] == False: return s.shape[0]
            else: return res

        def next_gram(gram_now, core_next, order='lr'):
            snext = core_next.shape            
            if order == 'lr':
                tmp = (gram_now.T @ core_next.reshape(
                    (snext[0], -1))).reshape((-1, snext[-1]))
                return tmp.T @ core_next.reshape((-1, snext[-1]))
            elif order == 'rl':
                tmp = (core_next.reshape(
                    (-1, snext[-1])) @ gram_now).reshape((snext[0], -1))
                return tmp @ core_next.reshape((snext[0], -1)).T
            else: print("Invalid choice")
        
        threshold = 1e-10 # Arbitrary
        dim = tn.dim()
        gr_list = [tn.value(dim-1) @ tn.value(dim-1).T]
        
        # Collect gram matrices from right to left
        for i in range(dim-2, -1, -1):
            gr_list.append(next_gram(gr_list[-1], tn.value(i), 'rl'))
        
        norm = np.sqrt(np.linalg.norm(gr_list[-1]))
        delta = eps * norm/(dim-1)**0.5
        gr_list = gr_list[::-1]
        
        for i in range(dim-1):
            sh = list(tn.value(i).shape)
            shp1 = list(tn.value(i+1).shape)
            
            gl = (tn.value(i).reshape((-1, sh[-1])).T @ 
                    tn.value(i).reshape((-1, sh[-1])))
            
            eigl, vl = np.linalg.eigh(gl)
            eigr, vr = np.linalg.eigh(gr_list[i+1])
            
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
            eiglm12[~maskl] = 1/eigl12[~maskl]
            eigrm12[~maskr] = 1/eigr12[~maskr]

            # eiglm12 = np.nan_to_num(eiglm12, nan=0, posinf=0, neginf=0)
            # eigrm12 = np.nan_to_num(eigrm12, nan=0, posinf=0, neginf=0)
            
            tmp = ((eigl12[:, np.newaxis] * vl.T) @ 
                    (vr * eigr12[np.newaxis, :]))

            u, s, v = np.linalg.svd(tmp)
            rk = min(tmp.shape[0], tmp.shape[1], eps_to_rank(s, delta))
            u = u[:, :rk]; s = s[:rk]; v = v[:rk, :]

            curr_val = (tn.value(i).reshape((-1, sh[-1])) @ 
                        vl @ (eiglm12[:, np.newaxis] * u)
                        )

            next_val = ((s[:, np.newaxis] * v * eigrm12[np.newaxis, :]) @ 
                        vr.T @ tn.value(i+1).reshape((shp1[0], -1))
                        )
            
            sh[-1] = rk
            shp1[0] = rk

            curr_val = curr_val.reshape(sh)
            next_val = next_val.reshape(shp1)

            tn.network.nodes[i]["tensor"].update_val_size(curr_val)
            tn.network.nodes[i + 1]["tensor"].update_val_size(next_val)
        
        return tn


# Rounding sum of TT cores
def get_columns(matrix, periodicity, consecutive, start):
    """
    Gets the columns of the matrix that are supposed to be right multiplied with the non-zero elements of H(X_n).T to avoid 
    unnecessary computation. 
    """
    indices = np.concatenate([np.arange(i, i + consecutive) for i in range(start, matrix.shape[1], periodicity)])
    return matrix[:, indices]


def next_gram_sum(gram_now, core_next, order='rl'):
    """
    Let's say that we are dealing with 's' summands in our TT sum.

    gram_now is a sigma r_i times sigma r_i matrix (i from 1 to s) where r_i represents the sum of rank of a particular
    TT core of the summands. For example, it could be the sum for the last TT core of every summand. 

    core_next is the list (of size s) of adjacent TT-core of all summands. For example, if gram_now corresponds to the last TT core
    of all the summands, then assuming order = rl, core_next will be a list of the penultimate cores of all the summands.

    order: 'lr' means left to right and 'rl' means right to left.
    """

    shnext = [s.shape for s in core_next]
    if order == 'rl':
        Rk = [s.shape[-1] for s in core_next]
        Rk_cumsum = np.cumsum([0]+Rk)
        Rk_sum = gram_now.shape[0]
        Rk1 = [s.shape[0] for s in core_next]
        Rk1_sum = np.sum(Rk1)
        Rk1_cumsum = np.cumsum([0]+Rk1)
        n = core_next[0].shape[1]
        tmp = np.zeros((Rk1_sum*n, Rk_sum))
        for i in range(len(core_next)):
            tmp[Rk1_cumsum[i]*n:Rk1_cumsum[i+1]*n, :] = (core_next[i].reshape(
                                                        (-1, shnext[i][-1])) @ 
                                                        gram_now[Rk_cumsum[i]:Rk_cumsum[i+1], :])
        
        tmp = tmp.reshape((Rk1_sum, n*Rk_sum))
        tmplist = [get_columns(tmp, Rk_sum, rk, Rk_cumsum[i]) for i, rk in enumerate(Rk)]
        tmp = np.zeros((Rk1_sum, Rk1_sum))
        for i, mat in enumerate(tmplist):
            tmp[:, Rk1_cumsum[i]:Rk1_cumsum[i+1]] = mat @ core_next[i].reshape(
                                                            (shnext[i][0], -1)).T
        return tmp
    
    # elif order == 'lr':
    #     corelist = []
    #     for i in range(len(core_next)):
    #         corelist.append(core_next[i].reshape((-1, shnext[-1])).T @ core_next[i].reshape((-1, shnext[-1])))
    #     return corelist
        
        

def round_ttsum(factors_list: list[TensorNetwork], 
                eps=1e-14, threshold=1e-10):

    def eps_to_rank(s, eps):
        l = (np.sqrt(np.cumsum(np.square(s[::-1])))[::-1]) <= eps
        res = np.argmax(l)
        if res == 0 and l[0] == False: return s.shape[0]
        else: return res

    dim = factors_list[0].dim()
    n_s = len(factors_list)

    for i in range(n_s):
        if not i: ttsum = factors_list[i]
        else: ttsum = ttsum + factors_list[i]

    gr_list = [ttsum.value(dim-1) @ ttsum.value(dim-1).T]
    
    # Collect gram matrices from right to left
    for i in range(dim-2, 0, -1):
        gr_list.append(next_gram_sum(gr_list[-1], 
                [f.value(i) for f in factors_list], 'rl'))
    
    tmp = ttsum.value(0)
    gr_list.append(np.sum((tmp @ gr_list[-1]) * tmp).reshape((1, 1)))
    
    norm = np.sqrt(np.linalg.norm(gr_list[-1]))
    delta = eps * norm/(dim-1)**0.5

    gr_list = gr_list[::-1]
    
    for i in range(dim-1):
        sh = list(ttsum.value(i).shape)
        shp1 = list(ttsum.value(i+1).shape)

        gl = (ttsum.value(i).reshape((-1, sh[-1])).T @ 
                ttsum.value(i).reshape((-1, sh[-1])))

        eigl, vl = np.linalg.eigh(gl)
        eigr, vr = np.linalg.eigh(gr_list[i+1])
        
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
        eiglm12[~maskl] = 1/eigl12[~maskl]
        eigrm12[~maskr] = 1/eigr12[~maskr]
        
        # eiglm12 = np.nan_to_num(eiglm12, nan=0, posinf=0, neginf=0)
        # eigrm12 = np.nan_to_num(eigrm12, nan=0, posinf=0, neginf=0)

        tmp = ((eigl12[:, np.newaxis] * vl.T) @ 
                (vr * eigr12[np.newaxis, :]))
        
        u, s, v = np.linalg.svd(tmp)
        rk = min(tmp.shape[0], tmp.shape[1], eps_to_rank(s, delta))
        u = u[:, :rk]; s = s[:rk]; v = v[:rk, :]
        
        curr_val = (ttsum.value(i).reshape((-1, sh[-1])) @ vl @ 
                    (eiglm12[:, np.newaxis] * u))
        
        next_val = ((s[:, np.newaxis] * v * eigrm12[np.newaxis, :]) @ vr.T @ 
                    ttsum.value(i+1).reshape((shp1[0], -1)))
      
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
        cores[0][:, :, np.newaxis], [indices_out[0], indices_in[0], rank_indices[0]]
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
            core = np.zeros((num_sum, indices_out[ii].size, indices_in[ii].size))
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
    cores: List[List[np.ndarray]],
    rank_name_prefix: str,
) -> TensorNetwork:
    """Apply sum of rank1 tt ops to a tt."""
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    tt_out = TensorNetwork()
    num_sum = len(cores)
    for ii, node_tt in enumerate(tt_in.network.nodes()):
        v = tt_in.network.nodes[node_tt]["tensor"].value
        if ii == 0:
            rank_indices = [Index(f"{rank_name_prefix}_r1", num_sum * v.shape[1])]

        if ii > 0 and ii < dim - 1:
            rank_indices.append(
                Index(f"{rank_name_prefix}_r{ii+1}", v.shape[2] * num_sum)
            )

        if ii == 0:
            core = np.zeros((indices_out[ii].size, v.shape[1] * num_sum))
            indices = [indices_out[ii], rank_indices[ii]]
            on_ind = 0
            for jj in range(num_sum):
                new_core = cores[jj][ii](v)
                # new_core = np.einsum('ij,jl->il', cores[jj][ii], v)
                n = v.shape[0]
                new_core = np.reshape(new_core, (n, -1))
                core[:, on_ind : on_ind + new_core.shape[1]] = new_core
                on_ind += new_core.shape[1]
        elif ii < dim - 1:
            core = np.zeros(
                (num_sum * v.shape[0], indices_out[ii].size, num_sum * v.shape[2])
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
                core[on_ind1 : on_ind1 + n1, :, on_ind2 : on_ind2 + n2] = new_core
                on_ind1 += n1
                on_ind2 += n2
        else:
            core = np.zeros((num_sum * v.shape[0], indices_out[ii].size))
            indices = [rank_indices[ii - 1], indices_out[ii]]
            on_ind = 0
            for jj in range(num_sum):
                # new_core = np.einsum('ij,mj->mi', cores[jj][ii], v)
                new_core = cores[jj][ii](v)
                # shape = new_core.shape
                # new_core = np.reshape(new_core, (shape[0]*shape[1], -1))
                core[on_ind : on_ind + new_core.shape[0], :] = new_core
                on_ind += new_core.shape[0]

        new_tensor = Tensor(core, indices)

        tt_out.add_node(ii, new_tensor)

        if ii > 0:
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
def gmres(
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

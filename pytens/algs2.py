"""Algorithms for coupling random variables and sampling them."""
import abc
import copy
from collections.abc import Sequence
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Any, TypeVar, Generic, Optional, Callable, Union, List, Tuple, Self
from dataclasses import dataclass
from collections import namedtuple
import numpy as np
import opt_einsum as oe
import networkx as nx
from .utils import deltaSVD

import logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

NodeName = Union[str, int]

@dataclass(frozen=True, eq=True)
class Index:
    """Class for denoting an index."""
    name: Union[str, int]
    size: int

    def with_new_size(self, new_size: int) -> Self:
        return Index(self.name, new_size)

    def with_new_name(self, name):
        return Index(name, self.size)

@dataclass# (frozen=True, eq=True)
class Tensor[T]:
    value: T
    indices: List[Index]

    def update_val_size(self, value: T):
        assert value.ndim == len(self.indices)
        self.value = value
        for ii, index in enumerate(self.indices):
            self.indices[ii] = index.with_new_size(value.shape[ii])
        return self

    def rename_indices(self, rename_map: Dict[str, str]):

        for ii, index in enumerate(self.indices):
            self.indices[ii] = index.with_new_name(rename_map[index.name])

        return self
        
    def concat_fill(self, other, indices_common) -> Self:
        """Concatenate two arrays.

        keep dimensions corresponding to indices_common the same. pad zeros on all other dimensions. new dimensions retain index names currently used, but have updated size
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
                new_index = Index(f"{index_here.name}",
                                  index_here.size + index_other.size)
                new_indices.append(new_index)

        # print("new shape = ", new_shape)
        # print("new_indices = ", new_indices)
        new_val = np.zeros(new_shape)

        ix1 = []
        ix2 = []
        for ii, index_here in enumerate(self.indices):
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

    def mult(self, other, indices_common) -> Self:
        """Outer product of two tensors except at common indices

        retain naming of self
        """
        shape_here = self.value.shape
        shape_other = other.value.shape

        assert len(shape_here) == len(shape_other)

        new_shape = []
        new_indices = []
        str1 = ''
        str2 = ''
        output_str = ''
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
                new_index = Index(f"{index_here.name}",
                                  index_here.size * index_other.size)
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
        estr = str1 + ',' + str2 + '->'+ output_str
        # print("estr", estr)


        new_val = np.einsum(estr, self.value, other.value)
        new_val = np.reshape(new_val, new_shape)
        tens = Tensor(new_val, new_indices)
        return tens    
        
        
        
@dataclass(frozen=True, eq=True)
class EinsumArgs:
    """Represent information about contraction in einsum list format"""
    input_str_map: Dict[NodeName, str]
    output_str: str
    output_str_index_map: Dict[str, Index]

    def replace_char(self, value, replacement):

        for key, vals in self.input_str_map.items():
            vals = vals.replace(value, replacement)
        self.output_str = self.output_str.replace(value, replacement)
        

class TensorNetwork:

    def __init__(self):
        self.network = nx.Graph()

    def add_node(self, name: NodeName, tensor: Tensor):
        self.network.add_node(name, tensor=tensor)

    def add_edge(self, name1: NodeName, name2: NodeName):
        self.network.add_edge(name1, name2)

    def value(self, node_name: NodeName):
        return self.network.nodes[node_name]['tensor'].value
    
    def all_indices(self) -> Counter:        
        indices = []
        for node, data in self.network.nodes(data=True):
            indices += data['tensor'].indices
        cnt = Counter(indices)
        return cnt

    def rename_indices(self, rename_map) -> Self:        
        indices = []
        for node, data in self.network.nodes(data=True):
            data['tensor'].rename_indices(rename_map)
        return self

    def free_indices(self) -> List[Index]:
        icount = self.all_indices()
        free_indices = [i for i, v in icount.items() if v == 1]
        return free_indices

    def get_contraction_index(self, node1, node2) -> List[Index]:
        ind1 = self.network.nodes[node1]['tensor'].indices
        ind2 = self.network.nodes[node2]['tensor'].indices
        inds = list(ind1) + list(ind2)
        cnt = Counter(inds)
        indices = [i for i, v in cnt.items() if v > 1]
        return indices
    
    def inner_indices(self) -> List[Index]:
        icount = self.all_indices()
        free_indices = [i for i, v in icount.items() if v > 1]
        return free_indices
    
    def ranks(self) -> List[int]:
        inner_indices = self.inner_indices()
        return [r.size for r in inner_indices]

    def einsum_args(self) -> EinsumArgs:
        """Compute einsum args

        Need to respect the edges, currently not using edges
        """
        all_indices = self.all_indices()
        free_indices = [i for i, v in all_indices.items() if v == 1]
        
        mapping = {name: chr(i+97) for i, name in enumerate(all_indices.keys())}
        input_str_map = {}
        for node, data in self.network.nodes(data=True):
            input_str_map[node] = ''.join([mapping[ind] for ind in data['tensor'].indices])
        output_str = ''.join([mapping[ind] for ind in free_indices])
        output_str_index_map = {}
        for ind in free_indices:
            output_str_index_map[mapping[ind]] = ind
            
        return EinsumArgs(input_str_map, output_str, output_str_index_map)

    def contract(self, eargs: Optional[EinsumArgs] = None):

        if eargs is None:
            eargs = self.einsum_args()

        estr = []
        arrs = []
        for key, val in eargs.input_str_map.items():
            arrs.append(self.network.nodes[key]['tensor'].value)
            estr.append(val)
        estr = ','.join(estr) + '->' + eargs.output_str # explicit
        # print(estr)
        # estr = ','.join(estr)
        logger.debug(f"Contraction string = {estr}")
        out = oe.contract(estr, *arrs, optimize='auto')
        indices = [eargs.output_str_index_map[s] for s in eargs.output_str]
        tens = Tensor(out, indices)
        return tens

    def __getitem__(self, ind) -> Self:
        """Evaluate at some elements.

        Assumes indices are provided in the order retrieved by
        TensorNetwork.free_indices()
        """
        free_indices = self.free_indices()
        
        new_network = TensorNetwork()
        for node, data in self.network.nodes(data=True):
            tens = data['tensor']
            ix = []
            new_indices = []
            for ii, local_ind in enumerate(tens.indices):
                try:
                    dim = free_indices.index(local_ind)
                    ix.append(ind[dim])
                    if not isinstance(ind[dim], int):
                        new_indices.append(local_ind)
                    
                except ValueError: # no dimension is in a free index
                    ix.append(slice(None))
                    new_indices.append(local_ind)
                    
            new_arr = tens.value[*ix]
            new_tens = Tensor(new_arr, new_indices)
            new_network.add_node(node, new_tens)
            
        for u, v in self.network.edges():
            new_network.add_edge(u, v)

        return new_network.contract()

    def attach(self, other, rename=("G", "H")) -> Self:
        
        # U = nx.union(copy.deepcopy(self.network),
        #              copy.deepcopy(other.network),
        #              rename=rename)

        new_self = copy.deepcopy(self)
        new_other = copy.deepcopy(other)
        
        U = nx.union(new_self.network,
                     new_other.network,
                     rename=rename)
        
        for n1, d1 in self.network.nodes(data=True):
            for n2, d2 in other.network.nodes(data=True):
                total_dim = len(d1['tensor'].indices) + len(d2['tensor'].indices)
                if len(set(list(d1['tensor'].indices) +
                           list(d2['tensor'].indices))) < total_dim:
                    U.add_edge(f'{rename[0]}{n1}', f'{rename[1]}{n2}')

        tn = TensorNetwork()
        tn.network = U

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
            U.nodes[f"{rename[0]}{n}"]['tensor'].rename_indices(rename_ix)

                
        all_indices = other.all_indices()
        free_indices = other.free_indices()
        rename_ix_o = {}
        for index in all_indices:
            if index in free_indices:
                rename_ix_o[index.name] = index.name
            else:
                rename_ix_o[index.name] = f"{rename[1]}{index.name}"

        for n in other.network.nodes():
            U.nodes[f"{rename[1]}{n}"]['tensor'].rename_indices(rename_ix_o)                
        # print("ATTACH TN: ", tn)
        # print("self is: ", self)
        return tn
    
    def dim(self):
        return len(self.free_indices())
    
    def scale(self, scale_factor) -> Self:
        for node, data in self.network.nodes(data=True):
            data['tensor'].value *= scale_factor
            break
        return self
    
    def inner(self, other) -> float:
        """Compute the inner product."""
        value = self.attach(other).contract().value
        return value

    def norm(self) -> float:
        # return np.sqrt(np.abs(self.inner(copy.deepcopy(self))))
        return np.sqrt(np.abs(self.inner(self)))

    def integrate(self, indices: Sequence[Index], weights: Sequence[
            Union[np.ndarray, float]]) -> Self:
        """Integrate over the chosen indices. So far just uses simpson rule."""

        out = self
        for weight, index in zip(weights, indices):
            if isinstance(weight, float):
                v = np.ones(index.size) * weight
            else:
                v = weight
            tens = vector(f'w_{index.name}', index, v)
            out = out.attach(tens, rename=('', ''))
        
        return out


    def __add__(self, other):
        """Add two tensor trains.

        New tensor has same names as self
        """
        assert nx.is_isomorphic(self.network, other.network)

        newTens = copy.deepcopy(self)
        free_indices = self.free_indices()
        for ii, (node1, node2) in enumerate(zip(self.network.nodes,
                                                other.network.nodes)):
            logger.debug(f"Adding: Node {node1} and Node {node2}")

            tens1 = self.network.nodes[node1]['tensor']
            tens2 = other.network.nodes[node2]['tensor']
            newTens.network.nodes[node1]['tensor'] = \
                tens1.concat_fill(tens2, free_indices)

        return newTens

    def __mul__(self, other):
        """Multiply two tensor trains.

        New tensor has same names as self
        """
        assert nx.is_isomorphic(self.network, other.network)

        newTens = copy.deepcopy(self)
        free_indices = self.free_indices()
        for ii, (node1, node2) in enumerate(zip(self.network.nodes,
                                                other.network.nodes)):
            logger.debug(f"Multiplying: Node {node1} and Node {node2}")

            tens1 = self.network.nodes[node1]['tensor']
            tens2 = other.network.nodes[node2]['tensor']
            newTens.network.nodes[node1]['tensor'] = \
                tens1.mult(tens2, free_indices)

        return newTens    
    
    
    def __str__(self):

        out  = "Nodes:\n"
        out += "------\n"
        for node, data in self.network.nodes(data=True):
            out += f"\t{node}: shape = {data['tensor'].value.shape}, indices = {[i.name for i in data['tensor'].indices]}\n"

        out += "Edges:\n"
        out += "------\n"
        for node1, node2, data in self.network.edges(data=True):
            out += f"\t{node1} -> {node2}\n"            

        return out

    def draw(self, ax=None):

        # Define color and shape maps
        color_map = {'A':'lightblue', 'B':'lightgreen'}
        shape_map = {'A':'o', 'B':'s'}
        size_map = {'A': 300, 'B': 100}
        node_groups = {'A': [], 'B': []}

        with_label = {'A': True, 'B': False}

        free_indices = self.free_indices()

        free_graph = nx.Graph()
        for index in free_indices:
            free_graph.add_node(f'{index.name}-f')

        new_graph = nx.compose(self.network, free_graph)
        for index in free_indices:
            name1 = f'{index.name}-f'
            for node, data in self.network.nodes(data=True):
                if index in data['tensor'].indices:
                    new_graph.add_edge(node, name1)

        
        pos = nx.planar_layout(new_graph)
        
        
        for node, data in self.network.nodes(data=True):
            node_groups['A'].append(node)

        for node in free_graph.nodes():
            node_groups['B'].append(node)

        for group, nodes in node_groups.items():

            nx.draw_networkx_nodes(new_graph, pos,
                                   ax=ax,
                                   nodelist=nodes,
                                   node_color=color_map[group],
                                   node_shape=shape_map[group],
                                   node_size=size_map[group]
                                   # with_label=with_label[group]
                                   )
            if group == 'A':
                node_labels = {node: node for node in node_groups['A']}
                nx.draw_networkx_labels(new_graph,
                                        pos,
                                        ax=ax,
                                        labels=node_labels,
                                        font_size=12)
            if group == 'B':
                node_labels = {node: node for node in node_groups['B']}
                nx.draw_networkx_labels(new_graph,
                                        pos,
                                        ax=ax,
                                        labels=node_labels,
                                        font_size=12)

        edge_labels = {}
        for u, v in self.network.edges():
            indices = self.get_contraction_index(u,v)
            labels = [i.name for i in indices]
            label = '-'.join(labels)
            edge_labels[(u,v)] = label
        nx.draw_networkx_edges(new_graph, pos, ax=ax)
        nx.draw_networkx_edge_labels(new_graph, pos, ax=ax, edge_labels=edge_labels)

def vector(name: Union[str, int],
           index: Index,
           value: np.ndarray,
           ) -> TensorNetwork:
    vec = TensorNetwork()
    vec.add_node(name, Tensor(value, [index]))
    return vec        
        
def rand_tt(indices: List[Index],
            ranks: List[int]) -> TensorNetwork:
    """Return a random tt."""

    dim = len(indices)
    assert len(ranks)+1 == len(indices)

    tt = TensorNetwork()
    
    r = [Index('r1', ranks[0])]
    tt.add_node(0, Tensor(
        np.random.randn(indices[0].size, ranks[0]),
        [indices[0], r[0]]
    ))

    core = 1
    for ii, index in enumerate(indices[1:-1]):
        r.append(Index(f'r{ii+2}', ranks[ii+1]))
        tt.add_node(core, Tensor(
            np.random.randn(ranks[ii], index.size, ranks[ii+1]),
            [r[ii], index, r[ii+1]]
        ))
        core += 1
        tt.add_edge(ii, ii+1)


    tt.add_node(dim-1, Tensor(
        np.random.randn(ranks[-1], indices[-1].size),
        [r[-1], indices[-1]]
    ))
    tt.add_edge(dim-2, dim-1)
    
    return tt


def tt_rank1(indices: List[Index],
             vals: List[np.ndarray]) -> TensorNetwork:
    """Return a random tt."""

    dim = len(indices)

    tt = TensorNetwork()
    
    r = [Index('r1', 1)]
    # print("vals[0] ", vals[0][:, np.newaxis])
    new_tens = Tensor(vals[0][:, np.newaxis], [indices[0], r[0]])
    tt.add_node(0, new_tens)
    # print("new_tens = ", new_tens.indices)
    
    core = 1
    for ii, index in enumerate(indices[1:-1]):
        r.append(Index(f'r{ii+2}', 1))
        new_tens = Tensor(vals[ii+1][np.newaxis, :, np.newaxis],
                          [r[ii], index, r[ii+1]])
        tt.add_node(core, new_tens)
        tt.add_edge(core-1, core)
        core += 1        


    tt.add_node(dim-1, Tensor(vals[-1][np.newaxis, :],
                              [r[-1], indices[-1]]))
    tt.add_edge(dim-2, dim-1)
    # print("tt_rank1 = ", tt)
    return tt

def tt_separable(indices: List[Index],
                 funcs: List[np.ndarray]) -> TensorNetwork:
    """Rank 2 function formed by summation of functoins of individual dimensions"""

    dim = len(indices)

    tt = TensorNetwork()
    ranks = []
    for ii, index in enumerate(indices):
        
        ranks.append(Index(f"r_{ii+1}", 2))
        if ii == 0:
            val = np.ones((index.size, 2))
            val[:, 0] = funcs[ii]

            tt.add_node(ii, Tensor(val, [index, ranks[-1]]))
        elif ii < dim-1:
            val = np.zeros((2, index.size, 2))
            val[0, :, 0] = 1.0
            val[1, :, 0] = funcs[ii]
            val[1, :, 1] = 1.0
            tt.add_node(ii, Tensor(val,
                                   [ranks[-2], index, ranks[-1]]))
        else:
            val = np.ones((2, index.size))
            val[1, :] = funcs[ii]
            tt.add_node(ii, Tensor(val,
                                   [ranks[-2], index]))
            
        if ii > 0:
            tt.add_edge(ii-1, ii)

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

    val1 = tn.network.nodes[node]['tensor'].value
    if val1.ndim == 3:
        # print("val1.shape = ", val1.shape)
        r, n, b = val1.shape
        val1 = np.reshape(val1, (r, n*b), 'F')
        # print("val1.T.shape = ", val1.T.shape)
        q, R = np.linalg.qr(val1.T, mode='reduced')
        if q.shape[1] < r:
            newq = np.zeros((q.shape[0], r))
            newq[:, :q.shape[1]] = q
            q = newq
            newr = np.zeros((r, R.shape[1]))
            newr[:R.shape[0], :] = R
            R = newr

        # print("q.shape = ", q.shape)
        # print("r.shape = ", R.shape)
        # print("r = ", r)
        # print("q shape = ", q.shape)
        new_val = np.reshape(q.T, (r, n, b), 'F')
        tn.network.nodes[node]['tensor'].update_val_size(new_val)
    else:
        q, R = np.linalg.qr(val1.T)
        new_val = q.T
        tn.network.nodes[node]['tensor'].update_val_size(new_val)


    val2 = tn.network.nodes[node-1]['tensor'].value
    new_val2 = np.einsum('...i,ij->...j', val2, R.T)
    tn.network.nodes[node-1]['tensor'].update_val_size(new_val2)

    return tn

def tt_round(tn, eps: float) -> TensorNetwork:
    """Round a tensor train.

    Nodes should be integers 0,1,2,...,dim-1
    """

    norm2 = tn.norm()
    dim = tn.dim()
    delta = eps / np.sqrt(dim-1) * norm2

    # cores = []
    # for node, data in tn.network.nodes(data=True):
    #     cores.append(node)

    # print("DIM = ", dim)
    out = tt_right_orth(tn, dim-1)
    for jj in range(dim-2, 0, -1):
        # print(f"orthogonalizing core {cores[jj]}")
        out = tt_right_orth(out, jj)

    # print("ON FORWARD SWEEP")
    for node, data in out.network.nodes(data=True):

        value = data['tensor'].value
        if value.ndim == 3:
            r1, n, r2a = value.shape
            val = np.reshape(value, (r1*n, r2a))
            u, s, v = deltaSVD(val, delta)
            # print("u shape = ", u.shape)
            # print("s shape = ", s.shape)
            # print("v shape = ", v.shape)
            v = np.dot(np.diag(s), v)
            r2 = u.shape[1]
            new_core = np.reshape(u, (r1, n, r2))
        else:
            n, r2a = value.shape
            u, s, v = deltaSVD(value, delta)
            v = np.dot(np.diag(s), v)
            r2 = u.shape[1]
            new_core = np.reshape(u, (n, r2))

        data['tensor'].update_val_size(new_core)
        
        # print("In here")
        val_old = out.network.nodes[node+1]['tensor'].value
        next_val = np.einsum('ij,jk...->ik...', v, val_old)
        out.network.nodes[node+1]['tensor'].update_val_size(next_val)

        if node == dim-2:
            break


    return out

def ttop_rank1(indices_in: List[Index], indices_out: List[Index],
               cores: List[np.ndarray],
               rank_name_prefix: str):
    """Rank 1 TT-op with op in the first dimension.

    """
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    TTop = TensorNetwork()

    rank_indices = [Index(f'{rank_name_prefix}_r1', 1)]
    A1tens = Tensor(cores[0][:, :, np.newaxis],
                    [indices_out[0],
                     indices_in[0],
                     rank_indices[0]])
    TTop.add_node(0, A1tens)
    for ii in range(1, dim):
        rank_indices.append(Index(f'{rank_name_prefix}_r{ii+1}', 1))
        if ii < dim-1:
            eye = cores[ii][np.newaxis, :, :, np.newaxis]
            eye_tens = Tensor(eye,
                              [rank_indices[ii-1],
                               indices_out[ii],
                               indices_in[ii],
                               rank_indices[ii]])
            TTop.add_node(ii, eye_tens)
        else:
            eye = cores[ii][np.newaxis, :, :]
            eye_tens = Tensor(eye, [rank_indices[ii-1],
                                    indices_out[ii],
                                    indices_in[ii]])
            TTop.add_node(ii, eye_tens)
        if ii == 1:
            TTop.add_edge(ii-1, ii)
        else:
            TTop.add_edge(ii-1, ii)
            
    return TTop


def ttop_rank2(indices_in: List[Index], indices_out: List[Index],
               cores_r1: List[np.ndarray],
               cores_r2: List[np.ndarray],
               rank_name_prefix: str):
    """Rank 1 TT-op with op in the first dimension.

    """
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    TTop = TensorNetwork()

    rank_indices = [Index(f'{rank_name_prefix}_r1', 2)]

    core = np.zeros((indices_out[0].size, indices_in[0].size, 2))
    core[:, :, 0] = cores_r1[0]
    core[:, :, 1] = cores_r2[0]
    
    A1tens = Tensor(core, [indices_out[0],
                           indices_in[0],
                           rank_indices[0]])
    
    TTop.add_node(0, A1tens)
    for ii in range(1, dim):
        rank_indices.append(Index(f'{rank_name_prefix}_r{ii+1}', 2))
        if ii < dim-1:

            core = np.zeros((2, indices_out[ii].size,
                             indices_in[ii].size, 2))
            core[0, :, :, 0] = cores_r1[ii]
            core[1, :, :, 1] = cores_r2[ii]

            Aitens = Tensor(core,
                            [rank_indices[ii-1],
                             indices_out[ii],
                             indices_in[ii],
                             rank_indices[ii]])
            TTop.add_node(ii, Aitens)
        else:

            core = np.zeros((2, indices_out[ii].size, indices_in[ii].size))
            core[0, :, :] = cores_r1[ii]
            core[1, :, :] = cores_r2[ii]
            Aitens = Tensor(core, [rank_indices[ii-1],
                                    indices_out[ii],
                                    indices_in[ii]])
            TTop.add_node(ii, Aitens)
        TTop.add_edge(ii-1, ii)
            
    return TTop


def ttop_apply(ttop: TensorNetwork, tt_in: TensorNetwork) -> TensorNetwork:
    """Apply a ttop to a tt tensor.

    # tt overwritten, same free_indices as before
    """
    tt = copy.deepcopy(tt_in)
    dim = tt.dim()
    for ii, (node_op, node_tt) in enumerate(zip(ttop.network.nodes(),
                                                tt.network.nodes())):
        

        op = ttop.network.nodes[node_op]['tensor'].value
        v = tt.network.nodes[node_tt]['tensor'].value
        # print(f"op shape: {node_op}", op.shape)
        # print(f"v shape: {node_tt}", v.shape)
        if ii == 0:
            new_core = np.einsum('ijk,jl->ilk', op, v)
            n = v.shape[0]
            new_core = np.reshape(new_core, (n,-1))
        elif ii < dim-1:
            new_core = np.einsum('ijkl,mkp->mijpl', op, v)
            shape = new_core.shape
            new_core = np.reshape(new_core, (shape[0]*shape[1],
                                             shape[2],
                                             shape[3]*shape[4]))
        else:
            new_core = np.einsum('ijk,mk->mij', op, v)
            shape = new_core.shape
            new_core = np.reshape(new_core, (shape[0]*shape[1], -1))

        tt.network.nodes[node_tt]['tensor'] =  \
            tt.network.nodes[node_tt]['tensor'].update_val_size(new_core)

    # print("After op = ")
    # print(tt)
    return tt
    
 
def gmres(op, # function from in to out
          rhs: TensorNetwork,
          x0: TensorNetwork,
          eps: float = 1e-5,
          round_eps: float=1e-10,
          maxiter:int = 100) -> TensorNetwork:
    """Perform GMRES.
    VERY HACKY
    """

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
            H = np.zeros((jj+2, jj+1))
        else:
            m, n = H.shape
            newH = np.zeros((m+1, n+1))
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
        for ii in range(jj+1):
            # print("ii = ", ii)
            H[ii, jj] = w.inner(v[ii])
            vv = copy.deepcopy(v[ii])
            vv.scale(-H[ii, jj])
            w = w + vv
        # print("inner w = ", w.inner(v[0]), w.inner(w))
        # print("H = ", H)
        # exit(1)
        w = tt_round(w, round_eps)
        H[jj+1, jj] = w.norm()        
        v.append(w.scale(1.0 / H[jj+1, jj]))
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

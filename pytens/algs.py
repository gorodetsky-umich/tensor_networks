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


@dataclass(frozen=True, eq=True)
class Index:
    """Class for denoting an index."""
    name: Union[str, int]
    size: Union[int, List[int]]
    ndim: int

@dataclass(frozen=True, eq=True)
class EinsumArgs:
    """Represent information about contraction in einsum list format"""
    input_str_map: Dict[Union[str, int], str]
    output_str: str
    # arr_list: List[np.ndarray]
    node_list: Optional[List[Union[str, int]]] # names of nodes in input arguments
    free_index_map: Dict[str, Index]
    
def compute_einsum_str(network: nx.DiGraph) -> EinsumArgs:
    """Compute the einsum contraction string.

    VERY IMPORTANT: assumes only one dimension contracted along each edge
    """

    logger.debug(f"---In compute_einsum_str---")
    edge_names = []
    free_edge = []
    free_node_indices = {}
    for u, v, d in network.edges(data=True):
        logger.debug(f"Edge: {u} -> {v}; Data = {d}")
        edge_names.append(d['name'].name)
        # print(f"v = {network.nodes(data=True)[v]}")
        if network.nodes(data=True)[v]['free_node'] is not None:
            free_edge.append(d['name'].name)
            free_node_indices[d['name'].name] = network.nodes(data=True)[v]['free_node']

    mapping = {name: chr(i+97) for i, name in enumerate(edge_names)}
    
    logger.debug("edge names = ", edge_names)
    logger.debug("free_node_indices = ", free_node_indices)
    logger.debug("mapping = ", mapping)
        # if data['free_node'] = False:
        #     names += data['mode_names']
    einsum_str_map = {node: [''] * data['ndim'] for node, data in network.nodes(data=True) if not data['free_node']}
    # output_index_map = {val: free_node_indices[key] for key, val in mapping.items() if key in free_node_indices}
    output_index_map = {val: free_node_indices[key] for key, val in mapping.items() if key in free_node_indices}
    logger.debug("output_index_map = ", output_index_map)
    # print(list(network.nodes(data=True)))
    # print(network.nodes(data=True)['x'])
    
    for node, data in network.nodes(data=True):        
        if not data['free_node']:
            # print(f"Node {node}")
            for neighbor in network[node]:
                edge = network.edges[node, neighbor]
                index = edge['name']
                for modes_involved in edge['dims']:  # assumes only loops once because same index used!
                    einsum_str_map[node][modes_involved[0]] = mapping[index.name]
                    if neighbor in einsum_str_map:
                        einsum_str_map[neighbor][modes_involved[1]] = mapping[index.name]
                    

    input_str_map = {}
    node_list = []
    for node, data in network.nodes(data=True):
        if not data['free_node']:
            input_str_map[node] = ''.join(einsum_str_map[node])
            node_list.append(node)



    
    # einsum_str = ','.join(einsum_str_lst) + '->'
    logger.debug("einsum_str_map = ", einsum_str_map)
    # Calculate the output subscript        
    s = ''.join([''.join(v) for v in list(einsum_str_map.values())])
    char_count = Counter(s)
    output_indices = [char for char in s if char_count[char] == 1]
    output_str = ''.join(output_indices)
    # einsum_str += output_subscript

    # print("einsum_str = ", einsum_str)
    # print("output_indices = ", output_indices)
    output_indices = {ind:output_index_map[ind] for ind in output_indices}
    # print("output_indices = ", index_list)

    # exit(1)

    # print("einsum_str", einsum_str)
    return EinsumArgs(input_str_map, output_str, node_list, output_indices)

class TensorNetwork:
    """Tensor Network Class."""

    def __init__(self, network, free_nodes) -> None:
        """Initialization."""
        self.network = network
        self.free_nodes = free_nodes

    def copy(self) -> Self:
        """Shallow copy."""
        return TensorNetwork(self.network, self.free_nodes)
    
    @classmethod
    def create_empty(cls) -> Self:
        network = nx.DiGraph()
        free_nodes = []
        return cls(network, free_nodes)

    def add_node(self, name: Union[str, int], value: np.ndarray, free_node: Optional[Index] = None) -> None:
        if free_node:
            self.free_nodes.append(free_node)
            self.network.add_node(free_node.name, value=None, ndim=None, free_node=free_node)
        else:
            self.network.add_node(name, value=value, ndim=value.ndim, free_node=free_node)
            
    
    def add_edge(self,
                 name1: Union[str, int],
                 name2: Union[str, int],
                 matched_dims:List[Tuple[int, int]],
                 name: Optional[Index] = None) -> None: 
        data = self.network.nodes(data=True)        
        if name is None:           
            size = [data[name1]['value'].shape[dim] for dim, dim2 in matched_dims]
            name_is = Index(f"r{name1}-{name2}", size, len(matched_dims))
        else:
            name_is = name

        # print(f"ADDING EDGE: {name1} -> {name2}")
        # print(f"data = {data[name1]['free_node'] == None} and {data[name2]['free_node'] is None}")
        free_edge = True
        if (data[name1]['free_node'] is None) and (data[name2]['free_node'] is None):
            free_edge = False
        # print(f"free_edge = {free_edge}")
            
        self.network.add_edge(name1, name2, dims=matched_dims, name=name_is, free_edge=free_edge)

    def value(self, node_name: Union[str, int]):
        return self.network.nodes(data=True)[node_name]['value']
        
    def inspect(self):
        for node, data in self.network.nodes(data=True):
            print(f"Node: {node}, Data: {dict((k,v) for k, v in data.items() if k != 'value')}")
        for u, v, data in self.network.edges(data=True):
            print(f"Edge from {u} to {v}, Data: {data}")

    def attach(self, other: Self) -> Self:
        """Combine two tensor networks.

        Networks are attached at the free indices .
        """

        new_network = nx.compose(self.network, other.network)
        # new_network = nx.union(self.network, other.network)
        new_tens = TensorNetwork.create_empty()
        new_tens.network = new_network        
        for free_node in self.free_nodes:

            if free_node in other.free_nodes:
                # print("free_node = ", free_node)
                node_to_attach_here = list(self.network.predecessors(free_node.name))[0]
                # print("node_to-attach_here", node_to_attach_here)
                edge_data_here = self.network.edges[node_to_attach_here, free_node.name]

                node_to_attach_there = list(other.network.predecessors(free_node.name))[0]
                edge_data_there = other.network.edges[node_to_attach_there, free_node.name]

                loc_here = edge_data_here['dims'][0][0]
                loc_there = edge_data_there['dims'][0][0]
                new_tens.network.remove_node(free_node.name)
                new_tens.add_edge(node_to_attach_here, node_to_attach_there, [(loc_here, loc_there)],
                                  name=edge_data_here['name'])
                
    
        new_tens.free_nodes = list(set(self.free_nodes + other.free_nodes))
        return new_tens

    def contract_along_edge(self, name1, name2):
        pass
    
    def _contract(self,
                  name: Union[str, int],
                  einsum_args: EinsumArgs,
                  arrs: Sequence[np.ndarray]) -> Self:
        
        input_str = ','.join([''.join(v) for v in list(einsum_args.input_str_map.values())])
        ein_str = input_str + '->' + einsum_args.output_str
        out = oe.contract(ein_str, *arrs, optimize='auto')
        # out = oe.contract(ein_str, *arrs, optimize='greedy')
        # out = oe.contract(ein_str, *arrs, optimize='dp')
        # out = oe.contract(ein_str, *arrs, optimize='auto-hq')
    
        return tensor(name, [einsum_args.free_index_map[node] for node in einsum_args.output_str], out)
               
    def contract(self, name) -> Self:
        """Reconstruct the tensor. Very expensive."""

        einsum_args = compute_einsum_str(self.network)
        # print("einsum_args = ", einsum_args)
        
        arrs = [self.network.nodes[node]['value'] for node in einsum_args.node_list]
        return self._contract(name, einsum_args, arrs)

    def integrate(self, indices: Sequence[Index], weights: Sequence[float]) -> Self:
        """Integrate over the chosen indices. So far just uses simpson rule."""

        out = self
        for weight, index in zip(weights, indices):
            # print("\n\n")
            assert index.ndim == 1
            v = np.ones(index.size) * weight
            tens = vector(f'w_{index.name}', index, v)
            # print("Attaching: ")
            # tens.inspect()
            out = out.attach(tens)
            # print("Result: ")
            # out.inspect()

        return out
        # return out.contract('integral')

    def rename(self, prefix) -> Self:
        new_labels = {}
        for node, data in self.network.nodes(data=True):
            # print(self.network.nodes(node))
            if data['free_node'] is None:
                new_labels[node] = f"{prefix}{node}"
            else:
                new_labels[node] = node
                
        # print("new_labels = ", new_labels)
        # NEED TO COPY TO KEEP NODE ORDERING
        self.network = nx.relabel_nodes(self.network, new_labels, copy=True)
        # self.network = nx.relabel_nodes(self.network, new_labels, copy=False)
        for u, v, d in self.network.edges(data=True):
            if d['free_edge'] is False:
                new_name = f"{prefix}{d['name'].name}"
                d['name'] = Index(new_name, d['name'].size, d['name'].ndim)
            
        return self

    def rename_free_nodes(self, free_nodes_map) -> Self:
        new_labels = {}
        for node, data in self.network.nodes(data=True):
            # print(self.network.nodes(node))
            if data['free_node'] is None:
                new_labels[node] = node 
            else:
                new_labels[node] = free_nodes_map[node]
                
        # print("new_labels = ", new_labels)
        # NEED TO COPY TO KEEP NODE ORDERING
        self.network = nx.relabel_nodes(self.network, new_labels, copy=True)
        # self.network = nx.relabel_nodes(self.network, new_labels, copy=False)
        for u, v, d in self.network.edges(data=True):
            if d['free_edge'] is True:
                new_name = free_nodes_map[d['name']]
                d['name'] = new_name
            
        return self    
        
    def scale(self, scale_factor) -> Self:
        for node, data in self.network.nodes(data=True):
            data['value'] *= scale_factor
            break
        return self

    def inner(self, other) -> float:
        """Compute inner product."""
        value = self.attach(other).contract('a').value('a')
        return value

    def ranks(self) -> List[int]:
        """Return ranks.

        Assumes one rank per edge
        """
        ranks = []
        for _, _, data in self.network.edges(data=True):
            if data['free_edge'] == False:
                size = data['name'].size
                if isinstance(size, int):
                    ranks.append(size)
                else:
                    ranks += size
                    
        return ranks
            

    def __getitem__(self, ind) -> Self:
        """Evaluate at some elements.

        Assumes each free-edge is only over 1 dimension.
        """
        # assert len(ind) == len(self.free_nodes)

        logger.debug(f"---In GetItem---: {ind}")

        # cop = self.copy()
        # print("cop = ", cop)
        con_args = compute_einsum_str(self.network)
        new_output_str = con_args.output_str
        new_str_map = con_args.input_str_map
        new_free_index_map = con_args.free_index_map
        arr_map = {}
        for ii, (element, index)  in enumerate(zip(ind, self.free_nodes)):
            logger.debug(f"\n{ii}: {index}")

            # if isinstance(element, int):
            #     print("---> Dimension will be removed!")
            
            parents = self.network.predecessors(index.name)
            for parent in parents:
                # print(f"Parent = {parent}")
                edge = self.network.edges[parent, index.name]
                # print(f"edge = {edge}")
                dim = edge['dims'][0][0]
                arr = self.network.nodes(data=True)[parent]['value']

                # https://stackoverflow.com/a/41418649
                indices = {dim: element}
                ix = [indices.get(d, slice(None)) for d in range(arr.ndim)]
                
                # print(f"\tix = {ix}")
                arr_map[parent] = arr[*ix]
                # print("arr = ", arr.shape)
                
                elem_char = con_args.input_str_map[parent][dim]                
                if isinstance(element, int):  # modify the contraction string
                    # print(f"Need to remove element {elem_char}")                    
                    new_str_map[parent] = new_str_map[parent].replace(elem_char, '')
                    # print(f"\t {new_str_map[parent]}")
                    # print(f"updated input map = {new_str_map}")
                    new_output_str = new_output_str.replace(elem_char, '')
                    del new_free_index_map[elem_char]
                else:
                    old_index = new_free_index_map[elem_char]
                    new_free_index_map[elem_char] = Index(old_index.name,  arr.shape[dim], old_index.ndim)
                
        # print("new_output_str = ", new_output_str)
        # print("new_input_map = ", new_str_map)
        # print("new_free_index_map = ", new_free_index_map)

        new_args = EinsumArgs(new_str_map, new_output_str, None, new_free_index_map)
        node_list = [arr_map[node] for node in new_str_map.keys()]
        return self._contract("eval", new_args, node_list)

        
    def draw_nodes(self, pos, ax=None):
        # Define color and shape maps
        color_map = {'A':'lightblue', 'B':'lightgreen'}
        shape_map = {'A':'o', 'B':'s'}
        size_map = {'A': 300, 'B': 100}
        node_groups = {'A': [], 'B': []}

        with_label = {'A': True, 'B': False}

        for node, data in self.network.nodes(data=True):
            if data['value'] is not None:
                node_groups['A'].append(node)
            else:
                node_groups['B'].append(node)

        for group, nodes in node_groups.items():

            nx.draw_networkx_nodes(self.network, pos,
                                   ax=ax,
                                   nodelist=nodes,
                                   node_color=color_map[group],
                                   node_shape=shape_map[group],
                                   node_size=size_map[group]
                                   # with_label=with_label[group]
                                   )
            if group == 'A':
                node_labels = {node: node for node in node_groups['A']}
                nx.draw_networkx_labels(self.network,
                                        pos,
                                        ax=ax,
                                        labels=node_labels,
                                        font_size=12)
            if group == 'B':
                node_labels = {node: node for node in node_groups['B']}
                nx.draw_networkx_labels(self.network, pos,
                                        ax=ax,
                                        labels=node_labels,
                                        font_size=12)

    def draw_edges(self, pos, ax=None):
        edge_indices = nx.get_edge_attributes(self.network, 'name')
        # print("edge_indices = ", edge_indices)
        edge_labels = {}
        for key, val in edge_indices.items():
            # print("key = ", key, "val = ", val)
            edge_labels[key] = val.name
        nx.draw_networkx_edges(self.network, pos, ax=ax)
        nx.draw_networkx_edge_labels(self.network, pos, ax=ax, edge_labels=edge_labels)
    

    def draw(self, pos=None, ax=None):
        if pos is None:
            # pos = nx.spring_layout(self.network)
            pos = nx.planar_layout(self.network)
        self.draw_nodes(pos, ax=ax)
        self.draw_edges(pos, ax=ax)

    def as_tt(self):
        return TensorTrain(self.network, self.free_nodes)
    

    @property
    def num_tensors(self):
        num = 0
        for node, data in self.network.nodes(data=True):
            if data['value'] is not None:
                num += 1
        return num

    @property
    def dim(self):
        return len(self.free_nodes)
    
    @property
    def nodes(self):
        return list(self.network.nodes)

    @property
    def edge_names(self):
        return self._edge_names

        
class TensorTrain(TensorNetwork):

    def __init__(self, network, free_nodes):
        super().__init__(network, free_nodes)

    def __add__(self, other: Self) -> Self:
        """Add two tensor trains."""
        assert nx.is_isomorphic(self.network, other.network)

        logger.info("---In add---")
        out = copy.deepcopy(self)
        dim = len(self.free_nodes)
        for ii, (node1, node2) in enumerate(zip(self.network.nodes, other.network.nodes)):

            logger.debug(f"node1, node2 = {node1}, {node2}")
            val1 = self.network.nodes[node1]['value']
            val2 = other.network.nodes[node2]['value']
            if val1 is not None:
                logger.debug("val1 shape = ", val1.shape)
                logger.debug("val2 shape = ", val2.shape)
                if ii == 0:                    
                    out.network.nodes[node1]['value'] = np.block([val1, val2])
                elif ii == dim-1:
                    out.network.nodes[node1]['value'] = np.block([[val1], [val2]])
                else:
                    a, b, c = val1.shape
                    d, e, f = val2.shape
                    out.network.nodes[node1]['value'] = np.zeros((a + d, b, c + f))

                    out.network.nodes[node1]['value'][:a, :, :c] = val1
                    out.network.nodes[node1]['value'][a:a+d, :, c:c+f] = val2
                    
        for u, v, d in out.network.edges(data=True):
            if d['free_edge'] is False:
                ind = d['name']
                new_size = out.network.nodes[u]['value'].shape[d['dims'][0][0]]
                new_ind = Index(ind.name, new_size, ind.ndim)
                d['name'] = new_ind
        

        return out

    def __mul__(self, other: Self) -> Self:
        """Multiply two tensor trains."""
        assert nx.is_isomorphic(self.network, other.network)

        out = copy.deepcopy(self)
        dim = len(self.free_nodes)
        for ii, (node1, node2) in enumerate(zip(self.network.nodes, other.network.nodes)):

            val1 = self.network.nodes[node1]['value']
            val2 = other.network.nodes[node2]['value']

            if val1 is not None:

                if ii == 0:
                    n = val1.shape[0]
                    out.network.nodes[node1]['value'] = np.einsum('ij,ik->ijk', val1, val2).reshape(n, -1)
                elif ii == dim-1:
                    n = val1.shape[1]
                    out.network.nodes[node1]['value'] = np.einsum('ji,ki->jki', val1, val2).reshape(-1, n)
                else:
                    n = val1.shape[1]
                    r1 = val1.shape[0] * val2.shape[0]
                    out.network.nodes[node1]['value'] = np.einsum('ijk,ljm->iljkm', val1, val2).reshape(r1, n, -1)

                    
        for u, v, d in out.network.edges(data=True):
            if d['free_edge'] is False:
                ind = d['name']
                new_size = out.network.nodes[u]['value'].shape[d['dims'][0][0]]
                new_ind = Index(ind.name, new_size, ind.ndim)
                d['name'] = new_ind
        

        return out    

    def right_orthogonalize(self, node) -> Self:
        """Right orthogonalize all but first core.

        A right orthogonal core has the r_{k-1} x nk r_k matrix
        R(Gk(ik)) = ( Gk(1) Gk(2) · · · Gk(nk) )
        having orthonormal rows so that

        sum G_k(i) G_k(i)^T  = I

        Leverages the fact that one parent is predecessor
        """
        val1 = self.network.nodes[node]['value']
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
            self.network.nodes[node]['value'] = new_val
        else:
            q, R = np.linalg.qr(val1.T)
            new_val = q.T
            self.network.nodes[node]['value'] = new_val


        predecessor = list(self.network.predecessors(node))
        assert len(predecessor) == 1
        pred = predecessor[0]
        
        val2 = self.network.nodes[pred]['value']
        new_val2 = np.einsum('...i,ij->...j', val2, R.T)
        self.network.nodes[pred]['value'] = new_val2
                        
        return self

    def norm(self, tol=1e-13) -> float:
        t2 = copy.deepcopy(self)
        t2 = t2.rename('t')
        inner = self.inner(t2)
        if np.abs(inner) < tol:
            return np.sqrt(np.abs(inner))
        else:
            # if inner < 0:
            #     print("inner = ", inner)
            return np.sqrt(inner + tol)
        
    def round(self, eps: float) -> Self:
        
        norm2 = self.norm()
        dim = self.num_tensors      
        delta = eps / np.sqrt(dim-1) * norm2

        cores = []
        for node, data in self.network.nodes(data=True):
            if data['free_node'] is None:
                cores.append(node)
                
        # print("DIM = ", dim)
        out = self.right_orthogonalize(cores[dim-1])
        for jj in range(dim-2, 0, -1):
            # print(f"orthogonalizing core {cores[jj]}")
            out = out.right_orthogonalize(cores[jj])

        ii = 0
        # print("ON FORWARD SWEEP")
        for node, data in out.network.nodes(data=True):

            # print(f"Forward sweep: {node}")
            if not data['free_node']:
                value = data['value']
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

                data['value'] = new_core
                    # ASSUMES ONLY 1 NODE NEDS MODIFICATION (TT Structure)
                for neighbor in out.network[node]:
                    # print("neighbor = ", neighbor)
                    if not out.network.nodes[neighbor]['free_node']:
                        # print("In here")
                        val_old = out.network.nodes[neighbor]['value']
                        # print("val_old shape =", val_old.shape)
                        # print("v.shape = ", v.shape)
                        next_val = np.einsum('ij,jk...->ik...', v, val_old)
                        out.network.nodes[neighbor]['value'] = next_val
                        
                        break 
                ii += 1
                if ii == dim-1:
                    break

        # make this its own code
        for u, v, d in out.network.edges(data=True):
            if d['free_edge'] is False:
                ind = d['name']
                new_size = out.network.nodes[u]['value'].shape[d['dims'][0][0]]
                new_ind = Index(ind.name, new_size, ind.ndim)
                d['name'] = new_ind
        self.network = out.network

        return self
    
    def draw_layout(self):
        """Get a gridded layout for the graph."""
        pos = {}
        cols = self.num_tensors
        row = 0        
        col = 0
        switched = False
        for ii, node in enumerate(self.network.nodes()):
            # row = (ii - 1) // cols
            # col = (ii - 1) % cols
            if ii == cols and switched is False:
                col = 0
                row = 1
                switched = True
            col += 1
            pos[node] = (col, -row)  # Negate row to flip the y-axis for better visualization
        return pos

# def Vector(Index, value):

def vector(name: Union[str, int],
           index: Index,
           value: np.ndarray,
           ) -> TensorNetwork:
    vec = TensorNetwork.create_empty()
    vec.add_node(name, value)
    vec.add_node(index.name, None, free_node=index)
    vec.add_edge(name, index.name, [(0,)], name=index)
    return vec


def tensor(name: Union[str, int],
           indices: List[Index],
           value: np.ndarray,
           ) -> TensorNetwork:
    tensor = TensorNetwork.create_empty()
    tensor.add_node(name, value)
    for ii, index in enumerate(indices):
        tensor.add_node(index.name, None, free_node=index)
        tensor.add_edge(name, index.name, [(ii,)], name=index)
    return tensor

class TensorTrainOp(TensorNetwork):

    def __init__(self, network, indices_in: List[Index], indices_out: List[Index]):
        super().__init__(network, indices_in + indices_out)
        self.indices_in = indices_in
        self.indices_out = indices_out

    @classmethod
    def create_empty(cls) -> Self:
        network = nx.DiGraph()
        free_nodes = []
        return cls(network, [], [])

    def apply(self, tt_in: TensorTrain) -> TensorTrain:
        """Assumes common ordering!
        """
        tt = copy.deepcopy(tt_in)
        dim = tt.dim
        for ii, (node_op, node_tt) in enumerate(zip(self.network.nodes(), tt.network.nodes())):
            if tt.network.nodes[node_tt]['free_node'] is None:

                op = self.network.nodes[node_op]['value']
                v = tt.network.nodes[node_tt]['value']
                # print("op shape ", op.shape)
                # print("v shape ", v.shape)
                if ii == 0:
                    new_core = np.einsum('ijk,jl->ikl', op, v)
                    n = v.shape[0]
                    new_core = np.reshape(new_core, (n,-1))
                elif ii < dim-1:
                    new_core = np.einsum('ijkl,mkp->mijlp', op, v)
                    shape = new_core.shape
                    new_core = np.reshape(new_core, (shape[0]*shape[1],
                                                     shape[2],
                                                     shape[3]*shape[4]))
                else:
                    new_core = np.einsum('ijk,mk->mij', op, v)
                    shape = new_core.shape
                    new_core = np.reshape(new_core, (shape[0]*shape[1], -1))
                tt.network.nodes[node_tt]['value'] = new_core
                
        for u, v, d in tt.network.edges(data=True):
            # if d['free_edge'] is False:
            ind = d['name']
            new_size = tt.network.nodes[u]['value'].shape[d['dims'][0][0]]
            new_ind = Index(ind.name, new_size, ind.ndim)
            d['name'] = new_ind
        return tt

        
def gmres(op, # function from in to out
          rhs: TensorTrain,
          x0: TensorTrain,
          eps: float = 1e-5,
          round_eps: float=1e-10,
          maxiter:int = 100) -> TensorTrain:
    """Perform GMRES.
    VERY HACKY
    """

    r0 = rhs + op(x0).scale(-1)
    r0 = r0.round(round_eps)
    beta = r0.norm()

    v1 = r0.scale(1.0 / beta)
    v = [v1]
    y = []
    H = None
    for jj in range(maxiter):
        # print(f"jj = {jj}")
        delta = round_eps

        w = op(v[jj]).round(delta)

        if H is None:
            H = np.zeros((jj+2, jj+1))
        else:
            m, n = H.shape
            newH = np.zeros((m+1, n+1))
            newH[:m, :n] = H
            H = newH
        # print(f"H shape = {H.shape}")
        for ii in range(jj+1):
            H[ii, jj] = w.inner(v[ii].rename('t-'))
            vv = copy.deepcopy(v[ii]).rename('t-')
            w = w + vv.scale(-H[ii, jj])
        # print("H = ", H)
        # w.inspect()
        w = w.round(round_eps)
        H[jj+1, jj] = w.norm()
        v.append(w.scale(1.0 / H[jj+1,jj]))

        e = np.zeros((H.shape[0]))
        e[0] = beta
        yy, resid, _, _ = np.linalg.lstsq(H, e)
        # print(f"Iteration {jj}: resid = {resid}")
        y.append(yy)
        # if resid < eps:                
        #     break

    x = copy.deepcopy(x0)
    for ii in range(jj):
        x = x + v[ii].scale(y[-1][ii])
    x = x.round(round_eps)
    r0 = rhs + op(x).scale(-1)
    resid = r0.norm()
    return x, resid

    
def rand_tt(name: Union[str, int],
            indices: List[Index],
            ranks: List[int]) -> TensorTrain:
    """Return a random tt."""

    dim = len(indices)

    tt = TensorTrain.create_empty()
    core_names = []
    for ii, index in enumerate(indices):

        core_name = f"{name}_{ii+1}"
        core_names.append(core_name)
        if ii == 0:
            tt.add_node(core_name, np.random.randn(index.size, ranks[1]))
        elif ii < dim-1:
            tt.add_node(core_name, np.random.randn(ranks[ii], index.size, ranks[ii+1]))
        else:
            tt.add_node(core_name, np.random.randn(ranks[ii], index.size))
            
        if ii == 1:
            tt.add_edge(core_names[ii-1], core_names[ii], [(1, 0)])
        elif ii > 1:
            tt.add_edge(core_names[ii-1], core_names[ii], [(2, 0)])

    for ii, (index, core_name) in enumerate(zip(indices, core_names)):
        tt.add_node(index.name, None, free_node=index)
        if ii == 0:
            tt.add_edge(core_name, index.name, [(0,)], name=index)
        else:
            tt.add_edge(core_name, index.name, [(1,)], name=index)

    return tt


def tt_separable(name: Union[str, int],
                 indices: List[Index],
                 funcs: List[np.ndarray]) -> TensorTrain:
    """Rank 2 function formed by summation of functoins of individual dimensions"""

    dim = len(indices)

    tt = TensorTrain.create_empty()
    core_names = []
    for ii, index in enumerate(indices):

        core_name = f"{name}_{ii+1}"
        core_names.append(core_name)
        if ii == 0:
            val = np.ones((index.size, 2))
            val[:, 0] = funcs[ii]
            tt.add_node(core_name, val)
        elif ii < dim-1:
            val = np.zeros((2, index.size, 2))
            val[0, :, 0] = 1.0
            val[1, :, 0] = funcs[ii]
            val[1, :, 1] = 1.0
            tt.add_node(core_name, val)
        else:
            val = np.ones((2, index.size))
            val[1, :] = funcs[ii]
            tt.add_node(core_name, val)
            
        if ii == 1:
            tt.add_edge(core_names[ii-1], core_names[ii], [(1, 0)])
        elif ii > 1:
            tt.add_edge(core_names[ii-1], core_names[ii], [(2, 0)])

    for ii, (index, core_name) in enumerate(zip(indices, core_names)):
        tt.add_node(index.name, None, free_node=index)
        if ii == 0:
            tt.add_edge(core_name, index.name, [(0,)], name=index)
        else:
            tt.add_edge(core_name, index.name, [(1,)], name=index)

    return tt

def tt_rank1(name: Union[str, int],
             indices: List[Index],
             funcs: List[np.ndarray]) -> TensorTrain:
    """Rank 2 function formed by summation of functoins of individual dimensions"""

    dim = len(indices)

    tt = TensorTrain.create_empty()
    core_names = []
    for ii, index in enumerate(indices):

        core_name = f"{name}_{ii+1}"
        core_names.append(core_name)
        if ii == 0:
            val = np.ones((index.size, 1))
            val[:, 0] = funcs[ii]
            tt.add_node(core_name, val)
        elif ii < dim-1:
            val = np.zeros((1, index.size, 1))
            val[0, :, 0] = funcs[ii]
            tt.add_node(core_name, val)
        else:
            val = np.ones((1, index.size))
            val[0, :] = funcs[ii]
            tt.add_node(core_name, val)
            
        if ii == 1:
            tt.add_edge(core_names[ii-1], core_names[ii], [(1, 0)])
        elif ii > 1:
            tt.add_edge(core_names[ii-1], core_names[ii], [(2, 0)])

    for ii, (index, core_name) in enumerate(zip(indices, core_names)):
        tt.add_node(index.name, None, free_node=index)
        if ii == 0:
            tt.add_edge(core_name, index.name, [(0,)], name=index)
        else:
            tt.add_edge(core_name, index.name, [(1,)], name=index)

    return tt


def ttop_1dim(indices_in: List[Index], indices_out: List[Index], A1d):
    """Rank 1 TT-op with op in the first dimension."""
    assert len(indices_in) == len(indices_out)
    dim = len(indices_in)
    TTop = TensorTrainOp.create_empty()
    
    TTop.add_node(0, A1d[:, :, np.newaxis])
    for ii in range(1, dim):
        if ii < dim-1:
            TTop.add_node(ii, np.eye(indices_in[ii].size)[np.newaxis, :, :, np.newaxis])
        else:
            TTop.add_node(ii, np.eye(indices_in[ii].size)[np.newaxis, :, :])
        if ii == 1:
            TTop.add_edge(ii-1, ii, [(2, 0)])
        else:
            TTop.add_edge(ii-1, ii, [(3, 0)])
            
    ## Add freenodes to hold output edges
    for ii, (index_in, index_out) in enumerate(zip(indices_in, indices_out)):
        TTop.add_node(index_in.name, None, free_node=index_in)
        TTop.add_node(index_out.name, None, free_node=index_out)
        if ii == 0:
            TTop.add_edge(ii, index_in.name, [(1,)], name = index_in)
            TTop.add_edge(ii, index_out.name, [(0,)], name = index_out)
        else:
            TTop.add_edge(ii, index_in.name, [(2,)], name = index_in)
            TTop.add_edge(ii, index_out.name, [(1,)], name = index_out)
            
    TTop.indices_in = indices_in
    TTop.indices_out = indices_out
    return TTop
    

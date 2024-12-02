"""Hierarchical Tucker."""

import numpy as np
import htucker as ht
import pickle as pckl

from math import ceil
import os
from warnings import warn


__all__ = [
    "HTucker",
    "TuckerCore",
    "TuckerLeaf",
    "hosvd",
    "truncated_svd",
    "create_permutations",
    "split_dimensions",
    "mode_n_unfolding",
    "createDimensionTree",
    "Tree",
    "Node"
]


class NotFoundError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
  
class TuckerCore:
    # Object for tucker cores. Planning to use it to create the recursive graph structure
    def __init__(self,core=None, parent=None, dims=None, idx=None) -> None:
        self.parent=parent
        self.core=core
        self.left=None
        self.right=None
        self.dims=dims
        self.core_idx=idx
        self.ranks=[]
        self.children = []

        if parent is None:
            self._isroot=True
        else:
            self._isroot=False

        self._isexpanded=False

    def get_ranks(self):
        if self.core is not None:
            self.ranks=list(self.core.shape)
    
    def contract_children(self):
        # Need to write another contraction code for n-ary splits
        _ldims=len(self.left.dims)+1
        _rdims=len(self.right.dims)+1
        
        left_str = ''.join([chr(idx) for idx in range(97,97+_ldims)])
        right_str = ''.join([chr(idx) for idx in range(97+_ldims,97+_ldims+_rdims)])
        if self._isroot:
            core_str = left_str[-1]+right_str[-1]
        else:    
            core_str = left_str[-1]+right_str[-1]+chr(97+_ldims+_rdims)

        result_str = core_str.replace(left_str[-1],left_str[:-1])
        result_str = result_str.replace(right_str[-1],right_str[:-1])
        self.core = np.einsum(
            ','.join([left_str,right_str,core_str])+'->'+result_str,
            self.left.core,self.right.core,self.core,optimize=True
            )
        pass

    def contract_children_dimension_tree(self):
        dimensions=[]
        matrices=[]
        for chld in self.children:
            if type(chld) is ht.TuckerLeaf:
                dimensions.append(2)
            else:
                dimensions.append(len(chld.dims)+1)
            matrices.append(chld.core)
        matrices.append(self.core)
        strings=[]
        core_string=""
        last_char=97
        for dims in dimensions:
            strings.append(''.join([chr(idx) for idx in range(last_char,last_char+dims)]))
            last_char+=dims
            core_string += strings[-1][-1]
        core_string+=chr(last_char)
        result_string=""
        for stri in strings:
            result_string += stri[:-1]
        result_string+=core_string[-1]
        if self.parent is not None:
            pass
        else: #We are contracting the root node
            # We need to adjust the einstein summation strings for root node
            result_string, core_string = result_string[:-1],core_string[:-1]
        self.core = eval(
            "np.einsum("+
            "'"+
            ",".join([
                ','.join(strings+[core_string])+'->'+result_string+"'",
                ",".join([f"matrices[{idx}]" for idx in range(len(matrices))]),
                'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
            )+
            ")"
        )

    @property
    def shape(self):
        return self.core.shape

class TuckerLeaf:
    def __init__(self, matrix=None, parent=None, dims=None, idx=None) -> None:
        self.parent=parent
        self.core=matrix #Refactoring this as core for consistency with TuckerCore
        self.dims=dims
        self.leaf_idx=idx
        if matrix is not None: self.rank=matrix.shape[1]
    @property
    def shape(self):
        return self.core.shape
    def get_ranks(self):
        if self.core is not None:
            self.rank=self.core.shape[-1]
        
class HTucker:

    # harded for 4d first
    def __init__(self):
        # TODO: Move initialization to initialize function entirely. 
        # This function is there to just create some of the necessary variables
        self._leaf_count = None
        self.leaves = None
        self.transfer_nodes = None
        self.root = None
        # self.nodes2Expand=[]
        self._iscompressed=False
        self._dimension_tree = None
        self.batch_dimension = None
        self.rtol = None 


    def initialize(self,tensor,dimension_tree=None, batch=False, batch_dimension=None):
        self.original_shape = list(tensor.shape)
        if batch: 
            self._leaf_count = len(self.original_shape)-1
            if batch_dimension is None:
                batch_dimension = len(self.original_shape)-1
            self.batch_dimension = batch_dimension
            self.original_shape = self.original_shape[:batch_dimension]+self.original_shape[batch_dimension+1:]
            self.batch_count = tensor.shape[batch_dimension]
        else:
            self.batch_count = 1
            self._leaf_count=len(self.original_shape)
        self._dimension_tree = dimension_tree
        self.leaves = [None]*self._leaf_count
        self.transfer_nodes = [None]*(self._leaf_count-2) #Root node not included here
        self.root = None
        self.nodes2Expand = []
        self._iscompressed = False
        self.allowed_error=0

        
    def compress_root2leaf(self, tensor=None, isroot=False): # isroot flag is subject to remove
        # TODO: Replace initial SVD with HOSVD -> Done, Requires testing -> Testing done
        # TODO: Create a structure for the HT -> 
        # TODO: Make compress() function general for n-dimensional tensors -> Done, need to check imbalanced trees (n=5) -> Testing done
        # TODO: Write a reconstruct() function. -> Done including testing.
        # TODO: Implement error truncated compression
        assert(self._iscompressed is False)
        _leaf_counter=0
        _node_counter=0

        #this if check looks unnecessary
        if self.root is None: isroot=True
        # This looks unnecessary

        if tensor is None:
            raise NotFoundError("No tensor is given. Please check if you provided correct input(s)")
        
        dims=list(tensor.shape)

        # initial split for the tensor
        left,right=split_dimensions(dims)
        
        self.root=TuckerCore(dims=dims)

        self.root.left=TuckerCore(parent=self.root, dims=left)
        self.root.right=TuckerCore(parent=self.root, dims=right)
        # Reshape initial tensor into a matrix for the first splt
        tensor=tensor.reshape(np.prod(left),np.prod(right), order='F')
        
        self.root.core, [self.root.left.core, self.root.right.core] = hosvd(tensor)
        # The reshapings below might be unnecessary, will look into those
        # self.root.left.core = self.root.left.core.reshape(left+[-1],order='F')
        # self.root.right.core = self.root.right.core.reshape(right+[-1],order='F')
        self.root.get_ranks()

        self.nodes2Expand.append(self.root.left)
        self.nodes2Expand.append(self.root.right)

        while self.nodes2Expand:

            node=self.nodes2Expand.pop(0)
            # if len(node.dims)==1:
            #     print(node.dims)
            #     continue
            left,right=split_dimensions(node.dims)



            node.core = node.core.reshape((np.prod(left),np.prod(right),-1), order='F')
            node.core, [lsv1, lsv2, lsv3] = hosvd(node.core)
            # Contract the third leaf with the tucker core for now
            node.core = np.einsum('ijk,lk->ijl',node.core,lsv3,optimize=True)
            node._isexpanded = True
            node.core_idx = _node_counter
            self.transfer_nodes[_node_counter] = node
            _node_counter += 1
            
            if len(left)==1:
                # i.e we have a leaf
                node.left=TuckerLeaf(matrix=lsv1,parent=node, dims=left, idx=_leaf_counter)
                self.leaves[_leaf_counter]=node.left
                _leaf_counter+=1
            else:
                node.left=TuckerCore(core=lsv1, parent=node, dims=left)
                self.nodes2Expand.append(node.left)

            if len(right)==1:
                node.right=TuckerLeaf(matrix=lsv2,parent=node, dims=right, idx=_leaf_counter)
                self.leaves[_leaf_counter]=node.right
                _leaf_counter+=1
            else:
                node.right=TuckerCore(core=lsv2, parent=node, dims=right)
                self.nodes2Expand.append(node.right)
        
        self._iscompressed=True
        return None
    
    def compress_leaf2root(self,tensor=None,dimension_tree=None):
        assert(self._iscompressed is False)
        if tensor is None:
            raise NotFoundError("No tensor is given. Please check if you provided correct input(s)")
        if (self._dimension_tree is None) and (dimension_tree is None):
            warn("No dimension tree is given, creating one with binary splitting now...")
            self._dimension_tree = createDimensionTree(tensor,2,1)
        else:
            self._dimension_tree = dimension_tree

        if self.rtol is not None:
            _num_total_svds=sum([len(items) for items in self._dimension_tree._level_items[1:]])-1
            # self.allowed_error=np.linalg.norm(tensor)*self.rtol/_num_total_svds # allowed error per svd step
            # print(np.sqrt(2*len(tensor.shape)-3),len(tensor.shape),_num_total_svds)
            # print(np.linalg.norm(tensor),self.rtol,np.sqrt(2*len(tensor.shape)-3))
            # self.allowed_error=np.linalg.norm(tensor)*self.rtol*np.sqrt(2*len(tensor.shape)-3)#/_num_total_svds # allowed error per svd step

            self.allowed_error=np.linalg.norm(tensor)*self.rtol/np.sqrt(2*len(tensor.shape)-3) # allowed error per svd
            # self.allowed_error=np.linalg.norm(tensor)*self.rtol/np.sqrt(2*len(tensor.shape)-3) *(_num_total_svds/len(tensor.shape))# allowed error per svd
            
            # self.allowed_error=np.linalg.norm(tensor)*self.rtol/np.sqrt(_num_total_svds)#/_num_total_svds # allowed error per svd step
            # self.allowed_error=np.linalg.norm(tensor)*self.rtol*(2+np.sqrt(2))*(np.sqrt(len(tensor.shape)))/_num_total_svds # allowed error per svd step
            # self.allowed_error=np.linalg.norm(tensor)*self.rtol/((2+np.sqrt(2))*(np.sqrt(len(tensor.shape))))*np.sqrt(_num_total_svds) # allowed error per svd step
            # self.allowed_error=np.linalg.norm(tensor)*self.rtol/(self._dimension_tree._depth) # allowed error per svd step
        
        ## Start with initial HOSVD to get the initial set of leaves
        existing_leaves=[]
        existing_nodes=[]
        node_idx = self._dimension_tree._nodeCount-1
        # print(self.allowed_error)
        # print(len(self._dimension_tree._level_items[-1]))
        # _ , leafs = hosvd(tensor,tol=self.allowed_error*len(self._dimension_tree._level_items[-1]))
        _ , leafs = hosvd(tensor,tol=self.allowed_error)
        # _ , leafs = hosvd(tensor,tol=self.allowed_error,dimensions=len(self._dimension_tree._level_items[-1]))
        # _ , leafs = hosvd(tensor,tol=self.allowed_error*len(self._dimension_tree._level_items[-1]),dimensions=len(self._dimension_tree._level_items[-1]))
        # _ , leafs = hosvd(tensor,tol=self.allowed_error)
        for li in self._dimension_tree._level_items[-1]:
            if li._isleaf:
                leaf_idx=li._dimension_index[0]
                self.leaves[leaf_idx]=ht.TuckerLeaf(matrix=leafs[leaf_idx],dims=li.val[0],idx=leaf_idx)
                li._ranks[0]=self.leaves[leaf_idx].shape[-1]
                li.real_node = self.leaves[leaf_idx]
                li.parent._ranks[li.parent._ranks.index(None)]=li.real_node.rank
        for lf in self.leaves:
            if (lf is not None) and (lf not in existing_leaves):
                existing_leaves.append(lf)
                tensor = mode_n_product(tensor,lf.core,[lf.leaf_idx,0])


        ## Niye last layer'i kaydettigini hatirla
        ## Sanirim mevcut layerin bir onceki layerla baglantisini yapmak icin var burada last layer
    
        last_layer=self._dimension_tree._level_items[-1]
        
        for layer in self._dimension_tree._level_items[1:-1][::-1]:
            new_shape=[]
            # burada ilk pass hangi dimensionun hangi dimensionla birlesecegini anlamak icin var
            deneme=[]
            for item in layer:
                # print(item._isleaf,item._ranks,item.val,item._dimension_index)
                deneme.extend(item._dimension_index)
                if item._isleaf:
                    new_shape.append(item.val[0])
                else:
                    for chld in item.children:
                        item.real_children.append(chld.real_node)
                    new_shape.append(np.prod(list(filter(lambda item: item is not None, item._ranks))))
            # dimension contractionun nasil olacagini planladiktan sonra reshape et ve HOSVD hesapla
            # if len(new_shape)>1:
            tensor=tensor.reshape(new_shape,order="F")
            # print(len(new_shape),new_shape)
            # print(len(layer),tensor.shape, deneme, len(deneme),new_shape,len(new_shape),item._dimension_index)
            # if len(new_shape)==2:
            #     _ , leafs = hosvd(tensor,tol=self.allowed_error)
            #     print(leafs[0].shape,leafs[1].shape)
            # else:
            #     # _ , leafs = hosvd(tensor,tol=self.allowed_error*len(new_shape))
            #     _ , leafs = hosvd(tensor,tol=self.allowed_error)
            _ , leafs = hosvd(tensor,tol=self.allowed_error)
            # _ , leafs = hosvd(tensor,tol=self.allowed_error*len(deneme))
            # else:
                # We are at the first level 
                # leafs = [tensor]
            # HOSVD hesaplandiktan sonra left singular vectorleri ilgili dimensionlarla contract etmek gerekiyor
            for item_idx, item in enumerate(layer):
                # LSV'leri ilgili dimensionlarla contract et
                # if len(layer)>1:
                tensor = mode_n_product(tensor,leafs[item_idx],[item_idx,0])
                # else:
                #     pass
                if item._isleaf:
                    # The current item is a leaf.
                    leaf_idx=item._dimension_index[0]
                    lf=ht.TuckerLeaf(matrix=leafs[item_idx],dims=item.val[0],idx=leaf_idx)
                    self.leaves[leaf_idx]=lf
                    item.real_node=lf
                    item._ranks[0] = lf.rank
                    item.parent._ranks[item.parent._ranks.index(None)] = lf.rank
                    pass
                else:
                    # The current item is a transfer node.
                    # Create tucker core and insert the transfer tensor.
                    item._ranks[item._ranks.index(None)]=leafs[item_idx].shape[-1]
                    item.parent._ranks[item.parent._ranks.index(None)]=leafs[item_idx].shape[-1]
                    # new_shape = 
                    node = ht.TuckerCore(
                        core=leafs[item_idx].reshape(item._ranks,order="F"),
                        dims=item.val.copy(),
                        idx=item._dimension_index.copy()
                        )
                    self.transfer_nodes[node_idx]=node
                    node._isexpanded = True
                    node_idx -=1
                    if len(layer) !=1: node._isroot = False
                    # Create 
                    # item.parent.real_children.append(node)
                    item.real_node = node
                    for chld in item.real_children:
                        node.children.append(chld)
                        chld.parent = node
                    for chld in item.children:
                        chld.real_parent = node
                    # self.transfer_nodes
                    node.get_ranks()
                    # np.prod(list(filter(lambda item: item is not None, sk._ranks)))
                    # learn the children and connect it to the current node. 
                    pass

            last_layer=layer
        layer = self._dimension_tree._level_items[0]
        item = layer[0]
        for chld in item.children:
            item.real_children.append(chld.real_node)
        node = ht.TuckerCore(
            core = tensor,
            dims = item.val.copy(),
            idx = item._dimension_index.copy()
        )
        self.root = node
        node._isexpanded = True
        item.real_node = node
        for chld in item.real_children:
            node.children.append(chld)
            chld.parent = node
        for chld in item.children:
            chld.real_parent = node
        node.get_ranks()
        self._iscompressed=True
        return None

    def compress_leaf2root_batch(self, tensor=None, dimension_tree=None, batch_dimension=None):
        assert(self._iscompressed is False)
        if tensor is None:
            raise NotFoundError("No tensor is given. Please check if you provided correct input(s)")
        if batch_dimension is None:
            warn("No batch dimension is given, assuming last dimension as batch dimension!")
            batch_dimension = len(tensor.shape)-1
        batch_count = tensor.shape[batch_dimension]
        if (self._dimension_tree is None) and (dimension_tree is None):
            warn("No dimension tree is given, creating one with binary splitting now...")
            tree_shape = list(tensor.shape)[:batch_dimension]+list(tensor.shape)[batch_dimension+1:]
            self._dimension_tree = createDimensionTree(tree_shape,2,1)
        else:
            self._dimension_tree = dimension_tree

        if self.rtol is not None:
            # _num_total_svds=sum([len(items) for items in self._dimension_tree._level_items[1:]])-1
            # self.allowed_error=np.linalg.norm(tensor)*self.rtol/np.sqrt(2*len(tensor.shape)-3) # allowed error per svd
            num_svds = 2*(len(tensor.shape)-1)-2
            tenNorm = np.linalg.norm(tensor)
            cur_norm = tenNorm
            total_allowed_error = tenNorm*self.rtol
            self.allowed_error=tenNorm*self.rtol/np.sqrt(num_svds) # allowed error per svd

            # TODO: here, the allowed error does not take the fact that one of the dimensions is the batch dimension and therefore will be ignored
        
        # print(num_svds,total_allowed_error,self.allowed_error,cur_norm,tenNorm)
        node_idx = self._dimension_tree._nodeCount-1

        hosvd_dimensions = [item._dimension_index[0] for item in self._dimension_tree._level_items[-1] if item._isleaf]
        leafs = hosvd_only_for_dimensions(
            tensor,
            tol = self.allowed_error,
            dims = hosvd_dimensions,
            # batch_dimension = batch_dimension, 
            contract = False,
        )
        for leaf_idx,li,leaf in zip(hosvd_dimensions,self._dimension_tree._level_items[-1],leafs):
            if li._isleaf:
                # leaf_idx=li._dimension_index[0]
                self.leaves[leaf_idx]=ht.TuckerLeaf(matrix=leaf,dims=li.val[0],idx=leaf_idx)
                li._ranks[0]=self.leaves[leaf_idx].shape[-1]
                li.real_node = self.leaves[leaf_idx]
                li.parent._ranks[li.parent._ranks.index(None)]=li.real_node.rank
            tensor = mode_n_product(tensor=tensor, matrix=leaf, modes=[leaf_idx,0])
        cur_norm = min(np.linalg.norm(tensor),tenNorm)
        # print()
        # print(
        #     num_svds,
        #     len(self._dimension_tree._level_items[-1]),
        #     tenNorm,
        #     cur_norm,
        #     total_allowed_error,
        #     self.allowed_error,
        #     (cur_norm**2/tenNorm**2),
        #     np.sqrt(1-(cur_norm**2/tenNorm**2)),
        #     max(((tenNorm**(2))-(cur_norm**(2))),0),
        #     )
        # print()

        num_svds -= len(self._dimension_tree._level_items[-1])
        for layer in self._dimension_tree._level_items[1:-1][::-1]:
            # print(cur_norm)
            self.allowed_error =  np.sqrt((total_allowed_error**(2)) - max(((tenNorm**(2))-(cur_norm**(2))),0))
            # print(self.allowed_error)
            self.allowed_error = self.allowed_error/np.sqrt(num_svds)
            # print(self.allowed_error)
            # print(num_svds,self.allowed_error,cur_norm,tenNorm)

            hosvd_dimensions = [item._dimension_index[0] for item in layer if item._isleaf]
            # _ , leafs = hosvd(tensor,tol=self.allowed_error)
            if hosvd_dimensions:
                # Compute missing leaves (if any)
                leafs = hosvd_only_for_dimensions(
                    tensor,
                    tol = self.allowed_error,
                    dims = hosvd_dimensions,
                    # batch_dimension = batch_dimension, 
                    contract = False,
                )
                leaf_ctr=0
                for item_idxx,item in enumerate(layer):
                    if item._isleaf:
                        leaf_idx=item._dimension_index[0]
                        self.leaves[leaf_idx]=ht.TuckerLeaf(matrix=leafs[leaf_ctr],dims=item.val[0],idx=leaf_idx)
                        item._ranks[0]=self.leaves[leaf_idx].shape[-1]
                        item.real_node = self.leaves[leaf_idx]
                        # item.parent._ranks[item.parent._ranks.index(None)]=item.real_node.rank
                        item.parent._ranks[item.parent.children.index(item)]=item.real_node.rank
                        leaf_ctr+=1
            # hosvd_dimensions = [item._dimension_index[0] for item in layer if not item._isleaf]
            
            new_shape=[]
            inter_shape=[]
            dimension_shift=0
            for item in layer:
                child_list=[]
                if item.children:
                    if item._dimension_index[0]<batch_dimension:
                        # print(len(item.children)-1)
                        dimension_shift+=len(item.children)-1
                    for chld in item.children:
                        child_list.append(chld.shape[-1])
                else:
                    child_list.append(item.shape[0])
                new_shape.append(np.prod(child_list))
                # inter_shape.append(np.prod(child_list))
            # print(batch_dimension-dimension_shift)
            
            batch_dimension-=dimension_shift
            new_shape.insert(batch_dimension,batch_count)
            tensor=tensor.reshape(new_shape,order="F")
            hosvd_dimensions = [item_idx for item_idx,item in enumerate(layer) if not item._isleaf]
            nodes = hosvd_only_for_dimensions(
                tensor,
                tol = self.allowed_error,
                dims = hosvd_dimensions,
                contract = False
            )
            node_ctr=0
            leaf_ctr=0
            for item_idx, item in enumerate(layer):
                if not item._isleaf:
                    # The current item is a node
                    item._ranks[item._ranks.index(None)]=nodes[node_ctr].shape[-1]
                    # item.parent._ranks[item.parent._ranks.index(None)]=nodes[node_ctr].shape[-1]
                    item.parent._ranks[item.parent.children.index(item)]=nodes[node_ctr].shape[-1]
                    node = ht.TuckerCore(
                        core=nodes[node_ctr].reshape(item._ranks,order="F"),
                        dims=item.val.copy(),
                        idx=item._dimension_index.copy()
                        )
                    self.transfer_nodes[node_idx]=node

                    tensor = mode_n_product(tensor,nodes[node_ctr],modes=[item_idx,0])
                    node._isexpanded = True
                    node_idx -=1
                    if len(layer) !=1: node._isroot = False
                    item.real_node = node
                    for chld in item.real_children:
                        node.children.append(chld)
                        chld.parent = node
                    for chld in item.children:
                        chld.real_parent = node
                    # self.transfer_nodes
                    node.get_ranks()

                    node_ctr += 1
                else:
                    # The current item is a leaf
                    tensor = mode_n_product(tensor,leafs[leaf_ctr],modes=[item_idx,0])
                    leaf_ctr += 1
                    # leaf_idx = item._dimension_index[0]
                    # self.leaves[leaf_idx] = ht.TuckerLeaf(matrix=leafs[leaf_idx],dims=item.val[0],idx=leaf_idx)
                    # item.ranks[0] = self.leaves[leaf_idx].shape[-1]
                    # item.real_node = self.leaves[leaf_idx]
                    # item.parent_.ranks[item.parent_ranks.index(None)] = item.real.node.rank
            # cur_norm=np.linalg.norm(tensor)
            cur_norm = min(np.linalg.norm(tensor),tenNorm)
            # print()
            # print(
            #     num_svds,
            #     len(layer),
            #     tenNorm,
            #     cur_norm,
            #     total_allowed_error,
            #     self.allowed_error,
            #     (cur_norm**2/tenNorm**2),
            #     np.sqrt(1-(cur_norm**2/tenNorm**2)),
            #     max(((tenNorm**(2))-(cur_norm**(2))),0),
            #     )
            # print()
            num_svds -= len(layer)
        layer = self._dimension_tree._level_items[0]
        item = layer[0]
        for chld in item.children:
            item.real_children.append(chld.real_node)
        node = ht.TuckerCore(
            core = tensor,
            dims = item.val.copy(),
            idx = item._dimension_index.copy()
        )
        self.root = node
        node._isexpanded = True
        item.real_node = node
        for chld in item.real_children:
            node.children.append(chld)
            chld.parent = node
        for chld in item.children:
            chld.real_parent = node
        node.get_ranks()
        self._iscompressed=True
    
        return None

    

    def reconstruct_all(self):
        # The strategy is to start from the last core and work the way up to the root.
        assert(self._iscompressed)
        _transfer_nodes=self.transfer_nodes.copy()
        if self._dimension_tree is None:
            while _transfer_nodes:
                node=_transfer_nodes.pop(-1)
                node.contract_children()
            self.root.contract_children()
        else:
            while _transfer_nodes:
                node=_transfer_nodes.pop(-1)
                node.contract_children_dimension_tree()
            self.root.contract_children_dimension_tree()

        self._iscompressed=False
        return None
    
    def project(self,new_tensor,batch=False, batch_dimension = None):
        assert(self._iscompressed is True)
        new_tensor_shape = new_tensor.shape
        if batch:
            if list(new_tensor_shape[:batch_dimension]+new_tensor_shape[batch_dimension+1:])!=self.original_shape:
                try:
                    shape = np.arange(len(new_tensor_shape)).tolist()
                    shape.pop(batch_dimension)
                    shape=shape+[batch_dimension]
                    new_tensor.transpose(shape)
                    new_tensor = new_tensor.reshape(self.original_shape+[-1],order="F")
                except ValueError:
                    warn(f"Presented tensor has shape {new_tensor.shape}, which is not compatible with {tuple(self.original_shape)}!")
        else:
            if list(new_tensor_shape)!=self.original_shape:
                try:
                    new_tensor = new_tensor.reshape(self.original_shape,order="F")
                    # 2+2
                except ValueError:
                    warn(f"Presented tensor has shape {new_tensor.shape}, which is not compatible with {tuple(self.original_shape)}!")
        
        for layer in self._dimension_tree._level_items[::-1][:-1]:
            idxCtr = 0
            strings=[]
            last_char=97
            dims = len(new_tensor.shape)
            coreString =[chr(idx) for idx in range(last_char,last_char+dims)]
            strings.append(''.join(coreString))
            last_char+=dims
            for itemIdx, item in enumerate(layer):
                if type(item.real_node) is ht.TuckerLeaf:
                    # strings.append(strings[0][item._dimension_index[0]]+chr(last_char))
                    # coreString[item._dimension_index[0]]=chr(last_char)
                    if (item in self._dimension_tree._level_items[-1]) and (item.real_node.leaf_idx != idxCtr):
                        idxCtr+=1 
                    strings.append(strings[0][idxCtr]+chr(last_char))
                    coreString[idxCtr]=chr(last_char)
                    last_char+=1
                    idxCtr+=1
                elif type(item.real_node) is ht.TuckerCore:
                    # icerde ayri minik bir counter tut, leaf olunca 1 core olunca 2 ilerlet olsun bitsin
                    # counter da her layerda sifirlansin.
                    contractionDims = len(item.shape)-1
                    strings.append(strings[0][idxCtr:idxCtr+contractionDims]+chr(last_char))
                    coreString[idxCtr]=chr(last_char)
                    last_char+=1
                    for stringIdx in range(1,contractionDims):
                        coreString[idxCtr+stringIdx]=""
                    # last_char += 1
                    idxCtr += contractionDims
                else:
                    ValueError(f"Unknown node type! {type(item)} is not known!")
            try:
                new_tensor = eval(
                    "np.einsum("+
                    "'"+
                    ",".join([
                        ','.join(strings)+'->'+"".join(coreString)+"'",'new_tensor',
                        ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                        'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                    )+
                    ")"
                )
            except ValueError:
                for ii, string in enumerate(strings):
                    tempstr = [*string]
                    for jj, chrs in enumerate(tempstr):
                            if ord(chrs)>ord("z"):
                                    strings[ii]=strings[ii].replace(chrs,chr(ord(chrs)-ord("z")+ord("A")-1),jj)
                for jj, chrs in enumerate(coreString):
                    if ord(chrs)>ord("z"):
                            coreString[jj]=chr(ord(chrs)-ord("z")+ord("A")-1)
                new_tensor = eval(
                    "np.einsum("+
                    "'"+
                    ",".join([
                        ','.join(strings)+'->'+"".join(coreString)+"'",'new_tensor',
                        ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                        'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                    )+
                    ")"
                )
        return new_tensor
    def reconstruct(self,core,batch=False):
        # if list(new_tensor.shape)!=self.original_shape:
        #     try:
        #         new_tensor = new_tensor.reshape(self.original_shape,order="F")
        #     except ValueError:
        #         warn(f"Presented tensor has shape {new_tensor.shape}, which is not compatible with {tuple(self.original_shape)}!")
        
        for layer in self._dimension_tree._level_items[1:]:
            # idxCtr = 0
            strings=[]
            last_char=97
            dims = len(core.shape)
            coreString =[chr(idx) for idx in range(last_char,last_char+dims)]
            strings.append(''.join(coreString))
            last_char+=dims
            for itemIdx, item in enumerate(layer):
                if type(item.real_node) is ht.TuckerLeaf:
                    tempStr = chr(last_char)
                    if (item in self._dimension_tree._level_items[-1]) and (item.real_node.leaf_idx != itemIdx):
                        strings.append(tempStr+strings[0][item.real_node.leaf_idx])
                        coreString[item.real_node.leaf_idx]=tempStr
                        # idxCtr+=1 
                    else:
                        strings.append(tempStr+strings[0][itemIdx])
                        coreString[itemIdx]=tempStr
                    last_char+=1
                    # idxCtr+=1
                elif type(item.real_node) is ht.TuckerCore:
                    contractionDims = len(item.shape)-1
                    tempStr="".join([chr(last_char+stringIdx) for stringIdx in range(0,contractionDims)])#Sorun olursa 0i 1 yap
                    strings.append(
                        tempStr+strings[0][itemIdx]
                        # strings[0][idxCtr:idxCtr+contractionDims]+chr(last_char)
                        )
                    coreString[itemIdx]=tempStr
                    last_char += contractionDims
                    # idxCtr += contractionDims
                else:
                    ValueError(f"Unknown node type! {type(item)} is not known!")
            try:
                core = eval(
                    "np.einsum("+
                    "'"+
                    ",".join([
                        ','.join(strings)+'->'+"".join(coreString)+"'",'core',
                        ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                        'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                    )+
                    ")"
                )
            except ValueError:
                for ii, string in enumerate(strings):
                    tempstr = [*string]
                    for jj, chrs in enumerate(tempstr):
                            if ord(chrs)>ord("z"):
                                    tempstr[jj]=chr(ord(chrs)-ord("z")+ord("A")-1)
                                    strings[ii]="".join(tempstr)
                for jj, chrs in enumerate(coreString):
                    if ord(chrs)>ord("z"):
                            coreString[jj]=chr(ord(chrs)-ord("z")+ord("A")-1)
                core = eval(
                    "np.einsum("+
                    "'"+
                    ",".join([
                        ','.join(strings)+'->'+"".join(coreString)+"'",'core',
                        ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                        'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                    )+
                    ")"
                )
            
        return core
    
    def incremental_update(self,new_tensor):
        if list(new_tensor.shape)!=self.original_shape:
            try:
                new_tensor = new_tensor.reshape(self.original_shape,order="F")
            except ValueError:
                warn(f"Presented tensor has shape {new_tensor.shape}, which is not compatible with {tuple(self.original_shape)}!")

        core = self.project(new_tensor)
        reconstruction = self.reconstruct(core)
        tenNorm = np.linalg.norm(new_tensor)
        if (np.linalg.norm(new_tensor-reconstruction)/tenNorm)<=self.rtol:
            warn('Current tensor network is sufficient, no need to update the cores.')
            return core
        
        allowed_error=tenNorm*self.rtol/np.sqrt(2*len(new_tensor.shape)-3)

        for layer in self._dimension_tree._level_items[::-1][:-1]:
            idxCtr = 0
            for itemIdx, item in enumerate(layer):
                2+2
                strings=[]
                last_char=97
                dims = len(new_tensor.shape)
                coreString =[chr(idx) for idx in range(last_char,last_char+dims)]
                strings.append(''.join(coreString))
                last_char+=dims
                
                contractionDims = len(item.shape)-1
                # Find Orthonormal vectors
                if type(item.real_node) is ht.TuckerLeaf:
                    2+2

                    strings.append(strings[0][idxCtr]+chr(last_char))
                    strings.append(chr(last_char+contractionDims)+chr(last_char))
                    coreString[idxCtr]=chr(last_char+contractionDims)

                    tempTens= eval(
                        "np.einsum("+
                        "'"+
                        ",".join([
                            ','.join(strings)+'->'+"".join(coreString)+"'",'new_tensor',
                            "item.real_node.core,item.real_node.core",
                            # ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                        )+
                        ")"
                    )
                
                    tempTens = tempTens-new_tensor
                    u,s,_ = np.linalg.svd(ht.mode_n_unfolding(tempTens,idxCtr),full_matrices=False)
                    idxCtr +=1                

                elif type(item.real_node) is ht.TuckerCore:
                    2+2
                    strings.append(strings[0][idxCtr:idxCtr+contractionDims]+chr(last_char))
                    strings.append(
                        "".join(
                        [chr(stringIdx+1) for stringIdx in range(last_char,last_char+contractionDims)]
                        )+chr(last_char)
                        )
                    for stringIdx in range(contractionDims):
                        coreString[idxCtr+stringIdx] = strings[-1][stringIdx]
                    tempTens= eval(
                        "np.einsum("+
                        "'"+
                        ",".join([
                            ','.join(strings)+'->'+"".join(coreString)+"'",'new_tensor',
                            "item.real_node.core,item.real_node.core",
                            # ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                        )+
                        ")"
                    )
                    tempTens = tempTens-new_tensor
                    new_shape = list(new_tensor.shape)
                    new_shape = new_shape[:idxCtr]+[np.prod(new_shape[idxCtr:idxCtr+contractionDims])]+new_shape[idxCtr+contractionDims:]
                    u,s,_ = np.linalg.svd(ht.mode_n_unfolding(tempTens.reshape(new_shape,order="F"),idxCtr),full_matrices=False)
                    idxCtr +=2



                else:
                    ValueError(f"Unknown node type! {type(item)} is not known!")

                # Core Updating
                u = u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
                # s = s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
                u_shape = list(item.shape)[:-1]+[-1]
                item.real_node.core = np.concatenate((item.real_node.core,u.reshape(u_shape,order="F")),axis=-1)
                item.real_node.get_ranks()

                # Rank Matching
                # TODO: Need to update this part after introducing "sibling index" for n-ary splits!
                if item.parent._dimension_index.index(item._dimension_index[0]) == 0: 
                    ranks=item.real_parent.ranks
                    item.real_parent.core = np.concatenate((item.real_parent.core,np.zeros([u.shape[-1]]+ranks[1:])),axis=0)
                else:
                    ranks=item.real_parent.ranks
                    item.real_parent.core = np.concatenate((item.real_parent.core,np.zeros([ranks[0],u.shape[-1]]+ranks[2:])),axis=1)
                item.real_parent.get_ranks()

            # Project Through Layer
            # TODO : Convert layerwise projection into a separate function.
            idxCtr = 0
            strings=[]
            last_char=97
            dims = len(new_tensor.shape)
            coreString =[chr(idx) for idx in range(last_char,last_char+dims)]
            strings.append(''.join(coreString))
            last_char+=dims
            for itemIdx, item in enumerate(layer):
                if type(item.real_node) is ht.TuckerLeaf:
                    # strings.append(strings[0][item._dimension_index[0]]+chr(last_char))
                    # coreString[item._dimension_index[0]]=chr(last_char)
                    if (item in self._dimension_tree._level_items[-1]) and (item.real_node.leaf_idx != idxCtr):
                        idxCtr+=1 
                    strings.append(strings[0][idxCtr]+chr(last_char))
                    coreString[idxCtr]=chr(last_char)
                    last_char+=1
                    idxCtr+=1
                elif type(item.real_node) is ht.TuckerCore:
                    # icerde ayri minik bir counter tut, leaf olunca 1 core olunca 2 ilerlet olsun bitsin
                    # counter da her layerda sifirlansin.
                    contractionDims = len(item.shape)-1
                    strings.append(strings[0][idxCtr:idxCtr+contractionDims]+chr(last_char))
                    coreString[idxCtr]=chr(last_char)
                    last_char+=1
                    for stringIdx in range(1,contractionDims):
                        coreString[idxCtr+stringIdx]=""
                    # last_char += 1
                    idxCtr += contractionDims
                else:
                    ValueError(f"Unknown node type! {type(item)} is not known!")
            new_tensor = eval(
                "np.einsum("+
                "'"+
                ",".join([
                    ','.join(strings)+'->'+"".join(coreString)+"'",'new_tensor',
                    ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                    'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                )+
                ")"
            )

        return new_tensor
    def incremental_update_batch(self,new_tensor, batch_dimension= None, append=True):
        assert(self._iscompressed is True)
        new_tensor_shape = new_tensor.shape
        if list(new_tensor_shape[:batch_dimension]+new_tensor_shape[batch_dimension+1:])!=self.original_shape:
            try:
                shape = np.arange(len(new_tensor_shape)).tolist()
                shape.pop(batch_dimension)
                shape=shape+[batch_dimension]
                new_tensor.transpose(shape)
                new_tensor = new_tensor.reshape(self.original_shape+[-1],order="F")
            except ValueError:
                2+2
                # warn(f"Presented tensor has shape {new_tensor.shape}, which is not compatible with {tuple(self.original_shape)}!")
        
        core = self.project(new_tensor,batch=True,batch_dimension=batch_dimension)

        tenNorm = np.linalg.norm(new_tensor)
        # print(tenNorm, np.linalg.norm(core), tenNorm*(1-self.rtol), tenNorm*np.sqrt(1-self.rtol**2))
        if np.linalg.norm(core)>= tenNorm*np.sqrt(1-self.rtol**2):
            self.batch_count+=new_tensor_shape[batch_dimension]#*(not append)
            if append:
                self.root.core = np.concatenate((self.root.core,core),axis=-1)
                self.root.get_ranks()
                return False
            else:
                return False
        # reconstruction = self.reconstruct(core)
        # rel_proj_error = (np.linalg.norm(new_tensor-reconstruction)/tenNorm)
        # # print(rel_proj_error,rel_proj_error<=self.rtol)
        # if rel_proj_error<=self.rtol:
        #     # print('Current tensor network is sufficient, no need to update the cores.')
        #     self.batch_count+=new_tensor_shape[batch_dimension]#*(not append)
        #     if append:
        #         self.root.core = np.concatenate((self.root.core,core),axis=-1)
        #         self.root.get_ranks()
        #         return False
        #     else:
        #         return False
        


        # allowed_error=tenNorm*self.rtol/np.sqrt(2*len(new_tensor.shape)-3)
        cur_norm = tenNorm
        total_allowed_error = tenNorm*self.rtol
        allowed_error = total_allowed_error
        num_svds = 2*(len(new_tensor.shape)-1)-2
        # print(num_svds)
        for layer in self._dimension_tree._level_items[::-1][:-1]:
            # print(num_svds)
            # print(allowed_error)
            allowed_error=allowed_error/np.sqrt(num_svds)
            # print(allowed_error)
            idxCtr = 0
            for itemIdx, item in enumerate(layer):
                2+2
                strings=[]
                last_char=97
                dims = len(new_tensor.shape)
                coreString =[chr(idx) for idx in range(last_char,last_char+dims)]
                strings.append(''.join(coreString))
                last_char+=dims
                
                contractionDims = len(item.shape)-1
                # Find Orthonormal vectors
                if type(item.real_node) is ht.TuckerLeaf:
                    2+2
                    if (item in self._dimension_tree._level_items[-1]) and (item.real_node.leaf_idx != idxCtr):
                        idxCtr+=1 
                    strings.append(strings[0][idxCtr]+chr(last_char))
                    strings.append(chr(last_char+contractionDims)+chr(last_char))
                    coreString[idxCtr]=chr(last_char+contractionDims)

                    tempTens= eval(
                        "np.einsum("+
                        "'"+
                        ",".join([
                            ','.join(strings)+'->'+"".join(coreString)+"'",'new_tensor',
                            "item.real_node.core,item.real_node.core",
                            # ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                        )+
                        ")"
                    )
                
                    tempTens = tempTens-new_tensor
                    u,s,_ = np.linalg.svd(ht.mode_n_unfolding(tempTens,idxCtr),full_matrices=False)
                    idxCtr +=1                

                elif type(item.real_node) is ht.TuckerCore:
                    2+2
                    strings.append(strings[0][idxCtr:idxCtr+contractionDims]+chr(last_char))
                    strings.append(
                        "".join(
                        [chr(stringIdx+1) for stringIdx in range(last_char,last_char+contractionDims)]
                        )+chr(last_char)
                        )
                    for stringIdx in range(contractionDims):
                        coreString[idxCtr+stringIdx] = strings[-1][stringIdx]
                    tempTens= eval(
                        "np.einsum("+
                        "'"+
                        ",".join([
                            ','.join(strings)+'->'+"".join(coreString)+"'",'new_tensor',
                            "item.real_node.core,item.real_node.core",
                            # ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                        )+
                        ")"
                    )
                    tempTens = tempTens-new_tensor
                    new_shape = list(new_tensor.shape)
                    new_shape = new_shape[:idxCtr]+[np.prod(new_shape[idxCtr:idxCtr+contractionDims])]+new_shape[idxCtr+contractionDims:]
                    try:
                        u,s,_ = np.linalg.svd(ht.mode_n_unfolding(tempTens.reshape(new_shape,order="F"),idxCtr),full_matrices=False)
                    except np.linalg.LinAlgError:
                        print("Numpy svd did not converge, using qr+svd")
                        q,r = np.linalg.qr(ht.mode_n_unfolding(tempTens.reshape(new_shape,order="F"),idxCtr))
                        u,s,_ = np.linalg.svd(r,full_matrices=False)
                        u = q@u
                    idxCtr +=2
                else:
                    ValueError(f"Unknown node type! {type(item)} is not known!")

                # Core Updating
                u = u[:,np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
                # s = s[np.cumsum((s**2)[::-1])[::-1]>(allowed_error)**2]
                u_shape = list(item.shape)[:-1]+[-1]
                item.real_node.core = np.concatenate((item.real_node.core,u.reshape(u_shape,order="F")),axis=-1)
                item.real_node.get_ranks()

                # Rank Matching
                # TODO: Need to update this part after introducing "sibling index" for n-ary splits!
                if item.parent._dimension_index.index(item._dimension_index[0]) == 0: 
                    ranks=item.real_parent.ranks
                    item.real_parent.core = np.concatenate((item.real_parent.core,np.zeros([u.shape[-1]]+ranks[1:])),axis=0)
                else:
                    ranks=item.real_parent.ranks
                    item.real_parent.core = np.concatenate((item.real_parent.core,np.zeros([ranks[0],u.shape[-1]]+ranks[2:])),axis=1)
                item.real_parent.get_ranks()

            # Project Through Layer
            # TODO : Convert layerwise projection into a separate function.
            idxCtr = 0
            strings=[]
            last_char=97
            dims = len(new_tensor.shape)
            coreString =[chr(idx) for idx in range(last_char,last_char+dims)]
            strings.append(''.join(coreString))
            last_char+=dims
            for itemIdx, item in enumerate(layer):
                if type(item.real_node) is ht.TuckerLeaf:
                    # strings.append(strings[0][item._dimension_index[0]]+chr(last_char))
                    # coreString[item._dimension_index[0]]=chr(last_char)
                    if (item in self._dimension_tree._level_items[-1]) and (item.real_node.leaf_idx != idxCtr):
                        idxCtr+=1 
                    strings.append(strings[0][idxCtr]+chr(last_char))
                    coreString[idxCtr]=chr(last_char)
                    last_char+=1
                    idxCtr+=1
                elif type(item.real_node) is ht.TuckerCore:
                    # icerde ayri minik bir counter tut, leaf olunca 1 core olunca 2 ilerlet olsun bitsin
                    # counter da her layerda sifirlansin.
                    contractionDims = len(item.shape)-1
                    strings.append(strings[0][idxCtr:idxCtr+contractionDims]+chr(last_char))
                    coreString[idxCtr]=chr(last_char)
                    last_char+=1
                    for stringIdx in range(1,contractionDims):
                        coreString[idxCtr+stringIdx]=""
                    # last_char += 1
                    idxCtr += contractionDims
                else:
                    ValueError(f"Unknown node type! {type(item)} is not known!")
            try:
                new_tensor = eval(
                    "np.einsum("+
                    "'"+
                    ",".join([
                        ','.join(strings)+'->'+"".join(coreString)+"'",'new_tensor',
                        ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                        'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                    )+
                    ")"
                )
            except ValueError:
                for ii, string in enumerate(strings):
                    tempstr = [*string]
                    for jj, chrs in enumerate(tempstr):
                            if ord(chrs)>ord("z"):
                                    strings[ii]=strings[ii].replace(chrs,chr(ord(chrs)-ord("z")+ord("A")-1),jj)
                for jj, chrs in enumerate(coreString):
                    if ord(chrs)>ord("z"):
                            coreString[jj]=chr(ord(chrs)-ord("z")+ord("A")-1)
                new_tensor = eval(
                    "np.einsum("+
                    "'"+
                    ",".join([
                        ','.join(strings)+'->'+"".join(coreString)+"'",'new_tensor',
                        ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                        'optimize=True,order="F"'] ## Bir sorun olursa buraya bak order="F" sonradan eklendi
                    )+
                    ")"
                )
            # print(len(layer),layer)
            num_svds -= len(layer)
            cur_norm = np.linalg.norm(new_tensor)
            # print(num_svds,new_tensor.shape)
            # print(
            #     tenNorm,
            #     cur_norm,
            #     allowed_error,
            #     (cur_norm**2/tenNorm**2),
            #     np.sqrt(1-(cur_norm**2/tenNorm**2)),
            #     allowed_error**2,
            #     (total_allowed_error**(2)),
            #     (tenNorm**(2)),
            #     (cur_norm**(2)),
            #     ((tenNorm**(2))-(cur_norm**(2))),
            #     (total_allowed_error**(2)) - ((tenNorm**(2))-(cur_norm**(2))),
            #     np.sqrt((total_allowed_error**(2)) - max(((tenNorm**(2))-(cur_norm**(2))),0))
            #     )
            # print(allowed_error)
            allowed_error = np.sqrt((total_allowed_error**(2)) - max(((tenNorm**(2))-(cur_norm**(2))),0))
            # print(allowed_error)

        
        self.batch_count+=new_tensor_shape[batch_dimension]#*(not append)
        if append:
            self.root.core = np.concatenate((self.root.core,new_tensor),axis=-1)
            self.root.get_ranks()
            return True
        else:
            return True

    def compress_sanity_check(self,tensor):
        # Commenting out below for now, might be needed later for checking
        # print("\n",tensor)
        # print("\n",tensor.shape)

        mat_tensor = np.reshape(tensor, (tensor.shape[0]*tensor.shape[1],
                                         tensor.shape[2]*tensor.shape[3]), order='F')

        [u, s, v] = truncated_svd(mat_tensor, 1e-8, full_matrices=False)
        # print("\n U")
        [core_l, lsv_l] = hosvd(u.reshape(tensor.shape[0],tensor.shape[1],-1, order='F'))
        # print("\n V")
        [core_r, lsv_r] = hosvd(v.reshape(-1, tensor.shape[2],tensor.shape[3], order='F'))

        # need an HOSVD tucker of u (and v) this (look at kolda paper)
        # print("\n",core_l)
        # print("\n",lsv_l[-1])
        core_l = np.einsum('ijk,lk->ijl', core_l, lsv_l[-1],optimize=True)
        core_r = np.einsum('ijk,li->ljk', core_r, lsv_r[0],optimize=True)
        # print("\n",core_l)
        # print("\n",core_r)
        top = np.diag(s)
        return (lsv_l[0], lsv_l[1], lsv_r[1], lsv_r[2], core_l, core_r, top)

    @property
    def compression_ratio(self):
        num_entries=np.prod(self.root.shape[:-1])*self.batch_count
        for tf in self.transfer_nodes:
            num_entries+=np.prod(tf.shape)
        for lf in self.leaves:
            num_entries+=np.prod(lf.shape)
        return np.prod(self.original_shape)*self.batch_count/num_entries

    def save(self, fileName, fileType="hto", directory = "./"):
        """
        This method saves the object (self) into a file with Hierarchal Tucker Object (.hto) format as default.
        
        Args:
            fileName (str): This argument provides the filename to save the object. The filename can include the type of the file.
            fileType (str, optional): This defines the type of the file. The default file type is "hto".
            directory (str, optional): This is the location where the file will be saved. Default location is the current directory.
            
        Raises:
            NameError: Raises Name Error if fileName has more than one '.' or unknown file extension is provided.
            NotImplementedError: Raises NotImplementedError when fileType is "npy" as it is currently not supported.
        """
        if len(fileName.split("."))==2:
            #File extension is given in the file name
            fileType = fileName.split(".")[1]
        elif len(fileName.split("."))>=2:
            raise NameError(f"Filename {fileName} can not have more than 1 '.'!")
        else:
            fileName = fileName+"."+fileType

        if fileType == "hto":
            # Save to a hierarcichal tucker object file
            with open(os.path.join(directory, fileName), 'wb') as f:
                pckl.dump(self, f)
        elif fileType == "npy":
            # TODO
            # Save htucker object to numpy arrays
            # This will require saving the dimension tree as well
            # Maybe you can adopt a layer-by-layer approach to that 
            raise NotImplementedError("This function is not implemented yet")
        else:
            raise NameError(f"Unknown file extension {fileType}")
    
    @staticmethod
    def load(file, directory = "./"):
        """
        This static method loads a Hierarchal Tucker Object from a file. The directory parameter takes the path to the file.
        
        Args:
            file (str): The name of the file that is going to be loaded. The directory path should be given separately. File argument should not contain path.
            directory (str, optional): Directory path to the file. By default, it is set to the current directory.
            
        Returns:
            It returns the loaded Hierarchal Tucker Object.
            
        Raises:
            AssertionError: Raises an assertion error if the file name contains a '/'.
            NameError: Raises Name Error if fileName has more than one '.' or unknown file extension is provided.
            NotImplementedError: Raises NotImplementedError when fileType is "npy" as it is currently not supported.
        """
        # File address should be given as directory variable
        assert len(file.split(os.sep))==1 , "Please give address as directory variable."
        if len(file.split("."))==2:
            #File extension is given in the file name
            _ , fileType = file.split(".")
        elif len(file.split("."))>=2:
            raise NameError(f"Filename {file} can not have more than 1 '.'!")
        
        if fileType == "hto":
            # File is a hierarcichal tucker object file
            with open(os.path.join(directory, file), 'rb') as f:
                return pckl.load(f)
        elif fileType == "npy":
            # TODO
            # Load htucker object using numpy arrays
            # This will require an accompanying dimension tree file
            raise NotImplementedError("This function is not implemented yet")
        else:
            raise NameError(f"Unknown file extension {fileType}")
        
        
def truncated_svd(a, truncation_threshold=None, full_matrices=True, compute_uv=True, hermitian=False):
    # print(a.shape)
    try:
        [u, s, v] = np.linalg.svd(a,
                                full_matrices=full_matrices,
                                compute_uv=compute_uv,
                                hermitian=False)
    except:
        q, r = np.linalg.qr(a)
        [u,s,v] = np.linalg.svd(r,
                                full_matrices=full_matrices,
                                compute_uv=compute_uv,
                                hermitian=False)
        u = q @ u
    if truncation_threshold == None:
        return [u, s, v]

    trunc = sum(s>=truncation_threshold)
    # print(truncation_threshold,s,trunc)
    u=u[:,:trunc]
    s=s[:trunc]
    v=v[:trunc,:]

    return [u, s, v]
        
def hosvd(tensor,rtol=None,tol=None,threshold=1e-8,norm=None,dimensions=None):
    
    # TODO: Arrange 

    # if norm is None:

    if tol is not None:
        tolerance = tol
    else:
        tolerance = 1e-8

    if (tol is None) and (rtol is not None):
        tensor_norm=np.linalg.norm(tensor)
        tolerance = tensor_norm*rtol
    if dimensions is None:
        ndims=len(tensor.shape)
    else:
        ndims=dimensions
        # tolerance=tolerance/np.sqrt(len(tensor.shape)/ndims)
    # print(ndims,len(tensor.shape))
    # print(tolerance)
    # tolerance=tolerance*1.7

    if len(tensor.shape) == 2:
        [u, s, v] = truncated_svd(tensor, truncation_threshold=threshold, full_matrices=False)

        # u = u[:,np.cumsum((s**2)[::-1])[::-1]>(tolerance/np.sqrt(ndims))**2]
        # v = v[np.cumsum((s**2)[::-1])[::-1]>(tolerance/np.sqrt(ndims))**2,:]
        # s = s[np.cumsum((s**2)[::-1])[::-1]>(tolerance/np.sqrt(ndims))**2]

        # u = u[:,np.cumsum((s**2)[::-1])[::-1]>(tolerance/ndims)**2]
        # v = v[np.cumsum((s**2)[::-1])[::-1]>(tolerance/ndims)**2,:]
        # s = s[np.cumsum((s**2)[::-1])[::-1]>(tolerance/ndims)**2]

        u = u[:,np.cumsum((s**2)[::-1])[::-1]>(tolerance)**2]
        v = v[np.cumsum((s**2)[::-1])[::-1]>(tolerance)**2,:]
        s = s[np.cumsum((s**2)[::-1])[::-1]>(tolerance)**2]
        return np.diag(s), [u, v.T]

    # permutations=create_permutations(ndims)
    permutations=create_permutations(len(tensor.shape))

    leftSingularVectors=[]
    singularValues=[]
    # print("\n",tensor.shape)
    # print(permutations)

    # May replace this with a combination of mode-n unfolding and truncated svd
    for dim , perm in enumerate(permutations):
        # print(dim,perm)

        # Swap next two lines if something breaks down
        # tempTensor=tensor.transpose(perm).reshape(tensor.shape[dim],-1)
        tempTensor = mode_n_unfolding(tensor,dim)

        # [u, s, v] = np.linalg.svd(tempTensor,full_matrices=False)
        [u, s, v] = truncated_svd(tempTensor,truncation_threshold=threshold,full_matrices=False)
        # np.cumsum((s**2)[::-1])[::-1]>(tolerance/ndims)**2
        # print(u.shape)
        # Automatic rank truncation, can later be replaced with the deltaSVD function
        
        # leftSingularVectors.append(u[:,np.cumsum((s**2)[::-1])[::-1]>(tolerance/ndims)**2])
        # singularValues.append(s[np.cumsum((s**2)[::-1])[::-1]>(tolerance/ndims)**2])
        
        # leftSingularVectors.append(u[:,np.cumsum((s**2)[::-1])[::-1]>(tolerance/np.sqrt(ndims))**2])
        # singularValues.append(s[np.cumsum((s**2)[::-1])[::-1]>(tolerance/np.sqrt(ndims))**2])

        # leftSingularVectors.append(u[:,np.cumsum((s**2)[::-1])[::-1]>(tolerance/ndims)**2])
        # singularValues.append(s[np.cumsum((s**2)[::-1])[::-1]>(tolerance/ndims)**2])

        # print(np.cumsum((s**2)[::-1])[::-1]>(tolerance)**2)
        leftSingularVectors.append(u[:,np.cumsum((s**2)[::-1])[::-1]>(tolerance)**2])
        singularValues.append(s[np.cumsum((s**2)[::-1])[::-1]>(tolerance)**2])

    
    for dim , u in enumerate(leftSingularVectors):
        # print(dim,u.shape)#,tensor.shape)
        tensorShape=list(tensor.shape)
        currentIndices=list(range(1,len(tensorShape)))
        currentIndices=currentIndices[:dim]+[0]+currentIndices[dim:]
        tensor=np.tensordot(u.T, tensor, axes=(1, dim)).transpose(currentIndices)
    # tensor = np.einsum('ij,kl,mn,op,ikmo->jlnp',leftSingularVectors[0],leftSingularVectors[1],leftSingularVectors[2],leftSingularVectors[3],tensor)
    # print(tensor.shape)
    # print()
    return tensor , leftSingularVectors
    # return tensor , singularValues, leftSingularVectors

def hosvd_only_for_dimensions(tensor,tol=None, rtol=None, threshold =1e-8, norm=None, dims=None, batch_dimension=None, contract=False):
    # TODO: Change name to hosvd and refactor other functions here.
    if dims is None:
        dims = list(tensor.shape)
    if batch_dimension is not None:
        dims.pop(batch_dimension)

    if (tol is None) and (rtol is not None):
        if norm is None:
            norm = np.linalg.norm(tensor)        
        tol = norm*rtol
    elif (tol is None) and (rtol is None):
        tol = 1e-8
    elif tol is not None:
        pass
    else:
        raise ValueError("Ben de ne oldugunu tam anlamadim ama bi' buraya bak.")
    
    if len(tensor.shape) == 2:
        [u, s, v] = truncated_svd(tensor, truncation_threshold=threshold, full_matrices=False)
        u = u[:,np.cumsum((s**2)[::-1])[::-1]>(tol)**2]
        v = v[np.cumsum((s**2)[::-1])[::-1]>(tol)**2,:]
        s = s[np.cumsum((s**2)[::-1])[::-1]>(tol)**2]
        return np.diag(s), [u, v.T]
    
    leftSingularVectors = []
    for dimension in dims:
        [u, s, v] = truncated_svd(
            mode_n_unfolding(tensor,dimension), truncation_threshold=threshold, full_matrices=False
        )
        leftSingularVectors.append(
            u[:,np.cumsum((s**2)[::-1])[::-1]>(tol)**2]
        )
    if contract:
        2+2
        for dimension,lsv in zip(dims,leftSingularVectors):
            dimension_flip = np.arange(len(tensor.shape)-1).tolist()
            dimension_flip.insert(dimension,len(tensor.shape)-1)
            tensor = np.tensordot(tensor,lsv,axes=[dimension,0]).transpose(dimension_flip)
        return tensor, leftSingularVectors
    
    return leftSingularVectors

def create_permutations(nDims):
    # Creates permutations to compute the matricizations
    permutations=[]
    dimensionList=list(range(nDims))
    for dim in dimensionList:
        copyDimensions=dimensionList.copy()
        firstDim=copyDimensions.pop(dim)
        permutations.append([firstDim]+copyDimensions)
    return permutations

def split_dimensions(dims):
        n_dims=len(dims)
        return dims[:ceil(n_dims/2)],dims[ceil(n_dims/2):]

def mode_n_unfolding(tensor,mode):
    # Computes mode-n unfolding/matricization of a tensor in the sense of Kolda&Bader
    # Assumes the mode is given in 0 indexed format
    nDims = len(tensor.shape)
    dims = [dim for dim in range(nDims)]
    modeIdx = dims.pop(mode)
    dims=[modeIdx]+dims
    tensor=tensor.transpose(dims)
    return tensor.reshape(tensor.shape[0],-1,order='F')

def mode_n_product(tensor:np.ndarray, matrix:np.ndarray, modes:list or tuple):
    dims=[idx for idx in range(len(tensor.shape)+len(matrix.shape)-2)]
    tensor_ax, matrix_ax = modes
    dims.pop(tensor_ax)
    dims.append(tensor_ax)
    tensor=np.tensordot(tensor,matrix,axes=modes)
    tensor=tensor.transpose(np.argsort(dims).tolist())
    return tensor

# def matrixTensorProduct(matrix,tensor,axes):
    
#     ax1,ax2=axes

#     matrixShape=list(matrix.shape)
#     tensorShape=list(tensor.shape)
#     assert(matrixShape[ax1]==tensorShape[ax2])

#     productShape=tensorShape.copy()

#     tensorIndices=list(range(len(tensorShape)))

#     order=tensorIndices.pop(ax2)
#     tensorIndices=[order]+tensorIndices

#     # tensor=tensor.transpose(tensorIndices).reshape(tensorShape[ax2],-1,order='F')
#     tensor=np.tensordot(matrix,tensor,axes=[ax1,ax2])
#     currentIndices=list(range(1,len(tensorShape)))
#     currentIndices=currentIndices[:ax2]+[0]+currentIndices[ax2:]


class Node:
    def __init__(self, val, children=None, parent=None) -> None:
        self.children = children or []
        self.val = val
        self.parent = parent
        self.real_children = []
        self.real_parent = []
        self.real_node = None
        self._ranks = []
        self._propagated = False
        self._isleaf = False
        self._level = None
        self._dimension_index = None

    def __str__(self) -> str:
        return self.children
    
    def adjust_ranks(self):
        if self.parent is None:
            #This is the root node
            if len(self._ranks)<len(self.children):
                diff =len(self.children)-len(self._ranks)
                self._ranks += [None]*diff
        else:
            # This is any node (incl. leaves)
            if len(self._ranks)<len(self.children)+1:
                diff =len(self.children)-len(self._ranks)+1
                self._ranks += [None]*diff
    
    @property
    def shape(self):
        if self.real_node:
            return self.real_node.shape
        else:
            return warn("No real node is defined.")

class Tree:
    def __init__(self) -> None:
        self.root = None
        self._depth = 0
        self._size = 0
        self._leafCount = 0
        self._nodeCount = 0
        self._leaves = []
        self._level_items = None 

    def findNode(self, node, key, check_propagation=False):
        if (node is None) or (node.val == key):
            return node
        for child in node.children:
            return_node = self.findNode(child, key,check_propagation=check_propagation)
            if return_node:
                if check_propagation and (not return_node._propagated):
                    return return_node
                elif check_propagation and return_node._propagated:
                    pass
                else:
                    return return_node
        return None

    def isEmpty(self):
        return self._size == 0

    def initializeTree(self, vals):
        # Initalizes the tree
        if self.root is None:
            if type(vals) is list:
                self.root = vals
            else:
                raise TypeError(f"Type: {type(vals)} is not known!")
        else:
            warn("Root node already implemented! Doing nothing.")

    def insertNode(self, val, parent=None, dim_index=None):
        newNode = Node(val)
        newNode._dimension_index=dim_index
        if parent is None: # No parent is given, i.e. Root node
            self.root = newNode
            self._depth = 1
            self._size = 1
            newNode._level = 0
            newNode.adjust_ranks()
        elif type(parent) is Node: # Parent is given directly as a node object
            parent.children.append(newNode)
            parent._propagated = True
            # parent._ranks+=[None]
            self._size += 1
            newNode.parent = parent
            newNode._level = parent._level+1
            parent.adjust_ranks()
        else: # Key/dimensions of the parent is given as input
            parentNode = self.findNode(self.root, parent,check_propagation=True)
            if not (parentNode):
                raise NotFoundError(f"No parent was found for parent name: {parent}")
            parentNode.children.append(newNode)
            parentNode._propagated = True
            # parentNode._ranks+=[None]
            self._size += 1
            newNode.parent = parentNode
            newNode._level = parentNode._level+1
            parentNode.adjust_ranks()
        if len(val)==1:
            newNode._isleaf = True
            newNode.adjust_ranks()
            self._leaves.append(newNode)
            self._leafCount+=1

    def get_max_depth(self):
        self._depth = 0
        for leaf in self._leaves:
            depth=0
            node = leaf
            while node.parent is not None:
                depth += 1
                node = node.parent
            if depth > self._depth:
                self._depth = depth
        return None
                
    def get_items_from_level(self):
        self._level_items=[]
        for _ in range(self._depth+1):
            self._level_items.append([])
        # for depth,items in enumerate(level_items):
        nodes2expand=[self.root]
        while nodes2expand:
            node = nodes2expand.pop(0)
            nodes2expand.extend(node.children)
            self._level_items[node._level].append(node)

    def toList(self):
        # Returns a list from the tree
        return None


def createDimensionTree(inp, numSplits, minSplitSize):
    ## FIXME: Dimension tree returns wrong case where dimension order repeats itself.
    if type(inp) is np.ndarray:
        dims = np.array(inp.shape)
    elif type(inp) is tuple or list:
        dims = np.array(inp)  # NOQA
    else:
        raise TypeError(f"Type: {type(inp)} is unsupported!!")
    dimensionTree = Tree()
    dimensionTree.insertNode(dims.tolist())
    # print(np.array(dimensionTree.root.val))
    dimensionTree.root._dimension_index = [idx for idx,_ in enumerate(dimensionTree.root.val)]
    nodes2expand = []
    nodes2expand.append(dimensionTree.root.val.copy())
    while nodes2expand:
        # BUG: searching just with node values return wrong results
        # FIXME: change dimensions in nodes2expand to a tuple of (dimensions,parent node) 
        #        to avoid confusion while searching in the dimension tree.
        #        Or maybe come up with a new createDimensionTree that suits n-ary splits better.
        # print(leaves)
        node2expand = nodes2expand.pop(0)
        node = dimensionTree.findNode(dimensionTree.root, node2expand, check_propagation=True)
        dim_split=np.array_split(np.array(node.val), numSplits)
        idx_split=np.array_split(np.array(node._dimension_index), numSplits)
        if (not node._propagated) and (len(node.val) > minSplitSize + 1):
            # for split in [data[x:x+10] for x in xrange(0, len(data), 10)]:
            for dims,indices in zip(dim_split,idx_split): # place zip here
                # print(dims)
                # tree.insertNode(split,node.val)
                # leaves.append(split)
                # dimensionTree.insertNode(dims.tolist(), node.val,dim_index=indices.tolist())
                dimensionTree.insertNode(dims.tolist(), node,dim_index=indices.tolist())
                nodes2expand.append(dims.tolist())
        elif (not node._propagated) and (len(node.val) > minSplitSize):
            # i.e. the node is a leaf
            # print(node.val)
            for dims,indices in zip(dim_split,idx_split): # place zip here
                # dimensionTree.insertNode(dims.tolist(), node.val, dim_index=indices.tolist())
                dimensionTree.insertNode(dims.tolist(), node, dim_index=indices.tolist())
    dimensionTree.get_max_depth()
    dimensionTree._nodeCount = dimensionTree._size-dimensionTree._leafCount-1 #last -1 is to subtract root node
    return dimensionTree


#     return None

        

        
def convert_to_base2(num):
    binary = bin(num)[2:]  # convert to binary string
    binary = binary.zfill(3)  # pad with leading zeros to ensure 3 digits
    binary_list = [int(bit) for bit in binary]
    return binary_list
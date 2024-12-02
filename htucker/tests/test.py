import unittest
import random

import numpy as np
import htucker as ht

from unittest.mock import patch, mock_open

seed =1905
np.random.seed(seed)
random.seed(seed)
        
class TestCase(unittest.TestCase):

    def setUp(self):

        num_dim = 4
        self.size = [11, 4, 7, 5]

        leaf_ranks = [3, 2, 5, 4]
        
        leafs = [np.random.randn(r, n) for r,n in zip(leaf_ranks, self.size)]

        transfer_ranks = [3, 6]

        transfer_tensors = [
            np.random.randn(leaf_ranks[0], leaf_ranks[1], transfer_ranks[0]),
            np.random.randn(leaf_ranks[2], leaf_ranks[3], transfer_ranks[1])
        ]

        root = np.random.randn(transfer_ranks[0], transfer_ranks[1])
        
        eval_left = np.einsum('ij,kl,ikr->jlr', leafs[0], leafs[1], transfer_tensors[0])
        eval_right = np.einsum('ij,kl,ikr->jlr', leafs[2], leafs[3], transfer_tensors[1])

        self.tensor = np.einsum('ijk,lmn,kn->ijlm',eval_left, eval_right, root)
        self.koldaTensor=np.array(
            [
                [
                    [1,13],
                    [4,16],
                    [7,19],
                    [10,22]
                    ],
                [
                    [2,14],
                    [5,17],
                    [8,20],
                    [11,23]
                    ],
                [
                    [3,15],
                    [6,18],
                    [9,21],
                    [12,24]
                    ]
                ]
                
        )
        self.tensor6d=create_nway_tensor(num_dim=6)
        self.tensor7d=create_nway_tensor(num_dim=7)
        
    # @unittest.skip("add_edge")
    def test_add_edge(self):

        # print("\n", self.tensor)
        self.assertEqual(self.size[0], self.tensor.shape[0])
        self.assertEqual(self.size[1], self.tensor.shape[1])
        self.assertEqual(self.size[2], self.tensor.shape[2])
        self.assertEqual(self.size[3], self.tensor.shape[3])

    # @unittest.skip("mode-n unfolding")        
    def test_kolda_unfolding(self):
        mode0=ht.mode_n_unfolding(self.koldaTensor,0)
        mode1=ht.mode_n_unfolding(self.koldaTensor,1)
        mode2=ht.mode_n_unfolding(self.koldaTensor,2)

        # TODO: Write assert statements for dimensions!

        mode0kolda=np.array([
            [1,4,7,10,13,16,19,22],
            [2,5,8,11,14,17,20,23],
            [3,6,9,12,15,18,21,24]
            ])
        mode1kolda=np.array([
            [1,2,3,13,14,15],
            [4,5,6,16,17,18],
            [7,8,9,19,20,21],
            [10,11,12,22,23,24]
        ])
        mode2kolda=np.array([
            [1,2,3,4,5,6,7,8,9,10,11,12],
            [13,14,15,16,17,18,19,20,21,22,23,24]
        ])
        self.assertTrue(np.allclose((mode0-mode0kolda),np.zeros_like(mode0)))
        self.assertTrue(np.allclose((mode1-mode1kolda),np.zeros_like(mode1)))
        self.assertTrue(np.allclose((mode2-mode2kolda),np.zeros_like(mode2)))
    
    def test_hosvd(self):
        core,matrices=ht.hosvd(self.tensor)
        reconstruction=np.einsum('ij,kl,mn,op,jlnp->ikmo',matrices[0],matrices[1],matrices[2],matrices[3],core)
        self.assertTrue(np.allclose((reconstruction-self.tensor),np.zeros_like(reconstruction)))

    def test_htucker_sanity_check_4d(self):
        tens=ht.HTucker()
        (leaf1, leaf2, leaf3, leaf4, nodel, noder, top) = tens.compress_sanity_check(self.tensor)
        # print('\n')
        # print(leaf1.shape,leaf2.shape,leaf3.shape,leaf4.shape)
        # print(nodel.shape,noder.shape)
        # print(top.shape)
        eval_left = np.einsum('ji,lk,ikr->jlr', leaf1, leaf2, nodel)
        eval_right = np.einsum('ij,kl,rjl->rik', leaf3, leaf4, noder)
        # print(eval_left.shape,eval_right.shape)
        tensor = np.einsum('ijk,lmn,kl->ijmn',eval_left, eval_right, top)
        # print(tensor-self.tensor)
        self.assertTrue(np.allclose((tensor-self.tensor),np.zeros_like(tensor)))

    def test_htucker_4d(self):
        tens=ht.HTucker()
        (leaf1, leaf2, leaf3, leaf4, nodel, noder, top) = tens.compress_sanity_check(self.tensor)
        tens.initialize(self.tensor)
        tens.compress_root2leaf(self.tensor)

        self.assertEqual(self.size[0], tens.leaves[0].core.shape[0])
        self.assertEqual(self.size[1], tens.leaves[1].core.shape[0])
        self.assertEqual(self.size[2], tens.leaves[2].core.shape[0])
        self.assertEqual(self.size[3], tens.leaves[3].core.shape[0])

        # Check rank consistency between left leaves and left core
        self.assertEqual(tens.leaves[0].core.shape[1], tens.transfer_nodes[0].core.shape[0])
        self.assertEqual(tens.leaves[1].core.shape[1], tens.transfer_nodes[0].core.shape[1])

        # Check rank consistency between right leaves and right core
        self.assertEqual(tens.leaves[2].core.shape[1], tens.transfer_nodes[1].core.shape[0])
        self.assertEqual(tens.leaves[3].core.shape[1], tens.transfer_nodes[1].core.shape[1])

        # Check if the leaves are same for 4d case
        # The exception handling is to cover singular vectors with flipped signs
        flippedSignLeft = False
        flippedSignRight = False
        flip1, flip2, flip3, flip4 = False, False, False, False
        try:
            self.assertTrue(np.allclose((leaf1-tens.leaves[0].core), np.zeros_like(leaf1)))
        except AssertionError:
            print("Flipped sign at leaf 1")
            self.assertTrue(np.allclose(leaf1-tens.leaves[0].core@(leaf1.T@tens.leaves[0].core),np.zeros_like(leaf1)))
            flippedSignLeft = True
            flip1 = True
        try:
            self.assertTrue(np.allclose((leaf2-tens.leaves[1].core), np.zeros_like(leaf2)))
        except AssertionError:
            print("Flipped sign at leaf 2")
            self.assertTrue(np.allclose(leaf2-tens.leaves[1].core@(leaf2.T@tens.leaves[1].core),np.zeros_like(leaf2)))
            flippedSignLeft = True
            flip2 = True
        try:
            self.assertTrue(np.allclose((leaf3-tens.leaves[2].core), np.zeros_like(leaf3)))
        except AssertionError:
            print("Flipped sign at leaf 3")
            self.assertTrue(np.allclose(leaf3-tens.leaves[2].core@(leaf3.T@tens.leaves[2].core),np.zeros_like(leaf3)))
            flippedSignRight = True
            flip3 = True
        try:
            self.assertTrue(np.allclose((leaf4-tens.leaves[3].core), np.zeros_like(leaf4)))
        except AssertionError:
            print("Flipped sign at leaf 4")
            self.assertTrue(np.allclose(leaf4-tens.leaves[3].core@(leaf4.T@tens.leaves[3].core),np.zeros_like(leaf4)))
            flippedSignRight = True
            flip4 = True

        # Check if the transfer cores are same for 4d case
        # Note that we need to swap axes for the hardcoded version since we always
        # keep the tucker rank at the last index

        # Note that if there's a sign flip detected in the previous block, we adjust the transfer nodes accordingly.
        if not flippedSignLeft:
            self.assertTrue(np.allclose((nodel-tens.transfer_nodes[0].core), np.zeros_like(nodel)))
        else:
            if flip1:
                nodel = np.einsum('ijk,il->ljk',nodel,(leaf1.T@tens.leaves[0].core))
            if flip2:
                nodel = np.einsum('ijk,jl->ilk',nodel,(leaf2.T@tens.leaves[1].core))
            self.assertTrue(np.allclose((nodel-tens.transfer_nodes[0].core), np.zeros_like(nodel)))
        if not flippedSignRight:
            self.assertTrue(np.allclose((noder.transpose(1,2,0)-tens.transfer_nodes[1].core), np.zeros_like(noder.transpose(1,2,0))))
        else:
            if flip3:
                noder = np.einsum('ijk,jl->ilk',noder,(leaf3.T@tens.leaves[2].core))
            if flip4:
                noder = np.einsum('ijk,kl->ijl',noder,(leaf4.T@tens.leaves[3].core))
            self.assertTrue(np.allclose((noder.transpose(1,2,0)-tens.transfer_nodes[1].core), np.zeros_like(noder.transpose(1,2,0))))

        # self.assertEqual(leaf3.shape[1], noder.shape[1])
        # self.assertEqual(leaf4.shape[1], noder.shape[2])        
        
        eval_left = np.einsum('ji,lk,ikr->jlr', tens.leaves[0].core, tens.leaves[1].core, tens.transfer_nodes[0].core)
        eval_right = np.einsum('ij,kl,jlm->ikm',tens.leaves[2].core, tens.leaves[3].core, tens.transfer_nodes[1].core)


        # print("eval_left.shape = ", eval_left.shape)
        # print("eval_right.shape = ", eval_right.shape)
        # print("top shape = ", top.shape)
        
        tensor = np.einsum('ijk,lmn,kn->ijlm',eval_left, eval_right, tens.root.core)
        
        # Check if we get the same shape as the original tensor
        self.assertEqual(self.size[0], tensor.shape[0])
        self.assertEqual(self.size[1], tensor.shape[1])
        self.assertEqual(self.size[2], tensor.shape[2])
        self.assertEqual(self.size[3], tensor.shape[3])
        
        # Check if we get the same tensor as the original tensor
        self.assertTrue(np.allclose((tensor-self.tensor),np.zeros_like(tensor)))
    
    def test_root2leaf_reconstruct_4d(self):
        np.random.seed(seed)
        tens=ht.HTucker()
        tens.initialize(self.tensor)
        tens.compress_root2leaf(self.tensor)
        tens.reconstruct()
        self.assertTrue(np.allclose((tens.root.core-self.tensor),np.zeros_like(self.tensor)))
        
    def test_leaf2root_reconstruct_4d(self):
        np.random.seed(seed)
        tens=ht.HTucker()
        tens.initialize(self.tensor)
        dim_tree = ht.createDimensionTree(tens.original_shape,2,1)
        dim_tree.get_items_from_level()
        tens.compress_leaf2root(self.tensor,dimension_tree=dim_tree)
        tens.reconstruct()
        self.assertTrue(np.allclose((tens.root.core-self.tensor),np.zeros_like(self.tensor)))
    
    def test_root2leaf_reconstruct_6d(self):
        np.random.seed(seed)
        # tensor = create_nway_tensor(num_dim=6)
        tensor = self.tensor6d
        tens=ht.HTucker()
        tens.initialize(tensor)
        tens.compress_root2leaf(tensor)
        tens.reconstruct()
        self.assertTrue(np.allclose((tens.root.core-tensor),np.zeros_like(tensor)))

    def test_leaf2root_root2leaf_compare_6d(self):

        np.random.seed(seed)
        tensor = create_nway_tensor(num_dim=6)

        tens_ltr=ht.HTucker()
        tens_ltr.initialize(tensor)
        dim_tree = ht.createDimensionTree(tens_ltr.original_shape,2,1)
        dim_tree.get_items_from_level()
        tens_ltr.compress_leaf2root(tensor,dimension_tree=dim_tree)

        tens_rtl = ht.HTucker()
        tens_rtl.initialize(tensor)
        tens_rtl.compress_root2leaf(tensor)
        self.assertTrue(np.allclose(tens_ltr.compression_ratio,tens_rtl.compression_ratio))
        2+2
        tens_rtl.leaves=[
            tens_rtl.leaves[2],
            tens_rtl.leaves[3],
            tens_rtl.leaves[0],
            tens_rtl.leaves[4],
            tens_rtl.leaves[5],
            tens_rtl.leaves[1]
        ]
        for ltr,rtl in zip(tens_ltr.leaves,tens_rtl.leaves):
            self.assertEqual(ltr.shape,rtl.shape)

        tens_rtl.transfer_nodes=[
            tens_rtl.transfer_nodes[1],
            tens_rtl.transfer_nodes[0],
            tens_rtl.transfer_nodes[3],
            tens_rtl.transfer_nodes[2]
        ]
        for ltr,rtl in zip(tens_ltr.transfer_nodes,tens_rtl.transfer_nodes):
            self.assertEqual(ltr.shape,rtl.shape)

    def test_leaf2root_reconstruct_7d(self):
        np.random.seed(seed)
        # tensor = create_nway_tensor(num_dim=7)
        tensor = self.tensor7d
        tens=ht.HTucker()
        tens.initialize(tensor)
        dim_tree = ht.createDimensionTree(tens.original_shape,2,1)
        dim_tree.get_items_from_level()
        tens.compress_leaf2root(tensor,dimension_tree=dim_tree)
        tens.reconstruct()
        self.assertTrue(np.allclose((tens.root.core-tensor),np.zeros_like(tensor)))

    def test_leaf2root_reconstruct_6d(self):
        np.random.seed(seed)
        # tensor = create_nway_tensor(num_dim=6)
        tensor = self.tensor6d
        tens=ht.HTucker()
        tens.initialize(tensor)
        dim_tree = ht.createDimensionTree(tens.original_shape,2,1)
        dim_tree.get_items_from_level()
        tens.compress_leaf2root(tensor,dimension_tree=dim_tree)
        tens.reconstruct()
        self.assertTrue(np.allclose((tens.root.core-tensor),np.zeros_like(tensor)))

    def test_root2leaf_reconstruct_7d(self):
        np.random.seed(seed)
        # tensor = create_nway_tensor(num_dim=7)
        tensor = self.tensor7d
        tens=ht.HTucker()
        tens.initialize(tensor)
        tens.compress_root2leaf(tensor)
        tens.reconstruct()
        self.assertTrue(np.allclose((tens.root.core-tensor),np.zeros_like(tensor)))
        
class TestSaveFunction(unittest.TestCase):
    @patch("pickle.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_function_with_file_extension(self, mock_file, mock_dump):
        instance = ht.HTucker()  # Assuming you have a default constructor
        instance.save("test.hto")

        # Assert the file was opened in write-binary mode
        mock_file.assert_called_once_with("./test.hto", "wb")

        # Assert the pickle.dump was called with the class instance and file handle
        mock_dump.assert_called_once_with(instance, mock_file.return_value)
        
    @patch("pickle.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_function_without_file_extension(self, mock_file, mock_dump):
        instance = ht.HTucker()  # Assuming you have a default constructor
        instance.save("test")

        # Assert the file was opened in write-binary mode
        mock_file.assert_called_once_with("./test.hto", "wb")

        # Assert the pickle.dump was called with the class instance and file handle
        mock_dump.assert_called_once_with(instance, mock_file.return_value)

    def test_raises_exception_when_filename_has_more_than_one_dot(self):
        instance = ht.HTucker()
        self.assertRaises(NameError, instance.save, "test.invalid.hto")

    def test_raises_exception_when_unknown_file_extension(self):
        instance = ht.HTucker()
        self.assertRaises(NameError, instance.save, "test.unknown")

    def test_raises_not_implemented_error_with_npy_file_extension(self):
        instance = ht.HTucker()
        self.assertRaises(NotImplementedError, instance.save, "test.npy")


class TestLoadFunction(unittest.TestCase):

    @patch("pickle.load")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_function(self, mock_file, mock_load):
        # Assuming load returns an instance of ht.HTucker
        mock_load.return_value = ht.HTucker()
        
        result = ht.HTucker.load("test.hto")
        
        mock_file.assert_called_once_with("./test.hto", "rb")
        mock_load.assert_called_once()
        self.assertIsInstance(result, ht.HTucker)

    def test_raises_assertion_error_when_file_contains_slash(self):
        with self.assertRaises(AssertionError):
            ht.HTucker.load("wrong/path.hto")

    def test_raises_name_error_when_more_than_one_dot_in_filename(self):
        with self.assertRaises(NameError):
            ht.HTucker.load("test.invalid.hto")

    def test_raises_name_error_with_unknown_file_extension(self):
        with self.assertRaises(NameError):
            ht.HTucker.load("test.unknown")

    def test_raises_not_implemented_error_with_npy_file_extension(self):
        with self.assertRaises(NotImplementedError):
             ht.HTucker.load("test.npy")

def create_nway_tensor(num_dim=None, dims=None):
    np.random.seed(seed)
    # Creates a random n-dimensional tensor
    if num_dim is None:
        num_dim = random.randint(4,7)
    # print(num_dim)

    if dims is None:
        dims = [random.randint(2,10) for _ in range(num_dim)]
    # print(dims)

    leaf_ranks = [random.randint(2,max_rank) for max_rank in dims]
    # print(leaf_ranks)

    num_transfer_nodes = num_dim-2
    transfer_ranks = [random.randint(min(leaf_ranks),max(leaf_ranks)) for _ in range(num_transfer_nodes)]
    
    # print("Initial transfer ranks:")
    # print(transfer_ranks)
    if len(transfer_ranks)<6:
        transfer_ranks.extend(leaf_ranks[len(transfer_ranks)-6:])
    # print("Transfer ranks after completing to a minimum of 6:")
    # print(transfer_ranks)


    # leafs = [np.random.randn(r, n) for r,n in zip(leaf_ranks, dims)]

    l_dims,r_dims=ht.split_dimensions(dims)

    l_root_rank=transfer_ranks.pop(0)
    r_root_rank=transfer_ranks.pop(0)
    root=ht.TuckerCore(core=np.random.randn(l_root_rank,r_root_rank),dims=dims)

    l_rank=transfer_ranks.pop(0)
    r_rank=transfer_ranks.pop(0)
    root.left=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,l_root_rank),parent=root,dims=l_dims)


    l_rank=transfer_ranks.pop(0)
    r_rank=transfer_ranks.pop(0)
    root.right=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,r_root_rank),parent=root,dims=r_dims)


    completed_cores = []
    completed_cores.append(root)
    completed_cores.append(root.left)
    completed_cores.append(root.right)

    transfer_cores = []
    transfer_cores.append(root.left)
    transfer_cores.append(root.right)


    while transfer_cores:
        node=transfer_cores.pop(-1)
        node.get_ranks()

        l_dims,r_dims=ht.split_dimensions(node.dims)


        # Create right child of the current node
        if len(r_dims)==1:
            leaf_rank=node.ranks[1]
            leaf_dim=r_dims[0]
            # print(leaf_rank,leaf_dim)
            node.right=ht.TuckerLeaf(matrix=np.random.randn(leaf_dim,leaf_rank),parent=node,dims=r_dims)

        elif len(r_dims)==2:
            r_rank=leaf_ranks.pop(-1)
            l_rank=leaf_ranks.pop(-1)
            node.right=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[1]),parent=node,dims=r_dims)
            transfer_cores.insert(0,node.right)
            completed_cores.append(node.right)

        elif len(r_dims)==3:
            r_rank=leaf_ranks.pop(-1)
            l_rank=transfer_ranks.pop(0)
            node.right=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[1]),parent=node,dims=r_dims)
            transfer_cores.insert(0,node.right)
            completed_cores.append(node.right)

        else:
            l_rank=transfer_ranks.pop(0)
            r_rank=transfer_ranks.pop(0)
            node.right=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[1]),parent=node,dims=r_dims)
            transfer_cores.insert(0,node.right)
            completed_cores.append(node.right)

        # Create left child of the current node
        if len(l_dims)==1:
            leaf_rank=node.ranks[0]
            leaf_dim=l_dims[0]
            # print(leaf_rank,leaf_dim)
            node.left=ht.TuckerLeaf(matrix=np.random.randn(leaf_dim,leaf_rank),parent=node,dims=l_dims)

        elif len(l_dims)==2:
            r_rank=leaf_ranks.pop(-1)
            l_rank=leaf_ranks.pop(-1)
            node.left=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[0]),parent=node,dims=l_dims)
            transfer_cores.insert(0,node.left)
            completed_cores.append(node.left)

        elif len(l_dims)==3:
            r_rank=leaf_ranks.pop(-1)
            l_rank=transfer_ranks.pop(0)
            node.left=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[0]),parent=node,dims=l_dims)
            transfer_cores.insert(0,node.left)
            completed_cores.append(node.left)

        else:
            l_rank=transfer_ranks.pop(0)
            r_rank=transfer_ranks.pop(0)
            node.left=ht.TuckerCore(core=np.random.randn(l_rank,r_rank,node.ranks[0]),parent=node,dims=l_dims)
            transfer_cores.insert(0,node.left)
            completed_cores.append(node.left)

    transfer_cores=completed_cores.copy()

    # Contract created nodes to a tensor
    while completed_cores:
        node=completed_cores.pop(-1)
        node.contract_children()

    return node.core

    # TODO: Write test for n-mode unfolding -> Done
    # TODO: Write test for n-dimensional tucker  
        
if __name__ == '__main__':
    unittest.main()

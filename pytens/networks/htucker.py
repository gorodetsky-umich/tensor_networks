"""Implementation of Hierarchical Tuckers with Cross Approximations."""

from typing import Sequence
import itertools

import numpy as np

from pytens.algs import TensorNetwork, Tensor


class TreeNode:
    def __init__(self, dims: Sequence[int], idxs: Sequence[int]):
        self.dims = dims
        self.idxs = idxs
        self.children = []

def create_dimension_tree(dims: Sequence[int], num_splits: int = 2, min_split_size: int = 1):
    dtree = TreeNode(dims, list(range(len(dims))))
    nodes2expand = []
    nodes2expand.append(dtree)
    while nodes2expand:
        node = nodes2expand.pop(0)
        dim_splits = np.array_split(np.array(node.dims), num_splits)
        idx_splits = np.array_split(np.array(node.idxs), num_splits)
        for ds, idxs in zip(dim_splits, idx_splits):
            child = TreeNode(ds, idxs)
            dtree.children.append(child)
            if len(ds) > min_split_size:
                nodes2expand.append(child)
    
    return dtree

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def slice_to_args(dims, slices):
    """Turn array slices into function arguments"""
    indices = []
    for d, s in zip(dims, slices):
        indices.append(np.arange(d)[s])

    return itertools.product(*indices)

def greedy_pivot_search(tensor_func, tensor_approx, indices, f, P, max_iters=3):
    # create an inital index from P
    print(P)
    idx = [p[-1] for p in P]
    for _ in range(max_iters):
        for mu in f:
            idx_wo_mu = idx[:]
            idx_wo_mu[mu] = slice(None)
            idx_wo_mu = tuple(idx_wo_mu)
            diff = np.abs(tensor_func[idx_wo_mu] - tensor_approx[idx_wo_mu])
            # print(f"difference for mu of {mu} is", diff)
            bests = np.argsort(diff)
            # print(diff)
            # print(np.flip(bests))
            for best_i in np.flip(bests):
                idx[mu] = best_i
                # print("update", idx_wo_mu, best_i)
                # print(list(zip(*P)))
                if tuple(idx) not in zip(*P):
                    # print("taking", idx)
                    break
                
                # print(idx, "is in", P)


        f_prime = [i for i in range(len(indices)) if i not in f]
        if len(f_prime) == 0:
            return idx
        idx_wo_f_prime = idx[:]
        for i in range(len(indices)):
            if i not in f:
                idx_wo_f_prime[i] = P[i] if len(P[i]) != 0 else [0]
            else:
                idx_wo_f_prime[i] = [idx[i]]

        idx_wo_f_prime = np.ix_(*idx_wo_f_prime)
        # print(idx_wo_f_prime, tensor_func[idx_wo_f_prime], tensor_approx[idx_wo_f_prime])
        diff = np.abs(tensor_func[idx_wo_f_prime] - tensor_approx[idx_wo_f_prime])
        diff = diff.transpose(tuple(f_prime) + tuple(f))
        diff = diff.reshape(-1, 1)
        max_fi = np.argmax(diff, axis=0)
        # print(diff.shape, "difference matrix")
        # print("maximum values in difference matrix", max_fi)
        # restore max_fi into multiple indices
        # i1 * k^{n-1} + i2 * k^{n-2} + k * i3 + ...
        # decode this integer as a base k integer
        max_indices = numberToBase(max_fi[0], len(P[f_prime[0]]))
        if len(max_indices) < len(f_prime):
            # prepend zeros so that they have the same length
            max_indices = [0] * (len(f_prime) - len(max_indices)) + max_indices

        for i, fi in enumerate(f_prime):
            idx[fi] = P[fi][max_indices[i]] if len(P[fi]) > 0 else 0

    return idx

def cross_approx(A_func, X, left, f, pivots, eps=0.1):
    # np.random.seed(1)
    # initialize P with random numbers
    gamma = 0
    cnt = 0
    max_residual = None
    while True:
        # sample new pivots
        idx = greedy_pivot_search(A_func, X, A_func.shape, f, pivots)
        assert len(idx) == len(A_func.shape)
        # print("Proposing pivot", idx)
        # take the left indices and do the computation
        left_pivs, right_pivs = [], []
        left_szs, right_szs = [], []
        right = []
        for i in range(A_func.ndim):
            if i in left:
                left_pivs.append(slice(idx[i], idx[i]+1))
                right_pivs.append(slice(None))
                left_szs.append(A_func.shape[i])
            else:
                left_pivs.append(slice(None))
                right_pivs.append(slice(idx[i], idx[i]+1))
                right_szs.append(A_func.shape[i])
                right.append(i)

            # if jj == 0 and len(pivots[i]) == 1:
            #     pivots[i] = [idx[i]]
            # else:
            pivots[i].append(idx[i])

    
        A_permute = tuple(left) + tuple(right)
        left_sz = int(np.prod(left_szs))
        right_sz = int(np.prod(right_szs))
        # if jj == 0:
        #     # print(right_pivs)
        #     # print(A_func[tuple(right_pivs)].shape)
        #     # print(A_func[tuple(left_pivs)].shape)
        #     # assert A_func[tuple(right_pivs)].shape == np.array(A_func.shape)[tuple(left)]
    
        #     X = A_func[tuple(right_pivs)].transpose(*left, *right).reshape(left_sz, -1) / A_func[*idx] @ A_func[tuple(left_pivs)].transpose(*left, *right).reshape(-1, right_sz)
        # else:
        gamma = A_func[*idx] - X[*idx]
        if gamma == 0:
            # randomly sample points from the function and evaluate the error
            num = np.sum(A_func.shape)
            points = [np.random.randint(0, sz, num//A_func.ndim) for sz in A_func.shape]
            errs = (A_func[*points] - X[*points]) ** 2
            nrms = np.sum(A_func[*points] ** 2)
            max_ix = np.argmax(np.array(errs))
            idx = [p[max_ix] for p in points]
            if np.sqrt(np.sum(errs)) <= eps * np.sqrt(nrms):
                break

            for i, p in enumerate(pivots):
                pivots[i][-1] = idx[i]
            
            continue
        #     # pivots = [[] for _ in A.shape]
        #     # X = np.zeros(A.shape)
        #     print("restart", idx, A[*idx], X[*idx])
        #     # return rank_one_update(A_func, X, left, f, pivots)
        #     return X, pivots
        # X_old = X.copy()
        # print(k, A_func[tuple(right_pivs)].shape)
        u = (A_func[tuple(right_pivs)] - X[tuple(right_pivs)]).transpose(*A_permute).reshape(left_sz, 1)
        v = (A_func[tuple(left_pivs)] - X[tuple(left_pivs)]).transpose(*A_permute).reshape(1, right_sz) / gamma
        X = X.transpose(*A_permute).reshape(left_sz, right_sz) + u @ v

        # print("before", X.shape)
        X = X.reshape(tuple(left_szs) + tuple(right_szs))
        # print("before transpose", X.shape)
        X = X.transpose(np.argsort(A_permute))
        # print("after", X.shape)
        # print("error", np.linalg.norm(A_func - X) / np.linalg.norm(A_func), np.max(np.abs(A_func - X)))

        # if np.linalg.norm(X_old - X) / np.linalg.norm(X_old) <= 1e-3:
        #     break
        # if np.linalg.norm(X_old - X) / np.linalg.norm(X_old) < 0.01:
        #     break
        # residual_norm = np.linalg.norm(u) * np.linalg.norm(v)
        if max_residual is None:
            max_residual = gamma
        # print(gamma, max_residual)
        # if gamma < 1e-6 * max_residual: # / (1 + 2 * len(pivots[0]) ** 0.5 * (len(left) * 0.5 + len(right) * 0.5)):
        #     break
        if np.linalg.norm(u) @ np.linalg.norm(v) < eps * np.linalg.norm(X):
            # randomly sample points from the function and evaluate the error
            num = np.sum(A_func.shape)
            points = [np.random.randint(0, sz, num//A_func.ndim) for sz in A_func.shape]
            errs = (A_func[*points] - X[*points]) ** 2
            nrms = np.sum(A_func[*points] ** 2)
            max_ix = np.argmax(np.array(errs))
            idx = [p[max_ix] for p in points]
            if np.sqrt(np.sum(errs)) <= eps * np.sqrt(nrms):
                break

            for i, p in enumerate(pivots):
                pivots[i][-1] = idx[i]

        # if abs(gamma) < 0.01 * abs(max_residual):
        #     break
        # print(gamma)
        # if abs(gamma) <= eps:
        #     break

    return X, pivots

class HTucker(TensorNetwork):
    def __init__(self, nodes):
        """Create the hierarchical tucker by levels of nodes"""
        super().__init__()

    @staticmethod
    def cross(data: Tensor, eps: float):
        """Create the hierarchical tucker by cross approximation."""
        dims = [ind.size for ind in data.indices]
        dtree = create_dimension_tree(dims, 2, 1)

        approx = np.zeros_like(data.value)
        pivots = [[] for _ in dims]
        for d in dims:
            if d % 2 == 0:
                f = [d, d+1]
            else:
                f = [d-1, d]
            _, p = cross_approx(data.value, approx, [d], f, pivots, eps=eps)
            pivots[d] = p[d]

    @staticmethod
    def svd(data: Tensor):
        pass

    @staticmethod
    def hosvd(data: Tensor):
        pass
from typing import *
import copy

import numpy as np
import opt_einsum as oe
import timeit
import time

gamma = 5e-6 # 0.0015
rho = 0.001
mu = 1.0

def shrink(a, b):
    return np.maximum(a - b, 0) + np.minimum(a + b, 0)

def unfold(X: np.ndarray, dims: List[int], mode: int):
    """Split the whole tensor into ik x i1*i2*...*ik-1*ik+1*...iN where k == mode
    """
    N = len(X.shape)
    if N <= mode:
        raise Exception(f"X has shape {X.shape} but try to unfold at mode {mode}")
    
    X = X.transpose((mode,) + tuple([i for i in range(N) if i != mode]))
    return X.reshape((dims[mode], -1))

def fold(X: np.ndarray, dims: List[int], mode: int):
    """Reshape the whole tensor from ik x i1*i2*...*ik-1*ik+1*...iN to i1 x i2 x ... x iN where k == mode
    """
    N = len(dims)
    if N <= mode:
        raise Exception(f"X has shape {X.shape} but try to unfold at mode {mode}")

    X = X.reshape((dims[mode],) + tuple(dims[:mode]) + tuple(dims[mode+1:]))
    permute_indices = np.argsort((mode,) + tuple([i for i in range(N) if i != mode]))
    # print(permute_indices)
    return X.transpose(permute_indices)

class FCTN:
    def __init__(self, target):
        self.target = target
        self.G = {}
        self.S = {}
        self.indices = self.target.shape
        self.N = len(self.indices)

    def initialize(self):
        print("======= Start =======")
        # initialize S and R by calculating the mean of all mode-(t,l) slices of the target
        ranks = np.zeros((self.N, self.N), dtype=int)
        for t in range(self.N):
            for l in range(t + 1, self.N):
                axises = [i for i in range(self.N) if i != t and i != l]
                x_tl = np.mean(self.target, axis=tuple(axises))
                # print(f"x_{t}_{l} shape:", x_tl.shape)
                _, s_tl, _ = np.linalg.svd(x_tl)
                # print(f"s_{t}_{l} shape:", s_tl.shape)
                # print(f"s_{t}_{l}:", s_tl)
                s_tl = shrink(s_tl, gamma * np.max(s_tl) / (np.abs(s_tl) + 1e-16))
                # print(f"after shrink s_{t}_{l}:", s_tl)
                s_tl = s_tl[s_tl != 0]
                self.S[(t, l)] = s_tl
                # s_tl = self.S[(t, l)]
                # print(f"after shrink s_{t}_{l}:", s_tl.shape)
                ranks[(t, l)] = len(s_tl)
                # ranks[t, l] = min(self.indices[t], self.indices[l])
                # self.S[(t, l)] = np.random.random(ranks[t, l])
                print(f"rank({t}, {l}) =", ranks[t, l])

        # initialize G
        for i, idx in enumerate(self.indices):
            dims = tuple(ranks[:i, i]) + (idx,) + tuple(ranks[i, i+1:])
            self.G[i] = np.ones(dims) / np.sqrt(idx)

    def _get_index_id(self, i, j):
        # if i == j, it returns the index Ii
        # if i != j, it returns the index Rij (i < j)
        if i > j:
            i, j = j, i

        return (self.N + self.N - i + 1) * i // 2 + j - i
    
    def _get_index(self, i, j):
        idx = self._get_index_id(i, j)
        if idx < 26:
            return chr(ord('a') + idx)
        elif idx < 52:
            return chr(ord('A') + idx - 26)
        elif idx < 62:
            return chr(ord('0') + idx - 52)
        else:
            raise Exception(f"Do not support index larger than 62 but get index {idx}")

    def _contract_rest(self, exclusions: List[Tuple[str, List[int]]]):
        """Compute the context tensor for a given node.
        
        This is equivalent to contracting all tensors but the given one.
        """
        # Given the max mode N, we know that each G connects with N other nodes.
        # Gi and Gj share the same index on the edge marked with Sij.
        # Therefore, there is a total of N*(N+1)/2 indices.
        free_indices = [self._get_index(i, i) for i in range(self.N)]
        free_dims = [self.indices[i] for i in range(self.N)]
        new_free_indices, new_free_dims = [], []
        core_tensors = list(range(self.N))
        diag_tensors = [(i, j) for i in range(self.N) for j in range(i+1, self.N)]
        
        # always output the result tensor with the exclusion indices at the beginning
        for factor, k in exclusions:
            if factor == 'G':
                # Gk: r1,k x r2,k x ... x ik x rk,k+1, x ... x rk,N
                # after exclusion, we have Gk(k): r1,k x r2,k x ... rk-1,k x rk,k+1 x ... x rk,N
                for i in range(self.N):
                    idx = self._get_index(i, k[0])
                    if i != k[0]:
                        new_free_indices.append(idx)
                        new_free_dims.append(self.G[k[0]].shape[i])

                core_idx = self._get_index(k[0], k[0])
                if core_idx in free_indices:
                    free_indices.remove(core_idx)
                    free_dims.pop(k[0])
                    core_tensors.pop(k[0])

            elif factor == 'S':
                idx = self._get_index(k[0], k[1])
                if idx not in free_indices:
                    # once we remove an S, there will be two same indices becoming free
                    new_free_indices.append(idx)
                    d = self.S[(k[0], k[1])].shape
                    new_free_dims.append(d)

                diag_tensors.remove((k[0], k[1]))

            else:
                raise Exception(f"Unsupported factor type {factor}")
            
        # contract the remaining tensors
        ein_inputs = []
        arrs = []
        for gi in core_tensors:
            arrs.append(self.G[gi])
            ein_inputs.append("".join([self._get_index(i, gi) for i in range(self.N)]))

        for si in diag_tensors:
            arrs.append(self.S[si])
            # arrs.append(np.diag(self.S[si]))
            idx = self._get_index(si[0], si[1])
            # ein_inputs.append("".join([idx, idx]))
            ein_inputs.append(idx)

        ein_outputs = "".join(new_free_indices + free_indices)
        ein_args = ",".join(ein_inputs) + "->" + ein_outputs
        # print(ein_args)
        out = oe.contract(ein_args, *arrs, optimize=True)

        # group the tensor into two parts
        if len(new_free_indices) == 0:
            return out
        else:
            return out.reshape((np.prod(new_free_dims), np.prod(free_dims)))

    def _contract_rest_g(self, k: int):
        G = copy.deepcopy(self.G)
        a = tuple(range(k + 1, self.N)) + tuple(range(k + 1))
        for i in range(self.N):
            if i == k:
                continue
            
            for j in range(i + 1, self.N):
                G[i] = np.tensordot(G[i], np.diag(self.S[(i, j)]), axes=([i + 1], [0]))

            if i > k:
                G[i] = np.tensordot(G[i], np.diag(self.S[(k, i)]), axes=([k], [0]))
                G[i] = G[i].transpose(tuple(range(k)) + (self.N - 1,) + tuple(range(k, self.N - 1)))

            # print("G", i, "before transpose", G[i].shape)
            G[i] = G[i].transpose(a)
            # print("G", i, "after transpose", G[i].shape)

        M_k = G[a[0]]
        m = [0]
        n = [1]
        for i in range(self.N - 2):
            # print(M_k.shape)
            # print(G[a[i + 1]].shape)
            M_k = np.tensordot(M_k, G[a[i + 1]], axes=(n, m))
            m = [j for j in range(i + 2)]
            n = [1 + j * (self.N - i - 1) for j in range(i + 2)]
            # print(m)
            # print(n)

        # print(M_k.shape)
        M_k = M_k.transpose(tuple(range(2 * (self.N - k - 1), 2 * (self.N - 1))) + tuple(range(2 * (self.N - k - 1))))
        
        c = np.zeros(self.N - 1, dtype=int)
        d = np.zeros(self.N - 1, dtype=int)
        for i in range(self.N - 1):
            c[i] = 2 * i
            d[i] = 2 * i + 1

        M_k = M_k.transpose(tuple(d) + tuple(c))
        M_k = M_k.reshape(np.prod(M_k.shape[:self.N - 1]), np.prod(M_k.shape[self.N - 1:]))

        return M_k

    def _contract_rest_s(self, t: int, l: int):
        G = copy.deepcopy(self.G)
        for i in range(self.N):
            if i == t:
                continue
            
            for j in range(i + 1, self.N):
                G[i] = np.tensordot(G[i], np.diag(self.S[(i, j)]), axes=([i + 1], [0]))

        for j in range(t + 1, l):
            G[t] = np.tensordot(G[t], np.diag(self.S[(t, j)]), axes=([t + 1], [0]))

        for j in range(l + 1, self.N):
            G[t] = np.tensordot(G[t], np.diag(self.S[(t, j)]), axes=([t + 2], [0]))

        G[t] = G[t].transpose(tuple(range(t + 1)) + tuple(range(t + 2, l + 1)) + (t + 1,) + tuple(range(l + 1, self.N)))
        G_t = unfold(G[t], G[t].shape, l)
        G_l = unfold(G[l], G[l].shape, t)
        H_tl = np.zeros((np.prod(self.indices), len(self.S[(t, l)])))

        for i in range(len(self.S[(t, l)])):
            g = copy.deepcopy(G)
            g_t_shape = list(G[t].shape)
            # print(g_t_shape)
            g_t_shape[l] = 1
            # print(G_t.shape)
            g[t] = fold(G_t[i, :], g_t_shape, l)
            g_l_shape = list(G[l].shape)
            g_l_shape[t] = 1
            g[l] = fold(G_l[i, :], g_l_shape, t)
            H_tl[:, i] = self.tnprod(g).reshape(-1)

        return H_tl
    
    def tnprod(self, g):
        m = np.array([1])
        n = [0]
        out = g[0]
        for i in range(self.N - 1):
            # print(out.shape, m)
            # print(g[i+1].shape, n)
            out = np.tensordot(out, g[i + 1], axes=(m, n))
            n.append(i + 1)
            if i > 0:
                m[1:] = m[1:] - np.array(range(1, i+1))
            m = np.append(m, 1 + (i + 1) * (self.N - i - 1))

        return out

    def decompose(self, max_iters=5000):
        start = time.time()
        X = self.target

        Q = {(t, l): np.zeros_like(self.S[(t, l)]) for t in range(self.N) for l in range(t+1, self.N)}
        P = {(t, l): np.zeros_like(self.S[(t, l)]) for t in range(self.N) for l in range(t+1, self.N)}
        for it in range(max_iters):
            old_X = copy.deepcopy(X)

            for k in range(self.N):
                # M = self._contract_rest([('G', [k])])
                M_k = self._contract_rest_g(k)
                # print(M_k)
                # print(M)
                # print(np.allclose(M_k, M))
                X_k = unfold(self.target, self.indices, k)
                G_hat_k = unfold(self.G[k], self.G[k].shape, k)
                # print("X_k:", X_k.shape)
                # print("M_k:", M_k.shape)
                G_k = (X_k @ M_k.T + rho * G_hat_k) @ np.linalg.pinv(M_k @ M_k.T + (mu + rho) * np.eye(M_k.shape[0]))
                # print("G_k:", G_k.shape)
                # print("Old_G_k:", self.G[k].shape)
                self.G[k] = fold(G_k, self.G[k].shape, k)

            s = copy.deepcopy(self.S)
            for t in range(self.N):
                for l in range(t+1, self.N):
                    lam = gamma * np.max(self.S[(t, l)]) * (rho + 1)
                    # H = self._contract_rest([('S', [t, l])])
                    H_tl = self._contract_rest_s(t, l)
                    # print(np.allclose(H, H_tl))
                    # print("H_tl", H_tl.shape)
                    # print("X", X.reshape(-1).shape)
                    # print("S_tl", self.S[(t, l)].shape)
                    # print("P_tl", P[(t, l)].shape)
                    # old_stl = self.S[(t, l)]
                    for _ in range(5):
                        Q[(t, l)] = np.linalg.pinv(H_tl.T @ H_tl + np.eye(H_tl.shape[1])) @ (H_tl.T @ self.target.reshape(-1) + s[(t, l)] + P[(t, l)])
                        s_left = (rho * self.S[(t, l)] + Q[(t, l)] - P[(t, l)]) / (rho + 1)
                        s_right = lam / (rho + 1)
                        s[(t, l)] = shrink(s_left, s_right)
                        P[(t, l)] = P[(t, l)] + s[(t, l)] - Q[(t, l)]

                    # self.S[(t, l)] = s[(t, l)]

            # delete zero elements in S
            for t in range(self.N):
                for l in range(t+1, self.N):
                    s_tl = s[(t, l)]
                    indices = np.flatnonzero(s[(t, l)])
                    self.S[(t, l)] = s_tl[indices]
                    # print("indices:", indices)
                    # print("G[t]:", self.G[t].shape, "l:", l)
                    # print("G[l]:", self.G[l].shape, "t:", t)
                    P[(t, l)] = P[(t, l)][indices]
                    Q[(t, l)] = Q[(t, l)][indices]
                    self.G[t] = np.take(self.G[t], indices, axis=l)
                    self.G[l] = np.take(self.G[l], indices, axis=t)

            g = copy.deepcopy(self.G)
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    g[i] = np.tensordot(g[i], np.diag(self.S[(i, j)]), axes=([i + 1], [0]))

            X = self.tnprod(g)
            # print(self.target)
            err = np.linalg.norm(X - old_X) / np.linalg.norm(old_X)
            re = np.linalg.norm(X - self.target) / np.linalg.norm(self.target)
            print("Iteration:", it, "Time:", time.time() - start, "Error:", err, "RE:", re)
            if err < 1e-5:
                break

            if it % 100 == 0:
                import sys
                sys.stdout.flush()

        print("======== Finish ========")
        for t in range(self.N):
            # print(self.G[t])
            for l in range(t+1, self.N):
                # print(self.S[(t, l)])
                s_tl = self.S[(t, l)]
                print(f"rank({t}, {l}) =", len(s_tl[s_tl!=0]))

if __name__ == "__main__":
    np.random.seed(1314)
    target = np.random.randn(16, 18, 20, 22)
    target_fctn = FCTN(target)
    target_fctn.G[0] = np.random.random((16, 4, 3, 2))
    target_fctn.G[1] = np.random.random((4, 18, 2, 2))
    target_fctn.G[2] = np.random.random((3, 2, 20, 3))
    target_fctn.G[3] = np.random.random((2, 2, 3, 22))
    target_fctn.S[(0, 1)] = np.random.random(4)
    target_fctn.S[(0, 2)] = np.random.random(3)
    target_fctn.S[(0, 3)] = np.random.random(2)
    target_fctn.S[(1, 2)] = np.random.random(2)
    target_fctn.S[(1, 3)] = np.random.random(2)
    target_fctn.S[(2, 3)] = np.random.random(3)
    target = target_fctn._contract_rest([])
    
    fctn = FCTN(target)
    # fctn.S[(0, 1)] = target_fctn.S[(0, 1)]
    # fctn.S[(0, 2)] = target_fctn.S[(0, 2)]
    # fctn.S[(0, 3)] = target_fctn.S[(0, 3)]
    # fctn.S[(1, 2)] = target_fctn.S[(1, 2)]
    # fctn.S[(1, 3)] = target_fctn.S[(1, 3)]
    # fctn.S[(2, 3)] = target_fctn.S[(2, 3)]

    fctn.initialize()
    fctn.decompose()
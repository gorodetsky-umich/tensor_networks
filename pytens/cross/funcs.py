"""Test functions for cross approximation"""

from time import sleep
from typing import List
from multiprocessing import Pool
import subprocess
import pickle
import os
import sys

import numpy as np

from pytens.types import Index
import pytens.algs as pt


class TensorFunc:
    """Base class for tensor functions."""

    def __init__(self, indices: List[Index]):
        self.d = len(indices)
        self.indices = indices
        self.name = "_func_"

    def index_to_args(self, indices: np.ndarray) -> np.ndarray:
        """Convert vectorized indices to function arguments"""
        indices = indices.astype(int)
        args = np.empty_like(indices, dtype=float)
        for i, ind in enumerate(self.indices):
            args[:, i] = np.array(ind.value_choices)[indices[:, i]]

        return args

    def size(self) -> int:
        res = 1
        for ind in self.indices:
            res *= ind.size
            
        return res

    @property
    def shape(self) -> List[int]:
        result = [0] * len(self.indices)
        for i, ind in enumerate(self.indices):
            if isinstance(ind.size, int):
                result[i] = ind.size
            elif isinstance(ind.size, tuple):
                result[i] = ind.size[-1]
            else:
                raise TypeError("Unsupported index size type")

        return result

    def cost(self) -> int:
        return int(np.prod(self.shape))

    def free_indices(self) -> List[Index]:
        return self.indices

    def run(self, args: np.ndarray):
        """Run the function over the given arguments."""
        raise NotImplementedError

    def __call__(self, indices: np.ndarray):
        # print("recording", indices.shape[0])
        args = self.index_to_args(indices)
        return self.run(args)

class CountingFunc(TensorFunc):
    """A tensor function with cache"""
    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.calls = np.empty((0, self.d))

    def num_calls(self) -> int:
        """The number of unique function calls"""
        return len(np.unique(self.calls, axis=0))

    def _run(self, args: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run(self, args: np.ndarray) -> np.ndarray:
        self.calls = np.concatenate([args, self.calls])
        return self._run(args)

def simulate_neutron_diffusion(program, args):
    """Simulate the call with the given arguments"""
    arg_strs = [str(a) for a in args]
    cmd = f"./scripts/neutron_diffusion/{program} {' '.join(arg_strs)}"
    # print("running", cmd)
    result = subprocess.run(cmd, capture_output=True, shell=True, check=True)
    return float(result.stdout)

class FuncNeutron(CountingFunc):
    """Class for neutron transport function as cross approximation source."""
    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.name = f"neutron_diffusion_{len(indices)}"
        self.cache = {}

    def _check_job_status(self) -> bool:
        current_jobs = os.popen("sq").read()
        return len(current_jobs.split("\n")) == 1

    def _dispatch_jobs(self, jobs):
        print(f"dispatching {len(jobs)} jobs")
        sys.stdout.flush()
        
        # write params to the file
        params_file = "scripts/neutron_diffusion/slurm/batch_params.txt"
        with open(params_file, "w", encoding="utf-8") as params_hd:
            for job in jobs:
                params_hd.write(" ".join(str(x) for x in job) + "\n")

        # modify the params
        template = "scripts/neutron_diffusion/slurm/batch_neutron_template.sh"
        with open(template, "r", encoding="utf-8") as template_hd:
            content = template_hd.read()
            content = content.replace("<num_total_tasks>", str(len(jobs)))
            
        script = "scripts/neutron_diffusion/slurm/batch_neutron.sh"
        with open(script, "w+", encoding="utf-8") as script_hd:
            script_hd.write(content)

        # run the slurm script with the given keys
        subprocess.run(f"sbatch {script} ./scripts/neutron_diffusion/a_{self.d}.out", check=False)

    def _run(self, args: np.ndarray) -> np.ndarray:
        jobs = set()
        for i in range(args.shape[0]):
            key = tuple(args[i].tolist())
            if key not in self.cache and key not in jobs:
                jobs.add(key)

        self._dispatch_jobs(jobs)

        while not self._check_job_status():
            sleep(10)

        for job in jobs:
            with open("_".join(str(x) for x in job), "r", encoding="utf-8") as job_result:
                self.cache[job] = float(job_result.read())

        results = np.empty((args.shape[0],), dtype=float)
        for i in range(args.shape[0]):
            key = tuple(args[i].tolist())
            results[i] = self.cache[key]

        # print(results)
        # print("cache size", len(self.cache))
        with open(f"output/neutron_diffusion_{self.d}.pkl", "wb") as cache_file:
            pickle.dump(self.cache, cache_file)

        return results

class FuncData(CountingFunc):
    """Class for data tensors as cross approximation targets."""

    def __init__(self, indices: List[Index], data: np.ndarray):
        super().__init__(indices)
        self.data = data

    def _run(self, args: np.ndarray) -> np.ndarray:
        return self.data[*args.astype(int).T]


class FuncTensorNetwork(CountingFunc):
    """Class for data tensors as cross approximation targets."""

    def __init__(self, indices: List[Index], net: "pt.TensorNetwork"):
        super().__init__(indices)
        self.net = net

    def _run(self, args: np.ndarray) -> np.ndarray:
        return self.net.evaluate(self.indices, args.astype(int))


    def cost(self) -> int:
        return self.net.cost()

class NodeFunc(TensorFunc):
    """Reduce the tensor function from one node to the complete one."""

    def __init__(self, indices, old_func, node_indices, ind_mapping):
        super().__init__(indices)
        self.old_func = old_func
        self.node_indices = node_indices
        self.ind_mapping = ind_mapping

    def index_to_args(self, indices: np.ndarray) -> np.ndarray:
        # convert node indices to the network indices
        old_free = self.old_func.indices
        indices = indices.astype(int)
        new_indices = np.empty((len(indices), len(old_free)))
        for i, ind in enumerate(self.node_indices):
            # find whether we should use the up_vals or down_vals by
            # checking whether it is connected with the parent or child
            # free indices, children, parent
            if ind not in self.ind_mapping:
                new_indices[:, old_free.index(ind)] = indices[:, i]
                continue

            inds, vals = self.ind_mapping[ind]
            inds_perm = [old_free.index(ind) for ind in inds]
            new_indices[:, inds_perm] = vals[indices[:, i]]

        return self.old_func.index_to_args(new_indices)

    def run(self, args: np.ndarray) -> np.ndarray:
        return self.old_func.run(args)


class SplitFunc(TensorFunc):
    """Reduce the tensor function after split into the old one."""

    def __init__(self, indices, old_func, ind_mapping):
        super().__init__(indices)
        self.old_func = old_func
        self.ind_mapping = ind_mapping

    def index_to_args(self, indices: np.ndarray) -> np.ndarray:
        indices = indices.astype(int)
        old_free = self.old_func.indices
        old_indices = np.empty((len(indices), len(old_free)), dtype=int)
        for i, ind in enumerate(old_free):
            if i not in self.ind_mapping:
                assert ind in self.indices, "index not found"
                j = self.indices.index(ind)
                old_indices[:, i] = indices[:, j]
            else:
                split_inds, split_sizes = self.ind_mapping[i]
                old_indices[:, i] = np.ravel_multi_index(tuple(indices[:, split_inds].T), split_sizes)
        # for split_op in self.split_ops:
        #     split_out = split_op.result
        #     if split_out is None:
        #         continue

        #     split_indices = np.empty((len(indices), len(split_out)), dtype=int)
        #     split_sizes = []
        #     for i, ind in enumerate(split_out):
        #         split_indices[:, i] = indices[:, self.indices.index(ind)]
        #         split_sizes.append(int(ind.size))

        #     before_split = old_free.index(split_op.index)
        #     old_indices[:, before_split] = np.ravel_multi_index(
        #         tuple(split_indices.T), tuple(split_sizes)
        #     )

        # turn indices into arguments
        return self.old_func.index_to_args(old_indices)

    def run(self, args: np.ndarray):
        return self.old_func.run(args)


class MergeFunc(TensorFunc):
    """Tensor function for merged indices."""

    def __init__(self, indices, old_func, ind_mapping):
        super().__init__(indices)
        self.old_func = old_func
        self.ind_mapping = ind_mapping

    def index_to_args(self, indices: np.ndarray):
        old_free = self.old_func.indices
        old_indices = np.empty((len(indices), len(old_free)), dtype=int)
        for i in range(indices.shape[1]):
            if i in self.ind_mapping:
                inds, sizes = self.ind_mapping[i]
                # replace indices with unraveled indices
                old_indices[:, inds] = np.stack(
                    np.unravel_index(indices[:, i], sizes), axis=-1
                )
            else:
                j = old_free.index(self.indices[i])
                old_indices[:, j] = indices[
                    :, i
                ]

        return self.old_func.index_to_args(old_indices)

    def run(self, args: np.ndarray):
        return self.old_func.run(args)

class PermuteFunc(TensorFunc):
    """Tensor functions for index permutation."""
    def __init__(self, indices, old_func, ind_unperm):
        super().__init__(indices)
        self.old_func = old_func
        self.ind_unperm = ind_unperm

    def index_to_args(self, indices: np.ndarray):
        # permute the indices back into the order before the permutation
        return self.old_func.index_to_args(indices[:, self.ind_unperm])

    def run(self, args: np.ndarray):
        return self.old_func.run(args)

class FuncAckley(CountingFunc):
    """Source: https://www.sfu.ca/~ssurjano/ackley.html"""

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-32.768, 32.768, ind.size))
            inds.append(new_ind)
        super().__init__(inds)
        self.name = "Ackley"

    def _run(self, args: np.ndarray):
        y1 = np.sqrt(np.sum(args**2, axis=1) / args.shape[1])
        y1 = -20 * np.exp(-0.2 * y1)

        y2 = np.sum(np.cos(2 * np.pi * args), axis=1)
        y2 = -np.exp(y2 / args.shape[1])

        y3 = 20 + np.exp(1.0)

        return y1 + y2 + y3


class FuncAlpine(CountingFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("6. Alpine Function 1"; Continuous, Non-Differentiable, Separable,
            Non-scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-10, 10, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        self.name = "Alpine"

    def _run(self, args: np.ndarray):
        return np.sum(np.abs(args * np.sin(args) + 0.1 * args), axis=1)


class FuncChung(CountingFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("34. Chung Reynolds Function"; Continuous, Differentiable,
            Partially-separable, Scalable, Unimodal).
    """

    def __init__(self, indices: List[Index]):
        # self.low = -10
        # self.range = 10 * 2
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-10, 10, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        self.name = "Chung"

    def _run(self, args: np.ndarray):
        return np.sum(args**2, axis=1) ** 2


class FuncDixon(CountingFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/dixonpr.html
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-10, 10, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = -10
        # self.range = 10 * 2
        self.name = "Dixon"

    def _run(self, args: np.ndarray):
        y1 = (args[:, 0] - 1) ** 2
        i = np.arange(2, self.d + 1)
        y2 = i * (2.0 * args[:, 1:] ** 2 - args[:, :-1]) ** 2
        y2 = np.sum(y2, axis=1)

        return y1 + y2


class FuncGriewank(CountingFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/griewank.html
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-100, 100, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = -100
        # self.range = 100 * 2
        self.name = "Griewank"

    def _run(self, args: np.ndarray):
        y1 = np.sum(args**2, axis=1) / 4000

        i = np.arange(1, self.d + 1)
        y2 = np.cos(args / np.sqrt(i))
        y2 = -np.prod(y2, axis=1)

        y3 = 1.0

        return y1 + y2 + y3


class FuncPathological(CountingFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("87. Pathological Function"; Continuous, Differentiable,
            Non-separable, Non-scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-100, 100, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = -100
        # self.range = 100 * 2
        self.name = "Pathological"

    def _run(self, args: np.ndarray):
        x1 = args[:, :-1]
        x2 = args[:, 1:]

        y1 = (np.sin(np.sqrt(100.0 * x1**2 + x2**2))) ** 2 - 0.5
        y2 = 1.0 + 0.001 * (x1**2 - 2.0 * x1 * x2 + x2**2) ** 2

        return np.sum(0.5 + y1 / y2, axis=1)


class FuncPinter(CountingFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("89. Pinter Function"; Continuous, Differentiable,
            Non-separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-10, 10, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = -10
        # self.range = 10 * 2
        self.name = "Pinter"

    def _run(self, args: np.ndarray):
        xm1 = np.hstack([args[:, -1].reshape(-1, 1), args[:, :-1]])
        xp1 = np.hstack([args[:, +1:], args[:, +0].reshape(-1, 1)])

        a = xm1 * np.sin(args) + np.sin(xp1)
        b = xm1**2 - 2.0 * args + 3.0 * xp1 - np.cos(args) + 1.0

        i = np.arange(1, self.d + 1)

        y1 = np.sum(i * args**2, axis=1)
        y2 = np.sum(20 * i * np.sin(a) ** 2, axis=1)
        y3 = np.sum(i * np.log10(1.0 + i * b**2), axis=1)

        return y1 + y2 + y3


class FuncQing(CountingFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("98. Qing Function"; Continuous, Differentiable, Separable
            Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(0, 500, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = 0
        # self.range = 500
        self.name = "Qing"

    def _run(self, args: np.ndarray):
        i = np.arange(1, args.shape[1] + 1)
        return np.sum((args**2 - i) ** 2, axis=1)


class FuncRastrigin(CountingFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/rastr.html
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-5.12, 5.12, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = -5.12
        # self.range = 5.12 * 2
        self.name = "Rastrigin"

    def _run(self, args: np.ndarray):
        y1 = 10.0 * self.d
        y2 = np.sum(args**2 - 10.0 * np.cos(2.0 * np.pi * args), axis=1)
        return y1 + y2


class FuncSchaffer(CountingFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("135. Schaffer Function F6"; Continuous, Differentiable,
            Non-Separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-100, 100, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = -100
        # self.range = 100 * 2
        self.name = "Schaffer"

    def _run(self, args: np.ndarray):
        z = args[:, :-1] ** 2 + args[:, 1:] ** 2
        y = 0.5 + (np.sin(np.sqrt(z)) ** 2 - 0.5) / (1.0 + 0.001 * z) ** 2
        return np.sum(y, axis=1)


class FuncSchwefel(CountingFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("127. Schwefel Function 2.26"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(0, 500, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = 0
        # self.range = 500
        self.name = "Schwefel"

    def _run(self, args: np.ndarray):
        return -np.sum(args * np.sin(np.sqrt(np.abs(args))), axis=1) / self.d


class FuncSphere(CountingFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/spheref.html
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-5.12, 5.12, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = -5.12
        # self.range = 5.12 * 2
        self.name = "Sphere"

    def _run(self, args: np.ndarray):
        return np.sum(args**2, axis=1)


class FuncSquares(CountingFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/sumsqu.html
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-10, 10, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = -10
        # self.range = 10 * 2
        self.name = "Squares"

    def _run(self, args: np.ndarray):
        i = np.arange(1, self.d + 1)
        return np.sum(i * args**2, axis=1)


class FuncTrigonometric(CountingFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("152. Trigonometric Function 1"; Continuous, Differentiable,
            Non-separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(0, np.pi, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = 0
        # self.range = np.pi
        self.name = "Trigonometric"

    def _run(self, args: np.ndarray):
        i = np.arange(1, self.d + 1)

        y1 = self.d
        y2 = -np.sum(np.cos(args), axis=1)
        y2 = np.hstack([y2.reshape(-1, 1)] * self.d)
        y3 = i * (1.0 - np.cos(args) - np.sin(args))

        return np.sum((y1 + y2 + y3) ** 2, axis=1)


class FuncWavy(CountingFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("164. W / Wavy Function"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.linspace(-np.pi, np.pi, ind.size))
            inds.append(new_ind)

        super().__init__(inds)
        # self.low = -np.pi
        # self.range = np.pi * 2
        self.name = "Wavy"

    def _run(self, args: np.ndarray):
        y = np.cos(10.0 * args) * np.exp(-(args**2) / 2)
        return 1.0 - np.sum(y, axis=1) / self.d


class FuncHilbert(CountingFunc):
    """
    Source:
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.arange(ind.size) + 1)
            inds.append(new_ind)

        super().__init__(inds)
        self.name = "Hilbert"

    def _run(self, args: np.ndarray):
        return 1.0 / np.sum(args + 1, axis=1)


class FuncSqSum(CountingFunc):
    """
    Source:
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.arange(ind.size) + 1)
            inds.append(new_ind)

        super().__init__(inds)
        self.name = "SqSum"

    def _run(self, args: np.ndarray):
        return 1.0 / np.sqrt(np.sum(args**2, axis=1))


class FuncExpSum(CountingFunc):
    """
    Source:
    """

    def __init__(self, indices: List[Index]):
        inds = []
        for ind in indices:
            new_ind = ind.with_new_rng(np.arange(ind.size) + 1)
            inds.append(new_ind)

        super().__init__(inds)
        self.name = "ExpSum"

    def _run(self, args: np.ndarray):
        return np.exp(-np.sqrt(np.sum(args**2, axis=1)))


class FuncToy1(CountingFunc):
    """The toy example 1 from the paper
    TODO: add paper info
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.name = "Toy1"

    def _run(self, args: np.ndarray):
        return np.exp(-4 * np.prod(args, axis=1) ** 2)


class FuncToy2(CountingFunc):
    """The toy example 2 from the paper
    TODO: add paper info
    """

    def __init__(self, indices: List[Index], b: int):
        super().__init__(indices)
        self.name = f"Toy2 (b={b})"
        self.b = b

    def _run(self, args: np.ndarray):
        return np.power(np.sum(np.power(args, self.b), axis=1), -1.0 / self.b)


class FuncTDE(CountingFunc):
    """The toy example 2 from the paper
    TODO: add paper info
    """

    def __init__(self, indices: List[Index], t: int, b: int, lam: int):
        super().__init__(indices)
        self.name = "Nonlinear TDE"
        self.t = t
        self.b = b
        self.lam = lam

    def _run(self, args: np.ndarray):
        return np.power(
            np.sum(np.power(args, self.b), axis=1) + np.exp(self.t * self.lam),
            -1.0 / self.b,
        )


class FuncAdvReact(CountingFunc):
    """The toy example 2 from the paper
    TODO: add paper info
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.name = "4D-AR"

    def _run(self, args: np.ndarray):
        return np.exp(-np.sum((2 * args - 0.5) ** 2, axis=1))


FUNCS = [
    FuncAckley,
    FuncAlpine,
    FuncChung,
    FuncDixon,
    FuncGriewank,
    FuncPathological,
    FuncPinter,
    FuncQing,
    FuncRastrigin,
    FuncSchaffer,
    FuncSchwefel,
    FuncSphere,
    FuncSquares,
    FuncTrigonometric,
    FuncWavy,
    FuncHilbert,
    FuncSqSum,
]

HARD_FUNCS = [
    FuncDixon,
    FuncPathological,
    FuncPinter,
    FuncQing,
    FuncSchaffer,
    FuncTrigonometric,
    FuncHilbert,
    FuncSqSum,
]

if __name__ == "__main__":
    # test neutron function
    f = FuncNeutron([Index("I1", 21, np.linspace(0, 2, 21)), Index("I2", 21, np.linspace(0, 2, 21)), Index("I3", 21, np.linspace(0, 2, 21))])
    f(np.array([[0, 10, 20], [10, 10, 20], [20, 10, 20]]))
    f(np.array([[0, 10, 20], [10, 10, 20], [20, 10, 10]]))
    f.pool.close()
    f.pool.join()

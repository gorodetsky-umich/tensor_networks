"""Test functions for cross approximation"""

from typing import List
from functools import lru_cache

import numpy as np

from pytens.types import Index
import pytens.algs as pt


class TensorFunc:
    """Base class for tensor functions."""

    def __init__(self, indices: List[Index]):
        self.d = len(indices)
        self.indices = indices
        self.stats = 0

    def index_to_args(self, indices: np.ndarray) -> np.ndarray:
        """Convert vectorized indices to function arguments"""
        indices = indices.astype(int)
        args = np.empty_like(indices, dtype=float)
        for i, ind in enumerate(self.indices):
            args[:, i] = np.array(ind.value_choices)[indices[:, i]]

        return args

    @property
    def shape(self) -> List[int]:
        result = [0] * self.d
        for i, ind in enumerate(self.indices):
            if isinstance(ind.size, int):
                result[i] = 1
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
        self.stats += indices.shape[0]
        args = self.index_to_args(indices)
        return self.run(args)


class FuncData(TensorFunc):
    """Class for data tensors as cross approximation targets."""

    def __init__(self, indices: List[Index], data: np.ndarray):
        super().__init__(indices)
        self.data = data

    def run(self, args: np.ndarray):
        return self.data[*args.astype(int).T]


class FuncTensorNetwork(TensorFunc):
    """Class for data tensors as cross approximation targets."""

    def __init__(self, indices: List[Index], net: "pt.TensorNetwork"):
        super().__init__(indices)
        self.net = net

    def run(self, args: np.ndarray):
        return self.net.evaluate(args.astype(int))


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

    def __init__(self, indices, old_func, var_mapping):
        super().__init__(indices)
        self.old_func = old_func
        self.var_mapping = var_mapping

    def index_to_args(self, indices: np.ndarray) -> np.ndarray:
        indices = indices.astype(int)
        old_free = self.old_func.indices
        old_indices = np.empty((len(indices), len(old_free)), dtype=int)
        for i, ind in enumerate(old_free):
            if i not in self.var_mapping:
                assert ind in self.indices, "index not found"
                j = self.indices.index(ind)
                old_indices[:, i] = indices[:, j]
            else:
                split_inds, split_sizes = self.var_mapping[i]
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

    def __init__(self, indices, old_func, var_mapping):
        super().__init__(indices)
        self.old_func = old_func
        self.var_mapping = var_mapping

    def index_to_args(self, indices: np.ndarray):
        old_free = self.old_func.indices
        real_indices = np.empty((len(indices), len(old_free)))
        for i in range(indices.shape[1]):
            if i in self.var_mapping:
                sizes, inds = self.var_mapping[i]
                # replace indices with unraveled indices
                real_indices[:, inds] = np.stack(
                    np.unravel_index(indices[:, i], sizes), axis=-1
                )
            else:
                j = old_free.index(self.indices[i])
                real_indices[:, j] = indices[
                    :, i
                ]

        return self.old_func.index_to_args(real_indices)

    def run(self, args: np.ndarray):
        return self.old_func.run(args)


class FuncAckley(TensorFunc):
    """Source: https://www.sfu.ca/~ssurjano/ackley.html"""

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -32.768
        # self.range = 32.768 * 2
        self.name = "Ackley"

    def run(self, args: np.ndarray):
        y1 = np.sqrt(np.sum(args**2, axis=1) / args.shape[1])
        y1 = -20 * np.exp(-0.2 * y1)

        y2 = np.sum(np.cos(2 * np.pi * args), axis=1)
        y2 = -np.exp(y2 / args.shape[1])

        y3 = 20 + np.exp(1.0)

        return y1 + y2 + y3


class FuncAlpine(TensorFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("6. Alpine Function 1"; Continuous, Non-Differentiable, Separable,
            Non-scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -10
        # self.range = 10 * 2
        self.name = "Alpine"

    def run(self, args: np.ndarray):
        return np.sum(np.abs(args * np.sin(args) + 0.1 * args), axis=1)


class FuncChung(TensorFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("34. Chung Reynolds Function"; Continuous, Differentiable,
            Partially-separable, Scalable, Unimodal).
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -10
        # self.range = 10 * 2
        self.name = "Chung"

    def run(self, args: np.ndarray):
        return np.sum(args**2, axis=1) ** 2


class FuncDixon(TensorFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/dixonpr.html
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -10
        # self.range = 10 * 2
        self.name = "Dixon"

    def run(self, args: np.ndarray):
        y1 = (args[:, 0] - 1) ** 2
        i = np.arange(2, self.d + 1)
        y2 = i * (2.0 * args[:, 1:] ** 2 - args[:, :-1]) ** 2
        y2 = np.sum(y2, axis=1)

        return y1 + y2


class FuncGriewank(TensorFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/griewank.html
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -100
        # self.range = 100 * 2
        self.name = "Griewank"

    def run(self, args: np.ndarray):
        y1 = np.sum(args**2, axis=1) / 4000

        i = np.arange(1, self.d + 1)
        y2 = np.cos(args / np.sqrt(i))
        y2 = -np.prod(y2, axis=1)

        y3 = 1.0

        return y1 + y2 + y3


class FuncPathological(TensorFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("87. Pathological Function"; Continuous, Differentiable,
            Non-separable, Non-scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -100
        # self.range = 100 * 2
        self.name = "Pathological"

    def run(self, args: np.ndarray):
        x1 = args[:, :-1]
        x2 = args[:, 1:]

        y1 = (np.sin(np.sqrt(100.0 * x1**2 + x2**2))) ** 2 - 0.5
        y2 = 1.0 + 0.001 * (x1**2 - 2.0 * x1 * x2 + x2**2) ** 2

        return np.sum(0.5 + y1 / y2, axis=1)


class FuncPinter(TensorFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("89. Pinter Function"; Continuous, Differentiable,
            Non-separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -10
        # self.range = 10 * 2
        self.name = "Pinter"

    def run(self, args: np.ndarray):
        xm1 = np.hstack([args[:, -1].reshape(-1, 1), args[:, :-1]])
        xp1 = np.hstack([args[:, +1:], args[:, +0].reshape(-1, 1)])

        a = xm1 * np.sin(args) + np.sin(xp1)
        b = xm1**2 - 2.0 * args + 3.0 * xp1 - np.cos(args) + 1.0

        i = np.arange(1, self.d + 1)

        y1 = np.sum(i * args**2, axis=1)
        y2 = np.sum(20 * i * np.sin(a) ** 2, axis=1)
        y3 = np.sum(i * np.log10(1.0 + i * b**2), axis=1)

        return y1 + y2 + y3


class FuncQing(TensorFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("98. Qing Function"; Continuous, Differentiable, Separable
            Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = 0
        # self.range = 500
        self.name = "Qing"

    def run(self, args: np.ndarray):
        i = np.arange(1, self.d + 1)
        return np.sum((args**2 - i) ** 2, axis=1)


class FuncRastrigin(TensorFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/rastr.html
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -5.12
        # self.range = 5.12 * 2
        self.name = "Rastrigin"

    def run(self, args: np.ndarray):
        y1 = 10.0 * self.d
        y2 = np.sum(args**2 - 10.0 * np.cos(2.0 * np.pi * args), axis=1)
        return y1 + y2


class FuncSchaffer(TensorFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("135. Schaffer Function F6"; Continuous, Differentiable,
            Non-Separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -100
        # self.range = 100 * 2
        self.name = "Schaffer"

    def run(self, args: np.ndarray):
        z = args[:, :-1] ** 2 + args[:, 1:] ** 2
        y = 0.5 + (np.sin(np.sqrt(z)) ** 2 - 0.5) / (1.0 + 0.001 * z) ** 2
        return np.sum(y, axis=1)


class FuncSchwefel(TensorFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("127. Schwefel Function 2.26"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = 0
        # self.range = 500
        self.name = "Schwefel"

    def run(self, args: np.ndarray):
        return -np.sum(args * np.sin(np.sqrt(np.abs(args))), axis=1) / self.d


class FuncSphere(TensorFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/spheref.html
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -5.12
        # self.range = 5.12 * 2
        self.name = "Sphere"

    def run(self, args: np.ndarray):
        return np.sum(args**2, axis=1)


class FuncSquares(TensorFunc):
    """
    Source: https://www.sfu.ca/~ssurjano/sumsqu.html
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -10
        # self.range = 10 * 2
        self.name = "Squares"

    def run(self, args: np.ndarray):
        i = np.arange(1, self.d + 1)
        return np.sum(i * args**2, axis=1)


class FuncTrigonometric(TensorFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("152. Trigonometric Function 1"; Continuous, Differentiable,
            Non-separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = 0
        # self.range = np.pi
        self.name = "Trigonometric"

    def run(self, args: np.ndarray):
        i = np.arange(1, self.d + 1)

        y1 = self.d
        y2 = -np.sum(np.cos(args), axis=1)
        y2 = np.hstack([y2.reshape(-1, 1)] * self.d)
        y3 = i * (1.0 - np.cos(args) - np.sin(args))

        return np.sum((y1 + y2 + y3) ** 2, axis=1)


class FuncWavy(TensorFunc):
    """
    Source: See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("164. W / Wavy Function"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        # self.low = -np.pi
        # self.range = np.pi * 2
        self.name = "Wavy"

    def run(self, args: np.ndarray):
        y = np.cos(10.0 * args) * np.exp(-(args**2) / 2)
        return 1.0 - np.sum(y, axis=1) / self.d


class FuncHilbert(TensorFunc):
    """
    Source:
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.name = "Hilbert"

    def run(self, args: np.ndarray):
        return 1.0 / np.sum(args, axis=1)


class FuncSqSum(TensorFunc):
    """
    Source:
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.name = "SqSum"

    def run(self, args: np.ndarray):
        return 1.0 / np.sqrt(np.sum(args**2, axis=1))


class FuncExpSum(TensorFunc):
    """
    Source:
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.name = "ExpSum"

    def run(self, args: np.ndarray):
        return np.exp(-np.sqrt(np.sum(args**2, axis=1)))


class FuncToy1(TensorFunc):
    """The toy example 1 from the paper
    TODO: add paper info
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.name = "Toy1"

    def run(self, args: np.ndarray):
        return np.exp(-4 * np.prod(args, axis=1) ** 2)


class FuncToy2(TensorFunc):
    """The toy example 2 from the paper
    TODO: add paper info
    """

    def __init__(self, indices: List[Index], b: int):
        super().__init__(indices)
        self.name = f"Toy2 (b={b})"
        self.b = b

    def run(self, args: np.ndarray):
        return np.power(np.sum(np.power(args, self.b), axis=1), -1.0 / self.b)


class FuncTDE(TensorFunc):
    """The toy example 2 from the paper
    TODO: add paper info
    """

    def __init__(self, indices: List[Index], t: int, b: int, lam: int):
        super().__init__(indices)
        self.name = "Nonlinear TDE"
        self.t = t
        self.b = b
        self.lam = lam

    def run(self, args: np.ndarray):
        return np.power(
            np.sum(np.power(args, self.b), axis=1) + np.exp(self.t * self.lam),
            -1.0 / self.b,
        )


class FuncAdvReact(TensorFunc):
    """The toy example 2 from the paper
    TODO: add paper info
    """

    def __init__(self, indices: List[Index]):
        super().__init__(indices)
        self.name = "4D-AR"

    def run(self, args: np.ndarray):
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
    FuncExpSum,
]

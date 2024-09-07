# from enum import Enum
# import numpy as np
# import numpy.typing as npt
# from .types import *

# BoundaryCondition = Enum('BoundaryCondition', ['None', 'D', 'N'])

# class Stencil1D:
#     """1D stencils."""
#
#     def __init__(self,
#                  N: int,
#                  bc_left: BoundaryCondition,
#                  bc_right: BoundaryCondition) -> None:

#         self.N = N
#         self.bc_left = bc_left
#         self.bc_right = bc_right
#         self.Arr: Optional[DoubelArr2d] = None

# def get_1d_stencil_plus(N: int, h: float):
#     """1D stencil."""
#     A = np.zeros((N, N))
#     for ii in range(1, N-1):
#         A[ii, ii] = 1
#         A[ii, ii-1] = -1

#     A /= h

from typing import Optional, Sequence, Tuple, List
from dataclasses import dataclass

import numpy as np
import scipy

from line_profiler import profile

from pytens.cross.funcs import TensorFunc


@dataclass
class CrossApproxState:
    """Static and dynamic states for cross approximation."""

    # static states
    lefts: Sequence[int]
    rights: Sequence[int]
    row_arg_space: np.ndarray
    col_arg_space: np.ndarray
    perm_order: np.ndarray

    # dynamic states
    rows: List[np.ndarray]
    cols: List[np.ndarray]
    selected_rows: np.ndarray
    selected_cols: np.ndarray
    ranks_and_errors: List[Tuple[int, float]]
    tensor: np.ndarray


def new_point(
    approx_state: CrossApproxState,
    validation_points: Tuple[np.ndarray, np.ndarray],
    err: np.ndarray,
) -> Tuple[int, int]:
    """Propose a new point to start the 2D cross approximation"""
    abs_err = np.abs(err)
    max_diff = np.argmax(abs_err).astype(int)
    i, j = validation_points[0][max_diff], validation_points[1][max_diff]
    while i in approx_state.selected_rows and j in approx_state.selected_cols:
        abs_err[max_diff] = -np.inf
        max_diff = np.argmax(abs_err).astype(int)
        i, j = validation_points[0][max_diff], validation_points[1][max_diff]

    # print(
    #     np.sqrt(err_sq),
    #     np.sqrt(nrm_sq),
    #     np.sqrt(err_sq) / np.sqrt(nrm_sq),
    #     eps,
    # )
    # approx_state.selected_rows = np.append(approx_state.selected_rows, i)
    # approx_state.selected_cols = np.append(approx_state.selected_cols, j)
    return int(i), int(j)


def end_cross_approx(
    err: float,
    k: int,
    eps: Optional[float] = None,
    max_k: Optional[int] = None,
):
    """Check whether the cross aproximation should terminate.

    It terminates when it either reaches the prescribed error bound or
    reaches the prescribed number of steps.
    """
    # print(err)
    if max_k is not None:
        if k >= max_k:
            return True
        elif eps is not None:
            return abs(err) <= eps
        else:
            return False

    if eps is not None:
        return abs(err) <= eps

    raise ValueError("either eps or max_k should be set for cross approx.")


def validation(
    approx_tensor: np.ndarray,
    valid_pts: Tuple[np.ndarray, np.ndarray],
    valid_vals: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Validate the tensor approximation over the given points."""

    approx_vals = approx_tensor[*valid_pts]
    assert valid_vals.shape == approx_vals.shape, (
        f"real values and approximate values have different shapes,"
        f"expect {valid_vals.shape}, but get {approx_vals.shape}"
    )
    validation_err = valid_vals - approx_vals
    # print(err)
    validation_err_sq = np.sum(validation_err**2)
    validation_nrm_sq = np.sum(valid_vals**2)
    return np.sqrt(validation_err_sq / validation_nrm_sq), validation_err


def product_args(row_args, col_args, rows=None, cols=None):
    """Compute the product of choices of arguments given rows or cols"""
    # print(row_args.shape, col_args.shape)
    assert len(row_args.shape) == 2
    assert len(col_args.shape) == 2
    rargs = row_args if rows is None else row_args[rows]
    cargs = col_args if cols is None else col_args[cols]
    row_cnt = len(rargs)
    col_cnt = len(cargs)
    # print(rargs.shape, cargs.shape)
    rargs = np.tile(rargs, (col_cnt, 1))
    cargs = np.repeat(cargs, row_cnt, axis=0)
    # print(row_args.shape, col_args.shape, rargs.shape, cargs.shape)
    args = np.concat([rargs, cargs], axis=1)
    return args


@profile
def cross(
    tensor_func: TensorFunc,
    approx_state: CrossApproxState,
    eps: Optional[float] = None,
    max_k: Optional[int] = None,
):
    """
    Implementation of cross approximation from Jonas Ballani, Lars Grasedyck,
    Melanie Kluge, Black box approximation of tensors in hierarchical Tucker
    format, Linear Algebra and its Applications, Volume 438, Issue 2, 2013,
    Pages 639-657, ISSN 0024-3795, https://doi.org/10.1016/j.laa.2011.08.010.

    It will modify the argument approx_state with results.
    """
    k = 0
    i, j = 0, 0
    old_i, old_j = None, None
    function = tensor_func
    row_args = approx_state.row_arg_space
    col_args = approx_state.col_arg_space
    reorder = approx_state.perm_order
    r = len(row_args)
    c = len(col_args)

    # generate the validation set by sampling from the domain
    valid_size = min(r + c, 5000)
    valid_x = np.random.choice(r, valid_size)
    valid_y = np.random.choice(c, valid_size)
    valid_pts = (valid_x, valid_y)
    args = np.concat((row_args[valid_x], col_args[valid_y]), axis=-1)
    args = args[:, reorder]
    valid_vals = function(args)

    approx_tensor = np.zeros((r, c), order="F")

    # start = time.time()
    # preallocate the space for arguments
    col_arg_num = col_args.shape[1]
    row_arg_num = row_args.shape[1]

    u_args = np.insert(
        row_args, row_arg_num, np.zeros((col_arg_num, 1)), axis=1
    )
    u_args = u_args[:, reorder]

    v_args = np.insert(col_args, 0, np.zeros((row_arg_num, 1)), axis=1)
    v_args = v_args[:, reorder]

    eval_cache = {}

    while max_k is None or k < max_k:
        # print("iter time:", time.time() - start)
        # start = time.time()
        uk, vk = np.empty(0), np.empty(0)
        masked_rows, masked_cols = np.empty(0), np.empty(0)
        row_vals, col_vals = np.empty(1), np.empty(1)
        gamma = 0.0
        for iter in range(3):
            if (-1, j) in eval_cache:
                col_vals = eval_cache[(-1, j)]
            else:
                for ii, jj in enumerate(reorder):
                    if jj >= row_arg_num:
                        u_args[:, ii] = col_args[j][jj - row_arg_num]

                col_vals = function(u_args)
                eval_cache[(-1, j)] = col_vals

            # if iter == 0 and k != 0:
            #     approx_state.cols.append(col_vals)

            approx_col_vals = approx_tensor[:, j]
            uk = col_vals - approx_col_vals
            # uk = uk.reshape(p, 1)

            abs_uk = np.abs(uk)
            # ensure we don't find the same i
            masked_rows = approx_state.selected_rows
            if len(masked_rows) > 0:
                abs_uk[masked_rows] = -np.inf

            i = np.argmax(abs_uk).astype(
                int
            )  # astype to get rid of type errors

            # print("choosing i =", i)

            # if i == old_i:
            #     break

            if (i, -1) in eval_cache:
                row_vals = eval_cache[(i, -1)]
            else:
                for ii, jj in enumerate(reorder):
                    if jj < row_arg_num:
                        v_args[:, ii] = row_args[i][jj]

                # print(v_args.shape)
                # print((p, q))
                row_vals = function(v_args)
                eval_cache[(i, -1)] = row_vals

            vk = row_vals - approx_tensor[i, :]

            # print("vk", (u[i:i+1, :]@v.T).shape)
            if vk[j] == 0:
                # print(f"zero {i,j}, checking convergence")
                # if max_k is not None and k >= max_k:
                #     approx_state.tensor = approx_tensor
                #     return 0

                eeps, valid_errs = validation(
                    approx_tensor, valid_pts, valid_vals
                )
                i, j = new_point(approx_state, valid_pts, valid_errs)
                # print("after check_cvg", i, j)
                continue

            gamma = vk[j]
            # print("gamma", gamma.shape)
            # uk = uk / gamma
            # print("vk", vk)
            abs_vk = np.abs(vk)
            masked_cols = approx_state.selected_cols
            if len(masked_cols) > 0:
                abs_vk[masked_cols] = -np.inf
            j = np.argmax(abs_vk).astype(int)
            # print("choosing j =", j)
            # vk = vk.reshape(q, 1)

            if i == old_i and j == old_j:
                break

            old_i = i
            old_j = j

        ger = scipy.linalg.get_blas_funcs("ger", [approx_tensor])
        # print(uk.shape, vk.shape, approx_tensor.shape)
        ger(1.0 / gamma, uk, vk, a=approx_tensor, overwrite_a=1)
        # print(u.shape, uk.shape, v.shape, vk.shape)
        # u_vec = np.einsum("ji,i->j", u[:k], uk)
        # v_vec = np.einsum("ji,i->j", v[:k], vk)
        # u_vec = u[:k] @ uk
        # v_vec = v[:k] @ vk
        # uv_dot = oe.contract(
        #     "ji,i,jk,k", u[:k], uk, v[:k], vk, optimize=True
        # )
        # nrm_sq = nrm_sq + 2 * uv_dot + err_sq
        # nrm = np.sqrt(nrm_sq).item()
        # # print(nrm, np.linalg.norm(approx_state.tensor))
        # # nrm = np.linalg.norm(approx_tensor)
        # u[k] = uk
        # v[k] = vk
        # u = np.concat([u, uk[:, None]], axis=1)
        # v = np.concat([v, vk[:, None]], axis=1)
        # print(k, gamma)
        # remember all ranks and errors in the state
        eeps, valid_errs = validation(approx_tensor, valid_pts, valid_vals)
        k += 1
        # penalty = (min(r, c) - k) ** 0.5
        approx_state.ranks_and_errors.append((k, eeps))
        approx_state.selected_rows = np.append(masked_rows, i)
        approx_state.selected_cols = np.append(masked_cols, j)
        # approx_state.rows.append(row_vals)
        # approx_state.cols.append(col_vals)

        if end_cross_approx(eeps, k, eps, max_k):
            # if err <= eps * nrm:
            approx_state.tensor = approx_tensor
            if max_k is not None and k >= max_k:
                # print(k, max_k)
                return eeps

            if eps is not None and eeps <= eps:
                return eeps

            i, j = new_point(approx_state, valid_pts, valid_errs)

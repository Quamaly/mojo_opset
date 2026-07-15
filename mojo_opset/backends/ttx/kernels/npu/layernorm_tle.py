import torch
import triton
import triton.language as tl
from triton.language.math import rsqrt
from .utils import libentry
import triton.runtime.driver as driver

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align, ceil_div, torch_to_triton_dtype


UB_SAFE_BUDGET_BYTES = 48 * 1024          

def _get_num_vector_cores():
    props = driver.active.utils.get_device_properties("npu")
    return props["num_vectorcore"]


def _calculate_optimal_tiling(n_cols, dtype):

    element_size = 4
    wb_bytes = 2 * n_cols * element_size
    one_row_bytes = n_cols * element_size


    if one_row_bytes + wb_bytes <= UB_SAFE_BUDGET_BYTES:

        max_m = (UB_SAFE_BUDGET_BYTES - wb_bytes) // one_row_bytes
        block_size_m = max(1, min(max_m, 24))
        align_unit = 32 // element_size  
        block_size_n = ((n_cols + align_unit - 1) // align_unit) * align_unit
        return block_size_n, block_size_m
    else:
        block_size_n = 2048
        if n_cols >= 7000:
            block_size_m = 2
        else:
            block_size_m = 4
        return block_size_n, block_size_m


@libentry()
@triton.jit
def _layernorm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, RSTD_ptr,
    stride_x_row, stride_y_row,
    n_rows, n_cols, eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    IS_INFERENCE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, num_programs):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows


        if BLOCK_SIZE_N >= n_cols:
            cols = tl.arange(0, BLOCK_SIZE_N)
            mask = cols < n_cols
            x = tl.load(X_ptr + rows_off[:, None] * stride_x_row + cols[None, :],
                        mask=rows_mask[:, None] & mask[None, :], other=0.0).to(tl.float32)

            sum_x = tl.sum(x, axis=1)
            sum_x2 = tl.sum(x * x, axis=1)
            mean = sum_x / n_cols
            var = sum_x2 / n_cols - mean * mean
            rstd = rsqrt(var + eps)

            if not IS_INFERENCE:
                tl.store(Mean_ptr + rows_off, mean, mask=rows_mask)
                tl.store(RSTD_ptr + rows_off, rstd, mask=rows_mask)

            w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]
            tl.store(Y_ptr + rows_off[:, None] * stride_y_row + cols[None, :],
                     y.to(Y_ptr.dtype.element_ty),
                     mask=rows_mask[:, None] & mask[None, :])
        else:
            sum_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            for col_offset in range(0, n_cols, BLOCK_SIZE_N):
                cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
                cols_mask = cols_off < n_cols
                block_mask = rows_mask[:, None] & cols_mask[None, :]
                x_chunk = tl.load(
                    X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :],
                    mask=block_mask, other=0.0
                ).to(tl.float32)
                sum_acc += tl.sum(x_chunk, axis=1)

            mean = sum_acc / n_cols
            if not IS_INFERENCE:
                tl.store(Mean_ptr + rows_off, mean, mask=rows_mask)

            var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            for col_offset in range(0, n_cols, BLOCK_SIZE_N):
                cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
                cols_mask = cols_off < n_cols
                block_mask = rows_mask[:, None] & cols_mask[None, :]
                x_chunk = tl.load(
                    X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :],
                    mask=block_mask, other=0.0
                ).to(tl.float32)
                x_centered = x_chunk - mean[:, None]
                var_acc += tl.sum(x_centered * x_centered, axis=1)

            var = var_acc / n_cols
            rstd = rsqrt(var + eps)
            if not IS_INFERENCE:
                tl.store(RSTD_ptr + rows_off, rstd, mask=rows_mask)

            for col_offset in range(0, n_cols, BLOCK_SIZE_N):
                cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
                cols_mask = cols_off < n_cols
                block_mask = rows_mask[:, None] & cols_mask[None, :]
                x_chunk = tl.load(
                    X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :],
                    mask=block_mask, other=0.0
                ).to(tl.float32)
                w_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)
                b_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)
                x_centered = x_chunk - mean[:, None]
                y_chunk = x_centered * rstd[:, None] * w_chunk[None, :] + b_chunk[None, :]
                tl.store(
                    Y_ptr + rows_off[:, None] * stride_y_row + cols_off[None, :],
                    y_chunk.to(Y_ptr.dtype.element_ty),
                    mask=block_mask,
                )


@libentry()
@triton.jit
def _layernorm_fwd_fullrow_hoist_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, RSTD_ptr,
    stride_x_row, stride_y_row,
    n_rows, n_cols, eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    IS_INFERENCE: tl.constexpr,
    IS_EXACT_N: tl.constexpr,
    IS_EXACT_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    cols = tl.arange(0, BLOCK_SIZE_N)

    if IS_EXACT_N:
        w = tl.load(W_ptr + cols).to(tl.float32)
        b = tl.load(B_ptr + cols).to(tl.float32)
    else:
        cols_mask = cols < n_cols
        w = tl.load(W_ptr + cols, mask=cols_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=cols_mask, other=0.0).to(tl.float32)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, num_programs):
        row_start = row_task_id * BLOCK_SIZE_M
        rows = row_start + tl.arange(0, BLOCK_SIZE_M)

        if IS_EXACT_M and IS_EXACT_N:
            x = tl.load(
                X_ptr + rows[:, None] * stride_x_row + cols[None, :]
            ).to(tl.float32)

            sum_x = tl.sum(x, axis=1)
            sum_x2 = tl.sum(x * x, axis=1)

            mean = sum_x / n_cols
            var = sum_x2 / n_cols - mean * mean
            rstd = rsqrt(var + eps)

            if not IS_INFERENCE:
                tl.store(Mean_ptr + rows, mean)
                tl.store(RSTD_ptr + rows, rstd)

            y = (x - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]

            tl.store(
                Y_ptr + rows[:, None] * stride_y_row + cols[None, :],
                y.to(Y_ptr.dtype.element_ty),
            )

        else:
            row_mask = rows < n_rows
            cols_mask = cols < n_cols
            block_mask = row_mask[:, None] & cols_mask[None, :]

            x = tl.load(
                X_ptr + rows[:, None] * stride_x_row + cols[None, :],
                mask=block_mask,
                other=0.0,
            ).to(tl.float32)

            sum_x = tl.sum(x, axis=1)
            sum_x2 = tl.sum(x * x, axis=1)

            mean = sum_x / n_cols
            var = sum_x2 / n_cols - mean * mean
            rstd = rsqrt(var + eps)

            if not IS_INFERENCE:
                tl.store(Mean_ptr + rows, mean, mask=row_mask)
                tl.store(RSTD_ptr + rows, rstd, mask=row_mask)

            y = (x - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]

            tl.store(
                Y_ptr + rows[:, None] * stride_y_row + cols[None, :],
                y.to(Y_ptr.dtype.element_ty),
                mask=block_mask,
            )


def _select_fullrow_hoist_config(n_rows, n_cols, dtype):

    if n_rows < 1024:
        return False, None, None, None, None


    if n_cols < 2048:
        return False, None, None, None, None

    if n_cols > 4096:
        return False, None, None, None, None

    align_unit = 8
    block_n = ((n_cols + align_unit - 1) // align_unit) * align_unit


    if block_n <= 2048:
        block_n = 2048
    elif block_n <= 3072:
        block_n = 3072
    elif block_n <= 4096:
        block_n = 4096
    else:
        return False, None, None, None, None


    block_m = 2

    if n_rows % block_m != 0:
        return False, None, None, None, None

    is_exact_n = (block_n == n_cols)
    is_exact_m = (n_rows % block_m == 0)

    return True, block_n, block_m, is_exact_n, is_exact_m

def layernorm_infer_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if weight is None or bias is None:
        raise ValueError("layernorm_infer_impl does not support weight or bias being None. Both must be provided.")

    shape = hidden_states.shape
    dim = shape[-1]
    x_2d = hidden_states.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    num_cores = _get_num_vector_cores()

    y = torch.empty_like(x_2d)
    mean = torch.empty(n_rows, dtype=hidden_states.dtype, device=hidden_states.device)
    rstd = torch.empty(n_rows, dtype=hidden_states.dtype, device=hidden_states.device)


    use_fast, block_n, block_m, is_exact_n, is_exact_m = _select_fullrow_hoist_config(
        n_rows, n_cols, hidden_states.dtype
    )

    if use_fast:
        grid = (num_cores,)

        _layernorm_fwd_fullrow_hoist_kernel[grid](
            x_2d, y, weight, bias, mean, rstd,
            x_2d.stride(0), y.stride(0),
            n_rows, n_cols, eps,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_M=block_m,
            IS_INFERENCE=True,
            IS_EXACT_N=is_exact_n,
            IS_EXACT_M=is_exact_m,
        )

        return y.reshape(*shape)

    BLOCK_SIZE_N, BLOCK_SIZE_M = _calculate_optimal_tiling(n_cols, hidden_states.dtype)

    num_row_tasks = ceil_div(n_rows, BLOCK_SIZE_M)
    grid = (min(num_cores, num_row_tasks),)

    _layernorm_fwd_kernel[grid](
        x_2d, y, weight, bias, mean, rstd,
        x_2d.stride(0), y.stride(0),
        n_rows, n_cols, eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        IS_INFERENCE=True,
    )

    return y.reshape(*shape)

def layernorm_fwd_impl(x, w, b, eps):
    shape = x.shape
    dim = shape[-1]
    x_2d = x.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    num_cores = _get_num_vector_cores()

    y = torch.empty_like(x_2d)
    mean = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    rstd = torch.empty(n_rows, dtype=x.dtype, device=x.device)


    use_fast, block_n, block_m, is_exact_n, is_exact_m = _select_fullrow_hoist_config(
        n_rows, n_cols, x.dtype
    )

    if use_fast:
        grid = (num_cores,)

        _layernorm_fwd_fullrow_hoist_kernel[grid](
            x_2d, y, w, b, mean, rstd,
            x_2d.stride(0), y.stride(0),
            n_rows, n_cols, eps,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_M=block_m,
            IS_INFERENCE=False,
            IS_EXACT_N=is_exact_n,
            IS_EXACT_M=is_exact_m,
        )

        return y.reshape(*shape), x_2d, mean, rstd


    BLOCK_SIZE_N, BLOCK_SIZE_M = _calculate_optimal_tiling(n_cols, x.dtype)

    num_row_tasks = ceil_div(n_rows, BLOCK_SIZE_M)
    grid = (min(num_cores, num_row_tasks),)

    _layernorm_fwd_kernel[grid](
        x_2d, y, w, b, mean, rstd,
        x_2d.stride(0), y.stride(0),
        n_rows, n_cols, eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        IS_INFERENCE=False,
    )

    return y.reshape(*shape), x_2d, mean, rstd
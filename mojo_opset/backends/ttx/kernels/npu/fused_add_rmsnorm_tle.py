from typing import Tuple

import torch
import triton
import triton.experimental.tle as xtle
import triton.language as tl

from .utils import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align
from mojo_opset.backends.ttx.kernels.utils import ceil_div

COL_BLOCKING_THRESHOLD = 2048
SMALL_HIDDEN_DIM_THRESHOLD = 2048
SINGLE_ROW_KERNEL_THRESHOLD = 8192
MAX_PARALLEL_ROW_PROGRAMS = 48
MID_LARGE_HIDDEN_DIM_THRESHOLD = 4096
MID_LARGE_GRID_LIMIT = 29
TINY_ROW_THRESHOLD = 4
TINY_HIDDEN_DIM_THRESHOLD = 512
SMALL_MULTILINE_ROW_THRESHOLD = 8

_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

TOKEN_BLOCK_SIZE_TABLE = {
    2048: 4,
    1024: 6,
    512: 8,
    256: 16,
    128: 20,
}


def rms_norm_fwd_heuristics(args):
    hidden_dim = args["n_cols"]
    if hidden_dim <= COL_BLOCKING_THRESHOLD:
        if hidden_dim in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[hidden_dim]

        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if hidden_dim <= dim_thresh:
                return block_size
        return 1
    else:
        return 4


def _fused_add_rmsnorm_grid(n_rows: int, n_cols: int) -> tuple[int]:
    if n_cols <= SINGLE_ROW_KERNEL_THRESHOLD:
        grid_limit = MAX_PARALLEL_ROW_PROGRAMS
        if n_cols > MID_LARGE_HIDDEN_DIM_THRESHOLD:
            grid_limit = MID_LARGE_GRID_LIMIT
        return (max(1, min(n_rows, grid_limit)),)

    block_size_m = rms_norm_fwd_heuristics({"n_cols": n_cols})
    num_row_tasks = ceil_div(n_rows, block_size_m)
    return (max(1, min(num_row_tasks, MAX_PARALLEL_ROW_PROGRAMS)),)


def _fused_add_rmsnorm_multiline_grid(n_rows: int, n_cols: int) -> tuple[int]:
    block_size_m = rms_norm_fwd_heuristics({"n_cols": n_cols})
    num_row_tasks = ceil_div(n_rows, block_size_m)
    return (max(1, min(num_row_tasks, MAX_PARALLEL_ROW_PROGRAMS)),)


@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_tiny_pre_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    rows_off = tl.arange(0, BLOCK_SIZE_M)
    cols_off = tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows_off < n_rows
    col_mask = cols_off < n_cols
    block_mask = row_mask[:, None] & col_mask[None, :]

    X_block = tl.load(X_ptr + rows_off[:, None] * X_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
    R_block = tl.load(R_ptr + rows_off[:, None] * R_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
    S_block = X_block + R_block
    tl.store(S_ptr + rows_off[:, None] * S_row_stride + cols_off[None, :], S_block, mask=block_mask)

    S_block_f32 = S_block.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(S_block_f32 * S_block_f32, axis=1) / n_cols + eps)
    W_block = tl.load(W_ptr + cols_off, mask=col_mask, other=0.0)
    Y_block = (S_block_f32 * rstd[:, None]).to(S_ptr.dtype.element_ty) * (W_block[None, :] + offset)
    tl.store(Y_ptr + rows_off[:, None] * Y_row_stride + cols_off[None, :], Y_block, mask=block_mask)


@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_tiny_post_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    rows_off = tl.arange(0, BLOCK_SIZE_M)
    cols_off = tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows_off < n_rows
    col_mask = cols_off < n_cols
    block_mask = row_mask[:, None] & col_mask[None, :]

    X_block = tl.load(X_ptr + rows_off[:, None] * X_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
    R_block = tl.load(R_ptr + rows_off[:, None] * R_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
    S_block = X_block + R_block
    S_block_f32 = S_block.to(tl.float32)

    rstd = tl.rsqrt(tl.sum(S_block_f32 * S_block_f32, axis=1) / n_cols + eps)
    W_block = tl.load(W_ptr + cols_off, mask=col_mask, other=0.0)
    Y_block = (S_block_f32 * rstd[:, None]).to(Y_ptr.dtype.element_ty) * (W_block[None, :] + offset)
    tl.store(Y_ptr + rows_off[:, None] * Y_row_stride + cols_off[None, :], Y_block, mask=block_mask)


@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_chunked_pre_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    for row_idx in range(pid, n_rows, grid_size):
        X_ptr_row = X_ptr + row_idx * X_row_stride
        R_ptr_row = R_ptr + row_idx * R_row_stride
        S_ptr_row = S_ptr + row_idx * S_row_stride
        Y_ptr_row = Y_ptr + row_idx * Y_row_stride

        var_acc = 0.0
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols

            X_chunk = tl.load(X_ptr_row + cols_off, mask=cols_mask, other=0.0)
            R_chunk = tl.load(R_ptr_row + cols_off, mask=cols_mask, other=0.0)
            S_chunk = X_chunk + R_chunk
            tl.store(S_ptr_row + cols_off, S_chunk, mask=cols_mask)
            S_chunk_f32 = S_chunk.to(tl.float32)
            var_acc += tl.sum(S_chunk_f32 * S_chunk_f32)

        rstd = tl.rsqrt(var_acc / n_cols + eps)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols

            S_chunk = tl.load(S_ptr_row + cols_off, mask=cols_mask, other=0.0)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            if casting_mode == _CASTING_MODE_GEMMA:
                S_chunk = S_chunk.to(tl.float32)
                W_chunk = W_chunk.to(tl.float32)
            elif casting_mode == _CASTING_MODE_LLAMA:
                S_chunk = S_chunk.to(tl.float32)

            if casting_mode == _CASTING_MODE_LLAMA:
                normed_S_chunk = (S_chunk * rstd).to(S_ptr.dtype.element_ty)
            else:
                normed_S_chunk = S_chunk * rstd

            Y_chunk = normed_S_chunk * (W_chunk + offset)
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_chunk = Y_chunk.to(S_ptr.dtype.element_ty)

            tl.store(Y_ptr_row + cols_off, Y_chunk, mask=cols_mask)


@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_chunked_post_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    for row_idx in range(pid, n_rows, grid_size):
        X_ptr_row = X_ptr + row_idx * X_row_stride
        R_ptr_row = R_ptr + row_idx * R_row_stride
        Y_ptr_row = Y_ptr + row_idx * Y_row_stride

        var_acc = 0.0
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols

            X_chunk = tl.load(X_ptr_row + cols_off, mask=cols_mask, other=0.0)
            R_chunk = tl.load(R_ptr_row + cols_off, mask=cols_mask, other=0.0)
            S_chunk = X_chunk + R_chunk
            S_chunk_f32 = S_chunk.to(tl.float32)
            var_acc += tl.sum(S_chunk_f32 * S_chunk_f32)

        rstd = tl.rsqrt(var_acc / n_cols + eps)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols

            X_chunk = tl.load(X_ptr_row + cols_off, mask=cols_mask, other=0.0)
            R_chunk = tl.load(R_ptr_row + cols_off, mask=cols_mask, other=0.0)
            S_chunk = X_chunk + R_chunk
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            if casting_mode == _CASTING_MODE_GEMMA:
                S_chunk = S_chunk.to(tl.float32)
                W_chunk = W_chunk.to(tl.float32)
            elif casting_mode == _CASTING_MODE_LLAMA:
                S_chunk = S_chunk.to(tl.float32)

            if casting_mode == _CASTING_MODE_LLAMA:
                normed_S_chunk = (S_chunk * rstd).to(Y_ptr.dtype.element_ty)
            else:
                normed_S_chunk = S_chunk * rstd

            Y_chunk = normed_S_chunk * (W_chunk + offset)
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_chunk = Y_chunk.to(Y_ptr.dtype.element_ty)

            tl.store(Y_ptr_row + cols_off, Y_chunk, mask=cols_mask)


@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_small_pre_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    if casting_mode == _CASTING_MODE_GEMMA:
        W_cached = W_chunk.to(tl.float32)
    else:
        W_cached = W_chunk

    for row_idx in xtle.dsa.parallel(pid, n_rows, grid_size):
        X_ptr_row = X_ptr + row_idx * X_row_stride
        R_ptr_row = R_ptr + row_idx * R_row_stride
        S_ptr_row = S_ptr + row_idx * S_row_stride
        Y_ptr_row = Y_ptr + row_idx * Y_row_stride

        X_chunk = tl.load(X_ptr_row + cols_off, mask=cols_mask, other=0.0)
        R_chunk = tl.load(R_ptr_row + cols_off, mask=cols_mask, other=0.0)
        S_chunk = X_chunk + R_chunk
        tl.store(S_ptr_row + cols_off, S_chunk, mask=cols_mask)

        S_chunk_f32 = S_chunk.to(tl.float32)
        var = tl.sum(S_chunk_f32 * S_chunk_f32) / n_cols
        rstd = tl.rsqrt(var + eps)

        if casting_mode == _CASTING_MODE_GEMMA:
            normed_S_chunk = S_chunk_f32 * rstd
            Y_chunk = normed_S_chunk * (W_cached + offset)
            Y_chunk = Y_chunk.to(S_ptr.dtype.element_ty)
        elif casting_mode == _CASTING_MODE_LLAMA:
            normed_S_chunk = (S_chunk_f32 * rstd).to(S_ptr.dtype.element_ty)
            Y_chunk = normed_S_chunk * (W_cached + offset)
        else:
            normed_S_chunk = S_chunk * rstd
            Y_chunk = normed_S_chunk * (W_cached + offset)

        tl.store(Y_ptr_row + cols_off, Y_chunk, mask=cols_mask)


@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_wide_pre_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    # Load the weight vector once per program and reuse it across rows.
    W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    if casting_mode == _CASTING_MODE_GEMMA:
        W_cached = W_chunk.to(tl.float32)
    else:
        W_cached = W_chunk

    for row_idx in xtle.dsa.parallel(pid, n_rows, grid_size):
        X_ptr_row = X_ptr + row_idx * X_row_stride
        R_ptr_row = R_ptr + row_idx * R_row_stride
        S_ptr_row = S_ptr + row_idx * S_row_stride
        Y_ptr_row = Y_ptr + row_idx * Y_row_stride

        X_chunk = tl.load(X_ptr_row + cols_off, mask=cols_mask, other=0.0)
        R_chunk = tl.load(R_ptr_row + cols_off, mask=cols_mask, other=0.0)
        S_chunk = X_chunk + R_chunk
        tl.store(S_ptr_row + cols_off, S_chunk, mask=cols_mask)

        S_chunk_f32 = S_chunk.to(tl.float32)
        var = tl.sum(S_chunk_f32 * S_chunk_f32) / n_cols
        rstd = tl.rsqrt(var + eps)

        if casting_mode == _CASTING_MODE_GEMMA:
            normed_S_chunk = S_chunk_f32 * rstd
            Y_chunk = normed_S_chunk * (W_cached + offset)
            Y_chunk = Y_chunk.to(S_ptr.dtype.element_ty)
        elif casting_mode == _CASTING_MODE_LLAMA:
            normed_S_chunk = (S_chunk_f32 * rstd).to(S_ptr.dtype.element_ty)
            Y_chunk = normed_S_chunk * (W_cached + offset)
        else:
            normed_S_chunk = S_chunk * rstd
            Y_chunk = normed_S_chunk * (W_cached + offset)

        tl.store(Y_ptr_row + cols_off, Y_chunk, mask=cols_mask)


@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_small_post_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    if casting_mode == _CASTING_MODE_GEMMA:
        W_cached = W_chunk.to(tl.float32)
    else:
        W_cached = W_chunk

    for row_idx in xtle.dsa.parallel(pid, n_rows, grid_size):
        X_ptr_row = X_ptr + row_idx * X_row_stride
        R_ptr_row = R_ptr + row_idx * R_row_stride
        Y_ptr_row = Y_ptr + row_idx * Y_row_stride

        X_chunk = tl.load(X_ptr_row + cols_off, mask=cols_mask, other=0.0)
        R_chunk = tl.load(R_ptr_row + cols_off, mask=cols_mask, other=0.0)
        S_chunk = X_chunk + R_chunk

        S_chunk_f32 = S_chunk.to(tl.float32)
        var = tl.sum(S_chunk_f32 * S_chunk_f32) / n_cols
        rstd = tl.rsqrt(var + eps)

        if casting_mode == _CASTING_MODE_GEMMA:
            normed_S_chunk = S_chunk_f32 * rstd
            Y_chunk = normed_S_chunk * (W_cached + offset)
            Y_chunk = Y_chunk.to(Y_ptr.dtype.element_ty)
        elif casting_mode == _CASTING_MODE_LLAMA:
            normed_S_chunk = (S_chunk_f32 * rstd).to(Y_ptr.dtype.element_ty)
            Y_chunk = normed_S_chunk * (W_cached + offset)
        else:
            normed_S_chunk = S_chunk * rstd
            Y_chunk = normed_S_chunk * (W_cached + offset)

        tl.store(Y_ptr_row + cols_off, Y_chunk, mask=cols_mask)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_large_pre_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        X_ptr_row_block = X_ptr + rows_off[:, None] * X_row_stride
        R_ptr_row_block = R_ptr + rows_off[:, None] * R_row_stride
        S_ptr_row_block = S_ptr + rows_off[:, None] * S_row_stride
        Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride

        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            block_mask = rows_mask[:, None] & (cols_off[None, :] < n_cols)

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            R_chunk = tl.load(R_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            S_chunk = X_chunk + R_chunk
            tl.store(S_ptr_row_block + cols_off[None, :], S_chunk, mask=block_mask)

            S_chunk_f32 = S_chunk.to(tl.float32)
            var_acc += tl.sum(S_chunk_f32 * S_chunk_f32, axis=1)

        rstd_vec = tl.rsqrt(var_acc / n_cols + eps)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            if casting_mode == _CASTING_MODE_GEMMA:
                S_chunk = S_chunk.to(tl.float32)
                W_chunk = W_chunk.to(tl.float32)
            elif casting_mode == _CASTING_MODE_LLAMA:
                S_chunk = S_chunk.to(tl.float32)

            if casting_mode == _CASTING_MODE_LLAMA:
                normed_S_chunk = (S_chunk * rstd_vec[:, None]).to(S_ptr.dtype.element_ty)
            else:
                normed_S_chunk = S_chunk * rstd_vec[:, None]

            Y_chunk = normed_S_chunk * (W_chunk[None, :] + offset)

            if casting_mode == _CASTING_MODE_GEMMA:
                Y_chunk = Y_chunk.to(S_ptr.dtype.element_ty)

            tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk, mask=block_mask)


@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_wide_post_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    # Load the weight vector once per program and reuse it across rows.
    W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    if casting_mode == _CASTING_MODE_GEMMA:
        W_cached = W_chunk.to(tl.float32)
    else:
        W_cached = W_chunk

    for row_idx in xtle.dsa.parallel(pid, n_rows, grid_size):
        X_ptr_row = X_ptr + row_idx * X_row_stride
        R_ptr_row = R_ptr + row_idx * R_row_stride
        Y_ptr_row = Y_ptr + row_idx * Y_row_stride

        X_chunk = tl.load(X_ptr_row + cols_off, mask=cols_mask, other=0.0)
        R_chunk = tl.load(R_ptr_row + cols_off, mask=cols_mask, other=0.0)
        S_chunk = X_chunk + R_chunk

        S_chunk_f32 = S_chunk.to(tl.float32)
        var = tl.sum(S_chunk_f32 * S_chunk_f32) / n_cols
        rstd = tl.rsqrt(var + eps)

        if casting_mode == _CASTING_MODE_GEMMA:
            normed_S_chunk = S_chunk_f32 * rstd
            Y_chunk = normed_S_chunk * (W_cached + offset)
            Y_chunk = Y_chunk.to(Y_ptr.dtype.element_ty)
        elif casting_mode == _CASTING_MODE_LLAMA:
            normed_S_chunk = (S_chunk_f32 * rstd).to(Y_ptr.dtype.element_ty)
            Y_chunk = normed_S_chunk * (W_cached + offset)
        else:
            normed_S_chunk = S_chunk * rstd
            Y_chunk = normed_S_chunk * (W_cached + offset)

        tl.store(Y_ptr_row + cols_off, Y_chunk, mask=cols_mask)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_large_post_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        X_ptr_row_block = X_ptr + rows_off[:, None] * X_row_stride
        R_ptr_row_block = R_ptr + rows_off[:, None] * R_row_stride
        Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride

        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            block_mask = rows_mask[:, None] & (cols_off[None, :] < n_cols)

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            R_chunk = tl.load(R_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            S_chunk = X_chunk + R_chunk
            var_acc += tl.sum(S_chunk.to(tl.float32) * S_chunk.to(tl.float32), axis=1)

        rstd_vec = tl.rsqrt(var_acc / n_cols + eps)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            R_chunk = tl.load(R_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            S_chunk = X_chunk + R_chunk
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            if casting_mode == _CASTING_MODE_GEMMA:
                S_chunk = S_chunk.to(tl.float32)
                W_chunk = W_chunk.to(tl.float32)
            elif casting_mode == _CASTING_MODE_LLAMA:
                S_chunk = S_chunk.to(tl.float32)

            if casting_mode == _CASTING_MODE_LLAMA:
                normed_S_chunk = (S_chunk * rstd_vec[:, None]).to(Y_ptr.dtype.element_ty)
            else:
                normed_S_chunk = S_chunk * rstd_vec[:, None]

            Y_chunk = normed_S_chunk * (W_chunk[None, :] + offset)

            if casting_mode == _CASTING_MODE_GEMMA:
                Y_chunk = Y_chunk.to(Y_ptr.dtype.element_ty)

            tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk, mask=block_mask)


@triton.heuristics({"BLOCK_SIZE_M": lambda args: ceil_div(4096, args["n_cols"])})
@libentry()
@triton.jit
def _fused_add_rmsnorm_bwd_kernel(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
    dX_ptr,
    dX_row_stride,
    S_ptr,
    S_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode: tl.constexpr,
    S_dtype: tl.constexpr,
    has_dS_out: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    dW_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    W_row = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    W_row_offset = W_row + offset

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        dY_block = tl.load(dY_ptr + rows_off[:, None] * dY_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
        S_block = tl.load(S_ptr + rows_off[:, None] * S_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
        rstd_vec = tl.load(RSTD_ptr + rows_off * RSTD_row_stride, mask=rows_mask, other=0.0)

        S_block_f32 = S_block.to(tl.float32)
        normed_S_block = S_block_f32 * rstd_vec[:, None]

        if casting_mode == _CASTING_MODE_LLAMA:
            m_block = (dY_block * W_row_offset[None, :]).to(tl.float32)
            dW_acc += tl.sum(dY_block * normed_S_block.to(S_dtype), axis=0)
        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_block_f32 = dY_block.to(tl.float32)
            W_row_offset_f32 = W_row_offset.to(tl.float32)
            m_block = dY_block_f32 * W_row_offset_f32[None, :]
            dW_acc += tl.sum(dY_block_f32 * normed_S_block, axis=0)
        else:
            m_block = dY_block * W_row_offset[None, :]
            dW_acc += tl.sum(dY_block * normed_S_block, axis=0)

        dot_product_vec = tl.sum(m_block * S_block_f32, axis=1)
        rstd_vec_sq = rstd_vec * rstd_vec

        term1 = rstd_vec[:, None] * m_block
        term2 = -(1 / n_cols) * rstd_vec_sq[:, None] * rstd_vec[:, None] * dot_product_vec[:, None] * S_block_f32

        grad_after_norm = term1 + term2

        dS_block = grad_after_norm
        if has_dS_out:
            dS_out_block = tl.load(
                dS_out_ptr + rows_off[:, None] * dS_out_row_stride + cols_off[None, :], mask=block_mask, other=0.0
            )
            dS_block += dS_out_block.to(dS_block.dtype)

        tl.store(dX_ptr + rows_off[:, None] * dX_row_stride + cols_off[None, :], dS_block.to(S_dtype), mask=block_mask)

    dW_ptr_prog = dW_ptr + pid * dW_row_stride + cols_off
    tl.store(dW_ptr_prog, dW_acc, mask=cols_mask)


def fused_add_rmsnorm_infer_impl(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    add_mode: str = "pre",
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = hidden_states.shape
    dim = shape[-1]
    hidden_states_2d = hidden_states.reshape(-1, dim)
    residual_2d = residual.reshape(-1, dim)
    n_rows, n_cols = hidden_states_2d.shape
    use_tiny_multiline_kernel = n_rows <= TINY_ROW_THRESHOLD and n_cols <= TINY_HIDDEN_DIM_THRESHOLD
    use_small_multiline_kernel = (
        not use_tiny_multiline_kernel
        and n_cols <= SMALL_HIDDEN_DIM_THRESHOLD
        and n_rows >= SMALL_MULTILINE_ROW_THRESHOLD
    )
    use_wide_single_row_kernel = (
        not use_tiny_multiline_kernel
        and not use_small_multiline_kernel
        and MID_LARGE_HIDDEN_DIM_THRESHOLD < n_cols <= SINGLE_ROW_KERNEL_THRESHOLD
    )
    use_single_row_kernel = (
        n_cols <= SINGLE_ROW_KERNEL_THRESHOLD
        and not use_tiny_multiline_kernel
        and not use_small_multiline_kernel
        and not use_wide_single_row_kernel
    )
    if use_small_multiline_kernel:
        BLOCK_SIZE_N = align(hidden_states, n_cols, VEC_ALIGN_BYTES)
    elif use_wide_single_row_kernel:
        BLOCK_SIZE_N = min(triton.next_power_of_2(n_cols), SINGLE_ROW_KERNEL_THRESHOLD)
    elif use_single_row_kernel:
        BLOCK_SIZE_N = min(triton.next_power_of_2(n_cols), SINGLE_ROW_KERNEL_THRESHOLD)
    elif use_tiny_multiline_kernel:
        BLOCK_SIZE_N = align(hidden_states, n_cols, VEC_ALIGN_BYTES)
    elif n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = COL_BLOCKING_THRESHOLD
    else:
        BLOCK_SIZE_N = align(hidden_states, n_cols, VEC_ALIGN_BYTES)

    if use_small_multiline_kernel:
        grid = _fused_add_rmsnorm_multiline_grid(n_rows, n_cols)
    else:
        grid = _fused_add_rmsnorm_grid(n_rows, n_cols)

    str_to_casting_mode = {"llama": 0, "gemma": 1, "none": -1}
    _casting_mode = str_to_casting_mode[casting_mode]

    Y = torch.empty_like(hidden_states_2d)
    if add_mode == "pre":
        S = torch.empty_like(hidden_states_2d)
        if use_tiny_multiline_kernel:
            _fused_add_rmsnorm_fwd_tiny_pre_kernel[(1,)](
                Y,
                Y.stride(0),
                S,
                S.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                BLOCK_SIZE_M=max(n_rows, 1),
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        elif use_small_multiline_kernel:
            _fused_add_rmsnorm_fwd_small_pre_kernel[grid](
                Y,
                Y.stride(0),
                S,
                S.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode=_casting_mode,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        elif use_wide_single_row_kernel:
            _fused_add_rmsnorm_fwd_wide_pre_kernel[grid](
                Y,
                Y.stride(0),
                S,
                S.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode=_casting_mode,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        elif use_single_row_kernel:
            _fused_add_rmsnorm_fwd_small_pre_kernel[grid](
                Y,
                Y.stride(0),
                S,
                S.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode=_casting_mode,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        else:
            _fused_add_rmsnorm_fwd_large_pre_kernel[grid](
                Y,
                Y.stride(0),
                S,
                S.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode=_casting_mode,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        return Y.reshape(*shape), S.reshape(*shape)

    if add_mode == "post":
        if use_tiny_multiline_kernel:
            _fused_add_rmsnorm_fwd_tiny_post_kernel[(1,)](
                Y,
                Y.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                BLOCK_SIZE_M=max(n_rows, 1),
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        elif use_small_multiline_kernel:
            _fused_add_rmsnorm_fwd_small_post_kernel[grid](
                Y,
                Y.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode=_casting_mode,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        elif use_wide_single_row_kernel:
            _fused_add_rmsnorm_fwd_wide_post_kernel[grid](
                Y,
                Y.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode=_casting_mode,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        elif use_single_row_kernel:
            _fused_add_rmsnorm_fwd_small_post_kernel[grid](
                Y,
                Y.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode=_casting_mode,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        else:
            _fused_add_rmsnorm_fwd_large_post_kernel[grid](
                Y,
                Y.stride(0),
                hidden_states_2d,
                hidden_states_2d.stride(0),
                residual_2d,
                residual_2d.stride(0),
                weight,
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode=_casting_mode,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        return Y.reshape(*shape), Y.reshape(*shape)

    raise ValueError(f"Invalid add_mode: {add_mode}. Must be 'pre' or 'post'.")


_fused_add_rmsnorm_fwd_kernel = _fused_add_rmsnorm_fwd_large_pre_kernel

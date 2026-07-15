import os

import torch
import torch.nn.functional as F
import triton
import triton.experimental.tle as xtle
import triton.language as tl

from .utils import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align

COL_BLOCKING_THRESHOLD = 2048
DEFAULT_TLE_SINGLE_PASS_MAX_COLS = 8192
DEFAULT_TLE_CHUNK_TILE_SIZE = COL_BLOCKING_THRESHOLD
DEFAULT_TLE_CHUNK_PIPELINE_STAGES = 2
MAX_PARALLEL_ROW_PROGRAMS = 48
LARGE_BLOCK_ROW_THRESHOLD = 64
MAX_SINGLE_PASS_ROW_BLOCK_SIZE_M = 8

TOKEN_BLOCK_SIZE_TABLE = {
    2048: 4,
    1024: 6,
    512: 8,
    256: 16,
    128: 20,
}


def _tle_single_pass_max_cols() -> int:
    value = os.environ.get("MOJO_RMSNORM_TLE_MAX_COLS", "").strip()
    if not value:
        return DEFAULT_TLE_SINGLE_PASS_MAX_COLS
    return int(value)


def _select_chunk_tile_size(n_rows: int, n_cols: int) -> int:
    env_value = os.environ.get("MOJO_RMSNORM_TLE_TILE_SIZE", "").strip()
    if env_value:
        return int(env_value)
    if n_cols > DEFAULT_TLE_SINGLE_PASS_MAX_COLS and n_rows < LARGE_BLOCK_ROW_THRESHOLD:
        if n_rows <= 16:
            return 4096
        return DEFAULT_TLE_SINGLE_PASS_MAX_COLS
    return min(triton.next_power_of_2(n_cols), DEFAULT_TLE_CHUNK_TILE_SIZE)


def _select_chunk_pipeline_stages(n_rows: int, n_cols: int) -> int:
    env_value = os.environ.get("MOJO_RMSNORM_TLE_PIPELINE_STAGES", "").strip()
    if env_value:
        return int(env_value)
    if n_cols > DEFAULT_TLE_SINGLE_PASS_MAX_COLS and n_rows <= 16:
        return 1
    return DEFAULT_TLE_CHUNK_PIPELINE_STAGES


def rms_norm_fwd_heuristics(args):
    env_value = os.environ.get("MOJO_RMSNORM_TLE_LARGE_BLOCK_SIZE_M", "").strip()
    if env_value:
        return int(env_value)
    hidden_dim = args["n_cols"]
    if hidden_dim <= COL_BLOCKING_THRESHOLD:
        if hidden_dim in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[hidden_dim]
        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if hidden_dim <= dim_thresh:
                return block_size
        return 1
    return 4


def single_pass_row_block_heuristics(args):
    env_value = os.environ.get("MOJO_RMSNORM_TLE_SINGLE_PASS_BLOCK_SIZE_M", "").strip()
    if env_value:
        return int(env_value)
    return min(rms_norm_fwd_heuristics(args), MAX_SINGLE_PASS_ROW_BLOCK_SIZE_M)


def _rmsnorm_large_grid(n_rows: int, n_cols: int) -> tuple[int]:
    if n_cols <= DEFAULT_TLE_SINGLE_PASS_MAX_COLS:
        return (max(1, min(n_rows, MAX_PARALLEL_ROW_PROGRAMS)),)
    block_size_m = rms_norm_fwd_heuristics({"n_cols": n_cols})
    num_row_tasks = (n_rows + block_size_m - 1) // block_size_m
    return (max(1, min(num_row_tasks, MAX_PARALLEL_ROW_PROGRAMS)),)


def _rmsnorm_single_pass_grid(n_rows: int) -> tuple[int]:
    num_vectorcore = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    return (max(1, min(num_vectorcore, n_rows)),)


def _rmsnorm_single_pass_row_block_grid(n_rows: int, n_cols: int) -> tuple[int]:
    num_vectorcore = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    block_size_m = single_pass_row_block_heuristics({"n_cols": n_cols})
    num_row_tasks = (n_rows + block_size_m - 1) // block_size_m
    return (max(1, min(num_vectorcore, num_row_tasks)),)


def _rmsnorm_single_pass_block_size_n(x: torch.Tensor, n_cols: int, single_pass_max_cols: int) -> int:
    if n_cols > COL_BLOCKING_THRESHOLD:
        return single_pass_max_cols
    return align(x, n_cols, VEC_ALIGN_BYTES)


def _should_use_single_pass_row_block(n_rows: int, n_cols: int) -> bool:
    return n_rows > 1 and n_cols <= COL_BLOCKING_THRESHOLD


def _rmsnorm_chunked_grid(n_rows: int) -> tuple[int]:
    num_vectorcore = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    return (max(1, min(num_vectorcore, n_rows)),)


def _should_use_large_block_kernel(n_rows: int, n_cols: int) -> bool:
    if n_cols <= DEFAULT_TLE_SINGLE_PASS_MAX_COLS:
        return False
    return n_rows >= LARGE_BLOCK_ROW_THRESHOLD


@libentry()
@triton.jit
def _rmsnorm_infer_tle_f32_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < n_cols
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for row_idx in range(pid, n_rows, grid_size):
        x_row_ptr = X_ptr + row_idx * stride_x_row + cols
        y_row_ptr = Y_ptr + row_idx * stride_y_row + cols

        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0).to(tl.float32)

        ss_acc = tl.sum(x_vals * x_vals, axis=0)
        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        y = x_vals * rrms * w
        tl.store(y_row_ptr, y, mask=mask)


@triton.heuristics({"BLOCK_SIZE_M": single_pass_row_block_heuristics})
@libentry()
@triton.jit
def _rmsnorm_infer_tle_row_block_f32_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < n_cols
    w = tl.load(W_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        x_row_block = X_ptr + rows[:, None] * stride_x_row
        y_row_block = Y_ptr + rows[:, None] * stride_y_row

        x = tl.load(x_row_block + cols[None, :], mask=block_mask, other=0.0).to(tl.float32)
        ss_acc = tl.sum(x * x, axis=1)
        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        y = x * rrms[:, None] * w[None, :]
        tl.store(y_row_block + cols[None, :], y, mask=block_mask)


@libentry()
@triton.jit
def _rmsnorm_infer_tle_f16_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < n_cols
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for row_idx in range(pid, n_rows, grid_size):
        x_row_ptr = X_ptr + row_idx * stride_x_row + cols
        y_row_ptr = Y_ptr + row_idx * stride_y_row + cols

        x = tl.load(x_row_ptr, mask=mask, other=0.0)
        x_vals = x.to(tl.float32)

        ss_acc = tl.sum(x_vals * x_vals, axis=0)
        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        y = x_vals * rrms * w
        tl.store(y_row_ptr, y.to(tl.float16), mask=mask)


@triton.heuristics({"BLOCK_SIZE_M": single_pass_row_block_heuristics})
@libentry()
@triton.jit
def _rmsnorm_infer_tle_row_block_f16_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < n_cols
    w = tl.load(W_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        x_row_block = X_ptr + rows[:, None] * stride_x_row
        y_row_block = Y_ptr + rows[:, None] * stride_y_row

        x = tl.load(x_row_block + cols[None, :], mask=block_mask, other=0.0)
        x_vals = x.to(tl.float32)
        ss_acc = tl.sum(x_vals * x_vals, axis=1)
        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        y = x_vals * rrms[:, None] * w[None, :]
        tl.store(y_row_block + cols[None, :], y.to(tl.float16), mask=block_mask)


@libentry()
@triton.jit
def _rmsnorm_infer_tle_bf16_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < n_cols
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for row_idx in range(pid, n_rows, grid_size):
        x_row_ptr = X_ptr + row_idx * stride_x_row + cols
        y_row_ptr = Y_ptr + row_idx * stride_y_row + cols

        x = tl.load(x_row_ptr, mask=mask, other=0.0)
        x_vals = x.to(tl.float32)

        ss_acc = tl.sum(x_vals * x_vals, axis=0)
        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        y = x_vals * rrms * w
        tl.store(y_row_ptr, y.to(tl.bfloat16), mask=mask)


@triton.heuristics({"BLOCK_SIZE_M": single_pass_row_block_heuristics})
@libentry()
@triton.jit
def _rmsnorm_infer_tle_row_block_bf16_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < n_cols
    w = tl.load(W_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        x_row_block = X_ptr + rows[:, None] * stride_x_row
        y_row_block = Y_ptr + rows[:, None] * stride_y_row

        x = tl.load(x_row_block + cols[None, :], mask=block_mask, other=0.0)
        x_vals = x.to(tl.float32)
        ss_acc = tl.sum(x_vals * x_vals, axis=1)
        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        y = x_vals * rrms[:, None] * w[None, :]
        tl.store(y_row_block + cols[None, :], y.to(tl.bfloat16), mask=block_mask)


@libentry()
@triton.jit
def _rmsnorm_infer_tle_chunked_f32_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_TILE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    tile_offsets = tl.arange(0, BLOCK_SIZE_TILE)

    for row_idx in range(pid, n_rows, grid_size):
        x_row_ptr = X_ptr + row_idx * stride_x_row
        y_row_ptr = Y_ptr + row_idx * stride_y_row

        ss_acc = tl.zeros((1,), dtype=tl.float32)
        for col_offset in xtle.dsa.pipeline(0, n_cols, BLOCK_SIZE_TILE, num_stages=PIPELINE_STAGES):
            cols = col_offset + tile_offsets
            mask = cols < n_cols
            x_vals = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            ss_acc += tl.sum(tl.where(mask, x_vals * x_vals, 0.0), axis=0)

        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        for col_offset in xtle.dsa.pipeline(0, n_cols, BLOCK_SIZE_TILE, num_stages=PIPELINE_STAGES):
            cols = col_offset + tile_offsets
            mask = cols < n_cols
            x_vals = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = x_vals * rrms * w
            tl.store(y_row_ptr + cols, y, mask=mask)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@libentry()
@triton.jit
def _rmsnorm_infer_tle_large_f32_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < n_rows
        x_row_block = X_ptr + rows[:, None] * stride_x_row
        y_row_block = Y_ptr + rows[:, None] * stride_y_row

        ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = cols < n_cols
            block_mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(x_row_block + cols[None, :], mask=block_mask, other=0.0).to(tl.float32)
            ss_acc += tl.sum(x * x, axis=1)

        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = cols < n_cols
            block_mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(x_row_block + cols[None, :], mask=block_mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            y = x * rrms[:, None] * w[None, :]
            tl.store(y_row_block + cols[None, :], y, mask=block_mask)


@libentry()
@triton.jit
def _rmsnorm_infer_tle_chunked_f16_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_TILE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    tile_offsets = tl.arange(0, BLOCK_SIZE_TILE)

    for row_idx in range(pid, n_rows, grid_size):
        x_row_ptr = X_ptr + row_idx * stride_x_row
        y_row_ptr = Y_ptr + row_idx * stride_y_row

        ss_acc = tl.zeros((1,), dtype=tl.float32)
        for col_offset in xtle.dsa.pipeline(0, n_cols, BLOCK_SIZE_TILE, num_stages=PIPELINE_STAGES):
            cols = col_offset + tile_offsets
            mask = cols < n_cols
            x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
            x_vals = x.to(tl.float32)
            ss_acc += tl.sum(tl.where(mask, x_vals * x_vals, 0.0), axis=0)

        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        for col_offset in xtle.dsa.pipeline(0, n_cols, BLOCK_SIZE_TILE, num_stages=PIPELINE_STAGES):
            cols = col_offset + tile_offsets
            mask = cols < n_cols
            x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
            x_vals = x.to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = x_vals * rrms * w
            tl.store(y_row_ptr + cols, y.to(tl.float16), mask=mask)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@libentry()
@triton.jit
def _rmsnorm_infer_tle_large_f16_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < n_rows
        x_row_block = X_ptr + rows[:, None] * stride_x_row
        y_row_block = Y_ptr + rows[:, None] * stride_y_row

        ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = cols < n_cols
            block_mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(x_row_block + cols[None, :], mask=block_mask, other=0.0)
            x_vals = x.to(tl.float32)
            ss_acc += tl.sum(x_vals * x_vals, axis=1)

        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = cols < n_cols
            block_mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(x_row_block + cols[None, :], mask=block_mask, other=0.0)
            x_vals = x.to(tl.float32)
            w = tl.load(W_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            y = x_vals * rrms[:, None] * w[None, :]
            tl.store(y_row_block + cols[None, :], y.to(tl.float16), mask=block_mask)


@libentry()
@triton.jit
def _rmsnorm_infer_tle_chunked_bf16_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_TILE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    tile_offsets = tl.arange(0, BLOCK_SIZE_TILE)

    for row_idx in range(pid, n_rows, grid_size):
        x_row_ptr = X_ptr + row_idx * stride_x_row
        y_row_ptr = Y_ptr + row_idx * stride_y_row

        ss_acc = tl.zeros((1,), dtype=tl.float32)
        for col_offset in xtle.dsa.pipeline(0, n_cols, BLOCK_SIZE_TILE, num_stages=PIPELINE_STAGES):
            cols = col_offset + tile_offsets
            mask = cols < n_cols
            x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
            x_vals = x.to(tl.float32)
            ss_acc += tl.sum(tl.where(mask, x_vals * x_vals, 0.0), axis=0)

        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        for col_offset in xtle.dsa.pipeline(0, n_cols, BLOCK_SIZE_TILE, num_stages=PIPELINE_STAGES):
            cols = col_offset + tile_offsets
            mask = cols < n_cols
            x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
            x_vals = x.to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = x_vals * rrms * w
            tl.store(y_row_ptr + cols, y.to(tl.bfloat16), mask=mask)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@libentry()
@triton.jit
def _rmsnorm_infer_tle_large_bf16_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < n_rows
        x_row_block = X_ptr + rows[:, None] * stride_x_row
        y_row_block = Y_ptr + rows[:, None] * stride_y_row

        ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = cols < n_cols
            block_mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(x_row_block + cols[None, :], mask=block_mask, other=0.0)
            x_vals = x.to(tl.float32)
            ss_acc += tl.sum(x_vals * x_vals, axis=1)

        rrms = tl.rsqrt(ss_acc / n_cols + eps)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = cols < n_cols
            block_mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(x_row_block + cols[None, :], mask=block_mask, other=0.0)
            x_vals = x.to(tl.float32)
            w = tl.load(W_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            y = x_vals * rrms[:, None] * w[None, :]
            tl.store(y_row_block + cols[None, :], y.to(tl.bfloat16), mask=block_mask)


def _supports_tle_rmsnorm(x: torch.Tensor, w: torch.Tensor) -> bool:
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    return (
        x.device.type == "npu"
        and w.device.type == "npu"
        and x.device == w.device
        and x.dtype in supported_dtypes
        and w.dtype in supported_dtypes
        and x.is_contiguous()
        and w.is_contiguous()
    )


def _launch_single_pass_kernel(
    x_2d: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    n_rows: int,
    n_cols: int,
    single_pass_max_cols: int,
) -> None:
    block_size_n = _rmsnorm_single_pass_block_size_n(x_2d, n_cols, single_pass_max_cols)

    if _should_use_single_pass_row_block(n_rows, n_cols):
        grid = _rmsnorm_single_pass_row_block_grid(n_rows, n_cols)
        if x_2d.dtype == torch.float32:
            _rmsnorm_infer_tle_row_block_f32_kernel[grid](
                x_2d,
                y,
                w,
                x_2d.stride(0),
                y.stride(0),
                n_rows=n_rows,
                n_cols=n_cols,
                eps=eps,
                BLOCK_SIZE_N=block_size_n,
            )
        elif x_2d.dtype == torch.float16:
            _rmsnorm_infer_tle_row_block_f16_kernel[grid](
                x_2d,
                y,
                w,
                x_2d.stride(0),
                y.stride(0),
                n_rows=n_rows,
                n_cols=n_cols,
                eps=eps,
                BLOCK_SIZE_N=block_size_n,
            )
        else:
            _rmsnorm_infer_tle_row_block_bf16_kernel[grid](
                x_2d,
                y,
                w,
                x_2d.stride(0),
                y.stride(0),
                n_rows=n_rows,
                n_cols=n_cols,
                eps=eps,
                BLOCK_SIZE_N=block_size_n,
            )
    else:
        grid = _rmsnorm_single_pass_grid(n_rows)
        if x_2d.dtype == torch.float32:
            _rmsnorm_infer_tle_f32_kernel[grid](
                x_2d,
                y,
                w,
                x_2d.stride(0),
                y.stride(0),
                n_rows=n_rows,
                n_cols=n_cols,
                eps=eps,
                BLOCK_SIZE_N=block_size_n,
            )
        elif x_2d.dtype == torch.float16:
            _rmsnorm_infer_tle_f16_kernel[grid](
                x_2d,
                y,
                w,
                x_2d.stride(0),
                y.stride(0),
                n_rows=n_rows,
                n_cols=n_cols,
                eps=eps,
                BLOCK_SIZE_N=block_size_n,
            )
        else:
            _rmsnorm_infer_tle_bf16_kernel[grid](
                x_2d,
                y,
                w,
                x_2d.stride(0),
                y.stride(0),
                n_rows=n_rows,
                n_cols=n_cols,
                eps=eps,
                BLOCK_SIZE_N=block_size_n,
            )


def _launch_large_block_kernel(
    x_2d: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    n_rows: int,
    n_cols: int,
    block_size_n: int,
) -> None:
    grid = _rmsnorm_large_grid(n_rows, n_cols)

    if x_2d.dtype == torch.float32:
        _rmsnorm_infer_tle_large_f32_kernel[grid](
            x_2d,
            y,
            w,
            x_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE_N=block_size_n,
        )
    elif x_2d.dtype == torch.float16:
        _rmsnorm_infer_tle_large_f16_kernel[grid](
            x_2d,
            y,
            w,
            x_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE_N=block_size_n,
        )
    else:
        _rmsnorm_infer_tle_large_bf16_kernel[grid](
            x_2d,
            y,
            w,
            x_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE_N=block_size_n,
        )


def _launch_chunked_kernel(
    x_2d: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    n_rows: int,
    n_cols: int,
    block_size_n: int,
    pipeline_stages: int,
) -> None:
    grid = _rmsnorm_chunked_grid(n_rows)

    if x_2d.dtype == torch.float32:
        _rmsnorm_infer_tle_chunked_f32_kernel[grid](
            x_2d,
            y,
            w,
            x_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE_TILE=block_size_n,
            PIPELINE_STAGES=pipeline_stages,
        )
    elif x_2d.dtype == torch.float16:
        _rmsnorm_infer_tle_chunked_f16_kernel[grid](
            x_2d,
            y,
            w,
            x_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE_TILE=block_size_n,
            PIPELINE_STAGES=pipeline_stages,
        )
    else:
        _rmsnorm_infer_tle_chunked_bf16_kernel[grid](
            x_2d,
            y,
            w,
            x_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE_TILE=block_size_n,
            PIPELINE_STAGES=pipeline_stages,
        )


def rmsnorm_infer_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert x.size(-1) == w.size(-1)
    shape = x.shape
    dim = shape[-1]
    x_2d = x.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    if not _supports_tle_rmsnorm(x, w):
        return F.rms_norm(x, [x.shape[-1]], weight=w, eps=eps)

    y = torch.empty_like(x_2d)
    tle_single_pass_max_cols = _tle_single_pass_max_cols()
    if n_cols <= tle_single_pass_max_cols:
        _launch_single_pass_kernel(x_2d, y, w, eps, n_rows, n_cols, tle_single_pass_max_cols)
    else:
        block_size_n = _select_chunk_tile_size(n_rows, n_cols)
        if _should_use_large_block_kernel(n_rows, n_cols):
            _launch_large_block_kernel(x_2d, y, w, eps, n_rows, n_cols, block_size_n)
        else:
            pipeline_stages = _select_chunk_pipeline_stages(n_rows, n_cols)
            _launch_chunked_kernel(x_2d, y, w, eps, n_rows, n_cols, block_size_n, pipeline_stages)

    return y.reshape(*shape)

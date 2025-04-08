import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device("cuda:0")


@triton.jit
def element_wise_multi_kernel(x_ptr, y_ptr, output_ptr, row_stride, n_cols, n_rows, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs
    for row_id in tl.range(row_start, n_rows, row_step):
        x_row_start_idx = x_ptr + row_id * row_stride
        y_row_start_idx = y_ptr + row_id * row_stride
        col_range = tl.arange(0, BLOCK_SIZE)
        mask = col_range < n_cols
        x_row_block = x_row_start_idx + col_range
        y_row_block = y_row_start_idx + col_range
        tl.load(x_row_block, mask=mask, other=0)
        tl.load(y_row_block, mask=mask, other=0)
        output = x_row_block * y_row_block
        output_idx = output_ptr + row_id * row_stride
        output_block = output_idx + col_range
        tl.store(output_block, output, mask=mask)

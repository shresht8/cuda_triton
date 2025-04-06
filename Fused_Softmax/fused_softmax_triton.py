import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device("cuda:0")
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_idx = input_ptr + row_idx * input_row_stride
        col_range = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_idx + col_range
        mask = col_range < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row,axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator,axis=0)
        softmax_output = numerator / denominator
        output_start_idx = output_ptr + row_idx * output_row_stride
        output_ptrs = output_start_idx + col_range
        tl.store(output_ptrs, softmax_output, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape
    num_warps = 8
    y = torch.empty_like(x)
    num_stages = 2
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1,))
    kernel._init_handles()
    n_regs_used = kernel.n_regs
    smem_used = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs_used * num_warps * WARP_SIZE)
    occupancy = min(occupancy, SIZE_SMEM // smem_used)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)
    print(num_programs)
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols)
    return y

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device("cuda:0")
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}
@triton.jit
def relu_kernel(x_ptr, output_ptr, num_rows, num_cols, row_stride, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_id in tl.range(row_start, num_rows, row_step):
        row_ptr = x_ptr + row_stride*row_id
        output_row_ptr = output_ptr + row_stride*row_id
        col_range = tl.arange(0,BLOCK_SIZE)
        mask=col_range<num_cols
        row_vals = tl.load(row_ptr+col_range,mask=mask)
        relu_row = tl.where(row_vals>0,row_vals,0)
        tl.store(output_row_ptr+col_range,relu_row,mask=mask)


def relu(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps= 8
    kernel=relu_kernel.warmup(x,output,n_rows,n_cols,x.stride(0),BLOCK_SIZE,grid=(1,))
    kernel._init_handles()
    n_regs_est=kernel.n_regs
    sm_est=kernel.metadata.shared
    print(sm_est)
    occupancy=NUM_REGS//(WARP_SIZE*num_warps*n_regs_est)
    occupancy=min(occupancy,SIZE_SMEM//sm_est) if sm_est >0 else occupancy
    print(occupancy)
    num_progs = NUM_SM*occupancy
    grid=(num_progs,1,1)
    kernel[grid](x,output,n_rows,n_cols,x.stride(0))
    return output

if __name__ == "__main__":
    # Set a seed for reproducibility
    torch.manual_seed(0)

    # Define matrix dimensions
    rows, cols = 128, 256

    # Create random matrix with both positive and negative values
    x = torch.randn((rows, cols), device=DEVICE)

    # Compute ReLU using PyTorch (for reference)
    expected_output = torch.relu(x)

    # Compute ReLU using your Triton kernel
    triton_output = relu(x)

    # Check if the results match
    max_diff = torch.max(torch.abs(expected_output - triton_output))
    print(f"Max difference between PyTorch and Triton implementations: {max_diff}")

    # Test with different sized matrices
    print("\nTesting with different matrix sizes:")
    for size in [(32, 32), (512, 128), (1024, 1024), (2048, 512)]:
        rows, cols = size
        x = torch.randn((rows, cols), device=DEVICE)

        expected_output = torch.relu(x)
        triton_output = relu(x)

        max_diff = torch.max(torch.abs(expected_output - triton_output))
        print(f"Size {size}: Max difference = {max_diff}")








import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device("cuda:0")


@triton.jit
def element_wise_multi_kernel(x_ptr, y_ptr, output_ptr, row_stride, n_cols, n_rows, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_id in tl.range(row_start, n_rows, row_step):
        x_row_start_idx = x_ptr + row_id * row_stride
        y_row_start_idx = y_ptr + row_id * row_stride
        col_range = tl.arange(0, BLOCK_SIZE)
        mask = col_range < n_cols
        x_row_block = x_row_start_idx + col_range
        y_row_block = y_row_start_idx + col_range
        x_values = tl.load(x_row_block, mask=mask)
        y_values = tl.load(y_row_block, mask=mask)
        output = x_values*y_values
        output_idx = output_ptr + row_id * row_stride
        output_block = output_idx + col_range
        tl.store(output_block, output, mask=mask)

def element_wise_mutli(x,y):
    assert x.shape == y.shape, "Input matrices must have the same shape"
    assert x.device == DEVICE and y.device == DEVICE, "Inputs must be on GPU"
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    num_programs = n_rows
    element_wise_multi_kernel[(num_programs,)](x,y,output,x.stride(0),n_cols,n_rows,BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    # Set a seed for reproducibility
    torch.manual_seed(0)

    # Define matrix dimensions
    rows, cols = 128, 256

    # Create random matrices on the GPU
    x = torch.rand((rows, cols), device=DEVICE)
    y = torch.rand((rows, cols), device=DEVICE)

    # Compute element-wise multiplication using PyTorch (for reference)
    expected_output = x * y

    # Compute element-wise multiplication using your Triton kernel
    triton_output = element_wise_mutli(x, y)

    # Check if the results match
    max_diff = torch.max(torch.abs(expected_output - triton_output))
    print(f"Max difference between PyTorch and Triton implementations: {max_diff}")

    # Test with different sized matrices
    print("\nTesting with different matrix sizes:")
    for size in [(32, 32), (512, 128), (1024, 1024), (2048, 512)]:
        rows, cols = size
        x = torch.rand((rows, cols), device=DEVICE)
        y = torch.rand((rows, cols), device=DEVICE)

        expected_output = x * y
        triton_output = element_wise_mutli(x, y)

        max_diff = torch.max(torch.abs(expected_output - triton_output))
        print(f"Size {size}: Max difference = {max_diff}")



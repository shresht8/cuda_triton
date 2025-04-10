import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device("cuda:0")

@triton.jit
def element_wise_mul_kernel(x_ptr, y_ptr, output_ptr, num_rows, num_cols, row_stride,
                            BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    for row_id in tl.range(row_start, num_rows, row_step):
        x_row_ptr = x_ptr + row_id*row_stride
        y_row_ptr = y_ptr + row_id*row_stride
        output_row_ptr = output_ptr + row_id*row_stride
        col_start = BLOCK_SIZE*col_pid
        col_range = col_start+tl.arange(0,BLOCK_SIZE)
        mask = col_range < num_cols
        x_values = tl.load(x_row_ptr+col_range,mask=mask)
        y_values = tl.load(y_row_ptr+col_range,mask=mask)
        output_vals = x_values*y_values
        tl.store(output_row_ptr + col_range, output_vals, mask=mask)


def element_wise_matrix_mul(x, y, split=None):
    if split is None:
        split = {'row_split': 2, 'col_split': 2}
    assert x.shape == y.shape, "Input matrices must have the same shape"
    assert x.device == DEVICE and y.device == DEVICE, "Inputs must be on GPU"
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_rows,meta['row_split']),triton.cdiv(n_cols,meta['col_split']))
    element_wise_mul_kernel[grid(split)](x,y,output,n_rows,n_cols,x.stride(0),split['col_split']//2)
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
    triton_output = element_wise_matrix_mul(x, y)

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
        triton_output = element_wise_matrix_mul(x, y)

        max_diff = torch.max(torch.abs(expected_output - triton_output))
        print(f"Size {size}: Max difference = {max_diff}")




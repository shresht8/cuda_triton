import torch
import triton
import triton.language as tl
import time
from element_wise_rowcol_splt import element_wise_mul_kernel
from element_wise_multi_triton import element_wise_multi_kernel
DEVICE = torch.device("cuda:0")

# Your first implementation
def element_wise_mutli_v1(x, y):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    num_programs = n_rows
    element_wise_multi_kernel[(num_programs,)](
        x, y, output, x.stride(0), n_cols, n_rows, BLOCK_SIZE=min(1024, n_cols)
    )
    return output


# Your second implementation
def element_wise_matrix_mul_v2(x, y, split=None):
    if split is None:
        split = {'row_split': 2, 'col_split': 2}
    assert x.shape == y.shape, "Input matrices must have the same shape"
    assert x.device == DEVICE and y.device == DEVICE, "Inputs must be on GPU"
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_rows, meta['row_split']), triton.cdiv(n_cols, meta['col_split']))
    element_wise_mul_kernel[grid(split)](x, y, output, n_rows, n_cols, x.stride(0), split['col_split'] // 2)
    return output


# PyTorch implementation
def element_wise_mul_pytorch(x, y):
    return x * y


def benchmark(sizes, implementations, num_runs=10):
    results = {}

    for size in sizes:
        rows, cols = size
        x = torch.rand((rows, cols), device=DEVICE)
        y = torch.rand((rows, cols), device=DEVICE)

        print(f"\nBenchmarking matrix size: {size}")
        size_results = {}

        for name, func in implementations.items():
            # Warmup
            for _ in range(3):
                _ = func(x, y)

            torch.cuda.synchronize()

            # Timing
            start_time = time.time()
            for _ in range(num_runs):
                output = func(x, y)
                torch.cuda.synchronize()
            end_time = time.time()

            avg_time = (end_time - start_time) / num_runs
            size_results[name] = avg_time
            print(f"{name}: {avg_time * 1000:.4f} ms")

            # Verify correctness
            expected = x * y
            max_diff = torch.max(torch.abs(expected - output))
            print(f"  Max difference: {max_diff}")

        results[size] = size_results

    return results


if __name__ == "__main__":
    sizes = [(32, 32), (128, 128), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]

    implementations = {
        "PyTorch": element_wise_mul_pytorch,
        "Triton V1 (Row-only)": element_wise_mutli_v1,
        "Triton V2 (2D grid)": element_wise_matrix_mul_v2
    }

    results = benchmark(sizes, implementations)

    # Print speedup summary
    print("\nSpeedup relative to PyTorch:")
    for size in sizes:
        pytorch_time = results[size]["PyTorch"]
        print(f"\nMatrix size: {size}")
        for name, time in results[size].items():
            if name != "PyTorch":
                speedup = pytorch_time / time
                print(f"{name}: {speedup:.2f}x")

    # Test different grid configurations for V2
    print("\nTesting different grid configurations for V2:")
    size = (2048, 2048)  # Choose a large size for testing
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    configs = [
        {'row_split': 1, 'col_split': 32},
        {'row_split': 32, 'col_split': 1},
        {'row_split': 2, 'col_split': 16},
        {'row_split': 16, 'col_split': 2},
        {'row_split': 4, 'col_split': 8},
        {'row_split': 8, 'col_split': 4},
    ]

    for config in configs:
        # Warmup
        for _ in range(3):
            _ = element_wise_matrix_mul_v2(x, y, config)

        torch.cuda.synchronize()

        # Timing
        start_time = time.time()
        for _ in range(10):
            output = element_wise_matrix_mul_v2(x, y, config)
            torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"Config {config}: {avg_time * 1000:.4f} ms")
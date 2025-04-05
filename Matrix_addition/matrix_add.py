import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid=tl.program_id(0)
    offset = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    output = x + y
    tl.store(output_ptr + offset, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output=torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements=output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE = 1024)
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')




    

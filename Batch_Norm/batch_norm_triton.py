import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = torch.device("cuda:0")

@triton.jit
def _batch_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    M,  # number of rows in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr, # BLOCK SIZE for number of rows during mean and var calculations
):
    col=tl.program_id(0)
    X+=col
    Y+=col
    # Compute the mean across the rows (batch)
    mean=0
    _mean=tl.zeros([BLOCK_SIZE],dtype=tl.float32)
    for off in range(0,M,BLOCK_SIZE):
        rows = off + tl.arange(0,BLOCK_SIZE)
        addresses = X + rows * stride
        a = tl.load(addresses, mask=rows<M, other=0.).to(tl.float32)
        _mean+=a
    mean=tl.sum(_mean,axis=0)/M
    # Compute the variance across the rows (batch)
    _var=tl.zeros([BLOCK_SIZE],dtype=tl.float32)
    var=0
    for off in range(0,M,BLOCK_SIZE):
        rows = off + tl.arange(0,BLOCK_SIZE)
        addresses = X + rows * stride
        x = tl.load(addresses, mask=rows<M, other=0.).to(tl.float32)
        x = tl.where(rows<M,x-mean,0.)
        _var+=x*x
    var=tl.sum(_var,axis=0)/M
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + col, mean)
    tl.store(Rstd + col, rstd)
    # Normalize and apply linear transformation
    for off in range(0,M,BLOCK_SIZE):
        rows = off + tl.arange(0, BLOCK_SIZE)
        mask = rows<M
        w = tl.load(W + col)
        b = tl.load(B + col)
        x = tl.load(X + rows * stride, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + rows * stride, y, mask=mask)

@triton.jit
def _batch_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             stride,  # how much to increase the pointer when moving by 1 row
                             M,  # number of rows in X
                             BLOCK_SIZE_M: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE_M)
    mask = rows < M
    X += col
    DY += col
    DX += col
    DW = DW + col
    DB = DB + col
    # Load data to SRAM
    x = tl.load(X + rows*stride, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + rows*stride, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + col).to(tl.float32)
    mean = tl.load(Mean + col)
    rstd = tl.load(Rstd + col)
    # Compute dx
    xhat = (x - mean) * rstd
    w_var=w*rstd/M
    c1 = M*dy
    c2 = tl.sum(dy, axis=0)
    c3 = M/(M-1)*xhat*tl.sum(dy*xhat,axis=0)
    dx=w_var*(c1-c2-c3)
    # Write dx
    tl.store(DX+rows*stride,dx,mask=mask)
    dw = tl.sum(dy*xhat,axis=0)
    db = tl.sum(dy,axis=0)
    tl.store(DW, dw)
    tl.store(DB, db)

class BatchNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((N, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((N, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(M))
        if M > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support batch dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _batch_norm_fwd_fused[(N, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), M, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        # allocate output
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _batch_norm_bwd_dx_fused[(N, )](  #
            dx, dy, dw, db, x, w, m, v,  #
            x_arg.stride(0), M,  #
            BLOCK_SIZE_M=ctx.BLOCK_SIZE,  #
            num_warps=ctx.num_warps)
        return dx, None, dw, db, None


batch_norm = BatchNorm.apply

def test_batch_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = batch_norm(x, weight, bias, eps)
    y_ref = torch.nn.functional.batch_norm(
        x, running_mean=None, running_var=None,
        weight=weight, bias=bias, training=True, eps=eps
    )
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)

if __name__ == '__main__':
    test_batch_norm(1151,8192,torch.float16)
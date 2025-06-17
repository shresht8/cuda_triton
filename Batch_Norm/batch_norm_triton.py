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
def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             Lock,  # pointer to the lock
                             stride,  # how much to increase the pointer when moving by 1 row
                             M,  # number of rows in X
                             GROUP_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE_M)
    mask = rows < M
    X += col
    DY += col
    DX += col
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = col % GROUP_SIZE_N
    Lock += lock_id
    Count = Lock + GROUP_SIZE_N
    DW = DW + lock_id * M + rows*stride
    DB = DB + lock_id * M + rows*stride
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
    # Accumulate partial sums for dw/db
    partial_dw = (dy*xhat).to(w.dtype)
    partial_db = dy.to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(Lock, 0)

@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):






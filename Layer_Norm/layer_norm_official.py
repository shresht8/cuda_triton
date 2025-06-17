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

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


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
                             N,  # number of columns in X
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0) # Token wise. (BxT) Tokens in total
    cols = tl.arange(0, BLOCK_SIZE_N) # Block wise processing (BLOCK_SIZE_N embedding dims processed)
    mask = cols < N # Mask for columns before total embedding dimension
    X += row * stride # To select the token to be processed
    DY += row * stride # row for DL/DY selected for processing - Total dim: (B,T,C)
    DX += row * stride # row for DL/DX selected for processing - Total dim
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M # selection of group to which row will belong to. Total groups: GROUP_SIZE_M
    Lock += lock_id # Pointer to the lock variable for this group
    Count = Lock + GROUP_SIZE_M # Pointer to the Count variable for this group
    DW = DW + lock_id * N + cols # Pointer to DW for the group row belongs to. Dims (C)
    DB = DB + lock_id * N + cols # Pointer to DB for the group row belongs to. Dims (C)
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32) # Loading x values to SRAM
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32) # Loading DY row to SRAM
    w = tl.load(W + cols, mask=mask).to(tl.float32) # Loading w values to SRAM
    mean = tl.load(Mean + row) # Loading already calculated (forward pass) mean from mean pointer
    rstd = tl.load(Rstd + row) # Loading already calculated std dev (forward pass) from pointer
    # Compute dx
    xhat = (x - mean) * rstd # Normalised x calculation in the SRAM
    wdy = w * dy # element wise multiplication b/w w and dy (Both have dimension size c)
    xhat = tl.where(mask, xhat, 0.) # masking xhat values with 0 for columns after last column
    wdy = tl.where(mask, wdy, 0.) # masking wdy values with 0 for columns after last column
    c1 = tl.sum(xhat * wdy, axis=0) / N # Calculating c1 constant value by summing  across the embedding dimension
    c2 = tl.sum(wdy, axis=0) / N # c2 constant calculation by summing across the embedding dimension
    dx = (wdy - (xhat * c1 + c2)) * rstd # dx calculation
    # Write dx
    tl.store(DX + cols, dx, mask=mask) # Writing dx to DX pointer
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype) # Calculating partial_dw value for this token.
    partial_db = (dy).to(w.dtype) # Calculating partial db value for this token
    while tl.atomic_cas(Lock, 0, 1) == 1: # When Lock value is 0 (released by other thread), it changes to 1 and
        # returns 0 (value before operation)
        pass
    count = tl.load(Count) # Load Count
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1) # First store involves storing partial_dw and partial_db to DW,DB
    else:
        partial_dw += tl.load(DW, mask=mask) # any other store, accumulate values stored + current value calculated
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask) # Store accumulated value back to DW
    tl.store(DB, partial_db, mask=mask)

    # need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(Lock, 0) # Release the lock (change to 0) for other threads/ kernels to capture it


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
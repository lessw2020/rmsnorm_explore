# titan
@triton.jit
def _rms_norm_fwd_kernel(
    X,
    stride_x,
    Y,
    stride_y,
    W,
    Rstd,
    eps,
    M,  # num rows
    N,  # num cols
    block_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_N)

    # Load input data and weights
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean and variance
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Store the reciprocal standard deviation
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    x_hat = x * rstd
    y = x_hat * w

    # Write output
    tl.store(Y + row * stride_y + cols, y, mask=mask)

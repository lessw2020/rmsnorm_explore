@triton.jit
def rms_forward_core(
        X,
        X_stride,
        Y, 
        Y_stride,
        W, 
        rstd,
        eps,
        M, 
        N, 
        BLOCK_SIZE: tl.constexpr,
):
    row_index = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # input and output pointer indexing
    x_row_pointer = X + row_index * X_stride
    y_row_pointer = Y + row_index * Y_stride

    # load data
    x_row = tl.load(x_row_ptr + cols, mask = mask, other = 0.0)
    w_row = tl.load(W + cols, mask = mask, other = 0.0)
    x_row = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
    w_row = tl.load(W + cols, mask=mask, other=0.0)

    # upscale
    x_row_fp32 = x_row_pointer.to(tl.float32)

    # compute RMS
    x_row_squared = x_row_fp32 * x_row_fp32
    mean_squared = tl.sum(x_row_squared, axis = 0) / N
    rstd_row = 1.0 / tl.sqrt(mean_squared + eps)

    # store rstd
    tl.store(rstd+row_index, rstd_row)

    # normalize and scale
    y_row = (x_row_fp32 * rstd_row).to(x_row.dtype) * w_row

    # save output
    tl.store(y_row_pointer + cols, y_row, mask = mask)

def liger_update(
        X, 
        X_stride,
        Y, 
        Y_stride, # output
        W, 
        W_stride, 
        rstd,
        rstd_stride, 
        eps, 
        n_cols,
        BLOCK_SIZE: tl.constexpr,

):
    row_index = tl.program_id(0)
    cols_offsets = tl.arange(0, BLOCK_SIZE )
    mask = cols_offsets < n_cols

    # Compute pointers for the current row
    Y_row_ptr = Y_ptr + row_idx * Y_row_stride
    X_row_ptr = X_ptr + row_idx * X_row_stride
    RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

    y_row_ptr = Y_ptr + row_index * Y_stride
    x_row_ptr = X_ptr + row_index * X_stride
    rstd_row_ptr = rstd + row_index * rsdt_stride

    # load inputs and weights
    x_row = tl.load(x_row_ptr + cols_offsets, mask = mask, other = 0.0)
    w_row = tl.load(w +cols_offsets, mask = mask, other = 0.0)

    orig_dtype = x_row.dtype
    x_row_fp32 = x_row.to(tl.float32)
    x_squared = x_row_fp32 * x_row_fp32
    mean_squared = tl.sum(x_squared, axis = 0) / n_cols

    rstd = rsqrt(mean_squared + eps)
    tl.store(rstd_row_ptr, rstd)

    x_norm = (x_row_fp32 * rstd).to(orig_dtype)
    y_row = x_norm * w_row
                      
    tl.store(y_row_ptr, y_row, mask=mask)

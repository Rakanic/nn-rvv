#include "ops/conv2D/conv2D.h"

#include <stddef.h>

/* Reshapes `in` [c_in × h_in × w_in] into `cols` [c_in*kh*kw × h_out*w_out]
 * for the standard im2col GEMM trick. Out-of-bounds positions get 0.0f
 * (zero padding). Scalar — bounds-checked per element, but the hot work in
 * conv2d_f32 is the subsequent f32_gemm_nobias which IS vectorized. */
void im2col_f32(const float *in, float *cols,
                size_t c_in, size_t h_in, size_t w_in,
                size_t kh, size_t kw, size_t stride, size_t padding,
                size_t h_out, size_t w_out)
{
    const size_t col_len = h_out * w_out;
    for (size_t ic = 0; ic < c_in; ic++) {
        for (size_t ki = 0; ki < kh; ki++) {
            for (size_t kj = 0; kj < kw; kj++) {
                size_t col_row = (ic * kh + ki) * kw + kj;
                float *col_ptr = cols + col_row * col_len;
                for (size_t oh = 0; oh < h_out; oh++) {
                    long ih = (long)oh * (long)stride - (long)padding + (long)ki;
                    for (size_t ow = 0; ow < w_out; ow++) {
                        long iw = (long)ow * (long)stride - (long)padding + (long)kj;
                        float v = 0.0f;
                        if (ih >= 0 && ih < (long)h_in &&
                            iw >= 0 && iw < (long)w_in) {
                            v = in[ic * h_in * w_in + (size_t)ih * w_in + (size_t)iw];
                        }
                        col_ptr[oh * w_out + ow] = v;
                    }
                }
            }
        }
    }
}

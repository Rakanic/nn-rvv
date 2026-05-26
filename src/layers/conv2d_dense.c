#include "nn_rvv/layers.h"
#include "ops/conv2D/conv2D.h"
#include "ops/matmul/matmul.h"
#include "ops/elementwise/elementwise.h"

#include <stddef.h>

/* Dense (non-depthwise) 2D convolution via im2col + GEMM:
 *
 *   im2col(in)             → cols[c_in*kh*kw × h_out*w_out]
 *   weight @ cols          → out [c_out × h_out*w_out]      (f32_gemm_nobias)
 *   + broadcast bias[c_out] over each output channel        (optional)
 *
 * The caller provides the cols scratch buffer
 *   scratch[c_in * kh * kw * h_out * w_out floats]
 * so the kernel itself does not malloc. */
void conv2d_f32(float *out,
                const float *in,
                const float *weight,
                const float *bias,
                float *cols_scratch,
                size_t c_in, size_t h_in, size_t w_in,
                size_t c_out, size_t kh, size_t kw,
                size_t stride, size_t padding)
{
    const size_t h_out = (h_in + 2 * padding - kh) / stride + 1;
    const size_t w_out = (w_in + 2 * padding - kw) / stride + 1;
    const size_t patch_size  = c_in * kh * kw;
    const size_t spatial_out = h_out * w_out;

    im2col_f32(in, cols_scratch,
               c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    /* weight[c_out × patch_size] @ cols[patch_size × spatial_out] */
    f32_gemm_nobias(c_out, spatial_out, patch_size,
                    weight, patch_size,
                    cols_scratch,
                    out, spatial_out, 1);

    if (bias) {
        /* Add bias broadcast over each output-channel row. */
        for (size_t oc = 0; oc < c_out; oc++) {
            scale_add_f32(out + oc * spatial_out,
                          out + oc * spatial_out,
                          1.0f, bias[oc], spatial_out);
        }
    }
}

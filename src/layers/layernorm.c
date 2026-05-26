#include "nn_rvv/layers.h"
#include "ops/reduce/reduce.h"

#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>

/* LayerNorm over one row of size n:
 *   mean    = (1/n) * sum(x)
 *   var     = (1/n) * sum((x - mean)^2)  =  E[x^2] - mean^2
 *   out[i]  = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
 *
 * Two reduction passes (sum + sum-of-squares) followed by one fused
 * affine pass. weight/bias may be NULL for plain LN (no scale/shift). */
void layer_norm_f32(float *out, const float *x,
                    const float *weight, const float *bias,
                    size_t n, float eps)
{
    const float inv_n = 1.0f / (float)n;
    float mean    = sum_f32(x, n) * inv_n;
    float sum_xsq = dot_f32(x, x, n);
    float var     = sum_xsq * inv_n - mean * mean;
    float inv_std = 1.0f / sqrtf(var + eps);

    size_t remaining = n;
    const float *xp = x;
    float *op = out;
    const float *wp = weight;
    const float *bp = bias;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m4(remaining);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(xp, vl);
        /* (x - mean) * inv_std */
        vfloat32m4_t vr = __riscv_vfsub_vf_f32m4(vx, mean, vl);
        vr              = __riscv_vfmul_vf_f32m4(vr, inv_std, vl);
        if (wp) {
            vfloat32m4_t vw = __riscv_vle32_v_f32m4(wp, vl);
            vr              = __riscv_vfmul_vv_f32m4(vr, vw, vl);
            wp += vl;
        }
        if (bp) {
            vfloat32m4_t vb = __riscv_vle32_v_f32m4(bp, vl);
            vr              = __riscv_vfadd_vv_f32m4(vr, vb, vl);
            bp += vl;
        }
        __riscv_vse32_v_f32m4(op, vr, vl);
        xp += vl; op += vl; remaining -= vl;
    }
}

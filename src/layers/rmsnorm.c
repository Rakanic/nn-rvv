#include "nn_rvv/layers.h"
#include "ops/elementwise/elementwise.h"
#include "ops/reduce/reduce.h"

#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>

/* out[i] = weight[i] * in[i] / sqrt(mean(in^2) + eps). Two vectorized
 * passes: sum-of-squares via dot_f32(in, in), then fused mul+scale. */
void rmsnorm_f32(float *out, const float *in, const float *weight, size_t size) {
    float ss = dot_f32(in, in, size);
    float scale = 1.0f / sqrtf(ss / (float)size + 1e-5f);

    size_t remaining = size;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m4(remaining);
        vfloat32m4_t vi = __riscv_vle32_v_f32m4(in, vl);
        vfloat32m4_t vw = __riscv_vle32_v_f32m4(weight, vl);
        vfloat32m4_t vo = __riscv_vfmul_vv_f32m4(vi, vw, vl);
        vo = __riscv_vfmul_vf_f32m4(vo, scale, vl);
        __riscv_vse32_v_f32m4(out, vo, vl);
        in += vl; weight += vl; out += vl; remaining -= vl;
    }
}

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <riscv_vector.h>

#include "nn_rvv/layers.h"

void relu6_int8(
    size_t channels,
    size_t inner_size,
    const float *input,
    int8_t *output,
    requantization_params_t requant_params
) {
    if (!input || !output || !requant_params.scale) {
        return;
    }

    const int8_t zp = (int8_t) requant_params.zero_point;
    const size_t channel_stride = inner_size;

    for (size_t c = 0; c < channels; c++) {
        float scale_c = requant_params.scale[c];
        if (scale_c <= 0.0f) {
            scale_c = 1.0f;
        }
        const float inv_scale = 1.0f / scale_c;

        const float *in_ch = input  + c * channel_stride;
        int8_t *out_ch     = output + c * channel_stride;
        size_t remaining = inner_size;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t vf = __riscv_vle32_v_f32m4(in_ch, vl);

            // Clamp in real domain
            vf = __riscv_vfmax_vf_f32m4(vf, 0.0f, vl);
            vf = __riscv_vfmin_vf_f32m4(vf, 6.0f, vl);

            // Quantize: round(clamped / scale) + zp
            vf = __riscv_vfmul_vf_f32m4(vf, inv_scale, vl);
            vint16m2_t q16 = __riscv_vfncvt_x_f_w_i16m2(vf, vl);
            q16 = __riscv_vadd_vx_i16m2(q16, zp, vl);

            // Clamp to int8 range then narrow
            q16 = __riscv_vmax_vx_i16m2(q16, -128, vl);
            q16 = __riscv_vmin_vx_i16m2(q16, 127, vl);
            vint8m1_t q8 = __riscv_vncvt_x_x_w_i8m1(q16, vl);
            __riscv_vse8_v_i8m1(out_ch, q8, vl);

            in_ch += vl;
            out_ch += vl;
            remaining -= vl;
        }
    }
}

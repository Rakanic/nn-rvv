#include "lib_layers.h"

#include <riscv_vector.h>
#include "stdio.h"

void quant_f32(
    size_t size, 
    float* input, 
    int8_t* output, 
    quantization_params_t qp
)
{
    register vfloat32m8_t vfacc0;
    register vint16m4_t vout0;
    vint8m2_t vout80;
    const int32_t output_zero_point = qp.zero_point;
    const int32_t output_min_less_zero_point = -128 - output_zero_point;
    const int32_t output_max_less_zero_point = 127 - output_zero_point;
    const float scale_inv = 1 / qp.scale;

    do {
        register size_t vl = __riscv_vsetvl_e32m8(size);
        vfacc0 = __riscv_vle32_v_f32m8(input, vl);
        vfacc0 = __riscv_vfmul_vf_f32m8(vfacc0, scale_inv, vl);
        vfacc0 = __riscv_vfmax_vf_f32m8(vfacc0, output_min_less_zero_point, vl);
        vfacc0 = __riscv_vfmin_vf_f32m8(vfacc0, output_max_less_zero_point, vl);
        vout0 = __riscv_vfncvt_x_f_w_i16m4(vfacc0, vl);
        vout0 = __riscv_vadd_vx_i16m4(vout0, (int16_t) output_zero_point, vl);
        vout80 = __riscv_vncvt_x_x_w_i8m2(vout0, vl);
        __riscv_vse8_v_i8m2(output, vout80, vl);

        input += vl;
        output += vl;
        size -= vl;
    } while (size != 0);
}
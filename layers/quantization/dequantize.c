#include "lib_layers.h"

#include <riscv_vector.h>
#include "stdio.h"

void dequant_f32(
    size_t size, 
    int8_t* input, 
    float* output, 
    quantization_params_t qp
)
{
    register vfloat32m8_t vfout0;
    register vint16m4_t vin0;
    register vint32m8_t vin320;
    const int32_t output_zero_point = qp.zero_point;
    const float scale = qp.scale;

    do {
        register size_t vl = __riscv_vsetvl_e8m2(size);
        vin0 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(input, vl), vl);
        vin320 = __riscv_vwsub_vx_i32m8(vin0, (int16_t) output_zero_point, vl);
        vfout0 = __riscv_vfcvt_f_x_v_f32m8(vin320, vl);
        vfout0 = __riscv_vfmul_vf_f32m8(vfout0, scale, vl);
        __riscv_vse32_v_f32m8(output, vfout0, vl);

        input += vl;
        output += vl;
        size -= vl;
    } while (size != 0);
}

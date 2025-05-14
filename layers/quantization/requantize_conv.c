#include "lib_layers.h"

#include <riscv_vector.h>
#include "stdio.h"
#include "stdint.h"

void requantize_2D(
    size_t size, 
    int32_t* input, 
    int8_t* output, 
    float scale, 
    int32_t zero_point
) 
{
    register vfloat32m8_t vfacc0;
    register vint32m8_t vacc0;
    register vint16m4_t vout0;
    vint8m2_t vout80;

    const int32_t output_min_less_zero_point = -128 - zero_point;
    const int32_t output_max_less_zero_point = 127 - zero_point;


    do {
        register size_t vl = __riscv_vsetvl_e32m8(size);
        vacc0 = __riscv_vle32_v_i32m8(input, vl);
        vfacc0 = __riscv_vfcvt_f_x_v_f32m8(vacc0, vl);
        vfacc0 = __riscv_vfmul_vf_f32m8(vfacc0, scale, vl);
        vfacc0 = __riscv_vfmax_vf_f32m8(vfacc0, output_min_less_zero_point, vl);
        vfacc0 = __riscv_vfmin_vf_f32m8(vfacc0, output_max_less_zero_point, vl);
        vout0 = __riscv_vfncvt_x_f_w_i16m4(vfacc0, vl);
        vout0 = __riscv_vadd_vx_i16m4(vout0, (int16_t) zero_point, vl);
        vout80 = __riscv_vncvt_x_x_w_i8m2(vout0, vl);
        __riscv_vse8_v_i8m2(output, vout80, vl);

        input += vl;
        output += vl;
        size -= vl;
    } while (size != 0);
}

void requantize_relu_2D(
    size_t size, 
    int32_t* input, 
    int8_t* output, 
    float scale, 
    int32_t zero_point
) 
{
    register vfloat32m8_t vfacc0;
    register vint32m8_t vacc0;
    register vint16m4_t vout0;
    vint8m2_t vout80;

    const int32_t output_min_less_zero_point = -128;
    const int32_t output_max_less_zero_point = 127;


    do {
        register size_t vl = __riscv_vsetvl_e32m8(size);
        vacc0 = __riscv_vle32_v_i32m8(input, vl);
        vfacc0 = __riscv_vfcvt_f_x_v_f32m8(vacc0, vl);
        vfacc0 = __riscv_vfmul_vf_f32m8(vfacc0, scale, vl);
        vfacc0 = __riscv_vfmax_vf_f32m8(vfacc0, 0.0f, vl);
        vout0 = __riscv_vfncvt_x_f_w_i16m4(vfacc0, vl);
        vout0 = __riscv_vmax_vx_i16m4(vout0, output_min_less_zero_point, vl);
        vout0 = __riscv_vmin_vx_i16m4(vout0, output_max_less_zero_point, vl);
        vout0 = __riscv_vadd_vx_i16m4(vout0, (int16_t) zero_point, vl);
        vout80 = __riscv_vncvt_x_x_w_i8m2(vout0, vl);
        __riscv_vse8_v_i8m2(output, vout80, vl);

        input += vl;
        output += vl;
        size -= vl;
    } while (size != 0);
}

void requant_outch_int32(
    size_t rows, size_t cols, 
    size_t channels,
    int32_t* input, 
    int8_t* output, 
    int relu,
    requantization_params_t rqp
)
{
    register size_t size = rows * cols;
    register int32_t zero_point = rqp.zero_point;

    for (size_t ch = 0; ch < channels; ch++) {

        if (!relu) {
            requantize_2D(size, input, output, rqp.scale[ch], zero_point);
        } else {
            requantize_relu_2D(size, input, output, rqp.scale[ch], zero_point);
        }

        output += size;
        input += size;
    }

}
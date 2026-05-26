#include "nn_rvv/layers.h"

#include <riscv_vector.h>
#include "stdio.h"

void residual_add(
    size_t rows, size_t cols, 
    size_t channels, 
    int8_t* a, int8_t* b, 
    int8_t* output, 
    requantization_params_t rqp
)
{
    size_t vl;
    size_t remaining;
    size_t channel_size = rows * cols;

    register vint8m1_t load_a;
    register vint8m1_t load_b;
    register vint16m2_t add_ab;
    register vint32m4_t add_ab32;
    register vfloat32m4_t vfacc;
    register vint16m2_t vout16;
    register vint8m1_t vout8;
    

    for (int c = 0; c < channels; c++) {
        remaining = channel_size;
        float scale = rqp.scale[c];
        do {
            vl = __riscv_vsetvl_e8m1(remaining);
            load_a = __riscv_vle8_v_i8m1(a, vl);
            load_b = __riscv_vle8_v_i8m1(b, vl);
            add_ab = __riscv_vwadd_vv_i16m2(load_a, load_b, vl);
            add_ab32 = __riscv_vwcvt_x_x_v_i32m4(add_ab, vl);
            vfacc = __riscv_vfcvt_f_x_v_f32m4(add_ab32, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(output, vout8, vl);

            a += vl;
            b += vl;
            output += vl;
            remaining -= vl;
        } while (remaining != 0);
    }
}
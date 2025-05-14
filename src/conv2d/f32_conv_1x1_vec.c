#include "conv2d.h"

#include <riscv_vector.h> 
#include <stdint.h>

void conv_1x1_f32_2d(
    size_t rows, size_t cols,
    float* input,
    float weight, 
    float bias,
    float* output
) {
    size_t vl;
    size_t n = rows * cols;
    register vfloat32m4_t v;
    register vfloat32m4_t vec_bias;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m4(n);
        v = __riscv_vle32_v_f32m4(input, vl);
        vec_bias = __riscv_vfmv_v_f_f32m4(bias, vl);
        v = __riscv_vfmacc_vf_f32m4(vec_bias, weight, v, vl);
        __riscv_vse32_v_f32m4(output, v, vl);
    
        // Advance pointers and decrease remaining count
        input       += vl;
        n         -= vl;
        output += vl;
    }
}

void conv_1x1_f32_2d_macc(
    size_t rows, size_t cols,
    float* input,
    float weight,
    float* output
) {
    size_t vl;
    size_t n = rows * cols;
    register vfloat32m4_t v;
    register vfloat32m4_t v1;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m4(n);
        v = __riscv_vle32_v_f32m4(input, vl);
        v1 = __riscv_vle32_v_f32m4(output, vl);
        v = __riscv_vfmacc_vf_f32m4(v1, weight, v, vl);
        __riscv_vse32_v_f32m4(output, v, vl);
    
        // Advance pointers and decrease remaining count
        input       += vl;
        n         -= vl;
        output += vl;
    }
}


void conv_1x1_f32(
    size_t rows, size_t cols, 
    size_t channels_in,
    size_t channels_out,
    float* input,
    const float* weights,
    float* output
) {
    register float* input_ptr;
    register float weight;
    register float bias_value; 
    register const float* bias = weights;
    weights += channels_out;
    
    for (size_t ch_o = 0; ch_o < channels_out; ch_o++) {
        bias_value = *(bias++);
        for (size_t ch_i = 0; ch_i < channels_in; ch_i++) {
            input_ptr = input + rows * cols * ch_i;
            weight = *(weights++);
            if (ch_i != 0) {
                conv_1x1_f32_2d_macc(rows, cols, input_ptr, weight, output);
            } else {
                conv_1x1_f32_2d(rows, cols, input_ptr, weight, bias_value, output);
            }
        }
        output += rows * cols;
    }
}
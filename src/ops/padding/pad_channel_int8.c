#include "ops/padding/padding.h"

#include <riscv_vector.h> 
#include <stdint.h>
#include "stdio.h"
#include "string.h"

void pad_input_channel(
    size_t input_cols, 
    size_t input_rows, 
    size_t x_padding, 
    size_t y_padding, 
    const int8_t* input, 
    int8_t* output
) 
{
    // printf("output: %p \n", output);
    register vint8m8_t temp;
    register size_t output_cols = input_cols + 2*x_padding;
    register size_t output_rows = input_rows + 2*y_padding;
    register size_t o_cols = output_cols - x_padding;
    register int32_t vl;
    do {
        register const int8_t* i = input;
        register int8_t* o = output; 
        vl = __riscv_vsetvl_e8m8(o_cols);
        register size_t rows = input_rows;

        // add padding in the top of vl-columns
        for (size_t i = 0; i < y_padding; i++) {
            __riscv_vse8_v_i8m8(o, __riscv_vmv_v_x_i8m8(0, vl), vl); o += output_cols;
        }

        do { 
            temp = __riscv_vle8_v_i8m8(i, vl);
            if (o_cols == output_cols - x_padding) {
                vint8m8_t zeros = __riscv_vmv_v_x_i8m8(0, vl);
                temp = __riscv_vslideup_vx_i8m8(zeros, temp, x_padding, vl);
            }
            __riscv_vse8_v_i8m8(o, temp, vl); o += output_cols;
            i += input_cols;
            rows -= 1;
        } while (rows != 0);

        // add padding in the bottom of vl-columns
        for (size_t i = 0; i < y_padding; i++) {
            __riscv_vse8_v_i8m8(o, __riscv_vmv_v_x_i8m8(0, vl), vl); o += output_cols;
        }

        input = (o_cols == output_cols - x_padding) ? input + vl - x_padding : input + vl;
        o_cols -= vl; 
        output = output + vl;

    } while (o_cols != 0);

    // printf("output: %p \n", output);
    for (size_t r = 0; r < output_rows; r++){
        vl = __riscv_vsetvl_e8m8(x_padding);
        __riscv_vse8_v_i8m8(output, __riscv_vmv_v_x_i8m8(0, vl), vl);
        output += output_cols;
    }
}

#include "transpose.h"

#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

void transpose_int8 (int8_t* input, int8_t* output, size_t rows, size_t cols) {
    size_t vl;
    vint8m8_t load0;

    int8_t* sp;
    int8_t* sp_2 = output;
    size_t r = rows;

    do {
        size_t c = cols;
        const int8_t* i = input + cols*(r - rows);
        sp = sp_2;
        sp_2 += 1;

        do {
            vl = __riscv_vsetvl_e8m8(c);
            load0 = __riscv_vle8_v_i8m8(i, vl);
            __riscv_vsse8_v_i8m8(sp, r, load0, vl);

            sp += vl*r;

            c -= vl;
            i += vl;

        } while (c != 0);

    rows -= 1;
    } while (rows != 0);
}
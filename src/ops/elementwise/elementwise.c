#include "elementwise.h"

#include <riscv_vector.h>

void fill_f32(float *y, float c, size_t n) {
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v = __riscv_vfmv_v_f_f32m4(c, vl);
        __riscv_vse32_v_f32m4(y, v, vl);
        y += vl; n -= vl;
    }
}

void axpy_f32(float *y, float a, const float *x, size_t n) {
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vy = __riscv_vle32_v_f32m4(y, vl);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x, vl);
        vy = __riscv_vfmacc_vf_f32m4(vy, a, vx, vl);
        __riscv_vse32_v_f32m4(y, vy, vl);
        y += vl; x += vl; n -= vl;
    }
}

void scale_add_f32(float *y, const float *x, float a, float b, size_t n) {
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x, vl);
        vx = __riscv_vfmul_vf_f32m4(vx, a, vl);
        vx = __riscv_vfadd_vf_f32m4(vx, b, vl);
        __riscv_vse32_v_f32m4(y, vx, vl);
        y += vl; x += vl; n -= vl;
    }
}

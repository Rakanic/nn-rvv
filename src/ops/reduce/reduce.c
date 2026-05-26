#include "reduce.h"

#include <riscv_vector.h>

/* Scalar accumulator across stripmine iterations — robust to tail policy. */
float dot_f32(const float *a, const float *b, size_t n) {
    float acc = 0.0f;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(b, vl);
        vfloat32m4_t vp = __riscv_vfmul_vv_f32m4(va, vb, vl);
        vfloat32m1_t vr = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vr = __riscv_vfredusum_vs_f32m4_f32m1(vp, vr, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(vr);
        a += vl; b += vl; n -= vl;
    }
    return acc;
}

float sum_f32(const float *x, size_t n) {
    float acc = 0.0f;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x, vl);
        vfloat32m1_t vr = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vr = __riscv_vfredusum_vs_f32m4_f32m1(vx, vr, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(vr);
        x += vl; n -= vl;
    }
    return acc;
}

float max_f32(const float *x, size_t n) {
    float acc = x[0];
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x, vl);
        vfloat32m1_t vr = __riscv_vfmv_s_f_f32m1(acc, 1);
        vr = __riscv_vfredmax_vs_f32m4_f32m1(vx, vr, vl);
        acc = __riscv_vfmv_f_s_f32m1_f32(vr);
        x += vl; n -= vl;
    }
    return acc;
}

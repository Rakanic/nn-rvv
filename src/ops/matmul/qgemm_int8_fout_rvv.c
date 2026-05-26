/*
 * qgemm_int8_fout_rvv.c
 *
 * Int8 × Int8 → Float32 GEMM kernels for the transposed matmul pattern
 * used in int8 transformer inference.
 *
 * Key differences from qgemm_int8_rvv.c:
 *   - Requantization: multiply int32 accumulator by a scalar float `scale`
 *     and store as float32 (no int8 narrowing, no per-channel scale array).
 *   - Input bias term: zero (weights are pre-converted from uint8 to int8 by
 *     subtracting 128 during the one-time transposition at startup, so the
 *     bias row in B_pack is all zeros and no zero-point correction is needed).
 *
 * Intended usage: matmul_t() in int8 borai (transformer inference)
 *   x(1,n_in) @ W_T(n_in, n_out) → xout(1, n_out)
 *   → int8_qgemm_fout(M=1, N=n_out, K=n_in, A=x_q, B=B_pack, C=xout,
 *                     scale = 1/(127*127))
 *
 * B_pack layout: [(K+1) × N] bytes
 *   Row 0       : N zero bytes  (bias / initial accumulator)
 *   Rows 1 .. K : N int8 bytes per row  (rows of W_T)
 */

#include "ops/matmul/matmul.h"
#include <riscv_vector.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * 1-row microkernel: M=1, vectorises over N (output_size).
 * This is the only path used in single-token transformer inference.
 * ------------------------------------------------------------------------- */
static void qgemm_i8_fout_1xm4(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const int8_t* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    float scale)
{
    const int8_t* a0 = a;
    float* c0 = c;

    size_t nr = nc;
    const int8_t* w_new = w;

    do {
        size_t vl = __riscv_vsetvl_e32m4(nc);
        nc -= vl;
        w_new = w + vl;

        /* Load bias row (int8 → int32) — bias is all zeros for borai */
        vint32m4_t vacc0 = __riscv_vwcvt_x_x_v_i32m4(
            __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl), vl);
        w += nr;

        size_t k = kc;
        do {
            const int8_t va0 = *a0++;
            vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(
                __riscv_vle8_v_i8m1(w, vl), vl);
            w += nr;
            vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
            k--;
        } while (k != 0);

        a0 -= kc;

        /* Requantize: int32 → float32 with scalar scale (no clamping needed) */
        vfloat32m4_t vfacc0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);
        vfacc0 = __riscv_vfmul_vf_f32m4(vfacc0, scale, vl);
        __riscv_vse32_v_f32m4(c0, vfacc0, vl);
        c0 += vl;

        w = w_new;
    } while (nc != 0);
}

/* -------------------------------------------------------------------------
 * 7-row microkernel: M=7, vectorises over N.
 * Included for completeness (batched inference).
 * ------------------------------------------------------------------------- */
static void qgemm_i8_fout_7xm4(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const int8_t* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    float scale)
{
    const int8_t* a0 = a;
    float* c0 = c;
    const int8_t* a1 = a0 + a_stride;
    float* c1 = (float*)((uintptr_t)c0 + cm_stride);
    const int8_t* a2 = a1 + a_stride;
    float* c2 = (float*)((uintptr_t)c1 + cm_stride);
    const int8_t* a3 = a2 + a_stride;
    float* c3 = (float*)((uintptr_t)c2 + cm_stride);
    const int8_t* a4 = a3 + a_stride;
    float* c4 = (float*)((uintptr_t)c3 + cm_stride);
    const int8_t* a5 = a4 + a_stride;
    float* c5 = (float*)((uintptr_t)c4 + cm_stride);
    const int8_t* a6 = a5 + a_stride;
    float* c6 = (float*)((uintptr_t)c5 + cm_stride);

    size_t nr = nc;
    const int8_t* w_new = w;

    do {
        size_t vl = __riscv_vsetvl_e32m4(nc);
        nc -= vl;
        w_new = w + vl;

        /* Load bias row (int8 → int32) and broadcast to all 7 rows */
        vint32m4_t vacc0 = __riscv_vwcvt_x_x_v_i32m4(
            __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl), vl);
        w += nr;
        vint32m4_t vacc1 = __riscv_vmv_v_v_i32m4(vacc0, vl);
        vint32m4_t vacc2 = __riscv_vmv_v_v_i32m4(vacc0, vl);
        vint32m4_t vacc3 = __riscv_vmv_v_v_i32m4(vacc0, vl);
        vint32m4_t vacc4 = __riscv_vmv_v_v_i32m4(vacc0, vl);
        vint32m4_t vacc5 = __riscv_vmv_v_v_i32m4(vacc0, vl);
        vint32m4_t vacc6 = __riscv_vmv_v_v_i32m4(vacc0, vl);

        size_t k = kc;
        do {
            const int8_t va0 = *a0++;
            const int8_t va1 = *a1++;
            const int8_t va2 = *a2++;
            const int8_t va3 = *a3++;
            const int8_t va4 = *a4++;
            const int8_t va5 = *a5++;
            const int8_t va6 = *a6++;
            vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(
                __riscv_vle8_v_i8m1(w, vl), vl);
            w += nr;
            vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
            vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
            vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
            vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
            vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
            vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
            vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);
            k--;
        } while (k != 0);

        a0 -= kc; a1 -= kc; a2 -= kc; a3 -= kc;
        a4 -= kc; a5 -= kc; a6 -= kc;

        /* Requantize: int32 → float32 with scalar scale */
#define STORE_FOUT(vacc, cp)                                                \
        do {                                                                 \
            vfloat32m4_t _vf = __riscv_vfcvt_f_x_v_f32m4((vacc), vl);     \
            _vf = __riscv_vfmul_vf_f32m4(_vf, scale, vl);                  \
            __riscv_vse32_v_f32m4((cp), _vf, vl);                           \
        } while (0)

        STORE_FOUT(vacc0, c0); c0 += vl;
        STORE_FOUT(vacc1, c1); c1 += vl;
        STORE_FOUT(vacc2, c2); c2 += vl;
        STORE_FOUT(vacc3, c3); c3 += vl;
        STORE_FOUT(vacc4, c4); c4 += vl;
        STORE_FOUT(vacc5, c5); c5 += vl;
        STORE_FOUT(vacc6, c6); c6 += vl;

#undef STORE_FOUT

        w = w_new;
    } while (nc != 0);
}

/* -------------------------------------------------------------------------
 * Public dispatcher: loops over M in tiles of 7 or 1.
 * ------------------------------------------------------------------------- */
void int8_qgemm_fout(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    float* C, size_t c_row_stride,
    size_t c_col_stride,
    float scale)
{
    const size_t cm_stride_bytes = c_row_stride;

    size_t row = 0;
    while (row < M) {
        size_t rows_left = M - row;

        if (rows_left >= 7) {
            qgemm_i8_fout_7xm4(
                7, N, K,
                A + row * a_row_stride, a_row_stride,
                B,
                C + row * (c_row_stride / sizeof(float)),
                cm_stride_bytes, c_col_stride,
                scale);
            row += 7;
        } else {
            qgemm_i8_fout_1xm4(
                1, N, K,
                A + row * a_row_stride, a_row_stride,
                B,
                C + row * (c_row_stride / sizeof(float)),
                cm_stride_bytes, c_col_stride,
                scale);
            row += 1;
        }
    }
}

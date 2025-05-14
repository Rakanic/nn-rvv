#include "matmul.h"

#include <riscv_vector.h> 
#include <stdint.h>

void xnn_f32_gemm_ukernel_7x4v_relu__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const float vmin = 0.0f;
  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);

  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);

  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);

  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);

  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);

  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);

  const size_t nr = nc;
  size_t vl = nr;
  const float* w_new = w;
  do {
    vl = __riscv_vsetvl_e32m4(nc);
    nc = nc - vl;
    w_new = w + vl;

    register vfloat32m4_t vacc0 =  __riscv_vle32_v_f32m4(w, vl);

    w = w + nr;
    register vfloat32m4_t vacc1 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc2 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc3 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc4 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc5 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    register vfloat32m4_t vacc6 =  __riscv_vmv_v_v_f32m4(vacc0, vl);

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;
      const float va4 = *a4++;
      const float va5 = *a5++;
      const float va6 = *a6++;
      register vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);

      w = w + nr;
      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, va3, vb, vl);
      vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, va4, vb, vl);
      vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, va5, vb, vl);
      vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, va6, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // store 7 x vl results to c
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = c0 + vl;
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, vmin, vl);
    __riscv_vse32_v_f32m4(c1, vacc1, vl);
    c1 = c1 + vl;
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, vmin, vl);
    __riscv_vse32_v_f32m4(c2, vacc2, vl);
    c2 = c2 + vl;
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, vmin, vl);
    __riscv_vse32_v_f32m4(c3, vacc3, vl);
    c3 = c3 + vl;
    vacc4 = __riscv_vfmax_vf_f32m4(vacc4, vmin, vl);
    __riscv_vse32_v_f32m4(c4, vacc4, vl);
    c4 = c4 + vl;
    vacc5 = __riscv_vfmax_vf_f32m4(vacc5, vmin, vl);
    __riscv_vse32_v_f32m4(c5, vacc5, vl);
    c5 = c5 + vl;
    vacc6 = __riscv_vfmax_vf_f32m4(vacc6, vmin, vl);
    __riscv_vse32_v_f32m4(c6, vacc6, vl);
    c6 = c6 + vl;
    a0 = (const float*) ((uintptr_t) a0 - kc);
    a1 = (const float*) ((uintptr_t) a1 - kc);
    a2 = (const float*) ((uintptr_t) a2 - kc);
    a3 = (const float*) ((uintptr_t) a3 - kc);
    a4 = (const float*) ((uintptr_t) a4 - kc);
    a5 = (const float*) ((uintptr_t) a5 - kc);
    a6 = (const float*) ((uintptr_t) a6 - kc);
    w = w_new;

  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_1x4v_relu__rvv(
  size_t mr,
  size_t nc,
  size_t kc,
  const float* a,
  size_t a_stride,
  const float* w,
  float* restrict c,
  size_t cm_stride,
  size_t cn_stride)
{
    const float vmin = 0.0f;
    const float* a0 = a;
    float* c0 = c;

    const size_t nr = nc;
    size_t vl = nr;
    const float* w_new = w;
    do {
        vl = __riscv_vsetvl_e32m4(nc);
        nc = nc - vl;
        w_new = w + vl;

        vfloat32m4_t vacc0 =  __riscv_vle32_v_f32m4(w, vl);

        w = w + nr;

        size_t k = kc;
        do {
            const float va0 = *a0++;
            vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);
            w = w + nr;
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
            k -= sizeof(float);
        } while (k != 0);
        // store 1 x vl results to c
        vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
        __riscv_vse32_v_f32m4(c0, vacc0, vl);

        c0 = c0 + vl;
        a0 = (const float*) ((uintptr_t) a0 - kc);
        w = w_new;
    } while (nc != 0);
}

void f32_gemm_relu(
  size_t M, size_t N, size_t K,
  const float* A, size_t a_row_stride,
  const float* B,
  float* C, size_t c_row_stride,
  size_t c_col_stride)
{
  const size_t kc_bytes = K * sizeof(float);
  const size_t a_stride_bytes = a_row_stride * sizeof(float);
  const size_t cm_stride_bytes = c_row_stride * sizeof(float);
  const size_t cn_stride_bytes = c_col_stride * sizeof(float);

  size_t row = 0;
  while (row < M) {
      size_t rows_left = M - row;

      if (rows_left >= 7) {
          xnn_f32_gemm_ukernel_7x4v_relu__rvv(
              7,
              N,
              kc_bytes,
              A + row * a_row_stride,
              a_stride_bytes,
              B,
              C + row * c_row_stride,
              cm_stride_bytes,
              cn_stride_bytes
          );
          row += 7;
      } else {
          xnn_f32_gemm_ukernel_1x4v_relu__rvv(
              1,
              N,
              kc_bytes,
              A + row * a_row_stride,
              a_stride_bytes,
              B,
              C + row * c_row_stride,
              cm_stride_bytes,
              cn_stride_bytes
          );
          row += 1;
      }
  }
}
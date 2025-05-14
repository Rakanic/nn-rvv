#include "matmul.h"

#include <stdint.h>
#include <riscv_vector.h> 

static inline void print_vint32m4(vint32m4_t vec, size_t vl) {
  // make a temporary array of exactly vl elements
  int32_t buf[vl];
  // spill the vector into the array
  __riscv_vse32_v_i32m4(buf, vec, vl);
  // print each lane
  for (size_t i = 0; i < vl; i++) {
      printf(" %d, ", buf[i]);
  }
  printf("\n");
}
static inline void print_vint16m2(vint16m2_t vec, size_t vl) {
  // make a temporary array of exactly vl elements
  int16_t buf[vl];
  // spill the vector into the array
  __riscv_vse16_v_i16m2(buf, vec, vl);
  // print each lane
  for (size_t i = 0; i < vl; i++) {
      printf(" %d, ", buf[i]);
  }
  printf("\n");
}

void qgemm_i8_i32_7xm4_int32bias (
    size_t mr,        // number of rows to process (1..7)
    size_t nc,        // number of columns to process
    size_t kc,        // number of "channels" or "inner dimension"
    const int8_t* a,  // input matrix A
    size_t a_stride,           // byte stride between consecutive rows of A
    const void* w,  // weights (B)
    int8_t* c,       // output matrix C (int32)
    size_t cm_stride,          // byte stride between consecutive rows of C
    size_t cn_stride,           // byte stride between consecutive columns-blocks of C 
    requantization_params_t requant_params // requantization parameters
)
{
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* start_weights = (const int8_t*) ((const int32_t*)w + nc);

  const int32_t output_min_less_zero_point = -128 - requant_params.zero_point;
  const int32_t output_max_less_zero_point = 127 - requant_params.zero_point;
  const int32_t output_zero_point = requant_params.zero_point;

  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);

  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);

  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);

  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);

  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);

  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  int8_t* c6 = (int8_t*) ((uintptr_t) c5 + cm_stride);

  // For NR="m4": we use a vsetvlmax for 32-bit int with LMUL=m4
  size_t nr = nc;
  size_t vl = nr;
  size_t scale_ptr = 0;
  const int32_t* w_new = (const int32_t*) w;
  // Loop over columns in chunks of VL
  do {
    // If fewer than nr columns remain, reduce VL
    vl = __riscv_vsetvl_e32m4(nc);
    nc -= vl;
    vint32m4_t vacc0 = __riscv_vle32_v_i32m4(w_new, vl);
    w = (const void*) start_weights;

    vint32m4_t vacc1 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc2 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc3 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc4 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc5 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc6 = __riscv_vmv_v_v_i32m4(vacc0, vl);

    // Multiply-accumulate across kc
    size_t k = kc;
    do {
      // Load 1 int8 from each row
      const int8_t va0 = *a0++;
      const int8_t va1 = *a1++;
      const int8_t va2 = *a2++;
      const int8_t va3 = *a3++;
      const int8_t va4 = *a4++;
      const int8_t va5 = *a5++;
      const int8_t va6 = *a6++;

      // Load one vector of int8 from the weights
      vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1((const int8_t* ) w, vl), vl);
      w = ((const int8_t*) w) + nr;

      // Perform widening multiply to int16, then add to the int32 accumulators
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
      vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
      vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
      vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;
    a1 -= kc;
    a2 -= kc;
    a3 -= kc;
    a4 -= kc;
    a5 -= kc;
    a6 -= kc;

    vfloat32m4_t vscale = __riscv_vle32_v_f32m4(requant_params.scale + scale_ptr, vl);
    // Now we store the results (int32) in the output C
    vfloat32m4_t vfacc0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);
    vfacc0 = __riscv_vfmul_vv_f32m4(vfacc0, vscale, vl);
    vfacc0 = __riscv_vfmax_vf_f32m4(vfacc0, output_min_less_zero_point, vl);
    vfacc0 = __riscv_vfmin_vf_f32m4(vfacc0, output_max_less_zero_point, vl);
    vint16m2_t vout0 = __riscv_vfncvt_x_f_w_i16m2(vfacc0, vl);
    vout0 = __riscv_vadd_vx_i16m2(vout0, (int16_t) output_zero_point, vl);
    vint8m1_t vout80 = __riscv_vncvt_x_x_w_i8m1(vout0, vl);
    __riscv_vse8_v_i8m1(c0, vout80, vl);
    c0 += vl;

    vfloat32m4_t vfacc1 = __riscv_vfcvt_f_x_v_f32m4(vacc1, vl);
    vfacc1 = __riscv_vfmul_vv_f32m4(vfacc1, vscale, vl);
    vfacc1 = __riscv_vfmax_vf_f32m4(vfacc1, output_min_less_zero_point, vl);
    vfacc1 = __riscv_vfmin_vf_f32m4(vfacc1, output_max_less_zero_point, vl);
    vint16m2_t vout1 = __riscv_vfncvt_x_f_w_i16m2(vfacc1, vl);
    vout1 = __riscv_vadd_vx_i16m2(vout1, (int16_t) output_zero_point, vl);
    vint8m1_t vout81 = __riscv_vncvt_x_x_w_i8m1(vout1, vl);
    __riscv_vse8_v_i8m1(c1, vout81, vl);
    c1 += vl;

    vfloat32m4_t vfacc2 = __riscv_vfcvt_f_x_v_f32m4(vacc2, vl);
    vfacc2 = __riscv_vfmul_vv_f32m4(vfacc2, vscale, vl);
    vfacc2 = __riscv_vfmax_vf_f32m4(vfacc2, output_min_less_zero_point, vl);
    vfacc2 = __riscv_vfmin_vf_f32m4(vfacc2, output_max_less_zero_point, vl);
    vint16m2_t vout2 = __riscv_vfncvt_x_f_w_i16m2(vfacc2, vl);
    vout2 = __riscv_vadd_vx_i16m2(vout2, (int16_t) output_zero_point, vl);
    vint8m1_t vout82 = __riscv_vncvt_x_x_w_i8m1(vout2, vl);
    __riscv_vse8_v_i8m1(c2, vout82, vl);
    c2 += vl;

    vfloat32m4_t vfacc3 = __riscv_vfcvt_f_x_v_f32m4(vacc3, vl);
    vfacc3 = __riscv_vfmul_vv_f32m4(vfacc3, vscale, vl);
    vfacc3 = __riscv_vfmax_vf_f32m4(vfacc3, output_min_less_zero_point, vl);
    vfacc3 = __riscv_vfmin_vf_f32m4(vfacc3, output_max_less_zero_point, vl);
    vint16m2_t vout3 = __riscv_vfncvt_x_f_w_i16m2(vfacc3, vl);
    vout3 = __riscv_vadd_vx_i16m2(vout3, (int16_t) output_zero_point, vl);
    vint8m1_t vout83 = __riscv_vncvt_x_x_w_i8m1(vout3, vl);
    __riscv_vse8_v_i8m1(c3, vout83, vl);
    c3 += vl;

    vfloat32m4_t vfacc4 = __riscv_vfcvt_f_x_v_f32m4(vacc4, vl);
    vfacc4 = __riscv_vfmul_vv_f32m4(vfacc4, vscale, vl);
    vfacc4 = __riscv_vfmax_vf_f32m4(vfacc4, output_min_less_zero_point, vl);
    vfacc4 = __riscv_vfmin_vf_f32m4(vfacc4, output_max_less_zero_point, vl);
    vint16m2_t vout4 = __riscv_vfncvt_x_f_w_i16m2(vfacc4, vl);
    vout4 = __riscv_vadd_vx_i16m2(vout4, (int16_t) output_zero_point, vl);
    vint8m1_t vout84 = __riscv_vncvt_x_x_w_i8m1(vout4, vl);
    __riscv_vse8_v_i8m1(c4, vout84, vl);
    c4 += vl;

    vfloat32m4_t vfacc5 = __riscv_vfcvt_f_x_v_f32m4(vacc5, vl);
    vfacc5 = __riscv_vfmul_vv_f32m4(vfacc5, vscale, vl);
    vfacc5 = __riscv_vfmax_vf_f32m4(vfacc5, output_min_less_zero_point, vl);
    vfacc5 = __riscv_vfmin_vf_f32m4(vfacc5, output_max_less_zero_point, vl);
    vint16m2_t vout5 = __riscv_vfncvt_x_f_w_i16m2(vfacc5, vl);
    vout5 = __riscv_vadd_vx_i16m2(vout5, (int16_t) output_zero_point, vl);
    vint8m1_t vout85 = __riscv_vncvt_x_x_w_i8m1(vout5, vl);
    __riscv_vse8_v_i8m1(c5, vout85, vl);
    c5 += vl;

    vfloat32m4_t vfacc6 = __riscv_vfcvt_f_x_v_f32m4(vacc6, vl);
    vfacc6 = __riscv_vfmul_vv_f32m4(vfacc6, vscale, vl);
    vfacc6 = __riscv_vfmax_vf_f32m4(vfacc6, output_min_less_zero_point, vl);
    vfacc6 = __riscv_vfmin_vf_f32m4(vfacc6, output_max_less_zero_point, vl);
    vint16m2_t vout6 = __riscv_vfncvt_x_f_w_i16m2(vfacc6, vl);
    vout6 = __riscv_vadd_vx_i16m2(vout6, (int16_t) output_zero_point, vl);
    vint8m1_t vout86 = __riscv_vncvt_x_x_w_i8m1(vout6, vl);
    __riscv_vse8_v_i8m1(c6, vout86, vl);
    c6 += vl;

    scale_ptr += vl;
    start_weights += vl;
    w_new += vl;
  } while (nc != 0);
}

void qgemm_i8_i32_1xm4_int32bias(
    size_t mr,        // number of rows to process (1..7)
    size_t nc,        // number of columns to process
    size_t kc,        // number of "channels" or "inner dimension"
    const int8_t* a,  // input matrix A
    size_t a_stride,           // byte stride between consecutive rows of A
    const void* w,  // weights (B)
    int8_t* c,       // output matrix C (int32)
    size_t cm_stride,          // byte stride between consecutive rows of C
    size_t cn_stride,           // byte stride between consecutive columns-blocks of C
    requantization_params_t requant_params 
)
{

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* start_weights = (const int8_t*) ((const int32_t*)w + nc);
  // printf("first element: %d \n", start_weights[0]);

  // float scale = qp_output.scale;
  const int32_t output_min_less_zero_point = -128 - requant_params.zero_point;
  const int32_t output_max_less_zero_point = 127 - requant_params.zero_point;
  const int32_t output_zero_point = requant_params.zero_point;

  size_t nr = nc;
  size_t vl = nr;
  const int32_t* w_new = (const int32_t*) w;
  size_t scale_ptr = 0;
  
  do {
    vl = __riscv_vsetvl_e32m4(nc);
    nc -= vl;
    vint32m4_t vacc0 = __riscv_vle32_v_i32m4(w_new, vl);
    w = (const void*) start_weights;

    size_t k = kc;
    do {
      const int8_t va0 = *a0++;
      vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1((const int8_t*) w, vl), vl);
      w = (const int8_t*) w + nr;
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl); 

      k -= 1;
    } while (k != 0);

    a0 -= kc;

    vfloat32m4_t vscale = __riscv_vle32_v_f32m4(requant_params.scale + scale_ptr, vl);
    vfloat32m4_t vfacc0 = __riscv_vfcvt_f_x_v_f32m4(vacc0, vl);
    vfacc0 = __riscv_vfmul_vv_f32m4(vfacc0, vscale, vl);
    vfacc0 = __riscv_vfmax_vf_f32m4(vfacc0, output_min_less_zero_point, vl);
    vfacc0 = __riscv_vfmin_vf_f32m4(vfacc0, output_max_less_zero_point, vl);
    vint16m2_t vout0 = __riscv_vfncvt_x_f_w_i16m2(vfacc0, vl);
    vout0 = __riscv_vadd_vx_i16m2(vout0, (int16_t) output_zero_point, vl);
    vint8m1_t vout80 = __riscv_vncvt_x_x_w_i8m1(vout0, vl);
    __riscv_vse8_v_i8m1(c0, vout80, vl);
    c0 += vl;
    
    scale_ptr += vl;
    start_weights += vl;
    w_new += vl;
  } while (nc != 0);
}

void int8_qgemm_int32bias(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const void* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params)
{
    const size_t kc_bytes = K;
    const size_t a_stride_bytes = a_row_stride;
    const size_t cm_stride_bytes = c_row_stride;
    const size_t cn_stride_bytes = c_col_stride;

    size_t row = 0;
    while (row < M) {
        size_t rows_left = M - row;

        if (rows_left >= 7) {
            qgemm_i8_i32_7xm4_int32bias(
                7,
                N,
                kc_bytes,
                A + row * a_row_stride,
                a_stride_bytes,
                B,
                C + row * c_row_stride,
                cm_stride_bytes,
                cn_stride_bytes,
                requant_params
            );
            row += 7;
        } else {
            qgemm_i8_i32_1xm4_int32bias(
                1,
                N,
                kc_bytes,
                A + row * a_row_stride,
                a_stride_bytes,
                B,
                C + row * c_row_stride,
                cm_stride_bytes,
                cn_stride_bytes,
                requant_params
            );
            row += 1;
        }
    }
}
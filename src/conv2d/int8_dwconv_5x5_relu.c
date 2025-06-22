#include "conv2d.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "util.h"
#include <riscv_vector.h> 

void int8_vec_conv2_5x5_relu(size_t rows, size_t cols, size_t a_stride, size_t b_stride,
    const int8_t *k, const int8_t *a, int8_t *b, int32_t bias, int32_t zero_point, float scale)
{
    int8_t k0  = k[ 0], k1  = k[ 1], k2  = k[ 2], k3  = k[ 3], k4  = k[ 4];
    int8_t k5  = k[ 5], k6  = k[ 6], k7  = k[ 7], k8  = k[ 8], k9  = k[ 9];
    int8_t k10 = k[10], k11 = k[11], k12 = k[12], k13 = k[13], k14 = k[14];
    int8_t k15 = k[15], k16 = k[16], k17 = k[17], k18 = k[18], k19 = k[19];
    int8_t k20 = k[20], k21 = k[21], k22 = k[22], k23 = k[23], k24 = k[24];

    const int8_t* ap; 
    const int8_t* ap_4; 
    const int8_t* ap_8; 
    const int8_t* ap_12; 
    const int8_t* ap_16; 
    int8_t* bp;
    size_t vl;
    int row_count;
    rows -= 4;

    float vout_min_minus_zp = 0;
    float vout_max_minus_zp = 127 - zero_point;

    do {
    ap = a;
    ap_4 = ap + 1; 
    ap_8 = ap + 2; 
    ap_12 = ap + 3; 
    ap_16 = ap + 4; 
    bp = b;
    vl = __riscv_vsetvl_e32m8(cols);
    row_count = (int) rows;

    register vint16m2_t vload0; 
    register vint16m2_t vload1; 
    register vint16m2_t vload2;
    register vint16m2_t vload3; 
    register vint16m2_t vload4;
    
    register vint32m4_t vrow0; 
    register vint32m4_t vrow1;
    register vint32m4_t vrow2; 
    register vint32m4_t vrow3;
    register vint32m4_t vbias = __riscv_vmv_v_x_i32m4(bias, vl);

    vfloat32m4_t vfacc; vint16m2_t vout16; vint8m1_t vout8; // temporary vectors for the quantization process before storing 
    

    // PROLOGUE
    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);

    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl); 

    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);

    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload3, vl);

    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload4, vl);

    ap += a_stride;
    ap_4 = ap + 1; 
    ap_8 = ap + 2; 
    ap_12 = ap + 3; 
    ap_16 = ap + 4;

    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload0, vl);

    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload1, vl); 

    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload2, vl);

    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload3, vl);

    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k9, vload4, vl);

    ap += a_stride;

    vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    ap_4 = ap + 1;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k10, vload0, vl);

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    ap_8 = ap + 2;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k11, vload1, vl);

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    ap_12 = ap + 3;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k12, vload2, vl);

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload3, vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    ap_16 = ap + 4;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k13, vload3, vl);

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload4, vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k14, vload4, vl);

    ap += a_stride;

    vrow2 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload0, vl);
    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    ap_4 = ap + 1;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k15, vload0, vl);

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k1, vload1, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload1, vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    ap_8 = ap + 2;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k16, vload1, vl);

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k2, vload2, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload2, vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    ap_12 = ap + 3;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k17, vload2, vl);

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k3, vload3, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload3, vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    ap_16 = ap + 4;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k18, vload3, vl);

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k4, vload4, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k9, vload4, vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k19, vload4, vl);

    ap += a_stride;

    vrow3 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k10, vload0, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k5, vload0, vl);
    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    ap_4 = ap + 1;

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k1, vload1, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k6, vload1, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k11, vload1, vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    ap_8 = ap + 2;

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k2, vload2, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k7, vload2, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k12, vload2, vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    ap_12 = ap + 3;

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k3, vload3, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k8, vload3, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k13, vload3, vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    ap_16 = ap + 4;

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k4, vload4, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k9, vload4, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k14, vload4, vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);

    // MAIN LOOP
    do {
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k20, vload0, vl);
    ap += a_stride;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k21, vload1, vl);
    ap_4 = ap + 1;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k22, vload2, vl);
    ap_8 = ap + 2;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k23, vload3, vl);
    ap_12 = ap + 3;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k24, vload4, vl);
    ap_16 = ap + 4;

    vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
    vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
    vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
    vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
    vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
    vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
    vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
    __riscv_vse8_v_i8m1(bp, vout8, vl);

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k15, vload0, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k16, vload1, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k17, vload2, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k18, vload3, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k19, vload4, vl);

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k10, vload0, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k11, vload1, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k12, vload2, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k13, vload3, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k14, vload4, vl);

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k5, vload0, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k6, vload1, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k7, vload2, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k8, vload3, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k9, vload4, vl);

    vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    bp += b_stride;
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload3, vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload4, vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);


    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload0, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload1, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload2, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload3, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k9, vload4, vl);

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k20, vload0, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k21, vload1, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k22, vload2, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k23, vload3, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k24, vload4, vl);

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k15, vload0, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k16, vload1, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k17, vload2, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k18, vload3, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k19, vload4, vl);

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k10, vload0, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k11, vload1, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k12, vload2, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k13, vload3, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k14, vload4, vl);

    ap += a_stride;
    ap_4 = ap + 1;
    ap_8 = ap + 2;
    ap_12 = ap + 3;
    ap_16 = ap + 4;

    vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
    vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
    vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
    vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
    vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
    vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
    vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
    __riscv_vse8_v_i8m1(bp, vout8, vl);

    vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    bp += b_stride;
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload3, vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload4, vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);

    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k10, vload0, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k11, vload1, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k12, vload2, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k13, vload3, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k14, vload4, vl);

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload0, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload1, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload2, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload3, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k9, vload4, vl);

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k20, vload0, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k21, vload1, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k22, vload2, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k23, vload3, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k24, vload4, vl);

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k15, vload0, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k16, vload1, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k17, vload2, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k18, vload3, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k19, vload4, vl);

    ap += a_stride;
    ap_4 = ap + 1;
    ap_8 = ap + 2;
    ap_12 = ap + 3;
    ap_16 = ap + 4;

    vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow2, vl);
    vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
    vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
    vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
    vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
    vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
    vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
    __riscv_vse8_v_i8m1(bp, vout8, vl);

    vrow2 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k1, vload1, vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    bp += b_stride;
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k2, vload2, vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k3, vload3, vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k4, vload4, vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);

    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k15, vload0, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k16, vload1, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k17, vload2, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k18, vload3, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k19, vload4, vl);

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k10, vload0, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k11, vload1, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k12, vload2, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k13, vload3, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k14, vload4, vl);

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k5, vload0, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k6, vload1, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k7, vload2, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k8, vload3, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k9, vload4, vl);

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k20, vload0, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k21, vload1, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k22, vload2, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k23, vload3, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k24, vload4, vl);

    ap += a_stride;
    ap_4 = ap + 1;
    ap_8 = ap + 2;
    ap_12 = ap + 3;
    ap_16 = ap + 4;


    vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow3, vl);
    vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
    vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
    vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
    vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
    vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
    vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
    __riscv_vse8_v_i8m1(bp, vout8, vl);


    vrow3 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k1, vload1, vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    bp += b_stride;
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k2, vload2, vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k3, vload3, vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k4, vload4, vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);

    row_count -= 4;

    } while (row_count > 0);

    // EPILOG
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k20, vload0, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k21, vload1, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k22, vload2, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k23, vload3, vl);
    vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k24, vload4, vl);
    ap += a_stride;

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k15, vload0, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k16, vload1, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k17, vload2, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k18, vload3, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k19, vload4, vl);
    ap_4 = ap + 1;

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k10, vload0, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k11, vload1, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k12, vload2, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k13, vload3, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k14, vload4, vl);
    ap_8 = ap + 2;

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k5, vload0, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k6, vload1, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k7, vload2, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k8, vload3, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k9, vload4, vl);
    ap_12 = ap + 3;
    ap_16 = ap + 4;


    vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
    vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
    vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
    vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
    vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
    vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
    vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
    __riscv_vse8_v_i8m1(bp, vout8, vl);
    
    bp += b_stride;

    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);

    ap += a_stride;
    ap_4 = ap + 1;

    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k20, vload0, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k21, vload1, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k22, vload2, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k23, vload3, vl);
    vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k24, vload4, vl);
    ap_8 = ap + 2;

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k15, vload0, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k16, vload1, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k17, vload2, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k18, vload3, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k19, vload4, vl);
    ap_12 = ap + 3;

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k10, vload0, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k11, vload1, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k12, vload2, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k13, vload3, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k14, vload4, vl);
    ap_16 = ap + 4;


    vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
    vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
    vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
    vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
    vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
    vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
    vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
    __riscv_vse8_v_i8m1(bp, vout8, vl);

    bp += b_stride;

    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);

    ap += a_stride;
    ap_4 = ap + 1;

    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k20, vload0, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k21, vload1, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k22, vload2, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k23, vload3, vl);
    vrow2 = __riscv_vwmacc_vx_i32m4(vrow2, k24, vload4, vl);
    ap_8 = ap + 2;

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k15, vload0, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k16, vload1, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k17, vload2, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k18, vload3, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k19, vload4, vl);
    ap_12 = ap + 3;
    ap_16 = ap + 4;


    vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow2, vl);
    vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
    vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
    vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
    vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
    vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
    vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
    __riscv_vse8_v_i8m1(bp, vout8, vl);

    bp += b_stride;

    vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
    vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_4, vl), vl);
    vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_8, vl), vl);
    vload3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_12, vl), vl);
    vload4 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_16, vl), vl);

    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k20, vload0, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k21, vload1, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k22, vload2, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k23, vload3, vl);
    vrow3 = __riscv_vwmacc_vx_i32m4(vrow3, k24, vload4, vl);


    vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow3, vl);
    vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
    vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
    vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
    vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
    vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
    vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
    __riscv_vse8_v_i8m1(bp, vout8, vl);

    a += vl;
    b += vl;
    cols -= vl;

    } while (cols != 0);
}

void dwconv_5x5_int8_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output, 
    requantization_params_t requant_params
) {
    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (rows + 4) * a_stride;
    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = rows * b_stride;

    const int8_t* w = (const int8_t*) ((const int32_t*) weights + channels);

    // copy_int8_to_tcm(input, a_channel_size);

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // int8_t bias = weights[ch];
        // The 3x3 kernel for this channel is stored starting at weights[channels] with 9 int8_ts per channel.
        const int8_t *k_ch = w + ch * 25;

        const int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;

        // Compute the convolution for this channel.
        int8_vec_conv2_5x5_relu(rows, cols, a_stride, b_stride, k_ch, a_ch, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
    }
}
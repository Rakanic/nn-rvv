#include "ops/conv2D/conv2D.h"
#include "ops/padding/padding.h"

#include <riscv_vector.h> 
#include <stdint.h>
#include "stdio.h"
#include "string.h"

void vec_conv_c_code(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    register size_t row_check = rows;
    row_check -= 2;
    register size_t row_count;
    register int rows_odd = rows & 1;

    register vint16m2_t vload0; 
    register vint16m2_t vload1; 
    register vint16m2_t vload2;
    register vint32m4_t vrow0; 
    register vint32m4_t vrow1; 
    vfloat32m4_t vfacc;
    register vint32m4_t vbias;
    vint16m2_t vout16;
    vint8m1_t vout8;

    register int16_t k0 = (int16_t) k[0]; register int16_t k1 = (int16_t) k[1]; register int16_t k2 = (int16_t) k[2];
    register int16_t k3 = (int16_t) k[3]; register int16_t k4 = (int16_t) k[4]; register int16_t k5 = (int16_t) k[5];
    register int16_t k6 = (int16_t) k[6]; register int16_t k7 = (int16_t) k[7]; register int16_t k8 = (int16_t) k[8];

    float vout_min_minus_zp = -128 - zero_point; 
    float vout_max_minus_zp = 127 - zero_point;  

    const int8_t* ap; const int8_t* ap_1; const int8_t* ap_2;
    int8_t* bp; 

    do {
        register size_t vl = __riscv_vsetvl_e32m4(cols);
        ap = a; ap_1 = ap + 1; ap_2 = ap + 2;
        bp = b; 
        row_count = row_check; 
        vbias = __riscv_vmv_v_x_i32m4(bias, vl);

        vl = __riscv_vsetvl_e8m1(cols);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl); ap_2 += a_stride;


        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl); ap += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl); ap_1 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl); ap_2 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);

        // printf("cols: %d \n", cols);

        do {
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl); ap_2 += a_stride;

            // print_vint32_m4(vrow0, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl); ap_2 += a_stride;             
            
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl);

            // print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);

            row_count -= 2;
        } while (row_count > 0 && (row_count != SIZE_MAX));

        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl);

        // print_vint32_m4(vrow0, vl);
        
        vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
        vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
        vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
        vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
        vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
        vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
        vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

        if (!rows_odd) {
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl); ap += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl); ap_1 += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl); ap_2 += a_stride;

            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            // print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl);

        }
        a += vl;
        b += vl;
        cols -= vl;

    } while (cols != 0);
    
}

void vec_conv_c_code_relu(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    register size_t row_check = rows;
    row_check -= 2;
    register size_t row_count;
    register int rows_odd = rows & 1;

    register vint16m2_t vload0; 
    register vint16m2_t vload1; 
    register vint16m2_t vload2;
    register vint32m4_t vrow0; 
    register vint32m4_t vrow1; 
    vfloat32m4_t vfacc;
    register vint32m4_t vbias;
    vint16m2_t vout16;
    vint8m1_t vout8;

    register int16_t k0 = (int16_t) k[0]; register int16_t k1 = (int16_t) k[1]; register int16_t k2 = (int16_t) k[2];
    register int16_t k3 = (int16_t) k[3]; register int16_t k4 = (int16_t) k[4]; register int16_t k5 = (int16_t) k[5];
    register int16_t k6 = (int16_t) k[6]; register int16_t k7 = (int16_t) k[7]; register int16_t k8 = (int16_t) k[8];

    float vout_min_minus_zp = 0; 
    float vout_max_minus_zp = 127 - zero_point;  

    const int8_t* ap; const int8_t* ap_1; const int8_t* ap_2;
    int8_t* bp; 


    do {
        register size_t vl = __riscv_vsetvl_e32m4(cols);
        ap = a; ap_1 = ap + 1; ap_2 = ap + 2;
        bp = b; 
        row_count = row_check; 
        vbias = __riscv_vmv_v_x_i32m4(bias, vl);

        vl = __riscv_vsetvl_e8m1(cols);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl); ap_2 += a_stride;


        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl); ap += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl); ap_1 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl); ap_2 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);

        // printf("cols: %d \n", cols);

        do {
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl); ap_2 += a_stride;

            // print_vint32_m4(vrow0, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl); ap_2 += a_stride;             
            
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl);

            // print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);

            row_count -= 2;
        } while (row_count > 0 && (row_count != SIZE_MAX));

        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl);

        // print_vint32_m4(vrow0, vl);
        
        vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
        vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
        vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
        vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
        vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
        vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
        vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

        if (!rows_odd) {
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl); ap += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl); ap_1 += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl); ap_2 += a_stride;

            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            // print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl);

        }
        a += vl;
        b += vl;
        cols -= vl;

    } while (cols != 0);
}

void vec_conv_c_code_stride2(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    register size_t row_check = rows;
    row_check -= 1;
    register size_t row_count;
    register int rows_odd = rows & 1;

    register vint16m2_t vload0; 
    register vint16m2_t vload1; 
    register vint16m2_t vload2;
    register vint32m4_t vrow0; 
    register vint32m4_t vrow1; 
    vfloat32m4_t vfacc;
    register vint32m4_t vbias;
    vint16m2_t vout16;
    vint8m1_t vout8;

    register int16_t k0 = (int16_t) k[0]; register int16_t k1 = (int16_t) k[1]; register int16_t k2 = (int16_t) k[2];
    register int16_t k3 = (int16_t) k[3]; register int16_t k4 = (int16_t) k[4]; register int16_t k5 = (int16_t) k[5];
    register int16_t k6 = (int16_t) k[6]; register int16_t k7 = (int16_t) k[7]; register int16_t k8 = (int16_t) k[8];

    float vout_min_minus_zp = -128 - zero_point; 
    float vout_max_minus_zp = 127 - zero_point;  

    const int8_t* ap; const int8_t* ap_1; const int8_t* ap_2;
    int8_t* bp; 

    do {
        register size_t vl = __riscv_vsetvl_e32m4(cols);
        ap = a; ap_1 = ap + 1; ap_2 = ap + 2;
        bp = b; 
        row_count = row_check; 
        vbias = __riscv_vmv_v_x_i32m4(bias, vl);

        vl = __riscv_vsetvl_e8m1(cols);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl); ap_2 += a_stride;


        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl); ap_2 += a_stride;

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);

        // printf("cols: %d \n", cols);

        do {
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl); ap_2 += a_stride;
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl); ap_2 += a_stride;             
            
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl);

            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);

            row_count -= 1;
        } while (row_count != 0);

        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl);
        
        vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
        vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
        vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
        vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
        vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
        vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
        vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

        a += 2*vl;
        b += vl;
        cols = cols - vl;

    } while (cols != 0);
    
}

void vec_conv_c_code_stride2_relu(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    register size_t row_check = rows;
    row_check -= 1;
    register size_t row_count;
    register int rows_odd = rows & 1;

    register vint16m2_t vload0; 
    register vint16m2_t vload1; 
    register vint16m2_t vload2;
    register vint32m4_t vrow0; 
    register vint32m4_t vrow1; 
    vfloat32m4_t vfacc;
    register vint32m4_t vbias;
    vint16m2_t vout16;
    vint8m1_t vout8;

    register int16_t k0 = (int16_t) k[0]; register int16_t k1 = (int16_t) k[1]; register int16_t k2 = (int16_t) k[2];
    register int16_t k3 = (int16_t) k[3]; register int16_t k4 = (int16_t) k[4]; register int16_t k5 = (int16_t) k[5];
    register int16_t k6 = (int16_t) k[6]; register int16_t k7 = (int16_t) k[7]; register int16_t k8 = (int16_t) k[8];

    float vout_min_minus_zp = 0; 
    float vout_max_minus_zp = 127 - zero_point;  

    const int8_t* ap; const int8_t* ap_1; const int8_t* ap_2;
    int8_t* bp; 

    do {
        register size_t vl = __riscv_vsetvl_e32m4(cols);
        ap = a; ap_1 = ap + 1; ap_2 = ap + 2;
        bp = b; 
        row_count = row_check; 
        vbias = __riscv_vmv_v_x_i32m4(bias, vl);

        vl = __riscv_vsetvl_e8m1(cols);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl); ap_2 += a_stride;


        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl); ap_2 += a_stride;

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);

        // printf("cols: %d \n", cols);

        do {
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl); ap_2 += a_stride;
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl); ap_2 += a_stride;             
            
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl);

            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);

            row_count -= 1;
        } while (row_count != 0);

        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl);

        // print_vint32_m4(vrow0, vl);
        
        vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
        vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
        vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
        vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
        vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
        vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
        vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

        a += 2*vl;
        b += vl;
        cols -= vl;

    } while (cols != 0);
}

void print_int8_matrix_(int8_t *arr, size_t rows, size_t cols)
{
    printf("matrix: \n");

    for (size_t r = 0; r < rows; r++) {
        printf("  [");
        for (size_t c = 0; c < cols; c++) {
            int idx = r * cols + c;
            printf("%4d", arr[idx]);
            if (c + 1 < cols) printf(", ");
        }
        printf("]\n");
    }
}

void dwconv_3x3_int8_VCO(
    size_t input_rows, size_t input_cols,
    size_t stride, size_t padding,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
) {
    const int8_t *in_conv; 
    int8_t in_conv_buf [input_rows + 2*padding][input_cols + 2*padding];
    size_t rows = (input_rows + 2*padding - 3)/stride + 1;
    size_t cols = (input_cols + 2*padding - 3)/stride + 1;

    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (input_rows) * a_stride;
    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = rows * b_stride;
    const int8_t* w = (const int8_t*) ((const int32_t*) weights + channels);

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 3x3 kernel for this channel is stored starting at weights[channels] with 9 floats per channel.
        const int8_t *k_ch = w + ch * 9;
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;

        // print_int8_matrix_(k_ch, 1, 9);

        if (padding) {
            pad_input_channel(input_cols, input_rows, padding, padding, (const int8_t*) a_ch, (int8_t*) in_conv_buf);
            in_conv = (int8_t*) in_conv_buf; 
        } else {
            in_conv = a_ch;
        }

        if (stride == 1) {
            vec_conv_c_code(rows, cols, a_stride + 2*padding, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        } else if (stride == 2) {
            vec_conv_c_code_stride2(rows, cols, a_stride + 2*padding, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        }

        // print_int8_matrix_(b_ch, rows, cols);
    }
}

void dwconv_3x3_int8_VCO_relu(
    size_t input_rows, size_t input_cols,
    size_t stride, size_t padding,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
) {
    const int8_t *in_conv; 
    int8_t in_conv_buf [input_rows + 2*padding][input_cols + 2*padding];
    size_t rows = (input_rows + 2*padding - 3)/stride + 1;
    size_t cols = (input_cols + 2*padding - 3)/stride + 1;

    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (input_rows) * a_stride;
    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = rows * b_stride;
    const int8_t* w = (const int8_t*) ((const int32_t*) weights + channels);

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 3x3 kernel for this channel is stored starting at weights[channels] with 9 floats per channel.
        const int8_t *k_ch = w + ch * 9;
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;

        if (padding) {
            pad_input_channel(input_cols, input_rows, padding, padding, (const int8_t*) a_ch, (int8_t*) in_conv_buf);
            in_conv = (int8_t*) in_conv_buf; 
        } else {
            in_conv = a_ch;
        }

        if (stride == 1) {
            vec_conv_c_code_relu(rows, cols, a_stride + 2*padding, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        } else if (stride == 2) {
            vec_conv_c_code_stride2_relu(rows, cols, a_stride + 2*padding, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        }
    }
}
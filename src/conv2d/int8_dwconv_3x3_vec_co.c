#include "conv2d.h"
#include "lib_kernels.h"

#include <riscv_vector.h> 
#include <stdint.h>
#include "stdio.h"
#include "string.h"
#define MMIO_BASE   0x08808000
#define CACHELINE           64

void print_vint32_m4(vint32m4_t vec, size_t n) {
    // Configure VL (vector length) for 32-bit elements, LMUL=4
    size_t vl = __riscv_vsetvl_e32m4(n);

    // Temporary buffer to hold vector contents (C99 VLA)
    int32_t buffer[vl];

    // Store vector elements into the buffer
    __riscv_vse32_v_i32m4(buffer, vec, vl);

    // Print each element
    for (size_t i = 0; i < vl; ++i) {
        printf("%d ", buffer[i]);
    }
    printf("\n");
}

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

    // printf("Bias: %d \n", bias);
    // printf("Kernel: \n");
    // for (int i=0; i < (3); i ++) {
    //   for (int j=0; j < (3); j++) {
    //         printf("%d ", k[i*(3) + j]);
    //   }
    //     printf("\n");
    // }

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

            print_vint32_m4(vrow0, vl);
            
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

            print_vint32_m4(vrow1, vl);
            
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
        } while (row_count != 0);

        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl);

        print_vint32_m4(vrow0, vl);
        
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

            print_vint32_m4(vrow1, vl);
            
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

void requantize_2D1(
    size_t size, 
    int32_t bias,
    int16_t* input, 
    int8_t* output, 
    float scale, 
    int32_t zero_point
) 
{
    register vfloat32m8_t vfacc0;
    register vint32m8_t vacc0;
    register vint16m4_t vout0;
    vint8m2_t vout80;

    const int32_t output_min_less_zero_point = -128 - zero_point;
    const int32_t output_max_less_zero_point = 127 - zero_point;


    do {
        register size_t vl = __riscv_vsetvl_e16m4(size);
        vacc0 = __riscv_vwcvt_x_x_v_i32m8(__riscv_vle16_v_i16m4(input, vl), vl);
        vacc0 = __riscv_vadd_vx_i32m8(vacc0, bias, vl);
        vfacc0 = __riscv_vfcvt_f_x_v_f32m8(vacc0, vl);
        vfacc0 = __riscv_vfmul_vf_f32m8(vfacc0, scale, vl);
        vfacc0 = __riscv_vfmax_vf_f32m8(vfacc0, output_min_less_zero_point, vl);
        vfacc0 = __riscv_vfmin_vf_f32m8(vfacc0, output_max_less_zero_point, vl);
        vout0 = __riscv_vfncvt_x_f_w_i16m4(vfacc0, vl);
        vout0 = __riscv_vadd_vx_i16m4(vout0, (int32_t) zero_point, vl);
        vout80 = __riscv_vncvt_x_x_w_i8m2(vout0, vl);
        __riscv_vse8_v_i8m2(output, vout80, vl);

        input += vl;
        output += vl;
        size -= vl;
    } while (size != 0);
}

// TODO: CALL CONV2D accelerator PROPPERLY
void dwconv_3x3_int8_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
) {
    __attribute__((aligned(CACHELINE))) int16_t out_conv[rows][cols];
    __attribute__((aligned(CACHELINE))) int8_t in_conv[rows+2][cols+2];

    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (rows + 2) * a_stride;

    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = rows * b_stride;
    const int8_t* w = (const int8_t*) ((const int32_t*) weights + channels);

    volatile uint64_t* STATUS_PTR = (volatile uint64_t *)(MMIO_BASE + 0x00);
    volatile uint8_t* READY_PTR = (volatile uint8_t *)(MMIO_BASE + 0x08);
    volatile uint64_t* SRC_ADDR_PTR = (volatile uint64_t *)(MMIO_BASE + 0x10);
    volatile uint64_t* DEST_ADDR_PTR = (volatile uint64_t *)(MMIO_BASE + 0x20);
    volatile uint64_t* INPUT_H_PTR = (volatile uint64_t *)(MMIO_BASE + 0x40);
    volatile uint64_t* INPUT_W_PTR = (volatile uint64_t *)(MMIO_BASE + 0x60);
    // volatile uint64_t* KERNEL_PTR = (volatile uint64_t *)(MMIO_BASE + 0x70);
    volatile uint8_t* KERNEL_SIZE_PTR = (volatile uint8_t *)(MMIO_BASE + 0x90);
    volatile uint8_t* RELU_PTR = (volatile uint8_t*) (MMIO_BASE + 0x98);
    volatile uint8_t* STRIDE_PTR = (volatile uint8_t*) (MMIO_BASE + 0xA0);

    *RELU_PTR = 0;
    *STRIDE_PTR = 1;
    *INPUT_H_PTR = rows + 2;
    *INPUT_W_PTR = cols + 2;
    *KERNEL_SIZE_PTR = 3;

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 3x3 kernel for this channel is stored starting at weights[channels] with 9 floats per channel.
        const int8_t *k_ch = w + ch * 9;
        int32_t bias = ((const int32_t*) weights)[ch];
        // alex here
        // memcpy((void*)(uintptr_t)KERNEL_PTR, w + ch*9, 9);
        
        int8_t *a_ch = input + ch * a_channel_size;
        for (int k = 0; k < rows+2; k ++) {
            for (int l = 0; l < cols + 2; l ++) {
                in_conv[k][l] = a_ch[k * (rows+2) + l];
            }
        }
        // printf("Accelerator input! \n");
        // for (int i = 0; i < rows+2; i ++) {
        //     for (int j = 0; j < cols+2; j ++) {
        //         printf("%d ", in_conv[i][j]);
        //     }
        //     printf("\n");
        // }
        // memcpy(in_conv, a_ch, (rows + 2) * (cols + 2));

        volatile int8_t* KP = (volatile uint8_t*)(MMIO_BASE + 0x70);
        for (int i = 0; i < 9; i++) {
            KP[i] = k_ch[i];    // guaranteed bus write, in order
        }

        // printf("weights: \n");
        // for (int i = 0; i < 3; i++) {
        //     for (int j = 0; j < 3; j++) {
        //         printf("%d, ", k_ch[i*3 + j]);
        //     }
        //     printf("\n");
        // }

        *SRC_ADDR_PTR = (uint64_t) in_conv;
        *DEST_ADDR_PTR = (uint64_t) out_conv;

        int8_t *b_ch = output + ch * b_channel_size;
        
        // for (size_t r = 0; r < rows + 2; r++)
        //     memcpy(&in_conv[r][0], a_ch + r*a_stride, cols + 2);
    

        // printf("scale values: %d \n", (int) (1/requant_params.scale[ch]));
        *READY_PTR = 0;
        // puts("Waiting on convolution");
        while (*READY_PTR == 0) ;
        // alex end
        // Compute the convolution for this channel.
        printf("Accelerator output! \n");
        for (int i = 0; i < rows; i ++) {
            for (int j = 0; j < cols; j ++) {
                printf("%d ", out_conv[i][j]);
            }
            printf("\n");
        }
        printf("bias: %d \n", bias);

        printf("vector output \n");
        vec_conv_c_code(rows, cols, a_stride, b_stride, k_ch, a_ch, b_ch, bias, requant_params.zero_point, requant_params.scale[ch]);

        // vec_conv_3x3_int8(rows, cols, a_stride, b_stride, k_ch, a_ch, b_ch, bias, requant_params.zero_point, requant_params.scale[ch]);
        requantize_2D1((rows)*(cols), bias, (int16_t*) out_conv, b_ch, requant_params.scale[ch], requant_params.zero_point);

    }
}

void dwconv_3x3_int8_VCO1(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
) {
    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (rows + 2) * a_stride;
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

        printf("weights: \n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%d, ", k_ch[i*3 + j]);
            }
            printf("\n");
        }

        // printf("scale values: %d \n", (int) (1/requant_params.scale[ch]));

        // Compute the convolution for this channel.
        // printf("bias: %d \n", weights[ch]);
        vec_conv_c_code(rows, cols, a_stride, b_stride, k_ch, a_ch, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        // vec_conv_3x3_int8(rows, cols, a_stride, b_stride, k_ch, a_ch, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        
    }
}


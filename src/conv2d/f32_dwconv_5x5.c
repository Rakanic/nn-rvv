#include <stdint.h>
#include <riscv_vector.h>


void vec_conv2_5x5(size_t rows, size_t cols, size_t a_stride, size_t b_stride,
                  const float *k, const float *a, float *b)
{
    float k0  = k[ 0], k1  = k[ 1], k2  = k[ 2], k3  = k[ 3], k4  = k[ 4];
    float k5  = k[ 5], k6  = k[ 6], k7  = k[ 7], k8  = k[ 8], k9  = k[ 9];
    float k10 = k[10], k11 = k[11], k12 = k[12], k13 = k[13], k14 = k[14];
    float k15 = k[15], k16 = k[16], k17 = k[17], k18 = k[18], k19 = k[19];
    float k20 = k[20], k21 = k[21], k22 = k[22], k23 = k[23], k24 = k[24];

    const float* ap; 
    const float* ap_4; 
    const float* ap_8; 
    const float* ap_12; 
    const float* ap_16; 
    float* bp;
    size_t vl;
    int row_count;
    rows -= 4;
    
    do {
        ap = a;
        ap_4 = ap + 1; 
        ap_8 = ap + 2; 
        ap_12 = ap + 3; 
        ap_16 = ap + 4; 
        bp = b;
        vl = __riscv_vsetvl_e32m8(cols);
        row_count = (int) rows;

        // PROLOGUE
        vfloat32m8_t vload0 = __riscv_vle32_v_f32m8(ap, vl);
        vfloat32m8_t vrow0 = __riscv_vfmul_vf_f32m8(vload0, k0, vl);

        vfloat32m8_t vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k1, vload1, vl); 

        vfloat32m8_t vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k2, vload2, vl);

        vfloat32m8_t vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k3, vload3, vl);

        vfloat32m8_t vload4 = __riscv_vle32_v_f32m8(ap_16, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k4, vload4, vl);

        ap += a_stride;
        ap_4 = ap + 1; 
        ap_8 = ap + 2; 
        ap_12 = ap + 3; 
        ap_16 = ap + 4;

        vload0 = __riscv_vle32_v_f32m8(ap, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k5, vload0, vl);

        vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k6, vload1, vl); 

        vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k7, vload2, vl);

        vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k8, vload3, vl);

        vload4 = __riscv_vle32_v_f32m8(ap_16, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k9, vload4, vl);

        ap += a_stride;

        vfloat32m8_t vrow1 = __riscv_vfmul_vf_f32m8(vload0, k0, vl);
        vload0 = __riscv_vle32_v_f32m8(ap, vl);
        ap_4 = ap + 1;
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k10, vload0, vl);

        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k1, vload1, vl);
        vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
        ap_8 = ap + 2;
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k11, vload1, vl);

        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k2, vload2, vl);
        vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
        ap_12 = ap + 3;
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k12, vload2, vl);

        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k3, vload3, vl);
        vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
        ap_16 = ap + 4;
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k13, vload3, vl);

        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k4, vload4, vl);
        vload4 = __riscv_vle32_v_f32m8(ap_16, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k14, vload4, vl);

        ap += a_stride;

        vfloat32m8_t vrow2 = __riscv_vfmul_vf_f32m8(vload0, k0, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k5, vload0, vl);
        vload0 = __riscv_vle32_v_f32m8(ap, vl);
        ap_4 = ap + 1;
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k15, vload0, vl);

        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k1, vload1, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k6, vload1, vl);
        vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
        ap_8 = ap + 2;
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k16, vload1, vl);

        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k2, vload2, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k7, vload2, vl);
        vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
        ap_12 = ap + 3;
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k17, vload2, vl);

        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k3, vload3, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k8, vload3, vl);
        vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
        ap_16 = ap + 4;
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k18, vload3, vl);

        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k4, vload4, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k9, vload4, vl);
        vload4 = __riscv_vle32_v_f32m8(ap_16, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k19, vload4, vl);

        ap += a_stride;

        vfloat32m8_t vrow3 = __riscv_vfmul_vf_f32m8(vload0, k0, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k10, vload0, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k5, vload0, vl);
        vload0 = __riscv_vle32_v_f32m8(ap, vl);
        ap_4 = ap + 1;

        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k1, vload1, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k6, vload1, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k11, vload1, vl);
        vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
        ap_8 = ap + 2;

        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k2, vload2, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k7, vload2, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k12, vload2, vl);
        vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
        ap_12 = ap + 3;

        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k3, vload3, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k8, vload3, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k13, vload3, vl);
        vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
        ap_16 = ap + 4;

        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k4, vload4, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k9, vload4, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k14, vload4, vl);
        vload4 = __riscv_vle32_v_f32m8(ap_16, vl);

        // MAIN LOOP
        do {
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k20, vload0, vl);
            ap += a_stride;
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k21, vload1, vl);
            ap_4 = ap + 1;
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k22, vload2, vl);
            ap_8 = ap + 2;
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k23, vload3, vl);
            ap_12 = ap + 3;
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k24, vload4, vl);
            ap_16 = ap + 4;

            __riscv_vse32_v_f32m8(bp, vrow0, vl);

            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k15, vload0, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k16, vload1, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k17, vload2, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k18, vload3, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k19, vload4, vl);

            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k10, vload0, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k11, vload1, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k12, vload2, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k13, vload3, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k14, vload4, vl);

            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k5, vload0, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k6, vload1, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k7, vload2, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k8, vload3, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k9, vload4, vl);

            vrow0 = __riscv_vfmul_vf_f32m8(vload0, k0, vl);
            vload0 = __riscv_vle32_v_f32m8(ap, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k1, vload1, vl);
            vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
            bp += b_stride;
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k2, vload2, vl);
            vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k3, vload3, vl);
            vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k4, vload4, vl);
            vload4 = __riscv_vle32_v_f32m8(ap_16, vl);


            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k5, vload0, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k6, vload1, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k7, vload2, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k8, vload3, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k9, vload4, vl);

            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k20, vload0, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k21, vload1, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k22, vload2, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k23, vload3, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k24, vload4, vl);

            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k15, vload0, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k16, vload1, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k17, vload2, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k18, vload3, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k19, vload4, vl);

            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k10, vload0, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k11, vload1, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k12, vload2, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k13, vload3, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k14, vload4, vl);

            ap += a_stride;
            ap_4 = ap + 1;
            ap_8 = ap + 2;
            ap_12 = ap + 3;
            ap_16 = ap + 4;

            __riscv_vse32_v_f32m8(bp, vrow1, vl);

            vrow1 = __riscv_vfmul_vf_f32m8(vload0, k0, vl);
            vload0 = __riscv_vle32_v_f32m8(ap, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k1, vload1, vl);
            vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
            bp += b_stride;
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k2, vload2, vl);
            vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k3, vload3, vl);
            vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k4, vload4, vl);
            vload4 = __riscv_vle32_v_f32m8(ap_16, vl);

            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k10, vload0, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k11, vload1, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k12, vload2, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k13, vload3, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k14, vload4, vl);

            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k5, vload0, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k6, vload1, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k7, vload2, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k8, vload3, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k9, vload4, vl);

            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k20, vload0, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k21, vload1, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k22, vload2, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k23, vload3, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k24, vload4, vl);

            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k15, vload0, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k16, vload1, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k17, vload2, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k18, vload3, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k19, vload4, vl);

            ap += a_stride;
            ap_4 = ap + 1;
            ap_8 = ap + 2;
            ap_12 = ap + 3;
            ap_16 = ap + 4;

            __riscv_vse32_v_f32m8(bp, vrow2, vl);

            vrow2 = __riscv_vfmul_vf_f32m8(vload0, k0, vl);
            vload0 = __riscv_vle32_v_f32m8(ap, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k1, vload1, vl);
            vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
            bp += b_stride;
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k2, vload2, vl);
            vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k3, vload3, vl);
            vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k4, vload4, vl);
            vload4 = __riscv_vle32_v_f32m8(ap_16, vl);

            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k15, vload0, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k16, vload1, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k17, vload2, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k18, vload3, vl);
            vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k19, vload4, vl);

            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k10, vload0, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k11, vload1, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k12, vload2, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k13, vload3, vl);
            vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k14, vload4, vl);

            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k5, vload0, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k6, vload1, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k7, vload2, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k8, vload3, vl);
            vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k9, vload4, vl);

            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k20, vload0, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k21, vload1, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k22, vload2, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k23, vload3, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k24, vload4, vl);

            ap += a_stride;
            ap_4 = ap + 1;
            ap_8 = ap + 2;
            ap_12 = ap + 3;
            ap_16 = ap + 4;

            __riscv_vse32_v_f32m8(bp, vrow3, vl);

            vrow3 = __riscv_vfmul_vf_f32m8(vload0, k0, vl);
            vload0 = __riscv_vle32_v_f32m8(ap, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k1, vload1, vl);
            vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
            bp += b_stride;
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k2, vload2, vl);
            vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k3, vload3, vl);
            vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
            vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k4, vload4, vl);
            vload4 = __riscv_vle32_v_f32m8(ap_16, vl);

            row_count -= 4;

        } while (row_count > 0);

        // EPILOG
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k20, vload0, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k21, vload1, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k22, vload2, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k23, vload3, vl);
        vrow0 = __riscv_vfmacc_vf_f32m8(vrow0, k24, vload4, vl);
        ap += a_stride;

        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k15, vload0, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k16, vload1, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k17, vload2, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k18, vload3, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k19, vload4, vl);
        ap_4 = ap + 1;

        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k10, vload0, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k11, vload1, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k12, vload2, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k13, vload3, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k14, vload4, vl);
        ap_8 = ap + 2;

        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k5, vload0, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k6, vload1, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k7, vload2, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k8, vload3, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k9, vload4, vl);
        ap_12 = ap + 3;
        ap_16 = ap + 4;

        __riscv_vse32_v_f32m8(bp, vrow0, vl);
        bp += b_stride;

        vload0 = __riscv_vle32_v_f32m8(ap, vl);
        vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
        vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
        vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
        vload4 = __riscv_vle32_v_f32m8(ap_16, vl);

        ap += a_stride;
        ap_4 = ap + 1;
        
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k20, vload0, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k21, vload1, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k22, vload2, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k23, vload3, vl);
        vrow1 = __riscv_vfmacc_vf_f32m8(vrow1, k24, vload4, vl);
        ap_8 = ap + 2;

        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k15, vload0, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k16, vload1, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k17, vload2, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k18, vload3, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k19, vload4, vl);
        ap_12 = ap + 3;

        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k10, vload0, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k11, vload1, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k12, vload2, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k13, vload3, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k14, vload4, vl);
        ap_16 = ap + 4;

        __riscv_vse32_v_f32m8(bp, vrow1, vl);
        bp += b_stride;

        vload0 = __riscv_vle32_v_f32m8(ap, vl);
        vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
        vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
        vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
        vload4 = __riscv_vle32_v_f32m8(ap_16, vl);

        ap += a_stride;
        ap_4 = ap + 1;

        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k20, vload0, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k21, vload1, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k22, vload2, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k23, vload3, vl);
        vrow2 = __riscv_vfmacc_vf_f32m8(vrow2, k24, vload4, vl);
        ap_8 = ap + 2;

        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k15, vload0, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k16, vload1, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k17, vload2, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k18, vload3, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k19, vload4, vl);
        ap_12 = ap + 3;
        ap_16 = ap + 4;

        __riscv_vse32_v_f32m8(bp, vrow2, vl);
        bp += b_stride;

        vload0 = __riscv_vle32_v_f32m8(ap, vl);
        vload1 = __riscv_vle32_v_f32m8(ap_4, vl);
        vload2 = __riscv_vle32_v_f32m8(ap_8, vl);
        vload3 = __riscv_vle32_v_f32m8(ap_12, vl);
        vload4 = __riscv_vle32_v_f32m8(ap_16, vl);

        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k20, vload0, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k21, vload1, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k22, vload2, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k23, vload3, vl);
        vrow3 = __riscv_vfmacc_vf_f32m8(vrow3, k24, vload4, vl);

        __riscv_vse32_v_f32m8(bp, vrow3, vl);

        a += vl;
        b += vl;
        cols -= vl;
        
    } while (cols != 0);
}

void *vec_conv_5x5 (size_t, size_t, size_t, size_t, const float*, const float*, float*);

void dwconv_5x5_f32_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    const float *input, 
    float *output
) {
    size_t a_channel_size = (rows + 4) * a_stride;
    // Each channel's output is rows x b_stride (b_stride is the columns)
    size_t b_channel_size = rows * b_stride;

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 5x5 kernel for this channel is stored starting at weights[channels] with 25 floats per channel.
        const float *k_ch = weights + channels + ch * 25;

        const float *a_ch = input + ch * a_channel_size;
        float *b_ch = output + ch * b_channel_size;

        // Compute the convolution for this channel using the assembly version.
        vec_conv_5x5(rows, cols, a_stride, b_stride, k_ch, a_ch, b_ch);
    }
}

//
// Vectorized 2D 5x5 convolution (illustrative example)
// Follows same structure, logic, and pipeline approach as the 3x3 code.
//
    .text
    .balign 4

    .global vec_conv_5x5_relu
/*
 * Calling convention (same as original):
 *     a0: size_t rows
 *     a1: size_t cols
 *     a2: size_t a_stride   (in floats)
 *     a3: size_t b_stride   (in floats)
 *     a4: const float *k    (25 floats for the 5x5 kernel)
 *     a5: const float *a    (input image)
 *     a6: float *b          (output image)
 *     fa0: float bias       (bias for this output channel)
 */

#define rows         a0
#define cols         a1
#define a_stride     a2
#define b_stride     a3
#define k            a4
#define a            a5
#define b            a6
#define bias         fa0

// We need more pointers for loads at +0,+4,+8,+12,+16
#define ap           t0
#define bp           t1
#define ap_4         t5
#define ap_8         t6
// For +12 and +16, we will reuse s2 and s3 (since the code already saves/restores s0, s1):
#define ap_12        s2
#define ap_16        s3

#define vlen         t2
#define row_count    t3
#define VLEN_stride  t4

#define row_check    s0
#define rows_odd     s1

// 25 scalar registers for the 5×5 kernel:
#define k0  ft0
#define k1  ft1
#define k2  ft2
#define k3  ft3
#define k4  ft4
#define k5  ft5
#define k6  ft6
#define k7  ft7
#define k8  ft8
#define k9  ft9
// We will use f registers from f10..f14, f15..f19, f20..f24, etc. to hold the rest:
#define k10 fs0
#define k11 fa1
#define k12 fa2
#define k13 fa3
#define k14 fa4
#define k15 fa5
#define k16 fa6
#define k17 fa7
#define k18 ft10
#define k19 ft11
#define k20 fs1
#define k21 fs2
#define k22 fs3
#define k23 fs4
#define k24 fs5
#define zero fs6

// Vector registers for loading and partial sums
// Just as in the 3×3 code, we use v0..v4 for loading, and v16..v20 for partial sums
#define vload0       v0
#define vload1       v4
#define vload2       v8
#define vload3       v12
#define vload4       v2
#define vrow0        v16
#define vrow1        v20
#define vrow2        v24
#define vrow3        v28

// Increase frame size to save/restore s2, s3 as well
#define FRAMESIZE 88   // 32 was for s0,s1; add 16 more bytes for s2,s3

//----------------------------------------------------------------------------
// vec_conv_5x5
//----------------------------------------------------------------------------
vec_conv_5x5_relu:
    addi    sp, sp, -FRAMESIZE
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    fsd     fs0, 32(sp)
    fsd     fs1, 40(sp)
    fsd     fs2, 48(sp)
    fsd     fs3, 56(sp)
    fsd     fs4, 64(sp)
    fsd     fs5, 72(sp)
    fsd     fs6, 80(sp)
    

    ////////////////////////////////////
    // Load the 25 kernel coefficients
    ////////////////////////////////////
    flw  k0,   0(k)
    flw  k1,   4(k)
    flw  k2,   8(k)
    flw  k3,  12(k)
    flw  k4,  16(k)

    flw  k5,  20(k)
    flw  k6,  24(k)
    flw  k7,  28(k)
    flw  k8,  32(k)
    flw  k9,  36(k)

    flw  k10,  40(k)
    flw  k11,  44(k)
    flw  k12,  48(k)
    flw  k13,  52(k)
    flw  k14,  56(k)

    flw  k15,  60(k)
    flw  k16,  64(k)
    flw  k17,  68(k)
    flw  k18,  72(k)
    flw  k19,  76(k)

    flw  k20,  80(k)
    flw  k21,  84(k)
    flw  k22,  88(k)
    flw  k23,  92(k)
    flw  k24,  96(k)

    fcvt.s.w zero, x0

    // Convert strides from "floats" to "bytes"
    slli   a_stride, a_stride, 2
    slli   b_stride, b_stride, 2

    // For a 5×5 conv, valid output rows = (rows - 4).
    // We still process 2 output rows per iteration => row_check = rows - 4
    mv     row_check, rows
    addi   row_check, row_check, -4

    // Whether we have an odd leftover row from the input
    // (mirroring the 3×3 code’s approach)
    andi   rows_odd, rows, 1

////////////////////////////////////////////////////////////////////////
//  Prolog: set up pointers, load first 4 input rows into vrow0, then
//          partially load the 5th row to begin vrow1, etc.  We keep
//          the same "two-row" pipeline style from the 3×3 version.
////////////////////////////////////////////////////////////////////////
loop_prolog:
    // ap -> row pointer
    mv    ap, a
    addi  ap_4,  ap,  4
    addi  ap_8,  ap,  8
    addi  ap_12, ap, 12
    addi  ap_16, ap, 16

    // bp -> output pointer
    mv    bp, b

    // row_count tracks how many pairs of output rows remain
    mv    row_count, row_check

    // Set vector length
    vsetvli  vlen, cols, e32, m2, ta, ma
    slli     VLEN_stride, vlen, 2

    ////////////////////////////////////////////
    // 1) Load row0 => partial accumulate in vrow0 with k0..k4
    ////////////////////////////////////////////
    vle32.v   vload0, (ap)       // row0[+0]
    vfmv.v.f vrow0, bias
    vfmacc.vf vrow0, k0, vload0

    vle32.v   vload1, (ap_4)     // row0[+4]
    vfmacc.vf vrow0, k1, vload1

    vle32.v   vload2, (ap_8)     // row0[+8]
    vfmacc.vf vrow0, k2, vload2

    vle32.v   vload3, (ap_12)    // row0[+12]
    vfmacc.vf vrow0, k3, vload3

    vle32.v   vload4, (ap_16)    // row0[+16]
    vfmacc.vf vrow0, k4, vload4

    // advance pointers to row1
    add   ap, ap, a_stride
    addi  ap_4,  ap,  4
    addi  ap_8,  ap,  8
    addi  ap_12, ap, 12
    addi  ap_16, ap, 16

    ////////////////////////////////////////////
    // 2) Load row1 => accumulate in vrow0 with k5..k9
    ////////////////////////////////////////////
    vle32.v   vload0, (ap)
    vfmacc.vf vrow0, k5, vload0

    vle32.v   vload1, (ap_4)
    vfmacc.vf vrow0, k6, vload1

    vle32.v   vload2, (ap_8)
    vfmacc.vf vrow0, k7, vload2

    vle32.v   vload3, (ap_12)
    vfmacc.vf vrow0, k8, vload3

    vle32.v   vload4, (ap_16)
    vfmacc.vf vrow0, k9, vload4

    // advance pointers to row2
    add   ap, ap, a_stride

    ////////////////////////////////////////////
    // 3) Load row2 => accumulate in vrow0 with k10..k14
    ////////////////////////////////////////////
    vfmv.v.f vrow1, bias
    vfmacc.vf vrow1, k0, vload0
    vle32.v   vload0, (ap)
    addi  ap_4,  ap,  4
    vfmacc.vf vrow0, k10, vload0

    vfmacc.vf vrow1, k1, vload1
    vle32.v   vload1, (ap_4)
    addi  ap_8,  ap,  8
    vfmacc.vf vrow0, k11, vload1

    vfmacc.vf vrow1, k2, vload2
    vle32.v   vload2, (ap_8)
    addi  ap_12, ap, 12
    vfmacc.vf vrow0, k12, vload2

    vfmacc.vf vrow1, k3, vload3
    vle32.v   vload3, (ap_12)
    addi  ap_16, ap, 16
    vfmacc.vf vrow0, k13, vload3

    vfmacc.vf vrow1, k4, vload4
    vle32.v   vload4, (ap_16)
    vfmacc.vf vrow0, k14, vload4

    // advance pointers to row3
    add   ap, ap, a_stride


    ////////////////////////////////////////////
    // 4) Load row3 => accumulate in vrow0 with k15..k19
    ////////////////////////////////////////////
    vfmv.v.f vrow2, bias
    vfmacc.vf vrow2, k0, vload0
    vfmacc.vf vrow1, k5, vload0
    vle32.v   vload0, (ap)
    addi  ap_4,  ap,  4
    vfmacc.vf vrow0, k15, vload0

    vfmacc.vf vrow2, k1, vload1
    vfmacc.vf vrow1, k6, vload1
    vle32.v   vload1, (ap_4)
    addi  ap_8,  ap,  8
    vfmacc.vf vrow0, k16, vload1

    vfmacc.vf vrow2, k2, vload2
    vfmacc.vf vrow1, k7, vload2
    vle32.v   vload2, (ap_8)
    addi  ap_12, ap, 12
    vfmacc.vf vrow0, k17, vload2

    vfmacc.vf vrow2, k3, vload3
    vfmacc.vf vrow1, k8, vload3
    vle32.v   vload3, (ap_12)
    addi  ap_16, ap, 16
    vfmacc.vf vrow0, k18, vload3
    
    vfmacc.vf vrow2, k4, vload4
    vfmacc.vf vrow1, k9, vload4
    vle32.v   vload4, (ap_16)
    vfmacc.vf vrow0, k19, vload4

    // advance pointers to row4
    add   ap, ap, a_stride

    ////////////////////////////////////////////
    // 5) Load row4 => begin partial sums in vrow1 with k0..k4
    ////////////////////////////////////////////
    vfmv.v.f vrow3, bias
    vfmacc.vf vrow3, k0, vload0
    vfmacc.vf vrow1, k10, vload0
    vfmacc.vf vrow2, k5, vload0
    vle32.v   vload0, (ap)
    addi  ap_4,  ap,  4

    vfmacc.vf vrow3, k1, vload1
    vfmacc.vf vrow2, k6, vload1
    vfmacc.vf vrow1, k11, vload1
    vle32.v   vload1, (ap_4)
    addi  ap_8,  ap,  8

    vfmacc.vf vrow3, k2, vload2
    vfmacc.vf vrow2, k7, vload2
    vfmacc.vf vrow1, k12, vload2
    vle32.v   vload2, (ap_8)
    addi  ap_12, ap, 12

    vfmacc.vf vrow3, k3, vload3
    vfmacc.vf vrow2, k8, vload3
    vfmacc.vf vrow1, k13, vload3
    vle32.v   vload3, (ap_12)
    addi  ap_16, ap, 16

    vfmacc.vf vrow3, k4, vload4
    vfmacc.vf vrow2, k9, vload4
    vfmacc.vf vrow1, k14, vload4
    vle32.v   vload4, (ap_16)
    beqz row_count, epilog

    
//////////////////////////////////////////////////////////////////////
//  Main Loop: process pairs of output rows until row_count <= 0
//////////////////////////////////////////////////////////////////////
conv_loop:
    //------------------------------------------------
    // Finish vrow0 with row4 => k20..k24
    //------------------------------------------------
    vfmacc.vf  vrow0, k20, vload0
    add ap, ap, a_stride
    vfmacc.vf  vrow0, k21, vload1
    addi  ap_4,  ap,  4
    vfmacc.vf  vrow0, k22, vload2
    addi  ap_8,  ap,  8
    vfmacc.vf  vrow0, k23, vload3
    addi  ap_12, ap, 12
    vfmacc.vf  vrow0, k24, vload4
    addi  ap_16, ap, 16

    // store vrow0 => first of the two output rows
    vfmax.vf vrow0, vrow0, zero
    vse32.v   vrow0, (bp)

    //------------------------------------------------
    // For vrow1, keep accumulating row4 => k5..k9
    //------------------------------------------------
    vfmacc.vf  vrow1, k15, vload0
    vfmacc.vf  vrow2, k10, vload0
    vfmacc.vf  vrow3, k5, vload0

    vfmacc.vf  vrow1, k16, vload1
    vfmacc.vf  vrow2, k11, vload1
    vfmacc.vf  vrow3, k6, vload1

    vfmacc.vf  vrow1, k17, vload2
    vfmacc.vf  vrow2, k12, vload2
    vfmacc.vf  vrow3, k7, vload2

    vfmacc.vf  vrow1, k18, vload3
    vfmacc.vf  vrow2, k13, vload3
    vfmacc.vf  vrow3, k8, vload3

    vfmacc.vf  vrow1, k19, vload4
    vfmacc.vf  vrow2, k14, vload4
    vfmacc.vf  vrow3, k9, vload4

    // load row5 => partial sums for vrow0 with k0..k4
    vfmv.v.f vrow0, bias
    vfmacc.vf vrow0, k0, vload0
    vle32.v   vload0, (ap)
    vfmacc.vf vrow0, k1, vload1
    vle32.v vload1, (ap_4)
    add bp, bp, b_stride
    vfmacc.vf vrow0, k2, vload2
    vle32.v vload2, (ap_8)
    vfmacc.vf vrow0, k3, vload3
    vle32.v vload3, (ap_12)
    vfmacc.vf vrow0, k4, vload4
    vle32.v vload4, (ap_16)


    vfmacc.vf vrow0, k5, vload0
    vfmacc.vf vrow1, k20, vload0
    vfmacc.vf vrow2, k15, vload0
    vfmacc.vf vrow3, k10, vload0
    add ap, ap, a_stride

    vfmacc.vf vrow0, k6, vload1
    vfmacc.vf  vrow1, k21, vload1
    vfmacc.vf  vrow2, k16, vload1
    vfmacc.vf  vrow3, k11, vload1
    addi ap_4, ap, 4

    vfmacc.vf vrow0, k7, vload2
    vfmacc.vf  vrow1, k22, vload2
    vfmacc.vf  vrow2, k17, vload2
    vfmacc.vf  vrow3, k12, vload2
    addi ap_8, ap, 8

    vfmacc.vf vrow0, k8, vload3
    vfmacc.vf  vrow1, k23, vload3
    vfmacc.vf  vrow2, k18, vload3
    vfmacc.vf  vrow3, k13, vload3
    addi ap_12, ap, 12

    vfmacc.vf vrow0, k9, vload4
    vfmacc.vf  vrow1, k24, vload4
    vfmacc.vf  vrow2, k19, vload4
    vfmacc.vf  vrow3, k14, vload4
    addi ap_16, ap, 16

    vfmax.vf vrow1, vrow1, zero
    vse32.v vrow1, (bp)


    vfmv.v.f vrow1, bias
    vfmacc.vf vrow1, k0, vload0
    vle32.v   vload0, (ap)
    vfmacc.vf vrow1, k1, vload1
    vle32.v vload1, (ap_4)
    add bp, bp, b_stride
    vfmacc.vf vrow1, k2, vload2
    vle32.v vload2, (ap_8)
    vfmacc.vf vrow1, k3, vload3
    vle32.v vload3, (ap_12)
    vfmacc.vf vrow1, k4, vload4
    vle32.v vload4, (ap_16)

    vfmacc.vf vrow0, k10, vload0
    vfmacc.vf vrow1, k5, vload0
    vfmacc.vf vrow2, k20, vload0
    vfmacc.vf vrow3, k15, vload0
    add ap, ap, a_stride

    vfmacc.vf vrow0, k11, vload1
    vfmacc.vf  vrow1, k6, vload1
    vfmacc.vf  vrow2, k21, vload1
    vfmacc.vf  vrow3, k16, vload1
    addi ap_4, ap, 4

    vfmacc.vf vrow0, k12, vload2
    vfmacc.vf  vrow1, k7, vload2
    vfmacc.vf  vrow2, k22, vload2
    vfmacc.vf  vrow3, k17, vload2
    addi ap_8, ap, 8

    vfmacc.vf vrow0, k13, vload3
    vfmacc.vf  vrow1, k8, vload3
    vfmacc.vf  vrow2, k23, vload3
    vfmacc.vf  vrow3, k18, vload3
    addi ap_12, ap, 12

    vfmacc.vf vrow0, k14, vload4
    vfmacc.vf  vrow1, k9, vload4
    vfmacc.vf  vrow2, k24, vload4
    vfmacc.vf  vrow3, k19, vload4
    addi ap_16, ap, 16

    vfmax.vf vrow2, vrow2, zero
    vse32.v vrow2, (bp)


    vfmv.v.f vrow2, bias
    vfmacc.vf vrow2, k0, vload0
    vle32.v   vload0, (ap)
    vfmacc.vf vrow2, k1, vload1
    vle32.v vload1, (ap_4)
    add bp, bp, b_stride
    vfmacc.vf vrow2, k2, vload2
    vle32.v vload2, (ap_8)
    vfmacc.vf vrow2, k3, vload3
    vle32.v vload3, (ap_12)
    vfmacc.vf vrow2, k4, vload4
    vle32.v vload4, (ap_16)

    vfmacc.vf vrow0, k15, vload0
    vfmacc.vf vrow1, k10, vload0
    vfmacc.vf vrow2, k5, vload0
    vfmacc.vf vrow3, k20, vload0
    add ap, ap, a_stride

    vfmacc.vf vrow0, k16, vload1
    vfmacc.vf  vrow1, k11, vload1
    vfmacc.vf  vrow2, k6, vload1
    vfmacc.vf  vrow3, k21, vload1
    addi ap_4, ap, 4

    vfmacc.vf vrow0, k17, vload2
    vfmacc.vf  vrow1, k12, vload2
    vfmacc.vf  vrow2, k7, vload2
    vfmacc.vf  vrow3, k22, vload2
    addi ap_8, ap, 8

    vfmacc.vf vrow0, k18, vload3
    vfmacc.vf  vrow1, k13, vload3
    vfmacc.vf  vrow2, k8, vload3
    vfmacc.vf  vrow3, k23, vload3
    addi ap_12, ap, 12

    vfmacc.vf vrow0, k19, vload4
    vfmacc.vf  vrow1, k14, vload4
    vfmacc.vf  vrow2, k9, vload4
    vfmacc.vf  vrow3, k24, vload4
    addi ap_16, ap, 16

    vfmax.vf vrow3, vrow3, zero
    vse32.v vrow3, (bp)

    vfmv.v.f vrow3, bias
    vfmacc.vf vrow3, k0, vload0
    vle32.v   vload0, (ap)
    vfmacc.vf vrow3, k1, vload1
    vle32.v vload1, (ap_4)
    add bp, bp, b_stride
    vfmacc.vf vrow3, k2, vload2
    vle32.v vload2, (ap_8)
    vfmacc.vf vrow3, k3, vload3
    vle32.v vload3, (ap_12)
    vfmacc.vf vrow3, k4, vload4
    vle32.v vload4, (ap_16)

	addi row_count, row_count, -4

	bgtz row_count, conv_loop



//////////////////////////////////////////////////////////////////////
//  Epilog: handle finishing up the pipeline plus leftover row if rows_odd != 0
//////////////////////////////////////////////////////////////////////

epilog:
    vfmacc.vf vrow0, k20, vload0
    vfmacc.vf vrow1, k15, vload0
    vfmacc.vf vrow2, k10, vload0
    vfmacc.vf vrow3, k5, vload0
    add ap, ap, a_stride

    vfmacc.vf vrow0, k21, vload1
    vfmacc.vf  vrow1, k16, vload1
    vfmacc.vf  vrow2, k11, vload1
    vfmacc.vf  vrow3, k6, vload1
    addi ap_4, ap, 4

    vfmacc.vf vrow0, k22, vload2
    vfmacc.vf  vrow1, k17, vload2
    vfmacc.vf  vrow2, k12, vload2
    vfmacc.vf  vrow3, k7, vload2
    addi ap_8, ap, 8

    vfmacc.vf vrow0, k23, vload3
    vfmacc.vf  vrow1, k18, vload3
    vfmacc.vf  vrow2, k13, vload3
    vfmacc.vf  vrow3, k8, vload3
    addi ap_12, ap, 12

    vfmacc.vf vrow0, k24, vload4
    vfmacc.vf  vrow1, k19, vload4
    vfmacc.vf  vrow2, k14, vload4
    vfmacc.vf  vrow3, k9, vload4
    addi ap_16, ap, 16

    vfmax.vf vrow0, vrow0, zero
    vse32.v vrow0, (bp)
    add bp, bp, b_stride
    vle32.v vload0, (ap)
    vle32.v vload1, (ap_4)
    vle32.v vload2, (ap_8)
    vle32.v vload3, (ap_12)
    vle32.v vload4, (ap_16)

    vfmacc.vf vrow1, k20, vload0
    vfmacc.vf vrow2, k15, vload0
    vfmacc.vf vrow3, k10, vload0
    add ap, ap, a_stride

    vfmacc.vf  vrow1, k21, vload1
    vfmacc.vf  vrow2, k16, vload1
    vfmacc.vf  vrow3, k11, vload1
    addi ap_4, ap, 4

    vfmacc.vf  vrow1, k22, vload2
    vfmacc.vf  vrow2, k17, vload2
    vfmacc.vf  vrow3, k12, vload2
    addi ap_8, ap, 8

    vfmacc.vf  vrow1, k23, vload3
    vfmacc.vf  vrow2, k18, vload3
    vfmacc.vf  vrow3, k13, vload3
    addi ap_12, ap, 12

    vfmacc.vf  vrow1, k24, vload4
    vfmacc.vf  vrow2, k19, vload4
    vfmacc.vf  vrow3, k14, vload4
    addi ap_16, ap, 16

    vfmax.vf vrow1, vrow1, zero
    vse32.v vrow1, (bp)
    add bp, bp, b_stride
    vle32.v vload0, (ap)
    vle32.v vload1, (ap_4)
    vle32.v vload2, (ap_8)
    vle32.v vload3, (ap_12)
    vle32.v vload4, (ap_16)


    vfmacc.vf vrow2, k20, vload0
    vfmacc.vf vrow3, k15, vload0
    add ap, ap, a_stride

    vfmacc.vf  vrow2, k21, vload1
    vfmacc.vf  vrow3, k16, vload1
    addi ap_4, ap, 4

    vfmacc.vf  vrow2, k22, vload2
    vfmacc.vf  vrow3, k17, vload2
    addi ap_8, ap, 8

    vfmacc.vf  vrow2, k23, vload3
    vfmacc.vf  vrow3, k18, vload3
    addi ap_12, ap, 12

    vfmacc.vf  vrow2, k24, vload4
    vfmacc.vf  vrow3, k19, vload4
    addi ap_16, ap, 16

    vfmax.vf vrow2, vrow2, zero
    vse32.v vrow2, (bp)
    add bp, bp, b_stride
    vle32.v vload0, (ap)
    vle32.v vload1, (ap_4)
    vle32.v vload2, (ap_8)
    vle32.v vload3, (ap_12)
    vle32.v vload4, (ap_16)


    vfmacc.vf vrow3, k20, vload0
    vfmacc.vf  vrow3, k21, vload1
    vfmacc.vf  vrow3, k22, vload2
    vfmacc.vf  vrow3, k23, vload3
    vfmacc.vf  vrow3, k24, vload4

    vfmax.vf vrow3, vrow3, zero
    vse32.v vrow3, (bp)


row_loop_complete:
    // Move a,b pointers to next column-block (just like the 3×3 code),
    // then decrement 'cols', repeating if needed
    add a, a, VLEN_stride
    add b, b, VLEN_stride

    sub cols, cols, vlen
    bnez cols, loop_prolog

exit:
    ld s0,  0(sp)
    ld s1,  8(sp)
    ld s2, 16(sp)
    ld s3, 24(sp)
    fld fs0, 32(sp)
    fld fs1, 40(sp)
    fld fs2, 48(sp)
    fld fs3, 56(sp)
    fld fs4, 64(sp)
    fld fs5, 72(sp)
    addi sp, sp, FRAMESIZE
    ret

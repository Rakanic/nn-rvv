#include "lib_layers.h"

#include "riscv_vector.h"
#include <stdint.h>

void int8_maxpool_ukernel_3x3__rvv_str1(
    size_t output_cols, 
    size_t output_rows,
    size_t input_cols,
    const int8_t* input,
    int8_t* output)
  {
    register vint8m2_t outv1;
    register vint8m2_t outv2;
    register size_t o_cols = output_cols;
  
    do {
      register const int8_t* i = input;
      register int8_t* o = output; 
      int32_t vl = __riscv_vsetvl_e8m2(o_cols);
      register size_t rows = output_rows - 2;
      register const int8_t* i0 = i;
      register const int8_t* i1 = i + 1;
      register const int8_t* i2 = i + 2;
  
      register vint8m2_t i0_f32v = __riscv_vle8_v_i8m2(i0, vl); i0 += input_cols;
      register vint8m2_t i1_f32v = __riscv_vle8_v_i8m2(i1, vl); i1 += input_cols;
      register vint8m2_t i2_f32v = __riscv_vle8_v_i8m2(i2, vl); i2 += input_cols;
      
      outv1 = __riscv_vmax_vv_i8m2(i2_f32v, __riscv_vmax_vv_i8m2(i0_f32v, i1_f32v, vl), vl);
      
      register vint8m2_t i3_f32v = __riscv_vle8_v_i8m2(i0, vl); i0 += input_cols;
      register vint8m2_t i4_f32v = __riscv_vle8_v_i8m2(i1, vl); i1 += input_cols;
      register vint8m2_t i5_f32v = __riscv_vle8_v_i8m2(i2, vl); i2 += input_cols;
  
      outv1 = __riscv_vmax_vv_i8m2(__riscv_vmax_vv_i8m2(outv1, i3_f32v, vl), __riscv_vmax_vv_i8m2(i4_f32v, i5_f32v, vl), vl);
      outv2 = __riscv_vmax_vv_i8m2(i5_f32v, __riscv_vmax_vv_i8m2(i4_f32v, i3_f32v, vl), vl);
  
      register vint8m2_t i6_f32v = __riscv_vle8_v_i8m2(i0, vl); i0 += input_cols;
      register vint8m2_t i7_f32v = __riscv_vle8_v_i8m2(i1, vl); i1 += input_cols;
      register vint8m2_t i8_f32v = __riscv_vle8_v_i8m2(i2, vl); i2 += input_cols;
  
      do {
        outv1 = __riscv_vmax_vv_i8m2(__riscv_vmax_vv_i8m2(outv1, i6_f32v, vl), __riscv_vmax_vv_i8m2(i7_f32v, i8_f32v, vl), vl);
        outv2 = __riscv_vmax_vv_i8m2(__riscv_vmax_vv_i8m2(outv2, i6_f32v, vl), __riscv_vmax_vv_i8m2(i7_f32v, i8_f32v, vl), vl);
        __riscv_vse8_v_i8m2(o, outv1, vl); o += output_cols; 
  
        outv1 = __riscv_vmax_vv_i8m2(i6_f32v, __riscv_vmax_vv_i8m2(i7_f32v, i8_f32v, vl), vl);
        
        i0_f32v = __riscv_vle8_v_i8m2(i0, vl); i0 += input_cols;
        i1_f32v = __riscv_vle8_v_i8m2(i1, vl); i1 += input_cols;
        i2_f32v = __riscv_vle8_v_i8m2(i2, vl); i2 += input_cols;
  
        outv1 = __riscv_vmax_vv_i8m2(__riscv_vmax_vv_i8m2(outv1, i0_f32v, vl), __riscv_vmax_vv_i8m2(i1_f32v, i2_f32v, vl), vl);
        outv2 = __riscv_vmax_vv_i8m2(__riscv_vmax_vv_i8m2(outv2, i0_f32v, vl), __riscv_vmax_vv_i8m2(i1_f32v, i2_f32v, vl), vl);
        __riscv_vse8_v_i8m2(o, outv2, vl); o += output_cols; 
  
        outv2 = __riscv_vmax_vv_i8m2(i0_f32v, __riscv_vmax_vv_i8m2(i1_f32v, i2_f32v, vl), vl);
  
        i6_f32v = __riscv_vle8_v_i8m2(i0, vl); i0 += input_cols;
        i7_f32v = __riscv_vle8_v_i8m2(i1, vl); i1 += input_cols;
        i8_f32v = __riscv_vle8_v_i8m2(i2, vl); i2 += input_cols;
  
        rows -= 2;
      } while (rows != 0);
  
      outv1 = __riscv_vmax_vv_i8m2(__riscv_vmax_vv_i8m2(outv1, i6_f32v, vl), __riscv_vmax_vv_i8m2(i7_f32v, i8_f32v, vl), vl);
      outv2 = __riscv_vmax_vv_i8m2(__riscv_vmax_vv_i8m2(outv2, i6_f32v, vl), __riscv_vmax_vv_i8m2(i7_f32v, i8_f32v, vl), vl);
      __riscv_vse8_v_i8m2(o, outv1, vl); o += output_cols;
  
      i0_f32v = __riscv_vle8_v_i8m2(i0, vl);
      i1_f32v = __riscv_vle8_v_i8m2(i1, vl);
      i2_f32v = __riscv_vle8_v_i8m2(i2, vl);
      outv2 = __riscv_vmax_vv_i8m2(__riscv_vmax_vv_i8m2(outv2, i0_f32v, vl), __riscv_vmax_vv_i8m2(i1_f32v, i2_f32v, vl), vl);
      __riscv_vse8_v_i8m2(o, outv2, vl); o += output_cols; 
  
      o_cols -= vl;
      input = input + vl; 
      output = output + vl;
    } while (o_cols != 0);
  
  }
  
void int8_maxpool_ukernel_3x3__rvv_str2(
    size_t output_cols, 
    size_t output_rows,
    size_t input_cols,
    const int8_t* input,
    int8_t* output)
  {
    register vint8m4_t outv1;
    register size_t o_cols = output_cols;
  
    do {
      register const int8_t* i = input;
      register int8_t* o = output; 
      int32_t vl = __riscv_vsetvl_e8m4(o_cols);
      register size_t rows = output_rows - 1;
      register const int8_t* i0 = i;
      register const int8_t* i1 = i + 1;
      register const int8_t* i2 = i + 2;
  
      register vint8m4_t i0_f32v = __riscv_vlse8_v_i8m4(i0, 4*2, vl); i0 += input_cols;
      register vint8m4_t i1_f32v = __riscv_vlse8_v_i8m4(i1, 4*2, vl); i1 += input_cols;
      register vint8m4_t i2_f32v = __riscv_vlse8_v_i8m4(i2, 4*2, vl); i2 += input_cols;
      
      outv1 = __riscv_vmax_vv_i8m4(i2_f32v, __riscv_vmax_vv_i8m4(i0_f32v, i1_f32v, vl), vl);
      
      register vint8m4_t i3_f32v = __riscv_vlse8_v_i8m4(i0, 4*2, vl); i0 += input_cols;
      register vint8m4_t i4_f32v = __riscv_vlse8_v_i8m4(i1, 4*2, vl); i1 += input_cols;
      register vint8m4_t i5_f32v = __riscv_vlse8_v_i8m4(i2, 4*2, vl); i2 += input_cols;
  
      do {
        // printf("rows: %d \n", rows);
        outv1 = __riscv_vmax_vv_i8m4(__riscv_vmax_vv_i8m4(outv1, i3_f32v, vl), __riscv_vmax_vv_i8m4(i4_f32v, i5_f32v, vl), vl);
  
        i0_f32v = __riscv_vlse8_v_i8m4(i0, 4*2, vl); i0 += input_cols;
        i1_f32v = __riscv_vlse8_v_i8m4(i1, 4*2, vl); i1 += input_cols;
        i2_f32v = __riscv_vlse8_v_i8m4(i2, 4*2, vl); i2 += input_cols;
  
        outv1 = __riscv_vmax_vv_i8m4(__riscv_vmax_vv_i8m4(outv1, i0_f32v, vl), __riscv_vmax_vv_i8m4(i1_f32v, i2_f32v, vl), vl);
        __riscv_vse8_v_i8m4(o, outv1, vl); o += output_cols; 
        outv1 = __riscv_vmax_vv_i8m4(i0_f32v, __riscv_vmax_vv_i8m4(i1_f32v, i2_f32v, vl), vl);
        
        i3_f32v = __riscv_vlse8_v_i8m4(i0, 4*2, vl); i0 += input_cols;
        i4_f32v = __riscv_vlse8_v_i8m4(i1, 4*2, vl); i1 += input_cols;
        i5_f32v = __riscv_vlse8_v_i8m4(i2, 4*2, vl); i2 += input_cols;
  
        rows -= 1;
      } while (rows != 0);
  
      outv1 = __riscv_vmax_vv_i8m4(__riscv_vmax_vv_i8m4(outv1, i3_f32v, vl), __riscv_vmax_vv_i8m4(i4_f32v, i5_f32v, vl), vl);
  
      i0_f32v = __riscv_vlse8_v_i8m4(i0, 4*2, vl); i0 += input_cols;
      i1_f32v = __riscv_vlse8_v_i8m4(i1, 4*2, vl); i1 += input_cols;
      i2_f32v = __riscv_vlse8_v_i8m4(i2, 4*2, vl); i2 += input_cols;
  
      outv1 = __riscv_vmax_vv_i8m4(__riscv_vmax_vv_i8m4(outv1, i0_f32v, vl), __riscv_vmax_vv_i8m4(i1_f32v, i2_f32v, vl), vl);
      __riscv_vse8_v_i8m4(o, outv1, vl);
  
      o_cols -= vl;
      input = input + 2*vl; 
      output = output + vl;
    } while (o_cols != 0);
  
  }
  
void int8_maxpool_ukernel_3x3__rvv_str3(
    size_t output_cols, 
    size_t output_rows,
    size_t input_cols,
    const int8_t* input,
    int8_t* output)
  {
    register vint8m4_t outv1;
    register size_t o_cols = output_cols;
    register vint8m4_t i0_f32v;
    register vint8m4_t i1_f32v;
    register vint8m4_t i2_f32v;
    register vint8m4_t i3_f32v;
    register vint8m4_t i4_f32v;
    register vint8m4_t i5_f32v;
    register vint8m4_t i6_f32v;
    register vint8m4_t i7_f32v;
    register vint8m4_t i8_f32v;
  
    do {
      register const int8_t* i = input;
      register int8_t* o = output; 
      int32_t vl = __riscv_vsetvl_e8m4(o_cols);
      register size_t rows = output_rows;
      register const int8_t* i0 = i;
      register const int8_t* i1 = i + 1;
      register const int8_t* i2 = i + 2;
  
      do {
        // printf("rows: %d \n", rows);
        i0_f32v = __riscv_vlse8_v_i8m4(i0, 1*3, vl); i0 += input_cols;
        i1_f32v = __riscv_vlse8_v_i8m4(i1, 1*3, vl); i1 += input_cols;
        i2_f32v = __riscv_vlse8_v_i8m4(i2, 1*3, vl); i2 += input_cols;
        i3_f32v = __riscv_vlse8_v_i8m4(i0, 1*3, vl); i0 += input_cols;
        i4_f32v = __riscv_vlse8_v_i8m4(i1, 1*3, vl); i1 += input_cols;
        i5_f32v = __riscv_vlse8_v_i8m4(i2, 1*3, vl); i2 += input_cols;
        i6_f32v = __riscv_vlse8_v_i8m4(i0, 1*3, vl); i0 += input_cols;
        i7_f32v = __riscv_vlse8_v_i8m4(i1, 1*3, vl); i1 += input_cols;
        i8_f32v = __riscv_vlse8_v_i8m4(i2, 1*3, vl); i2 += input_cols;
  
        vint8m4_t max01_f32v = __riscv_vmax_vv_i8m4(i0_f32v, i1_f32v, vl);
        vint8m4_t max23_f32v = __riscv_vmax_vv_i8m4(i2_f32v, i3_f32v, vl);
        vint8m4_t max45_f32v = __riscv_vmax_vv_i8m4(i4_f32v, i5_f32v, vl);
        vint8m4_t max67_f32v = __riscv_vmax_vv_i8m4(i6_f32v, i7_f32v, vl);
        vint8m4_t max018_f32v = __riscv_vmax_vv_i8m4(max01_f32v, i8_f32v, vl);
  
        outv1 = __riscv_vmax_vv_i8m4(__riscv_vmax_vv_i8m4(max23_f32v, max45_f32v, vl), __riscv_vmax_vv_i8m4(max67_f32v, max018_f32v, vl), vl);
        __riscv_vse8_v_i8m4(o, outv1, vl); o += output_cols; 
        rows -= 1;
      } while (rows != 0);
  
      o_cols -= vl;
      input = input + 3*vl; 
      output = output + vl;
    } while (o_cols != 0);
  
  }
  
void maxpool_int8(
    size_t output_rows, size_t output_cols, 
    size_t input_rows, size_t input_cols,
    size_t channels,
    size_t stride,
    int8_t *input, 
    int8_t *output)
  {
    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = input_rows * input_cols;
    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = output_rows * output_cols;
  
    if (stride == 1) {
      for (size_t ch = 0; ch < channels; ch++) {
  
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;
  
        // Compute the convolution for this channel.
        int8_maxpool_ukernel_3x3__rvv_str1(output_cols, output_rows, input_cols, a_ch, b_ch);
      }
    } else if (stride == 2) {
      for (size_t ch = 0; ch < channels; ch++) {
  
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;
  
        // Compute the convolution for this channel.
        int8_maxpool_ukernel_3x3__rvv_str2(output_cols, output_rows, input_cols, a_ch, b_ch);
      }
    }
    else if (stride == 3) {
      // f32_maxpool_ukernel_9__rvv_u2v(output_rows*output_cols, input_cols, input_rows, 9, channels, input, output);
      for (size_t ch = 0; ch < channels; ch++) {
  
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;
  
        // Compute the convolution for this channel.
        int8_maxpool_ukernel_3x3__rvv_str3(output_cols, output_rows, input_cols, a_ch, b_ch);
      }
    }
    }
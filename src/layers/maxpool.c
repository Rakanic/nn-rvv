#include "nn_rvv/layers.h"
#include "ops/pooling/maxpool.h"

#include "riscv_vector.h"
#include <stdint.h>

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

void maxpool_f32(
    size_t output_rows, size_t output_cols, 
    size_t input_rows, size_t input_cols,
    size_t channels,
    size_t stride,
    float *input, 
    float *output
  ) {
    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = input_rows * input_cols;
    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = output_rows * output_cols;
  
    if (stride == 1) {
      for (size_t ch = 0; ch < channels; ch++) {
  
        float *a_ch = input + ch * a_channel_size;
        float *b_ch = output + ch * b_channel_size;
  
        // Compute the convolution for this channel.
        f32_maxpool_ukernel_3x3__rvv_str1(output_cols, output_rows, input_cols, a_ch, b_ch);
      }
    } else if (stride == 2) {
      for (size_t ch = 0; ch < channels; ch++) {
  
        float *a_ch = input + ch * a_channel_size;
        float *b_ch = output + ch * b_channel_size;
  
        // Compute the convolution for this channel.
        f32_maxpool_ukernel_3x3__rvv_str2(output_cols, output_rows, input_cols, a_ch, b_ch);
      }
    }
    else if (stride == 3) {
      // f32_maxpool_ukernel_9__rvv_u2v(output_rows*output_cols, input_cols, input_rows, 9, channels, input, output);
      for (size_t ch = 0; ch < channels; ch++) {
  
        float *a_ch = input + ch * a_channel_size;
        float *b_ch = output + ch * b_channel_size;
  
        // Compute the convolution for this channel.
        f32_maxpool_ukernel_3x3__rvv_str3(output_cols, output_rows, input_cols, a_ch, b_ch);
      }
    }
}
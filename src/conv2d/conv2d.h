#ifndef CONV2D_H
#define CONV2D_H

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#include "stdio.h"
#include <stdint.h>
#include "quantization_params.h"

void *vec_conv (size_t, size_t, size_t, size_t, const float*, const float*, float*, float);
void *vec_conv_relu (size_t, size_t, size_t, size_t, const float*, const float*, float*, float);
void *vec_conv_3x3_int8(size_t, size_t, size_t, size_t, const int8_t*, const int8_t*, int8_t*, int32_t, float, float);
void *vec_conv_3x3_int8_relu(size_t, size_t, size_t, size_t, const int8_t*, const int8_t*, int8_t*, int16_t, float, float);
void *vec_conv_5x5 (size_t, size_t, size_t, size_t, const float*, const float*, float*, float);
void *vec_conv_5x5_relu (size_t, size_t, size_t, size_t, const float*, const float*, float*, float);

void dwconv_3x3_f32_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    float *input, 
    float *output
);

void dwconv_3x3_f32_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    float *input, 
    float *output
);

void dwconv_3x3_f32_VCH(
    size_t rows, size_t cols, 
    size_t channels,
    float* input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment
);

void dwconv_3x3_f32_VCH_relu(
    size_t rows, size_t cols, 
    size_t channels,
    float* input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment
);

void dwconv_3x3_int8_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
);

void dwconv_3x3_int8_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
);

void conv_1x1_f32(
    size_t rows, size_t cols, 
    size_t channels_in,
    size_t channels_out,
    float* input,
    const float* weights,
    float* output
);

void conv_1x1_relu_f32(
    size_t rows, size_t cols, 
    size_t channels_in,
    size_t channels_out,
    float* input,
    const float* weights,
    float* output
);

void conv_1x1_f32_2d(
    size_t rows, size_t cols,
    float* input,
    float weight, 
    float bias,
    float* output
);

void conv_1x1_f32_2d_macc(
    size_t rows, size_t cols,
    float* input,
    float weight, 
    float* output
);

void conv_1x1_relu_f32_2d_macc(
    size_t rows, size_t cols,
    float* input,
    float weight, 
    float* output
);

void conv_1x1_relu_f32_2d(
    size_t rows, size_t cols,
    float* input,
    float weight, 
    float bias,
    float* output
);

void dwconv_5x5_f32_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,      // weights: first 'channels' bias values, then 25 weights per channel
    const float *input, 
    float *output
);

void dwconv_5x5_f32_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,      // weights: first 'channels' bias values, then 25 weights per channel
    const float *input, 
    float *output
);

void dwconv_5x5_int8_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output, 
    requantization_params_t requant_params
);

void dwconv_5x5_int8_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output, 
    requantization_params_t requant_params
);

#endif
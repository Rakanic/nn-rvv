#ifndef NN_RVV_OPS_CONV2D_H
#define NN_RVV_OPS_CONV2D_H

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "nn_rvv/layers.h"

/* ---- External (assembly) entry points: standard non-depthwise 2D conv ---- */
void *vec_conv (size_t, size_t, size_t, size_t, const float*, const float*, float*, float);
void *vec_conv_relu (size_t, size_t, size_t, size_t, const float*, const float*, float*, float);
void *vec_conv_3x3_int8(size_t, size_t, size_t, size_t, const int8_t*, const int8_t*, int8_t*, int32_t, float, float);
void *vec_conv_3x3_int8_relu(size_t, size_t, size_t, size_t, const int8_t*, const int8_t*, int8_t*, int16_t, float, float);
void *vec_conv_5x5 (size_t, size_t, size_t, size_t, const float*, const float*, float*, float);
void *vec_conv_5x5_relu (size_t, size_t, size_t, size_t, const float*, const float*, float*, float);

/* ---- Depthwise 3x3 f32 (vec-channel-output and vec-channel-height) ---- */
void dwconv_3x3_f32_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,
    float *input,
    float *output
);

void dwconv_3x3_f32_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,
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

/* ---- Depthwise 3x3 int8 (vec-nn authoritative; honors padding/stride) ---- */
void dwconv_3x3_int8_VCO(
    size_t input_rows, size_t input_cols,
    size_t stride, size_t padding,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,
    int8_t *input,
    int8_t *output,
    requantization_params_t requant_params
);

void dwconv_3x3_int8_VCO_relu(
    size_t input_rows, size_t input_cols,
    size_t stride, size_t padding,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,
    int8_t *input,
    int8_t *output,
    requantization_params_t requant_params
);

/* ---- Pointwise 1x1 f32 ---- */
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

/* ---- Depthwise 5x5 f32 ---- */
void dwconv_5x5_f32_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,
    const float *input,
    float *output
);

void dwconv_5x5_f32_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,
    const float *input,
    float *output
);

/* ---- Depthwise 5x5 int8 ---- */
void dwconv_5x5_int8_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,
    int8_t *input,
    int8_t *output,
    requantization_params_t requant_params
);

void dwconv_5x5_int8_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,
    int8_t *input,
    int8_t *output,
    requantization_params_t requant_params
);

#endif /* NN_RVV_OPS_CONV2D_H */

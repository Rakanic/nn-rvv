#ifndef LAYERS_H
#define LAYERS_H

#include "lib_kernels.h"

void fully_connected_f32 (
    size_t input_size, 
    size_t output_size, 
    size_t batches, 
    float* input, 
    const float* weights_with_bias,
    float* output, 
    int relu
);

void fully_connected_int8 (
    size_t input_size, 
    size_t output_size, 
    size_t batches, 
    int8_t* input, 
    const int8_t* weights_with_bias,
    int8_t* output, 
    int relu, 
    quantization_params_t qp_input, 
    quantization_params_t qp_weights, 
    quantization_params_t qp_output 
);

void quant_fully_connected_int8 (
    size_t input_size, 
    size_t output_size, 
    size_t batches, 
    int8_t* input, 
    const void* weights_with_bias,
    int8_t* output, 
    int relu, int bias32,
    requantization_params_t requant_params 
);

void dequant_f32(
    size_t size, 
    int8_t* input, 
    float* output, 
    quantization_params_t qp
);

void quant_f32(
    size_t size, 
    float* input, 
    int8_t* output, 
    quantization_params_t qp
);

void requant_outch_int32(
    size_t rows, size_t cols, 
    size_t channels,
    int32_t* input, 
    int8_t* output, 
    int relu,
    requantization_params_t rqp
);

void softmax_vec(
    const float *i, 
    float *o, 
    size_t channels,
    size_t innerSize
);

void conv2D_3x3_f32 (
    size_t H, size_t W,
    size_t Cin, size_t Cout,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const float *dw_weights,  // length = Cin*(1 + 9)
    const float *pw_weights,  // length = Cout*Cin
    float *input,       // CHW: [Cin][H][W]
    float *output,            // CHW: [Cout][H_out][W_out]
    int relu_dw,
    int relu_pw
);

void conv2D_3x3_int8 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const void *dw_weights,  // length = Cin*(1 + 9)
    int8_t *input,       // CHW: [Cin][H][W]
    int8_t *output,            // CHW: [Cout][H_out][W_out]
    int relu,
    requantization_params_t requant_params_dwconv
);

void conv_1x1_int8(
    size_t rows, size_t cols, 
    size_t channels_in,
    size_t channels_out,
    int8_t* input,
    const void* weights,
    int8_t* output, 
    int relu,
    requantization_params_t rqp
);

void maxpool_f32(
    size_t output_rows, size_t output_cols, 
    size_t input_rows, size_t input_cols,
    size_t channels,
    size_t stride,
    float *input, 
    float *output
);

void maxpool_int8(
    size_t output_rows, size_t output_cols, 
    size_t input_rows, size_t input_cols,
    size_t channels,
    size_t stride,
    int8_t *input, 
    int8_t *output
);



#endif
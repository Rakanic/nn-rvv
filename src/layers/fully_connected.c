#include "nn_rvv/layers.h"
#include "ops/matmul/matmul.h"

#include <stdint.h>

void fully_connected_f32 (
    size_t input_size, 
    size_t output_size, 
    size_t batches, 
    float* input, 
    const float* weights_with_bias,
    float* output, 
    int relu
) {
    if (relu) {
        f32_gemm_relu(
            batches, output_size, input_size, 
            input, input_size, 
            weights_with_bias, 
            output, output_size, 1);
    } else {
        f32_gemm(
            batches, output_size, input_size, 
            input, input_size, 
            weights_with_bias, 
            output, output_size, 1);
    }
}

void fully_connected_f32_nobias (
    size_t input_size, 
    size_t output_size, 
    size_t batches, 
    float* input, 
    const float* weights_with_bias,
    float* output, 
    int relu
) {
    f32_gemm_nobias(
        batches, output_size, input_size, 
        input, input_size, 
        weights_with_bias, 
        output, output_size, 1);
}

void quant_fully_connected_int8_t(
    size_t input_size,
    size_t output_size,
    size_t batches,
    const int8_t* input,
    const void* weights_t_pack,
    float* output,
    float scale)
{
    int8_qgemm_fout(
        batches, output_size, input_size,
        input, input_size,
        (const int8_t*)weights_t_pack,
        output, output_size * sizeof(float), 1,
        scale);
}

void quant_fully_connected_int8 (
    size_t input_size, 
    size_t output_size, 
    size_t batches, 
    int8_t* input, 
    const void* weights_with_bias,
    int8_t* output, 
    int relu, int bias32,
    requantization_params_t requant_params 
) {

    if (bias32) {
        if (relu) {
            int8_qgemm_int32bias_relu(
                batches, output_size, input_size, 
                input, input_size, 
                weights_with_bias, 
                output, output_size, 1,
                requant_params);
        } else {
            int8_qgemm_int32bias(
                batches, output_size, input_size, 
                input, input_size, 
                weights_with_bias, 
                output, output_size, 1,
                requant_params);
        }
    } else {
        if (relu) {
            int8_qgemm_relu(
                batches, output_size, input_size, 
                input, input_size, 
                weights_with_bias, 
                output, output_size, 1,
                requant_params);
        } else {
            int8_qgemm(
                batches, output_size, input_size, 
                input, input_size, 
                weights_with_bias, 
                output, output_size, 1,
                requant_params);
        }
    }
}
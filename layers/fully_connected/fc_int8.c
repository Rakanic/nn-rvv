#include "lib_layers.h"

#include <stdint.h>

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
) {
    if (relu) {
        int8_gemm_relu(
            batches, output_size, input_size, 
            input, input_size, 
            weights_with_bias, 
            output, output_size, 1,
            qp_input, qp_weights, qp_output);
    } else {
        int8_gemm(
            batches, output_size, input_size, 
            input, input_size, 
            weights_with_bias, 
            output, output_size, 1,
            qp_input, qp_weights, qp_output);
    }
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
    // printf("bias: \n");
    // for (int k = 0; k < output_size; k++) {
    //     printf("%d ", ((int32_t*) weights_with_bias)[k]);
    // }

    // printf("weights: \n");
    // for (int i = 0; i < input_size; i ++) {
    //     for (int j = 0; j < output_size; j ++) {
    //         printf("%d ", ((int8_t*) weights_with_bias)[4*output_size + i*output_size + j]);
    //     }
    // }
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
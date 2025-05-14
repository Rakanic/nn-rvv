#include "lib_layers.h"

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
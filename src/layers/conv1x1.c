#include "nn_rvv/layers.h"
#include <stdint.h>

#include "ops/matmul/matmul.h"

void conv_1x1_int8(
    size_t rows, size_t cols, 
    size_t channels_in,
    size_t channels_out,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    int8_t* input,
    const void* weights,
    int8_t* output, 
    int relu,
    requantization_params_t rqp
) {

    if (relu) {
        int8_qgemm_int32bias_conv1x1_relu(
            channels_out, rows*cols, channels_in, 
            weights, channels_in,
            input, 
            output, rows*cols, 1,
            rqp);
    } else {
        int8_qgemm_int32bias_conv1x1(
            channels_out, rows*cols, channels_in, 
            weights, channels_in,
            input, 
            output, rows*cols, 1,
            rqp);
    }
}
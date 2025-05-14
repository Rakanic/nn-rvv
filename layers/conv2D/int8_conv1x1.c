#include "lib_layers.h"
#include <stdint.h>

void conv_1x1_int8(
    size_t rows, size_t cols, 
    size_t channels_in,
    size_t channels_out,
    int8_t* input,
    const void* weights,
    int8_t* output, 
    int relu,
    requantization_params_t rqp
) {
    int8_t buffer[channels_in*rows*cols];
    transpose_int8(input, buffer, channels_in, rows*cols);

    int8_t temp_output[channels_out*rows*cols];
    if (relu) {
        int8_qgemm_int32bias_relu(
            rows*cols, channels_out, channels_in, 
            buffer, channels_in,
            weights, 
            temp_output, channels_out, 1,
            rqp);
    } else {
        int8_qgemm_int32bias(
            rows*cols, channels_out, channels_in, 
            buffer, channels_in,
            weights, 
            temp_output, channels_out, 1,
            rqp);
    }

    transpose_int8(temp_output, output, rows*cols, channels_out);
}


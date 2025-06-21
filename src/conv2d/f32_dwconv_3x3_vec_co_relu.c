#include "conv2d.h"

#include <riscv_vector.h> 
#include <stdint.h>


void dwconv_3x3_f32_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    float *input, 
    float *output
) {
    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (rows + 2) * a_stride;
    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = rows * b_stride;

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 3x3 kernel for this channel is stored starting at weights[channels] with 9 floats per channel.
        const float *k_ch = weights + channels + ch * 9;

        float *a_ch = input + ch * a_channel_size;
        float *b_ch = output + ch * b_channel_size;

        // Compute the convolution for this channel.
        vec_conv_relu(rows, cols, a_stride, b_stride, k_ch, a_ch, b_ch, weights[ch]);
    }
}


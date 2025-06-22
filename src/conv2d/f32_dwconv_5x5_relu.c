#include <stdint.h>
#include <riscv_vector.h>
#include "conv2d.h"

void dwconv_5x5_f32_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const float *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    const float *input, 
    float *output
) {
    size_t a_channel_size = (rows + 4) * a_stride;
    // Each channel's output is rows x b_stride (b_stride is the columns)
    size_t b_channel_size = rows * b_stride;

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 5x5 kernel for this channel is stored starting at weights[channels] with 25 floats per channel.
        const float *k_ch = weights + channels + ch * 25;

        const float *a_ch = input + ch * a_channel_size;
        float *b_ch = output + ch * b_channel_size;

        // Compute the convolution for this channel using the assembly version.
        vec_conv_5x5_relu(rows, cols, a_stride, b_stride, k_ch, a_ch, b_ch, weights[ch]);
    }
}

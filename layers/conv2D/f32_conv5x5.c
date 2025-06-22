#include "lib_layers.h"

#include <stdint.h>

void dw_conv2D_5x5_f32 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const float *dw_weights,  // length = Cin*(1 + 9)
    float *input,       // CHW: [Cin][H][W]
    float *output,            // CHW: [Cout][H_out][W_out]
    int relu_dw
) {
    size_t H_out = (H - 5)/stride + 1;
    size_t W_out = (W - 5)/stride + 1;

    if (!relu_dw) {
        dwconv_5x5_f32_VCO(
            H_out, W_out, 
            Cin, 
            W, W_out, 
            dw_weights, 
            input, 
            output
        );
    } else {
        dwconv_5x5_f32_VCO_relu(
            H_out, W_out, 
            Cin, 
            W, W_out, 
            dw_weights, 
            input, 
            output
        );
    }
}
#include "lib_layers.h"

#include <stdint.h>

void conv2D_1x1_f32 (
    size_t H, size_t W,
    size_t Cin, size_t Cout,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const float *pw_weights,  // length = Cout*Cin
    float *input,       // CHW: [Cin][H][W]
    float *output,            // CHW: [Cout][H_out][W_out]
    int relu_pw
) {
    if (!relu_pw) {
        conv_1x1_f32(
            H, W, 
            Cin, Cout, 
            input, 
            pw_weights, 
            output
        );
    } else {
        conv_1x1_relu_f32(
            H, W, 
            Cin, Cout, 
            input, 
            pw_weights, 
            output
        );
    }


}
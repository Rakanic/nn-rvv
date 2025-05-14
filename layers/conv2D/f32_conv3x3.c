#include "lib_layers.h"

#include <stdint.h>

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
) {
    size_t H_out = (H - 3)/stride + 1;
    size_t W_out = (W - 3)/stride + 1;

    float depthwise_conv_output[Cin*H_out*W_out];

    if (!relu_dw) {
        if (W_out > 0) {
            dwconv_3x3_f32_VCO(
                H_out, W_out, 
                Cin, 
                W, W_out, 
                dw_weights, 
                input, 
                depthwise_conv_output
            );
        } else {
            dwconv_3x3_f32_VCH(
                H_out, W_out, 
                Cin,
                input,
                dw_weights,
                depthwise_conv_output, 
                1, 0
            );
        }
    } else {
        if (W_out > 0) {
            dwconv_3x3_f32_VCO_relu(
                H_out, W_out, 
                Cin, 
                W, W_out, 
                dw_weights, 
                input, 
                depthwise_conv_output
            );
        } else {
            dwconv_3x3_f32_VCH_relu(
                H_out, W_out, 
                Cin,
                input,
                dw_weights,
                depthwise_conv_output, 
                1, 0
            );
        }
    }

    if (!relu_pw) {
        conv_1x1_f32(
            H_out, W_out, 
            Cin, Cout, 
            depthwise_conv_output, 
            pw_weights, 
            output
        );
    } else {
        conv_1x1_relu_f32(
            H_out, W_out, 
            Cin, Cout, 
            depthwise_conv_output, 
            pw_weights, 
            output
        );
    }


}
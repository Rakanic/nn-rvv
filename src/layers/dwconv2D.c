#include "nn_rvv/layers.h"
#include <stdint.h>

#include "ops/conv2D/conv2D.h"


void dwconv2D_3x3_int8 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const void *dw_weights,  // length = Cin*(1 + 9)
    int8_t *input,       // CHW: [Cin][H][W]
    int8_t *output,            // CHW: [Cout][H_out][W_out]
    int relu,
    requantization_params_t requant_params_dwconv
) {
    size_t H_out = (H + 2*padding - 3)/stride + 1;
    size_t W_out = (W + 2*padding - 3)/stride + 1;


    if (!relu) {
        dwconv_3x3_int8_VCO(
            H, W,
            stride, padding, 
            Cin, 
            W, W_out, 
            dw_weights, 
            input, 
            output, 
            requant_params_dwconv
        );
    } else {
        dwconv_3x3_int8_VCO_relu(
            H, W,
            stride, padding, 
            Cin, 
            W, W_out, 
            dw_weights, 
            input, 
            output, 
            requant_params_dwconv
        );
    }
}
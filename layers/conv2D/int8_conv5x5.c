#include "lib_layers.h"

#include <stdint.h>


void dw_conv2D_5x5_int8 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const void *dw_weights,  // length = Cin*(1 + 25)
    int8_t *input,       // CHW: [Cin][H][W]
    int8_t *output,            // CHW: [Cout][H_out][W_out]
    int relu,
    requantization_params_t requant_params_dwconv
) {
    size_t H_out = (H - 5)/stride + 1;
    size_t W_out = (W - 5)/stride + 1;


    if (!relu) {
        dwconv_5x5_int8_VCO(
            H_out, W_out, 
            Cin, 
            W, W_out, 
            dw_weights, 
            input, 
            output, 
            requant_params_dwconv
        );
    } else {
        dwconv_5x5_int8_VCO_relu(
            H_out, W_out, 
            Cin, 
            W, W_out, 
            dw_weights, 
            input, 
            output, 
            requant_params_dwconv
        );
    }
}
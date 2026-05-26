#include "nn_rvv/layers.h"
#include "ops/matmul/matmul.h"

#include <stdint.h>

/* Legacy unquantized int8 FC path that takes separate quantization params
 * per operand (input/weights/output) rather than per-channel requantization
 * parameters. Kept for backward compatibility with older models built before
 * the per-channel requantization flow landed.
 *
 * The quant-requant variant (`quant_fully_connected_int8`) lives in
 * fully_connected.c. */
void fully_connected_int8 (
    size_t input_size,
    size_t output_size,
    size_t batches,
    int8_t* input,
    const int8_t* weights_with_bias,
    int8_t* output,
    int relu,
    quantization_params_t qp_input,
    quantization_params_t qp_weights,
    quantization_params_t qp_output
) {
    if (relu) {
        int8_gemm_relu(
            batches, output_size, input_size,
            input, input_size,
            weights_with_bias,
            output, output_size, 1,
            qp_input, qp_weights, qp_output);
    } else {
        int8_gemm(
            batches, output_size, input_size,
            input, input_size,
            weights_with_bias,
            output, output_size, 1,
            qp_input, qp_weights, qp_output);
    }
}

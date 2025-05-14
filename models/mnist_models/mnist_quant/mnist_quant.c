#include <stdio.h>
#include "model_params.h"   // Contains:
//   int8_t  layer0_wb_q   [64 * (784 + 1)];  // weights+bias for layer0, quantized per-tensor
//   int8_t  layer1_wb_q   [10  * ( 64 + 1)];  // weights+bias for layer1, quantized per-tensor
//   quantization_params_t qp_input;           // { scale, zero_point } for the network input
//   quantization_params_t qp_w0, qp_out0;     // { scale, zero_point } for layer0 weights & outputs
//   quantization_params_t qp_w1, qp_out1;     // { scale, zero_point } for layer1 weights & outputs
#include "input_data.h"          // float input[BATCHES * 784];
#include "lib_layers.h"          // declarations for fully_connected_int8, quant_f32, dequant_f32, softmax_vec

int main(void) {
    //------------------------------------------------------------------------
    // Buffers for quantized data and intermediate float results
    //------------------------------------------------------------------------
    // Quantized int8 buffers
    static int8_t input_q    [BATCHES * 784];  // quantized network input
    static int8_t dense0_q   [BATCHES *  64];  // quantized output of layer0
    static int8_t logits_q   [BATCHES *  10];  // quantized output of layer1 (logits)
    // Float buffer for dequantized logits (for softmax)
    static float  logits_f32 [BATCHES *  10];
    // Float buffer for softmax probabilities
    static float  probs      [BATCHES *  10];

    unsigned long cycles_start, cycles_end, instr_start, instr_end;
    asm volatile ("rdcycle %0" : "=r" (cycles_start));
    asm volatile ("rdinstret %0" : "=r" (instr_start));

    //------------------------------------------------------------------------
    // 1) Quantize the raw float input once, before the first layer
    //------------------------------------------------------------------------
    // Converts each float in `input` to int8 according to qp_input
    quant_f32(
        /* size */    BATCHES * 784,
        /* input */   input,
        /* output */  input_q,
        /* qparams */ qp_input
    );

    //------------------------------------------------------------------------
    // 2) Layer 0: Fully-connected 784 → 64 with fused ReLU in the kernel
    //------------------------------------------------------------------------
    // - Takes int8 input_q
    // - Uses quantized weights+bias layer0_wb_q
    // - Writes int8 output dense0_q
    // - `relu=1` turns on ReLU (clamp at zero) inside the requantization step
    fully_connected_int8(
        /* input_size */   784,
        /* output_size */  64,
        /* batches */      BATCHES,
        /* input */        input_q,
        /* w/b packed */   layer0_wb_q,
        /* output */       dense0_q,
        /* relu */         1, 0,
        /* qp input */     qp_input,
        /* qp weights */   qp_w0,
        /* qp output */    qp_out0
    );

    //------------------------------------------------------------------------
    // 3) Layer 1: Fully-connected 64 → 10 (produces logits, no activation)
    //------------------------------------------------------------------------
    // - Input quant params is the output params of layer0 (qp_out0)
    fully_connected_int8(
        /* input_size */   64,
        /* output_size */  10,
        /* batches */      BATCHES,
        /* input */        dense0_q,
        /* w/b packed */   layer1_wb_q,
        /* output */       logits_q,
        /* relu */         0, 0,
        /* qp input */     qp_out0,
        /* qp weights */   qp_w1,
        /* qp output */    qp_out1
    );

    //------------------------------------------------------------------------
    // 4) Dequantize logits back to float for softmax
    //------------------------------------------------------------------------
    dequant_f32(
        /* size */    BATCHES * 10,
        /* input */   logits_q,
        /* output */  logits_f32,
        /* qparams */ qp_out1
    );

    //------------------------------------------------------------------------
    // 5) Softmax (in float) and Prediction
    //------------------------------------------------------------------------
    for (size_t b = 0; b < BATCHES; b++) {
        // Compute softmax over logits_f32[b*10 .. b*10+9]
        softmax_vec(
            &logits_f32[b * 10],
            &probs    [b * 10],
            /* length */ 10,
            /* stride */ 1
        );
    }

    asm volatile ("fence");
    asm volatile ("rdcycle %0" : "=r" (cycles_end));
    asm volatile ("rdinstret %0" : "=r" (instr_end));

    printf("  Execution cycles:      %lu\n", cycles_end   - cycles_start);
    printf("  Instructions executed: %lu\n\n", instr_end    - instr_start);

    //------------------------------------------------------------------------
    // 6) Print out probabilities and predicted classes
    //------------------------------------------------------------------------
    for (size_t b = 0; b < BATCHES; b++) {
        int   predicted = 0;
        float max_prob  = probs[b * 10];
        for (int c = 1; c < 10; c++) {
            if (probs[b * 10 + c] > max_prob) {
                max_prob  = probs[b * 10 + c];
                predicted = c;
            }
        }
        printf("Input %d: Predicted digit %d, probabilities:", b, predicted);
        for (int c = 0; c < 10; c++) {
            // print as integer percentages
            printf(" %d", (int)(100.0f * probs[b * 10 + c]));
        }
        printf("\n");
    }

    return 0;
}
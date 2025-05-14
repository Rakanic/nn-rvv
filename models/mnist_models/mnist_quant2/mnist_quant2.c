#include <stdio.h>
#include "model_params.h"   // Now contains:
//   int8_t                     layer0_wb_q[64*(784+1)];
//   int8_t                     layer1_wb_q[10*(64+1)];
//   quantization_params_t      qp_input;
//   requantization_params_t    rq_layer0;    // { scale = s_in * s_w0 / s_out0, zero_point = zp_out0 }
//   requantization_params_t    rq_layer1;    // { scale = s_out0 * s_w1 / s_out1, zero_point = zp_out1 }
#include "input_data.h"           // float input[BATCHES * 784];
#include "lib_layers.h"           // quant_f32, quant_fully_connected_int8, dequant_f32, softmax_vec

int main(void) {
    //------------------------------------------------------------------------
    // Buffers for quantized data and intermediate float results
    //------------------------------------------------------------------------
    static int8_t input_q  [BATCHES * 784];
    static int8_t dense0_q [BATCHES *  64];
    static int8_t logits_q [BATCHES *  10];
    static float  logits_f32[BATCHES *  10];
    static float  probs     [BATCHES *  10];

    unsigned long cyc0, cyc1, ins0, ins1;
     asm volatile ("rdcycle %0"   : "=r"(cyc0));
     asm volatile ("rdinstret %0" : "=r"(ins0));

    //------------------------------------------------------------------------
    // 1) Quantize float input → int8[input_q]
    //------------------------------------------------------------------------
    quant_f32(
        /* size */   BATCHES * 784,
        /* input */  input,
        /* output */ input_q,
        /* qparams*/ qp_input
    );

    //------------------------------------------------------------------------
    // 2) Layer 0: 784→64, ReLU fused, using requant layer0
    //------------------------------------------------------------------------
    quant_fully_connected_int8(
        /* input_size */  784,
        /* output_size */ 64,
        /* batches */     BATCHES,
        /* input */       input_q,
        /* w/b */        layer0_wb_q,
        /* output */      dense0_q,
        /* relu */        1, 0, 
        /* rq params */   rq_layer0
    );

    //------------------------------------------------------------------------
    // 3) Layer 1: 64→10, no activation, requant layer1
    //------------------------------------------------------------------------
    quant_fully_connected_int8(
        /* input_size */  64,
        /* output_size */ 10,
        /* batches */     BATCHES,
        /* input */       dense0_q,
        /* w/b */        layer1_wb_q,
        /* output */      logits_q,
        /* relu */        0, 0, 
        /* rq params */   rq_layer1
    );

    //------------------------------------------------------------------------
    // 4) Dequantize logits → float for softmax
    //------------------------------------------------------------------------
    dequant_f32(
        /* size */   BATCHES * 10,
        /* input */  logits_q,
        /* output */ logits_f32,
        /* qparams*/ qp_output
        // or, if you still have qp_out1 in your header:
        // qp_out1
    );

    asm volatile ("fence");
    asm volatile ("rdcycle %0"   : "=r"(cyc1));
    asm volatile ("rdinstret %0" : "=r"(ins1));

    printf("Execution cycles      : %lu\n", cyc1 - cyc0);
    printf("Instructions executed : %lu\n", ins1 - ins0);

    //------------------------------------------------------------------------
    // 5) Softmax & print results
    //------------------------------------------------------------------------
    for (size_t b = 0; b < BATCHES; b++) {
        softmax_vec(
            &logits_f32[b * 10],
            &probs    [b * 10],
            /* length */ 10,
            /* stride */ 1
        );
        // find max
        int   pred = 0;
        float maxp = probs[b * 10];
        for (int c = 1; c < 10; c++) {
            if (probs[b*10 + c] > maxp) {
                maxp = probs[b*10 + c];
                pred = c;
            }
        }
        printf("Input %d: Predicted %d (probs:", b, pred);
        for (int c = 0; c < 10; c++) {
            printf(" %d", (int)(probs[b*10 + c]*100.0f));
        }
        printf(" )\n");
    }

    return 0;
}
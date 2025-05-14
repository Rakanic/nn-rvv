#include <stdio.h>
#include "model_params.h"   // Now contains:
//   int8_t                     layer0_wb_q[64*(784+1)];
//   int8_t                     layer1_wb_q[10*(64+1)];
//   quantization_params_t      qp_input;
//   requantization_params_t    rq_layer0;    // { scale = s_in * s_w0 / s_out0, zero_point = zp_out0 }
//   requantization_params_t    rq_layer1;    // { scale = s_out0 * s_w1 / s_out1, zero_point = zp_out1 }
#include "input_data.h"           // float input[BATCHES / 2  * 784];
#include "lib_layers.h"           // quant_f32, quant_fully_connected_int8, dequant_f32, softmax_vec
#include "encoding.h"


volatile uint32_t go       = 0;   /* 0 = parked, 1 = start work      */
volatile uint32_t done     = 0;   /* hart-1 sets to 1 when finished  */
static float probs2 [BATCHES / 2  *  10];

void __main(void)
{
    const uint32_t me = read_csr(mhartid);   /* == 1 */
    static int8_t input_q2  [BATCHES / 2  * 784];
    static int8_t dense0_q2 [BATCHES / 2  *  64];
    static int8_t logits_q2 [BATCHES / 2  *  10];
    static float  logits2_f32[BATCHES / 2  *  10];
    // printf("hello from corqe %d \n", me);
    

    while (1) {
        // printf("hello from core %d \n", me);
        while (!go) ;

        quant_f32(
            /* size */   BATCHES / 2  * 784,
            /* input */  input + BATCHES / 2  * 784,
            /* output */ input_q2,
            /* qparams*/ qp_input
        );
    
        //------------------------------------------------------------------------
        // 2) Layer 0: 784→64, ReLU fused, using requant layer0
        //------------------------------------------------------------------------
        quant_fully_connected_int8(
            /* input_size */  784,
            /* output_size */ 64,
            /* BATCHES / 2  */ BATCHES / 2 ,
            /* input */       input_q2,
            /* w/b */        layer0_wb_q,
            /* output */      dense0_q2,
            /* relu */        1, 0,
            /* rq params */   rq_layer0
        );
    
        //------------------------------------------------------------------------
        // 3) Layer 1: 64→10, no activation, requant layer1
        //------------------------------------------------------------------------
        quant_fully_connected_int8(
            /* input_size */  64,
            /* output_size */ 10,
            /* BATCHES / 2  */ BATCHES / 2 ,
            /* input */       dense0_q2,
            /* w/b */        layer1_wb_q,
            /* output */      logits_q2,
            /* relu */        0, 0,
            /* rq params */   rq_layer1
        );
    
        //------------------------------------------------------------------------
        // 4) Dequantize logits → float for softmax
        //------------------------------------------------------------------------
        dequant_f32(
            /* size */   BATCHES / 2  * 10,
            /* input */  logits_q2,
            /* output */ logits2_f32,
            /* qparams*/ qp_output
            // or, if you still have qp_out1 in your header:
            // qp_out1
        );

        for (size_t b2 = 0; b2 < BATCHES / 2; b2++) {
            softmax_vec(
                &logits2_f32[b2 * 10],
                &probs2    [b2 * 10],
                /* length */ 10,
                /* stride */ 1
            );
        }

        done = 1;                         /* signal hart-0 we’re done   */
         /* park until next round      */
    }
}

int main(void) {
    //------------------------------------------------------------------------
    // Buffers for quantized data and intermediate float results
    //------------------------------------------------------------------------
    printf("hello from core 0 \n");
    static int8_t input_q  [BATCHES / 2  * 784];
    static int8_t dense0_q [BATCHES / 2  *  64];
    static int8_t logits_q [BATCHES / 2  *  10];
    static float  logits_f32[BATCHES / 2  *  10];
    static float  probs     [BATCHES / 2  *  10];

    unsigned long cyc0, cyc1, ins0, ins1;
    
    __sync_synchronize();
    go = 1;
    asm volatile ("rdcycle %0"   : "=r"(cyc0));
    asm volatile ("rdinstret %0" : "=r"(ins0));
    
    //------------------------------------------------------------------------
    // 1) Quantize float input → int8[input_q]
    //------------------------------------------------------------------------
    quant_f32(
        /* size */   BATCHES / 2  * 784,
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
        /* BATCHES / 2  */ BATCHES / 2 ,
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
        /* BATCHES / 2  */ BATCHES / 2 ,
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
        /* size */   BATCHES / 2  * 10,
        /* input */  logits_q,
        /* output */ logits_f32,
        /* qparams*/ qp_output
        // or, if you still have qp_out1 in your header:
        // qp_out1
    );

    for (size_t b = 0; b < BATCHES / 2; b++) {
        softmax_vec(
            &logits_f32[b * 10],
            &probs    [b * 10],
            /* length */ 10,
            /* stride */ 1
        );
    }

    while (!done) ;
    go = 0;

    asm volatile ("fence");
    asm volatile ("rdcycle %0"   : "=r"(cyc1));
    asm volatile ("rdinstret %0" : "=r"(ins1));

    printf("Execution cycles      : %lu\n", cyc1 - cyc0);
    printf("Instructions executed : %lu\n", ins1 - ins0);

    //------------------------------------------------------------------------
    // 5) Softmax & print results
    //------------------------------------------------------------------------
    for (size_t b = 0; b < BATCHES; b++) {
        if (b < BATCHES / 2) {
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
        } else {
            size_t b2 = b - (BATCHES / 2);
            int   pred = 0;
            float maxp = probs2[b2 * 10];
            for (int c = 1; c < 10; c++) {
                if (probs2[b2*10 + c] > maxp) {
                    maxp = probs2[b2*10 + c];
                    pred = c;
                }
            }
            printf("Input %d: Predicted %d (probs:", b, pred);
            for (int c = 0; c < 10; c++) {
                printf(" %d", (int)(probs2[b2*10 + c]*100.0f));
            }
            printf(" )\n");
        }
        
    }

    return 0;
}
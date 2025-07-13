/********************************************************************
 * main_quant.c – quantised Conv→Pool→Conv→Pool→FC→FC(+softmax)
 *                Uses the RVV int8 kernels with combined bias+weights blobs.
 ********************************************************************/
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "model_params.h"    /* defines uint8_t dw0_wb[], pw0_wb[], etc. */
#include "data.h"      /* defines float input[] */
#include "lib_layers.h"      /* conv2D_3x3_int8, maxpool_int8, quant_fully_connected_int8, ... */
#define TCM 0
#define TCM_ADDRESS 0x78000000

/* -------- helper ------------------------------------------------- */
static int argmax10(const float *v)
{
    int idx = 0;
    float mx = v[0];
    for (int i = 1; i < 10; ++i) {
        if (v[i] > mx) {
            mx = v[i];
            idx = i;
        }
    }
    return idx;
}

/* -------- static buffers (int8 unless commented) ----------------- */
#if (!TCM)

    static int8_t  input_q   [BATCHES * 28 * 28];
    static int8_t  conv0_out [1 * 26 * 26];
    static int8_t  pw0_out   [16 * 26 * 26];
    static int8_t  pool0_out [16 *  8 *  8];

    static int8_t  conv1_out [16 *  6 *  6];
    static int8_t  pw1_out   [6 * 6 * 32];
    static int8_t  pool1_out [32 *  2 *  2];   /* = 128 int8 */

    static int8_t  dense0_q  [BATCHES * 32];
    static int8_t  logits_q  [BATCHES * 10];

    static float   logits_f32[BATCHES * 10];
    static float   probs      [BATCHES * 10];

#else 
    static int8_t*  input_q = (int8_t*)TCM_ADDRESS;
    static int8_t  conv0_out [16 * 26 * 26];
    static int8_t*  pool0_out = (int8_t*)(TCM_ADDRESS + 28*28);
    static int8_t*  conv1_out  = (int8_t*)(TCM_ADDRESS + 16*8*8 + 28*28);
    static int8_t*  pool1_out  = (int8_t*)(TCM_ADDRESS + 32*6*6 + 16*8*8 + 28*28);
    static int8_t*  dense0_q   = (int8_t*)(TCM_ADDRESS + 32*2*2 + 32*6*6 + 16*8*8 + 28*28);
    static int8_t*  logits_q   = (int8_t*)(TCM_ADDRESS + 32 + 32*2*2 + 32*6*6 + 16*8*8 + 28*28);
    
    static float*   logits_f32 = (float*)(TCM_ADDRESS + 10 + 32 + 32*2*2 + 32*6*6 + 16*8*8 + 28*28 + 2);
    static float*   probs      = (float*)(TCM_ADDRESS + 50 + 32 + 32*2*2 + 32*6*6 + 16*8*8 + 28*28 + 2); // 3210 elems ends in address: 0x78000C8A

#endif

int main(void)
{
    // printf("probs: %p", probs );
    int i = 0;
    while (i < 20) {
        /* cycle counter ------------------------------------------------ */
        unsigned long cyc0, cyc1, ins0, ins1;
        asm volatile("rdcycle %0"   : "=r"(cyc0));
        asm volatile("rdinstret %0" : "=r"(ins0));

        /* --------------------------------------------------------------
           0) Quant input
           -------------------------------------------------------------- */
        quant_f32(
            BATCHES * 28*28,
            input + i*28*28,
            input_q,
            qp_input
        );

        /* optional: print quantized input for debugging */
        // printf("Input to conv: \n");
        // for (int r = 0; r < 28; r++) {
        //     for (int c = 0; c < 28; c++) {
        //         printf("%d, ", (int) input_q[r*28 + c]);
        //     }
        //     printf("\n");
        // }

        /* --------------------------------------------------------------
           1) Conv-0 : 1×28×28 → 16×26×26   (DW)
           -------------------------------------------------------------- */
        conv2D_3x3_int8(
            /* H,W          */ 28, 28,
            /* Cin          */ 1,
            /* stride       */ 1,
            /* padding      */ 0,
            /* weights      */ (const void*) dw0_wb_q, 
            /* io           */ input_q, conv0_out,
            /* relu         */ 0,
            /* rq params    */ rq_conv0_dw
        );

        /* optional: print conv0_out */
        // printf("output conv 0\n");
        // for (size_t ch = 0; ch < 1; ch++) {
        //     printf("Channel %d:\n", ch);
        //     for (size_t r = 0; r < 26; r++) {
        //         for (size_t c = 0; c < 26; c++) {
        //             size_t idx = ch*26*26 + r*26 + c;
        //             printf("%d ", conv0_out[idx]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        conv_1x1_int8(
            26, 26, 
            1, 16,
            1, 0, 
            conv0_out, 
            (const void*) pw0_wb_q,
            pw0_out, 
            1, 
            rq_conv0_pw
        );
        /* optional: print pw0_out */
        // printf("output pointwise 0\n");
        // for (size_t ch = 0; ch < 16; ch++) {
        //     printf("Channel %d:\n", ch);
        //     for (size_t r = 0; r < 26; r++) {
        //         for (size_t c = 0; c < 26; c++) {
        //             size_t idx = ch*26*26 + r*26 + c;
        //             printf("%d ", pw0_out[idx]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        /* --------------------------------------------------------------
           2) MaxPool-0 : 3×3,str=3  –> 16×8×8
           -------------------------------------------------------------- */
        maxpool_int8(
            /* out rows,cols */ 8, 8,
            /* in  rows,cols */ 26, 26,
            /* channels      */ 16,
            /* stride        */ 3,
            pw0_out, pool0_out
        );

        /* optional: print pool0_out */
        // printf("output maxpool 0\n");
        // for (size_t ch = 0; ch < 16; ch++) {
        //     printf("Channel %d:\n", ch);
        //     for (size_t r = 0; r < 8; r++) {
        //         for (size_t c = 0; c < 8; c++) {
        //             size_t idx = ch*64 + r*8 + c;
        //             printf("%d ", pool0_out[idx]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        /* --------------------------------------------------------------
           3) Conv-1 : 16×8×8 → 32×6×6
           -------------------------------------------------------------- */
        conv2D_3x3_int8(
            /* H,W          */ 8, 8,
            /* Cin,Cout     */ 16,
            /* stride       */ 1,
            /* padding      */ 0,
            (const void*) dw1_wb_q,
            pool0_out, conv1_out,
            0,
            rq_conv1_dw
        );

        // Print output of conv1_out
        // printf("output conv 1\n");
        // for (size_t ch = 0; ch < 16; ch++) {
        //     printf("Channel %zu:\n", ch);
        //     for (size_t r = 0; r < 6; r++) {
        //     for (size_t c = 0; c < 6; c++) {
        //         size_t idx = ch*36 + r*6 + c;
        //         printf("%d ", conv1_out[idx]);
        //     }
        //     printf("\n");
        //     }
        //     printf("\n");
        // }

        conv_1x1_int8(
            6, 6, 
            16, 32,
            1, 0, 
            conv1_out, 
            (const void*) pw1_wb_q,
            pw1_out,
            1, 
            rq_conv1_pw
        );

        // Print output of pw1_out
        // printf("output pointwise 1\n");
        // for (size_t ch = 0; ch < 32; ch++) {
        //     printf("Channel %zu:\n", ch);
        //     for (size_t r = 0; r < 6; r++) {
        //     for (size_t c = 0; c < 6; c++) {
        //         size_t idx = ch*36 + r*6 + c;
        //         printf("%d ", pw1_out[idx]);
        //     }
        //     printf("\n");
        //     }
        //     printf("\n");
        // }

        /* --------------------------------------------------------------
           4) MaxPool-1 : 3×3,str=3  –> 32×2×2
           -------------------------------------------------------------- */
        maxpool_int8(
            /* out rows,cols */ 2, 2,
            /* in  rows,cols */ 6, 6,
            /* channels      */ 32,
            /* stride        */ 3,
            pw1_out, pool1_out
        );

        // printf("output maxpool 0\n");
        // for (size_t ch = 0; ch < 32; ch++) {
        //     printf("Channel %d:\n", ch);
        //     for (size_t r = 0; r < 2; r++) {
        //         for (size_t c = 0; c < 2; c++) {
        //             size_t idx = ch*4 + r*2 + c;
        //             printf("%d ", pool1_out[idx]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        // printf("input fully connected 0\n");
        // for (size_t ch = 0; ch < 1; ch++) {
        //     printf("Channel %d:\n", ch);
        //     for (size_t r = 0; r < 1; r++) {
        //         for (size_t c = 0; c < 128; c++) {
        //             size_t idx = ch*128 + r*128 + c;
        //             printf("%d, ", pool1_out[idx]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        /* --------------------------------------------------------------
           5) FC-0 : 128 → 32   (+ReLU)
           -------------------------------------------------------------- */
        quant_fully_connected_int8(
            /* in, out       */ 128, 32,
            /* batches       */ 1,
            /* input         */ pool1_out,
            /* weights+bias  */ (const void*) fc0_wb_q,
            /* output        */ dense0_q,
            /* relu flag     */ 1, 1,
            /* rq params     */ rq_fc0
        );

        // printf("output fully connected 0\n");
        // for (size_t ch = 0; ch < 1; ch++) {
        //     printf("Channel %d:\n", ch);
        //     for (size_t r = 0; r < 1; r++) {
        //         for (size_t c = 0; c < 32; c++) {
        //             size_t idx = ch*32 + r*32 + c;
        //             printf("%d, ", dense0_q[idx]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        /* --------------------------------------------------------------
           6) FC-1 : 32 → 10    (logits only)
           -------------------------------------------------------------- */
        quant_fully_connected_int8(
            32, 10,
            1,
            dense0_q,
            (const void*) fc1_wb_q,
            logits_q,
            0, 1,
            rq_fc1
        );

        // printf("output fully connected 1\n");
        // for (size_t ch = 0; ch < 1; ch++) {
        //     printf("Channel %d:\n", ch);
        //     for (size_t r = 0; r < 1; r++) {
        //         for (size_t c = 0; c < 10; c++) {
        //             size_t idx = ch*10 + r*10 + c;
        //             printf("%d, ", logits_q[idx]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        /* --------------------------------------------------------------
           7) Dequant logits → float, Softmax, print
           -------------------------------------------------------------- */
        dequant_f32(
            BATCHES * 10,
            logits_q,
            logits_f32,
            qp_logits
        );
        for (size_t b = 0; b < BATCHES; b++) {
            softmax_vec(&logits_f32[b*10], &probs[b*10], 10, 1);
        }

        asm volatile("fence");
        asm volatile("rdcycle %0"   : "=r"(cyc1));
        asm volatile("rdinstret %0" : "=r"(ins1));

        printf("Cycles      : %lu\n", cyc1 - cyc0);
        printf("Instructions: %lu\n", ins1 - ins0);

        for (size_t b = 0; b < BATCHES; b++) {
            int pred = argmax10(&probs[b*10]);
            printf("Sample %d → %d  probs:", i, pred);
            for (int c = 0; c < 10; c++) {
                printf(" %d", (int)(probs[b*10 + c]*100));
            }
            printf("\n");
        }

        i++;
    }

    return 0;
}
/********************************************************************
 * main.c – run a   Conv→Pool→Conv→Pool→FC→FC(+softmax)  network
 *            using the RVV layers you already implemented.
 ********************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <stddef.h>
 #include "model_params.h"
 #include "input_data.h"
 #include "lib_layers.h"      // prototypes for the 4 C primitives
 #include "encoding.h"
 
 /* --------------------------------------------------------------
    Network topology (valid padding everywhere)
       Input               :  1 × 28 × 28
       Conv3×3,16,relu     : 16 × 26 × 26
       MaxPool3×3,str=3    : 16 ×  8 ×  8
       Conv3×3,32,relu     : 32 ×  6 ×  6
       MaxPool3×3,str=3    : 32 ×  2 ×  2
       Flatten             : 128
       FC 128→32,relu
       FC  32→10,softmax
 ----------------------------------------------------------------- */
 volatile uint32_t go       = 0;   /* 0 = parked, 1 = start work      */
 volatile uint32_t done     = 0;   /* hart-1 sets to 1 when finished  */
 static float probs2 [BATCHES  *  10];


 void __main(void) 
 {
    const uint32_t me = read_csr(mhartid);
    static float conv0_out2 [1 * 26 * 26];
    static float pw0_out2   [16 * 26 * 26]; // pointwise conv output
    static float pool0_out2 [16 *  8 *  8];
       
    static float conv1_out2 [16 *  6 *  6];
    static float pw1_out2   [32 *  6 *  6]; // pointwise conv output
    static float pool1_out2 [32 *  2 *  2];   // = 128 floats
       
    static float dense0_out2[BATCHES * 32];
    static float logits2    [BATCHES * 10];
    size_t j = 1;


    while (1) {
        while (!go) ;

        dw_conv2D_3x3_f32(
            28, 28,                       /* H,W              */
            1,                            /* Cin,Cout         */
            1,                            /* stride           */
            0,                            /* padding VALID    */
            dw0,                          /* weights          */
            input + j*784,                /* input  CHW       */
            conv0_out2,                    /* output CHW       */
            /* relu_dw = */ 0
        );

        conv2D_1x1_f32(
            26, 26,                       /* H,W              */
            1, 16,                        /* Cin,Cout         */
            1, 0,                         /* stride, padding  */
            pw0,                          /* weights          */
            conv0_out2,                   /* input  CHW       */
            pw0_out2,                     /* output CHW       */
            /* relu = */ 1                /* ReLU after PW    */
        );
    
        /* ---------------- MaxPool‑0 : 3×3,str=3 --------------------------- */
        maxpool_f32(
            8, 8,                         /* output rows,cols  */
            26, 26,                       /* input  rows,cols  */
            16,                           /* channels          */
            3,                            /* stride            */
            pw0_out2, pool0_out2
        );
    
        /* ---------------- Conv‑1 : 16×8×8 → 32×6×6 ------------------------ */
        dw_conv2D_3x3_f32(
            8, 8,                         /* H,W               */
            16,                           /* Cin,Cout          */
            1,                            /* stride            */
            0,                            /* padding VALID     */
            dw1,                          /* weights           */
            pool0_out2,                   /* input             */
            conv1_out2,                   /* output            */
            0                             /* ReLU only after PW*/
        );

        conv2D_1x1_f32(
            6, 6,                         /* H,W              */
            16, 32,                       /* Cin,Cout         */
            1, 0,                         /* stride, padding  */
            pw1,                          /* weights          */
            conv1_out2,                    /* input  CHW       */
            pw1_out2,                      /* output CHW       */
            /* relu = */ 1                /* ReLU after PW    */
     );
    
        /* ---------------- MaxPool‑1 : 3×3,str=3 → 32×2×2 ----------------- */
        maxpool_f32(
            2, 2,                         /* out rows,cols     */
            6, 6,                         /* in  rows,cols     */
            32,                           /* channels          */
            3,                            /* stride            */
            pw1_out2, pool1_out2
        );
    
        /* ---------------- FC‑0 : 128 → 32  + ReLU ------------------------- */
        fully_connected_f32(
            128, 32,
            BATCHES,
            pool1_out2,        /* input 128 floats       */
            fc0,            /* bias||weights          */
            dense0_out2,
            /* relu */ 1
        );
    
        /* ---------------- FC‑1 : 32 → 10  (logits) ------------------------ */
        fully_connected_f32(
            32, 10,
            BATCHES,
            dense0_out2,
            fc1,
            logits2,
            /* relu */ 0
        );
    
        /* ---------------- Softmax per sample ----------------------------- */
        for (size_t b = 0; b < BATCHES; ++b) {
            softmax_vec(&logits2[b*10], &probs2[b*10], 10, 1);
        }

        done = 1;
        j += 2;
        go = 0;

    }
 }


 /* helper: argmax over 10 vals */
 static int argmax10(const float *v)
 {
     int idx = 0;
     float mx = v[0];
     for (int i = 1; i < 10; ++i) {
         if (v[i] > mx) { mx = v[i]; idx = i; }
     }
     return idx;
 }

 
 int main(void)
 {

     static float conv0_out [1 * 26 * 26];
    static float pw0_out   [16 * 26 * 26]; // pointwise conv output
     static float pool0_out [16 *  8 *  8];
        
     static float conv1_out [16 *  6 *  6];
     static float pw1_out   [32 *  6 *  6]; // pointwise conv output
     static float pool1_out [32 *  2 *  2];   // = 128 floats
        
     static float dense0_out[BATCHES * 32];
     static float logits    [BATCHES * 10];
     static float probs     [BATCHES * 10];

     printf("hello from core 0 \n");

     int i = 0;
     while (i < 18) {

     __sync_synchronize();
     go = 1;

     unsigned long cyc0, cyc1, ins0, ins1;
     asm volatile ("rdcycle %0"   : "=r"(cyc0));
     asm volatile ("rdinstret %0" : "=r"(ins0));
     
 
     /* ---------------- Conv‑0 : 1×28×28 → 16×26×26 -------------------- */
     dw_conv2D_3x3_f32(
         28, 28,                       /* H,W              */
         1,                            /* Cin,Cout         */
         1,                            /* stride           */
         0,                            /* padding VALID    */
         dw0,                          /* weights          */
         input + i*784,                /* input  CHW       */
         conv0_out,                    /* output CHW       */
         /* relu_dw = */ 0
     );

     conv2D_1x1_f32(
         26, 26,                       /* H,W              */
         1, 16,                        /* Cin,Cout         */
         1, 0,                         /* stride, padding  */
         pw0,                          /* weights          */
         conv0_out,                    /* input  CHW       */
         pw0_out,                      /* output CHW       */
         /* relu = */ 1                /* ReLU after PW    */
     ); 
 
     /* ---------------- MaxPool‑0 : 3×3,str=3 --------------------------- */
     maxpool_f32(
         8, 8,                         /* output rows,cols  */
         26, 26,                       /* input  rows,cols  */
         16,                           /* channels          */
         3,                            /* stride            */
         pw0_out, pool0_out
     );
 
     /* ---------------- Conv‑1 : 16×8×8 → 32×6×6 ------------------------ */
     dw_conv2D_3x3_f32(
         8, 8,                         /* H,W               */
         16,                           /* Cin,Cout          */
         1,                            /* stride            */
         0,                            /* padding VALID     */
         dw1,                          /* weights           */
         pool0_out,                    /* input             */
         conv1_out,                    /* output            */
         0                             /* ReLU only after PW*/
     );

     conv2D_1x1_f32(
         6, 6,                         /* H,W              */
         16, 32,                       /* Cin,Cout         */
         1, 0,                         /* stride, padding  */
         pw1,                          /* weights          */
         conv1_out,                    /* input  CHW       */
         pw1_out,                      /* output CHW       */
         /* relu = */ 1                /* ReLU after PW    */
     );

 
     /* ---------------- MaxPool‑1 : 3×3,str=3 → 32×2×2 ----------------- */
     maxpool_f32(
         2, 2,                         /* out rows,cols     */
         6, 6,                         /* in  rows,cols     */
         32,                           /* channels          */
         3,                            /* stride            */
         pw1_out, pool1_out
     );
 
     /* ---------------- FC‑0 : 128 → 32  + ReLU ------------------------- */
     fully_connected_f32(
         128, 32,
         BATCHES,
         pool1_out,        /* input 128 floats       */
         fc0,            /* bias||weights          */
         dense0_out,
         /* relu */ 1
     );
 
     /* ---------------- FC‑1 : 32 → 10  (logits) ------------------------ */
     fully_connected_f32(
         32, 10,
         BATCHES,
         dense0_out,
         fc1,
         logits,
         /* relu */ 0
     );
 
     /* ---------------- Softmax per sample ----------------------------- */
     for (size_t b = 0; b < BATCHES; ++b) {
         softmax_vec(&logits[b*10], &probs[b*10], 10, 1);
     }
 
     while (!done) ;
     done = 0;
     asm volatile ("fence");
     asm volatile ("rdcycle %0"   : "=r"(cyc1));
     asm volatile ("rdinstret %0" : "=r"(ins1));
 
     printf("Execution cycles      : %lu\n", cyc1 - cyc0);
     printf("Instructions executed : %lu\n", ins1 - ins0);
 
     /* ---------------- Print predictions ------------------------------ */
     for (size_t b = 0; b < BATCHES; ++b) {
         int pred = argmax10(&probs[b*10]);
         printf("Input %d → Predicted digit %d, probs:", i, pred);
         for (int c = 0; c < 10; ++c)
             printf(" %d", (int)(100*probs[b*10+c]));
         printf(" \n");
     }
     for (size_t b = 0; b < BATCHES; ++b) {
        int pred = argmax10(&probs2[b*10]);
        printf("Input %d → Predicted digit %d, probs:", i + 1, pred);
        for (int c = 0; c < 10; ++c)
            printf(" %d", (int)(100*probs2[b*10+c]));
        printf(" \n");
    }
     i+=2;
    }
     return 0;
 }
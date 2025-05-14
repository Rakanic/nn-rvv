#include <stdio.h>
#include "model_params.h"   // Contains layer0 (64 parameters), layer1 arrays, etc.
#include "input_data.h"     // Contains 'input' array for 19 inputs (each image has 784 floats)
#include "lib_layers.h"   

int main(void)
{
    /* Intermediate activation buffers */
    static float dense0_out[BATCHES * 64]; /* after Dense-0 + ReLU */
    static float logits    [BATCHES * 10];  /* output logits from Dense-1 */
    static float probs     [BATCHES * 10];  /* softmax probabilities */

    unsigned long cycles_start, cycles_end, instr_start, instr_end;
    asm volatile ("rdcycle %0" : "=r" (cycles_start));
    asm volatile ("rdinstret %0" : "=r" (instr_start));

    /* ---------------- Layer 0: 784 → 64 + ReLU ------------------------- */
    // fully_connected_f32 takes: input_dim, output_dim, batches, input, weight bias array, output, activation flag
    fully_connected_f32(784, 64, BATCHES, input, layer0, dense0_out, 1);

    /* ---------------- Layer 1: 64 → 10 (logits) ------------------------ */
    fully_connected_f32(64, 10, BATCHES, dense0_out, layer1, logits, 0);

    /* ---------------- Softmax per batch --------------------------------- */
    for (size_t b = 0; b < BATCHES; ++b) {
        softmax_vec(&logits[b * 10], &probs[b * 10], 10, 1);
    }

    asm volatile ("fence");
    asm volatile ("rdcycle %0" : "=r" (cycles_end));
    asm volatile ("rdinstret %0" : "=r" (instr_end));

    printf("  Execution cycles:      %lu\n", cycles_end - cycles_start);
    printf("  Instructions executed: %lu\n\n", instr_end - instr_start);

    /* ---------------- Print probabilities and predicted classes ------------------------------- */
    for (size_t b = 0; b < BATCHES; ++b) {
        int predicted = 0;
        float max_prob = probs[b * 10];
        for (int c = 1; c < 10; ++c) {
            if (probs[b * 10 + c] > max_prob) {
                max_prob = probs[b * 10 + c];
                predicted = c;
            }
        }
        printf("Input %d: Predicted digit %d, probabilities: ", b, predicted);
        for (int c = 0; c < 10; ++c) {
            printf("%d ", (int) (100 * probs[b * 10 + c]));
        }
        printf("\n");
    }

    return 0;
}
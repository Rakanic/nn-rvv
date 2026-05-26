#ifndef NN_RVV_OPS_REDUCE_H
#define NN_RVV_OPS_REDUCE_H

#include <stddef.h>

#include <stdint.h>

/* sum_i a[i] * b[i] */
float dot_f32(const float *a, const float *b, size_t n);

/* sum_i a[i] * bf16_to_f32(b[i])  — `b` is a row of bf16 weights. */
float f32_bf16_dot(const float *a, const uint16_t *b_bf16, size_t n);

/* sum_i x[i] */
float sum_f32(const float *x, size_t n);

/* max_i x[i] (precondition: n > 0) */
float max_f32(const float *x, size_t n);

#endif

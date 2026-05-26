#ifndef NN_RVV_OPS_REDUCE_H
#define NN_RVV_OPS_REDUCE_H

#include <stddef.h>

/* sum_i a[i] * b[i] */
float dot_f32(const float *a, const float *b, size_t n);

/* sum_i x[i] */
float sum_f32(const float *x, size_t n);

/* max_i x[i] (precondition: n > 0) */
float max_f32(const float *x, size_t n);

#endif

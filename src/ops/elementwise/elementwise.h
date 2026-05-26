#ifndef NN_RVV_OPS_ELEMENTWISE_H
#define NN_RVV_OPS_ELEMENTWISE_H

#include <stddef.h>

/* y[i] = c                                    */
void fill_f32(float *y, float c, size_t n);

/* y[i] += a * x[i]                            */
void axpy_f32(float *y, float a, const float *x, size_t n);

/* y[i] = a * x[i] + b                         */
void scale_add_f32(float *y, const float *x, float a, float b, size_t n);

#endif

// quantization_params.h
#ifndef QUANTIZATION_PARAMS_H
#define QUANTIZATION_PARAMS_H

#include <stdint.h>

typedef struct {
    float   scale;
    int32_t zero_point;
} quantization_params_t;

typedef struct {
    float*  scale;
    int32_t zero_point;
} requantization_params_t;

#endif // QUANTIZATION_PARAMS_H
#include "stdio.h"
#include <stdint.h>

void int8_maxpool_ukernel_3x3__rvv_str1(
    size_t output_cols, 
    size_t output_rows,
    size_t input_cols,
    const int8_t* input,
    int8_t* output
);

void int8_maxpool_ukernel_3x3__rvv_str2(
    size_t output_cols, 
    size_t output_rows,
    size_t input_cols,
    const int8_t* input,
    int8_t* output
);

void int8_maxpool_ukernel_3x3__rvv_str3(
    size_t output_cols, 
    size_t output_rows,
    size_t input_cols,
    const int8_t* input,
    int8_t* output
);

void f32_maxpool_ukernel_3x3__rvv_str1(
    size_t output_cols, 
    size_t output_rows,
    size_t input_cols,
    const float* input,
    float* output
);

void f32_maxpool_ukernel_3x3__rvv_str2(
    size_t output_cols, 
    size_t output_rows,
    size_t input_cols,
    const float* input,
    float* output
);

void f32_maxpool_ukernel_3x3__rvv_str3(
    size_t output_cols, 
    size_t output_rows,
    size_t input_cols,
    const float* input,
    float* output
);
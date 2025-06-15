#ifndef MATMUL_H
#define MATMUL_H

#include "stdio.h"
#include <stdint.h>
#include "quantization_params.h"

void f32_gemm_relu(
    size_t M, size_t N, size_t K,
    const float* A, size_t a_row_stride,
    const float* B,
    float* C, size_t c_row_stride,
    size_t c_col_stride);

void f32_gemm(
    size_t M, size_t N, size_t K,
    const float* A, size_t a_row_stride,
    const float* B,
    float* C, size_t c_row_stride,
    size_t c_col_stride);

void int8_gemm_relu(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    quantization_params_t qp_input, 
    quantization_params_t qp_weights, 
    quantization_params_t qp_output);

void int8_gemm(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    quantization_params_t qp_input, 
    quantization_params_t qp_weights, 
    quantization_params_t qp_output);

void int8_qgemm_relu(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);

void int8_qgemm(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);


void int8_qgemm_nobias(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);


void int8_qgemm_relu_nobias(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);

void int8_qgemm_int32bias_relu(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const void* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);

void int8_qgemm_int32bias(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const void* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);

void int8_qgemm_int32bias_conv1x1(
    size_t M, size_t N, size_t K,
    const void* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);

void int8_qgemm_int32bias_conv1x1_relu(
    size_t M, size_t N, size_t K,
    const void* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);
#endif
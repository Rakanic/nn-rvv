#ifndef NN_RVV_OPS_MATMUL_H
#define NN_RVV_OPS_MATMUL_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "nn_rvv/layers.h"

/* ---- f32 GEMM (vec-nn authoritative) ---- */
void f32_gemm(
  size_t M, size_t N, size_t K,
  const float* A, size_t a_row_stride,
  const float* B,
  float* C, size_t c_row_stride,
  size_t c_col_stride);

void f32_gemm_relu(
  size_t M, size_t N, size_t K,
  const float* A, size_t a_row_stride,
  const float* B,
  float* C, size_t c_row_stride,
  size_t c_col_stride);

void f32_gemm_nobias(
  size_t M, size_t N, size_t K,
  const float* A, size_t a_row_stride,
  const float* B,
  float* C, size_t c_row_stride,
  size_t c_col_stride);

/* ---- Legacy int8 GEMM (separate quantization params per operand) ---- */
void int8_gemm(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    quantization_params_t qp_input,
    quantization_params_t qp_weights,
    quantization_params_t qp_output);

void int8_gemm_relu(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    quantization_params_t qp_input,
    quantization_params_t qp_weights,
    quantization_params_t qp_output);

/* ---- Quantized int8 GEMM with per-channel requantization ---- */
void int8_qgemm(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);

void int8_qgemm_relu(
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

/* ---- int32-bias variants (weights+bias packed; A is int8) ---- */
void int8_qgemm_int32bias(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const void* B,
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

/* ---- Conv1x1-specialized int32-bias variants (A is packed weights, B is activations) ---- */
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

/*
 * int8_qgemm_fout — Int8×Int8 → Float32 GEMM.
 *
 * Same structure as int8_qgemm but requantizes the int32 accumulator to
 * float32 using a single scalar `scale` (no per-channel scale array, no
 * clamping to int8 range).
 *
 * Intended for the transposed matmul in int8 transformer inference:
 *   x_q(1,K) @ W_T_pack → xout(1,N)  with scale = 1/(127*127)
 *
 * B layout: [(K+1) × N] int8 bytes
 *   Row 0       : N zeros (zero bias, because weights are pre-converted
 *                  from uint8 to int8 by subtracting 128 at startup)
 *   Rows 1 .. K : N int8 bytes per row (rows of W_T)
 */
void int8_qgemm_fout(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    float* C, size_t c_row_stride,
    size_t c_col_stride,
    float scale);

#endif /* NN_RVV_OPS_MATMUL_H */

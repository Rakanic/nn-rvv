#ifndef NN_RVV_LAYERS_H
#define NN_RVV_LAYERS_H

#include <stdint.h>
#include <stddef.h>

/*---------------------------------------------*/
/*                                             */
/* Quantization helpers                        */
/*                                             */
/*---------------------------------------------*/
typedef struct {
    float   scale;
    int32_t zero_point;
} quantization_params_t;

typedef struct {
    float*  scale;
    int32_t zero_point;
} requantization_params_t;

void quant_f32(
    size_t size,
    float* input,
    int8_t* output,
    quantization_params_t qp
);

void dequant_f32(
    size_t size,
    int8_t* input,
    float* output,
    quantization_params_t qp
);

void requant_outch_int32(
    size_t rows, size_t cols,
    size_t channels,
    int32_t* input,
    int8_t* output,
    int relu,
    requantization_params_t rqp
);

/*---------------------------------------------*/
/*                                             */
/* Transpose                                   */
/*                                             */
/*---------------------------------------------*/
void transpose_int8 (int8_t* input, int8_t* output, size_t rows, size_t cols);
void transpose_f32  (const float *input, float *output, size_t rows, size_t cols);

/*---------------------------------------------*/
/*                                             */
/* Normalization                               */
/*                                             */
/*---------------------------------------------*/
/* RMS norm: out[i] = weight[i] * in[i] / sqrt(mean(in^2) + eps). */
void rmsnorm_f32(float *out, const float *in, const float *weight, size_t size);

/* LayerNorm:  out[i] = (in[i] - mean) / sqrt(var + eps) * weight[i] + bias[i].
 * weight or bias may be NULL (plain LN). */
void layer_norm_f32(float *out, const float *in,
                    const float *weight, const float *bias,
                    size_t size, float eps);

/*---------------------------------------------*/
/*                                             */
/* Activations                                 */
/*                                             */
/*---------------------------------------------*/
void silu_f32(float *x, size_t n);              /* x[i] = x[i] / (1 + exp(-x[i]))                  */
void gelu_f32(float *x, size_t n);              /* GELU (tanh approximation), in place              */

/* Fused SwiGLU multiply: out[i] = SiLU(gate_up[2i]) * gate_up[2i+1], i in [0, inter). */
void swiglu_multiply_f32(float *out, const float *gate_up, size_t inter);

/*---------------------------------------------*/
/*                                             */
/* Positional encoding                         */
/*                                             */
/*---------------------------------------------*/
/* RoPE: rotate q[n_heads * head_size] and k[n_kv_heads * head_size] in place
 * by the per-pair angle for token position `pos` (llama2 convention). */
void rope_f32(
    float *q, float *k,
    size_t n_heads, size_t n_kv_heads, size_t head_size,
    size_t pos
);

/* NeoX-style RoPE: rotates the two halves of each head against each other.
 *   x [n_heads * head_dim] for one seq position; cos/sin are per-seq tables
 *   of length head_dim (we read the first head_dim/2 entries). */
void rope_neox_apply_f32(float *x,
                         const float *cos_vals, const float *sin_vals,
                         size_t n_heads, size_t head_dim);

/*---------------------------------------------*/
/*                                             */
/* Dense 2D Convolution (im2col + GEMM)        */
/*                                             */
/*---------------------------------------------*/
/* Dense (non-depthwise) Conv2D with optional bias.
 *
 *   in           [c_in × h_in × w_in]                  CHW input
 *   weight       [c_out × c_in × kh × kw]              dense kernel
 *   bias         [c_out] or NULL                       per-output-channel bias
 *   cols_scratch [c_in*kh*kw × h_out*w_out]            caller-provided
 *   out          [c_out × h_out × w_out]               CHW output
 *
 * h_out, w_out are derived from h_in/w_in/kh/kw/stride/padding (VALID/SAME).
 * The kernel does not allocate. */
void conv2d_f32(float *out,
                const float *in,
                const float *weight,
                const float *bias,
                float *cols_scratch,
                size_t c_in, size_t h_in, size_t w_in,
                size_t c_out, size_t kh, size_t kw,
                size_t stride, size_t padding);

/*---------------------------------------------*/
/*                                             */
/* Fully Connected Layers                      */
/*                                             */
/*---------------------------------------------*/
void fully_connected_f32 (
    size_t input_size,
    size_t output_size,
    size_t batches,
    float* input,
    const float* weights_with_bias,
    float* output,
    int relu
);

void fully_connected_f32_nobias (
    size_t input_size,
    size_t output_size,
    size_t batches,
    float* input,
    const float* weights,
    float* output,
    int relu
);

/* Legacy unquantized int8 FC path (separate quantization params per operand). */
void fully_connected_int8 (
    size_t input_size,
    size_t output_size,
    size_t batches,
    int8_t* input,
    const int8_t* weights_with_bias,
    int8_t* output,
    int relu,
    quantization_params_t qp_input,
    quantization_params_t qp_weights,
    quantization_params_t qp_output
);

void quant_fully_connected_int8 (
    size_t input_size,
    size_t output_size,
    size_t batches,
    int8_t* input,
    const void* weights_with_bias,
    int8_t* output,
    int relu, int bias32,
    requantization_params_t requant_params
);

/*
 * quant_fully_connected_int8_t — transposed int8 fully-connected layer.
 *
 * Equivalent semantics to quant_fully_connected_int8 but expects the weight
 * matrix already transposed and packed, so that the vectorized dimension is
 * output_size (N) rather than the batch dimension.
 *
 * Changes vs. quant_fully_connected_int8:
 *   - Requantization: single scalar `scale` applied to the int32 accumulator
 *     to produce float32 output (no per-channel scale, no int8 narrowing).
 *   - Input bias term: zero — weights must be pre-converted from uint8 to
 *     int8 by subtracting 128 before calling (done once at model load time),
 *     so no zero-point correction is required at inference.
 *
 * weights_t_pack layout: [(input_size+1) × output_size] int8 bytes
 *   Row 0             : output_size zero bytes  (zero bias)
 *   Rows 1..input_size: rows of W_T as signed int8 (= original_uint8 − 128)
 *
 * Typical call for single-token transformer inference (batches=1):
 *   quant_fully_connected_int8_t(n, d, 1, x_q, w_t_pack, xout,
 *                                1.0f / (127.0f * 127.0f));
 */
void quant_fully_connected_int8_t(
    size_t input_size,
    size_t output_size,
    size_t batches,
    const int8_t* input,
    const void* weights_t_pack,
    float* output,
    float scale
);

/*---------------------------------------------*/
/*                                             */
/* 2D Convolution Layers                       */
/*                                             */
/*---------------------------------------------*/

/* Int8 depthwise 3x3 conv (vec-nn dispatcher; honors padding). */
void dwconv2D_3x3_int8 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const void *dw_weights,  // length = Cin*(1 + 9)
    int8_t *input,       // CHW: [Cin][H][W]
    int8_t *output,            // CHW: [Cout][H_out][W_out]
    int relu,
    requantization_params_t requant_params_dwconv
);

/* Int8 depthwise 3x3 conv (legacy nn-rvv dispatcher; ignores padding arg). */
void conv2D_3x3_int8 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const void *dw_weights,  // length = Cin*(1 + 9)
    int8_t *input,       // CHW: [Cin][H][W]
    int8_t *output,            // CHW: [Cout][H_out][W_out]
    int relu,
    requantization_params_t requant_params_dwconv
);

void dw_conv2D_3x3_f32 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding,
    const float *dw_weights,  // length = Cin*(1 + 9)
    float *input,
    float *output,
    int relu_dw
);

void dw_conv2D_5x5_f32 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding,
    const float *dw_weights,  // length = Cin*(1 + 25)
    float *input,
    float *output,
    int relu_dw
);

void dw_conv2D_5x5_int8 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding,
    const void *dw_weights,  // length = Cin*(1 + 25)
    int8_t *input,
    int8_t *output,
    int relu,
    requantization_params_t requant_params_dwconv
);

/*---------------------------------------------*/
/*                                             */
/* Pointwise Convolution Layers                */
/*                                             */
/*---------------------------------------------*/
void conv_1x1_int8(
    size_t rows, size_t cols,
    size_t channels_in,
    size_t channels_out,
    size_t stride,
    size_t padding,
    int8_t* input,
    const void* weights,
    int8_t* output,
    int relu,
    requantization_params_t rqp
);

void conv2D_1x1_f32 (
    size_t H, size_t W,
    size_t Cin, size_t Cout,
    size_t stride,
    size_t padding,
    const float *pw_weights,  // length = Cout*Cin
    float *input,
    float *output,
    int relu_pw
);

/*---------------------------------------------*/
/*                                             */
/* Pooling Layers.                             */
/*                                             */
/*---------------------------------------------*/
void maxpool_int8(
    size_t output_rows, size_t output_cols,
    size_t input_rows, size_t input_cols,
    size_t channels,
    size_t stride,
    int8_t *input,
    int8_t *output
);

void maxpool_f32(
    size_t output_rows, size_t output_cols,
    size_t input_rows, size_t input_cols,
    size_t channels,
    size_t stride,
    float *input,
    float *output
);


/*---------------------------------------------*/
/*                                             */
/* Softmax                                     */
/*                                             */
/*---------------------------------------------*/
void softmax_vec(
    const float *i,
    float *o,
    size_t channels,
    size_t innerSize
);

/* In-place vectorized 1D softmax over n floats. */
void softmax_f32(float *x, size_t n);

/*---------------------------------------------*/
/*                                             */
/* Vector primitives                           */
/*                                             */
/*---------------------------------------------*/
/* Elementwise (vectorized, single-hart). */
void fill_f32(float *y, float c, size_t n);                                 /* y[i] = c                       */
void axpy_f32(float *y, float a, const float *x, size_t n);                 /* y[i] += a * x[i]               */
void scale_add_f32(float *y, const float *x, float a, float b, size_t n);   /* y[i] =  a * x[i] + b           */

/* Reductions (vectorized, single-hart). */
float dot_f32(const float *a, const float *b, size_t n);                    /* sum_i a[i] * b[i]              */
float f32_bf16_dot(const float *a, const uint16_t *b_bf16, size_t n);       /* sum_i a[i] * bf16_to_f32(b[i]) */
float sum_f32(const float *x, size_t n);                                    /* sum_i x[i]                     */
float max_f32(const float *x, size_t n);                                    /* max_i x[i] (n > 0)             */

/*---------------------------------------------*/
/*                                             */
/* Attention                                   */
/*                                             */
/*---------------------------------------------*/
/* Causal multi-head self-attention for a single query token.
 *
 *   q              : [n_heads * head_dim]
 *   K_cache, V_cache: position-major  [seq_len][n_kv_heads][head_dim]
 *                    (caller passes the per-layer base pointer; the kernel
 *                     strides between positions by n_kv_heads * head_dim).
 *   pos            : current position (0-indexed); attention over [0, pos]
 *   scratch_scores : per-hart-safe scratch (heads write disjoint slices)
 *   scores_stride  : floats between consecutive head slots in scratch_scores
 *                    (e.g. seq_len if you have a fixed-size per-head buffer,
 *                     or pos+1 if you pack tightly)
 *   out            : [n_heads * head_dim]
 *
 * GQA: n_heads must be a multiple of n_kv_heads; query head h reads from
 * kv head h * n_kv_heads / n_heads.
 *
 * attention_f32 runs serially on the calling hart.
 * attention_mc_f32 splits heads across NN_RVV_N_HARTS via parallel_for. */
void attention_f32(
    const float *q,
    const float *K_cache,
    const float *V_cache,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos,
    float *scratch_scores, size_t scores_stride,
    float *out
);

void attention_mc_f32(
    const float *q,
    const float *K_cache,
    const float *V_cache,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos,
    float *scratch_scores, size_t scores_stride,
    float *out
);


/*---------------------------------------------*/
/*                                             */
/* Residual Add                                */
/*                                             */
/*---------------------------------------------*/

void residual_add(
    size_t rows, size_t cols,
    size_t channels,
    int8_t* a, int8_t* b,
    int8_t* output,
    requantization_params_t rqp
);

/*---------------------------------------------*/
/*                                             */
/* Activation Layers                           */
/*                                             */
/*---------------------------------------------*/
void relu6_int8(
    size_t channels,
    size_t inner_size,
    const float *input,
    int8_t *output,
    requantization_params_t requant_params
);

#endif /* NN_RVV_LAYERS_H */

#include "nn_rvv/layers.h"
#include "nn_rvv/threading.h"
#include "ops/elementwise/elementwise.h"
#include "ops/reduce/reduce.h"

#include <stddef.h>
#include <math.h>

/* Per-head causal self-attention for a single query token.
 *
 * Layout (matches llama2.c / borai KV cache):
 *   K_cache, V_cache: position-major  [seq_len][n_kv_heads][head_dim]
 *     -> stride between consecutive positions = n_kv_heads * head_dim
 *
 * GQA: query head h reads kv head h_kv = h / (n_heads / n_kv_heads).
 *
 * Vectorization: head_dim is the contiguous (innermost) dim, so dot_f32 /
 * axpy_f32 vectorize over head_dim with unit-stride loads. We do pos+1
 * such ops per head; vectorizing instead on t would require strided loads
 * across the n_kv_heads*head_dim stride, which on most cores is slower
 * than the contiguous variant even though pos+1 is larger. */
static inline void attention_head_f32(
    size_t h,
    const float *q,
    const float *K_cache,
    const float *V_cache,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos,
    float *scratch_scores, size_t scores_stride,
    float inv_sqrt_hd,
    float *out)
{
    size_t kv_mul        = n_heads / n_kv_heads;
    size_t h_kv          = h / kv_mul;
    size_t kv_pos_stride = n_kv_heads * head_dim;

    const float *q_h   = q       + h    * head_dim;
    const float *K_h0  = K_cache + h_kv * head_dim;   /* &K[0][h_kv][0] */
    const float *V_h0  = V_cache + h_kv * head_dim;
    float       *out_h = out     + h    * head_dim;
    float       *att   = scratch_scores + h * scores_stride;

    for (size_t t = 0; t <= pos; t++) {
        att[t] = dot_f32(q_h, K_h0 + t * kv_pos_stride, head_dim) * inv_sqrt_hd;
    }

    softmax_f32(att, pos + 1);

    fill_f32(out_h, 0.0f, head_dim);
    for (size_t t = 0; t <= pos; t++) {
        axpy_f32(out_h, att[t], V_h0 + t * kv_pos_stride, head_dim);
    }
}

void attention_f32(
    const float *q,
    const float *K_cache,
    const float *V_cache,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos,
    float *scratch_scores, size_t scores_stride,
    float *out)
{
    const float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);
    for (size_t h = 0; h < n_heads; h++) {
        attention_head_f32(h, q, K_cache, V_cache,
                           n_heads, n_kv_heads, head_dim,
                           pos, scratch_scores, scores_stride, inv_sqrt_hd,
                           out);
    }
}

typedef struct {
    const float *q, *K_cache, *V_cache;
    size_t n_heads, n_kv_heads, head_dim, pos;
    float *scratch_scores;
    size_t scores_stride;
    float inv_sqrt_hd;
    float *out;
} attention_ctx_t;

static void attention_chunk(size_t head_begin, size_t head_end, void *vctx) {
    attention_ctx_t *c = (attention_ctx_t *)vctx;
    for (size_t h = head_begin; h < head_end; h++) {
        attention_head_f32(h, c->q, c->K_cache, c->V_cache,
                           c->n_heads, c->n_kv_heads, c->head_dim,
                           c->pos, c->scratch_scores, c->scores_stride,
                           c->inv_sqrt_hd, c->out);
    }
}

void attention_mc_f32(
    const float *q,
    const float *K_cache,
    const float *V_cache,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos,
    float *scratch_scores, size_t scores_stride,
    float *out)
{
    attention_ctx_t ctx = {
        .q = q, .K_cache = K_cache, .V_cache = V_cache,
        .n_heads = n_heads, .n_kv_heads = n_kv_heads, .head_dim = head_dim,
        .pos = pos,
        .scratch_scores = scratch_scores, .scores_stride = scores_stride,
        .inv_sqrt_hd = 1.0f / sqrtf((float)head_dim),
        .out = out,
    };
    nn_rvv_parallel_for(n_heads, attention_chunk, &ctx);
}

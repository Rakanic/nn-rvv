#include "nn_rvv/layers.h"
#include "nn_rvv/threading.h"

#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>

/* RoPE rotary positional embedding (llama2 convention):
 *   for each head, for each pair (i, i+1):
 *     angle = pos / 10000^(i / head_size)
 *     (q[i], q[i+1]) = R(angle) @ (q[i], q[i+1])
 *     (k[i], k[i+1]) = R(angle) @ (k[i], k[i+1])     (only kv heads)
 *
 * We iterate the pair index j (= i/2) on the outer; for each j, the angle
 * is the same across all heads. We then apply a single scalar 2D rotation
 * to all (n_heads / n_kv_heads) head-instances using strided RVV loads
 * over the head dimension. */
static inline void rope_apply_pair(
    float *vec_e, float *vec_o, size_t n_heads_to_rotate,
    size_t head_size, float c, float s)
{
    const size_t head_stride_bytes = head_size * sizeof(float);
    size_t remaining = n_heads_to_rotate;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m4(remaining);
        vfloat32m4_t ve = __riscv_vlse32_v_f32m4(vec_e, head_stride_bytes, vl);
        vfloat32m4_t vo = __riscv_vlse32_v_f32m4(vec_o, head_stride_bytes, vl);
        /* e' = e*c - o*s,  o' = e*s + o*c */
        vfloat32m4_t e2 = __riscv_vfmul_vf_f32m4(ve, c, vl);
        e2 = __riscv_vfnmsac_vf_f32m4(e2, s, vo, vl);
        vfloat32m4_t o2 = __riscv_vfmul_vf_f32m4(ve, s, vl);
        o2 = __riscv_vfmacc_vf_f32m4(o2, c, vo, vl);
        __riscv_vsse32_v_f32m4(vec_e, head_stride_bytes, e2, vl);
        __riscv_vsse32_v_f32m4(vec_o, head_stride_bytes, o2, vl);
        vec_e += vl * head_size;
        vec_o += vl * head_size;
        remaining -= vl;
    }
}

typedef struct {
    float *q, *k;
    size_t n_heads, n_kv_heads, head_size, pos;
} rope_ctx_t;

static void rope_chunk(size_t j_begin, size_t j_end, void *vctx) {
    rope_ctx_t *c = (rope_ctx_t *)vctx;
    for (size_t j = j_begin; j < j_end; j++) {
        size_t i = 2 * j;
        float freq = 1.0f / powf(10000.0f, (float)i / (float)c->head_size);
        float angle = (float)c->pos * freq;
        float cos_v = cosf(angle), sin_v = sinf(angle);
        rope_apply_pair(c->q + i, c->q + i + 1, c->n_heads,
                        c->head_size, cos_v, sin_v);
        rope_apply_pair(c->k + i, c->k + i + 1, c->n_kv_heads,
                        c->head_size, cos_v, sin_v);
    }
}

/* Rotates q[n_heads * head_size] and k[n_kv_heads * head_size] in place.
 * pair-index loop is parallelized across NN_RVV_N_HARTS via parallel_for. */
void rope_f32(
    float *q,
    float *k,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_size,
    size_t pos)
{
    rope_ctx_t ctx = {
        .q = q, .k = k,
        .n_heads = n_heads, .n_kv_heads = n_kv_heads,
        .head_size = head_size, .pos = pos,
    };
    nn_rvv_parallel_for(head_size / 2, rope_chunk, &ctx);
}

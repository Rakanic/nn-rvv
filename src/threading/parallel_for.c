/*
 * parallel_for.c - Public threading API on top of the nnrvv_hthread runtime.
 *
 * Only compiled into the library when NN_RVV_N_HARTS > 1. For N_HARTS == 1
 * the implementation lives in nn_rvv/threading.h as a header-only inline.
 */
#include "nn_rvv/threading.h"

#if NN_RVV_N_HARTS > 1

#include "hthread.h"

typedef struct {
    nn_rvv_parfor_fn body;
    void *ctx;
    size_t begin;
    size_t end;
} parfor_chunk_t;

static void parfor_chunk_runner(void *arg) {
    parfor_chunk_t *c = (parfor_chunk_t *)arg;
    c->body(c->begin, c->end, c->ctx);
}

void nn_rvv_threading_init(void) {
    nnrvv_hthread_init();
}

void nn_rvv_parallel_for(size_t n, nn_rvv_parfor_fn body, void *ctx) {
    if (n == 0 || body == 0) {
        return;
    }

    /* Fast path: too little work to bother dispatching. The 8 ceiling is a
     * crude lower bound — splitting fewer than ~N_HARTS units of work into
     * N_HARTS shards mostly costs dispatch overhead for no gain. */
    if (n < (size_t)NN_RVV_N_HARTS) {
        body((size_t)0, n, ctx);
        return;
    }

    parfor_chunk_t chunks[NN_RVV_N_HARTS];
    size_t per_hart = n / (size_t)NN_RVV_N_HARTS;
    size_t extra    = n % (size_t)NN_RVV_N_HARTS;
    size_t off = 0;
    for (uint32_t h = 0; h < NN_RVV_N_HARTS; h++) {
        size_t sz = per_hart + (h < extra ? 1u : 0u);
        chunks[h].body  = body;
        chunks[h].ctx   = ctx;
        chunks[h].begin = off;
        chunks[h].end   = off + sz;
        off += sz;
    }

    /* Dispatch chunks 1..N-1 to remote harts. */
    for (uint32_t h = 1; h < NN_RVV_N_HARTS; h++) {
        nnrvv_hthread_issue(h, parfor_chunk_runner, &chunks[h]);
    }

    /* Run chunk 0 on the calling hart (typically hart 0). */
    parfor_chunk_runner(&chunks[0]);

    /* Wait for the remote chunks to finish before returning. */
    for (uint32_t h = 1; h < NN_RVV_N_HARTS; h++) {
        nnrvv_hthread_join(h);
    }
}

#endif /* NN_RVV_N_HARTS > 1 */

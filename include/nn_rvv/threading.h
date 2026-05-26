/*
 * nn_rvv/threading.h - Lightweight multi-hart threading for nn-rvv kernels.
 *
 * Threading is selected at compile time via NN_RVV_N_HARTS (default: 1).
 *   - NN_RVV_N_HARTS == 1: parallel_for is a single inline call to the body
 *     and no threading runtime is compiled into the library.
 *   - NN_RVV_N_HARTS  > 1: the work-stealing runtime in src/threading/ is
 *     compiled in. The build must be linked against a platform that
 *     provides the bare-metal CLINT MSIP wakeup path (i.e. a multi-hart
 *     RISC-V target with the parent project's `clint` + `rocketcore`
 *     drivers). Hart 0 is the calling thread; harts 1..N_HARTS-1 sit in a
 *     work-stealing scheduler installed as the linker's secondary-hart
 *     entry point (`__main`).
 *
 * Usage:
 *
 *   #include "nn_rvv/threading.h"
 *
 *   int main(void) {
 *       nn_rvv_threading_init();   // once, on hart 0, before any parallel work
 *       // ... kernels in nn-rvv now internally dispatch across harts ...
 *   }
 *
 * Kernels that do not parallelize stay correct in all modes: parallel_for
 * either splits [0, n) into N_HARTS chunks or runs the whole range inline,
 * so callers do not need to know whether a particular kernel is threaded.
 */
#ifndef NN_RVV_THREADING_H
#define NN_RVV_THREADING_H

#include <stddef.h>
#include <stdint.h>

#ifndef NN_RVV_N_HARTS
#define NN_RVV_N_HARTS 1
#endif

#if NN_RVV_N_HARTS < 1
#  error "NN_RVV_N_HARTS must be >= 1"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Parallel-for body. Receives the half-open range [begin, end) to process
 * and the user-provided context pointer. */
typedef void (*nn_rvv_parfor_fn)(size_t begin, size_t end, void *ctx);

#if NN_RVV_N_HARTS == 1

/* Single-core fast path: header-only inlines. No runtime is linked. */
static inline void nn_rvv_threading_init(void) { }

static inline void nn_rvv_parallel_for(size_t n, nn_rvv_parfor_fn body, void *ctx) {
    body((size_t)0, n, ctx);
}

#else

/* Multi-hart path: real implementation lives in src/threading/parallel_for.c. */

/* Initialize the threading runtime. Must be called once from hart 0 before
 * any parallel_for. Idempotent. */
void nn_rvv_threading_init(void);

/* Split [0, n) into NN_RVV_N_HARTS contiguous chunks and run body() on each.
 * Blocks until every chunk completes. Hart 0 runs chunk 0 inline; remote
 * harts run chunks 1..N-1 via the work-stealing runtime. */
void nn_rvv_parallel_for(size_t n, nn_rvv_parfor_fn body, void *ctx);

#endif /* NN_RVV_N_HARTS */

#ifdef __cplusplus
}
#endif

#endif /* NN_RVV_THREADING_H */

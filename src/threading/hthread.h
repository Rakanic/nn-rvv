/*
 * hthread.h - Internal work-stealing runtime for nn-rvv.
 *
 * Adapted from thread-lib's hthread (parent repo) but namespaced under
 * `nnrvv_hthread_*` so that it can coexist with the parent's thread-lib
 * symbols in the same binary (the linker still picks ONE __main between
 * the two; see CMakeLists for guidance).
 *
 * This is a PRIVATE header — kernels and consumers should go through
 * nn_rvv/threading.h (parallel_for), not these primitives.
 */
#ifndef NN_RVV_INTERNAL_HTHREAD_H
#define NN_RVV_INTERNAL_HTHREAD_H

#include <stdint.h>

#include "nn_rvv/threading.h"

/* Platform headers — only available when the parent project supplies the
 * RISC-V drivers + chip-config (clint, rocketcore, chip-config). chip_config.h
 * is where the chip-specific MMIO instance pointers like `CLINT` come from
 * (the `clint` driver only declares the CLINT_Type struct; the actual
 * `((CLINT_Type *)CLINT_BASE)` macro is per-chip). Compiled in only when
 * N_HARTS > 1. */
#include "chip_config.h"

#ifndef NN_RVV_WSQ_SIZE
#define NN_RVV_WSQ_SIZE 64
#endif

typedef struct {
    void (*fn)(void *);
    void *arg;
    uint32_t owner;
    uint32_t flags;
} nnrvv_htask_t;

typedef struct {
    volatile uint32_t top;
    volatile uint32_t bottom;
    nnrvv_htask_t tasks[NN_RVV_WSQ_SIZE];
} nnrvv_wsdeque_t;

#define NNRVV_HTHREAD_TASK_STEALABLE (1u << 0)

void nnrvv_hthread_init(void);
void nnrvv_hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void nnrvv_hthread_dispatch(void (*fn)(void *), void *arg);
void nnrvv_hthread_join(uint32_t hartid);
void nnrvv_hthread_barrier(void);

#endif /* NN_RVV_INTERNAL_HTHREAD_H */

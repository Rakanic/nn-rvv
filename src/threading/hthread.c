/*
 * hthread.c - Work-stealing runtime ported from thread-lib into nn-rvv.
 *
 * Per-hart deques, atomic steal, and CLINT MSIP wakeups. Hart 0 issues
 * tasks; harts 1..N-1 sit in the worker loop installed as __main (the
 * secondary-hart entry point from glossy/crt0). Symbols prefixed with
 * `nnrvv_` to avoid collision with the parent's thread-lib.
 *
 * This file is only compiled into the library when NN_RVV_N_HARTS > 1.
 */
#include "nn_rvv/threading.h"

#if NN_RVV_N_HARTS > 1

#include "hthread.h"

#define NN_RVV_HTHREAD_COOKIE 0x4E4E5256u /* "NNRV" */

static nnrvv_wsdeque_t deques[NN_RVV_N_HARTS];
static volatile uint32_t deque_locks[NN_RVV_N_HARTS] __attribute__((aligned(64)));
static volatile uint32_t pending_tasks[NN_RVV_N_HARTS] __attribute__((aligned(64)));
static volatile uint32_t dispatch_rr __attribute__((aligned(64))) = 0;
static volatile uint32_t runtime_cookie __attribute__((aligned(64))) = 0;

static volatile uint32_t barrier_count = 0;
static volatile uint32_t barrier_epoch = 0;

static inline void lock_deque(uint32_t hartid) {
    while (__sync_lock_test_and_set(&deque_locks[hartid], 1u) != 0u) {
        while (deque_locks[hartid] != 0u) {
            asm volatile("nop");
        }
    }
}

static inline void unlock_deque(uint32_t hartid) {
    __sync_lock_release(&deque_locks[hartid]);
}

static inline void run_task(const nnrvv_htask_t *task) {
    task->fn(task->arg);
    __sync_synchronize();
    __sync_fetch_and_sub(&pending_tasks[task->owner], 1u);
}

static inline void wake_hart(uint32_t hartid) {
    CLINT->MSIP[hartid] = 1;
}

static inline void wake_other_harts(uint32_t self) {
    for (uint32_t h = 0; h < NN_RVV_N_HARTS; ++h) {
        if (h == self) {
            continue;
        }
        wake_hart(h);
    }
}

static inline void ws_push(uint32_t hartid, const nnrvv_htask_t *task) {
    nnrvv_wsdeque_t *dq = &deques[hartid];

    while (1) {
        lock_deque(hartid);

        uint32_t t = dq->top;
        uint32_t b = dq->bottom;
        if ((b - t) < NN_RVV_WSQ_SIZE) {
            dq->tasks[b & (NN_RVV_WSQ_SIZE - 1)] = *task;
            __sync_synchronize();
            __sync_fetch_and_add(&pending_tasks[task->owner], 1u);
            dq->bottom = b + 1u;
            unlock_deque(hartid);
            break;
        }

        unlock_deque(hartid);
        asm volatile("nop");
    }
}

static inline int ws_pop(uint32_t hartid, nnrvv_htask_t *out) {
    nnrvv_wsdeque_t *dq = &deques[hartid];
    lock_deque(hartid);

    uint32_t t = dq->top;
    uint32_t b = dq->bottom;
    if (t == b) {
        unlock_deque(hartid);
        return 0;
    }

    b -= 1u;
    *out = dq->tasks[b & (NN_RVV_WSQ_SIZE - 1u)];
    dq->bottom = b;

    unlock_deque(hartid);
    return 1;
}

static inline int ws_steal(uint32_t victim, nnrvv_htask_t *out) {
    nnrvv_wsdeque_t *dq = &deques[victim];
    lock_deque(victim);

    uint32_t t = dq->top;
    uint32_t b = dq->bottom;

    if (t == b) {
        unlock_deque(victim);
        return 0;
    }

    nnrvv_htask_t task = dq->tasks[t & (NN_RVV_WSQ_SIZE - 1u)];
    if ((task.flags & NNRVV_HTHREAD_TASK_STEALABLE) == 0u) {
        unlock_deque(victim);
        return 0;
    }

    dq->top = t + 1u;
    unlock_deque(victim);

    *out = task;
    return 1;
}

void nnrvv_hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg) {
    if (hartid >= NN_RVV_N_HARTS || fn == 0) {
        return;
    }

    nnrvv_htask_t t = {
        .fn = fn,
        .arg = arg,
        .owner = hartid,
        .flags = 0u,
    };

    ws_push(hartid, &t);
    wake_hart(hartid);
}

void nnrvv_hthread_dispatch(void (*fn)(void *), void *arg) {
    if (fn == 0) {
        return;
    }

    uint32_t self = (uint32_t)READ_CSR("mhartid");
    uint32_t target = __sync_fetch_and_add(&dispatch_rr, 1u) % NN_RVV_N_HARTS;

    if (target == self) {
        fn(arg);
        return;
    }

    nnrvv_htask_t t = {
        .fn = fn,
        .arg = arg,
        .owner = target,
        .flags = NNRVV_HTHREAD_TASK_STEALABLE,
    };

    ws_push(target, &t);
    wake_other_harts(self);
}

void nnrvv_hthread_join(uint32_t hartid) {
    if (hartid >= NN_RVV_N_HARTS) {
        return;
    }

    uint32_t self = (uint32_t)READ_CSR("mhartid");
    nnrvv_htask_t task;

    while (__atomic_load_n(&pending_tasks[hartid], __ATOMIC_ACQUIRE) != 0u) {
        if (self == hartid) {
            if (ws_pop(hartid, &task)) {
                run_task(&task);
                continue;
            }
        } else {
            if (ws_steal(hartid, &task)) {
                run_task(&task);
                continue;
            }
        }

        wake_hart(hartid);
        asm volatile("nop");
    }
}

void nnrvv_hthread_barrier(void) {
    uint32_t epoch = barrier_epoch;
    uint32_t arrived = __sync_add_and_fetch(&barrier_count, 1u);

    if (arrived == NN_RVV_N_HARTS) {
        barrier_count = 0u;
        __sync_synchronize();
        barrier_epoch = epoch + 1u;
    } else {
        while (barrier_epoch == epoch) {
            asm volatile("nop");
        }
    }

    __sync_synchronize();
}

void nnrvv_hthread_init(void) {
    runtime_cookie = 0u;
    dispatch_rr = 0u;
    barrier_count = 0u;
    barrier_epoch = 0u;

    for (uint32_t i = 0; i < NN_RVV_N_HARTS; i++) {
        deques[i].top = 0;
        deques[i].bottom = 0;
        pending_tasks[i] = 0;
        deque_locks[i] = 0;
        CLINT->MSIP[i] = 0u;
    }

    __sync_synchronize();
    runtime_cookie = NN_RVV_HTHREAD_COOKIE;
    __sync_synchronize();
}

/* Secondary-hart entry point installed by glossy's crt0.S. Overrides
 * glossy's weak default (which does wfi). Each non-hart-0 core sits here
 * stealing work until power-off. */
void __main(void) {
    uint32_t mhartid = (uint32_t)READ_CSR("mhartid");
    nnrvv_htask_t task;
    uint32_t idle_spins = 0;

    while (1) {
        if (__atomic_load_n(&runtime_cookie, __ATOMIC_ACQUIRE) != NN_RVV_HTHREAD_COOKIE) {
            asm volatile("nop");
            continue;
        }

        int did_work = 0;

        nnrvv_wsdeque_t *my_dq = &deques[mhartid];
        if (my_dq->top != my_dq->bottom) {
            if (ws_pop(mhartid, &task)) {
                run_task(&task);
                did_work = 1;
                idle_spins = 0;
            }
        }

        if (!did_work) {
            for (uint32_t victim = 0; victim < NN_RVV_N_HARTS; victim++) {
                if (victim == mhartid) {
                    continue;
                }
                if (deques[victim].top != deques[victim].bottom) {
                    if (ws_steal(victim, &task)) {
                        run_task(&task);
                        did_work = 1;
                        idle_spins = 0;
                        break;
                    }
                }
            }
        }

        if (!did_work) {
            idle_spins++;
            for (uint32_t j = 0; j < (idle_spins < 64u ? idle_spins : 64u); j++) {
                asm volatile("nop");
            }
        }
    }
}

#endif /* NN_RVV_N_HARTS > 1 */

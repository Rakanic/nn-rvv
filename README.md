# nn-rvv

A library of RVV 1.0 neural-network kernels with a built-in multi-hart
work-stealing runtime. Designed to live as a third-party component inside the
UCB-BAR [Baremetal-IDE](https://github.com/ucb-bar/Baremetal-IDE) parent
project.

Public surface is a single header (`nn_rvv/layers.h`) plus an optional
threading header (`nn_rvv/threading.h`). The kernels themselves expose
`parallel_for` internally, so any matmul/attention/rope/etc. call fans out
across `NN_RVV_N_HARTS` harts automatically. Reference example programs
(MNIST variants, ...) live separately at `nn-rvv-examples/` in the parent.

## Kernels

| Layer | f32 | int8 | Notes |
|---|:---:|:---:|---|
| Fully connected | ✅ | ✅ | `fully_connected_f32` + nobias / transposed-weight variants; `fully_connected_int8`, `quant_fully_connected_int8{,_t}` |
| Depthwise conv 3×3 / 5×5 | ✅ | ✅ | `dw_conv2D_3x3_f32`, `dwconv2D_3x3_int8`, `dw_conv2D_5x5_*` |
| Pointwise conv 1×1 | ✅ | ✅ | `conv2D_1x1_f32`, `conv_1x1_int8` |
| Standard conv 3×3 / 5×5 (asm) | ✅ | ✅ | `vec_conv*` family |
| Max pool 3×3 (str 1/2/3) | ✅ | ✅ | `maxpool_f32`, `maxpool_int8` |
| Softmax | ✅ | — | `softmax_f32` (1D, vectorized), `softmax_vec` (multi-channel) |
| RMSNorm | ✅ | — | `rmsnorm_f32` |
| RoPE | ✅ | — | `rope_f32` (multi-hart) |
| Attention (causal MHA + GQA) | ✅ | — | `attention_f32` (serial), `attention_mc_f32` (multi-hart) |
| Quantize / dequantize / requantize | — | ✅ | `quant_f32`, `dequant_f32`, `requant_outch_int32` |
| Transpose | ✅ | ✅ | `transpose_f32`, `transpose_int8` |
| ReLU6, residual add, padding | — | ✅ | int8 helpers |

Reusable building blocks (`dot_f32`, `axpy_f32`, `fill_f32`, `max_f32`,
`sum_f32`, `scale_add_f32`) live under `src/ops/{reduce,elementwise}` for
internal use by new kernels.

## Runtime

Single compile-time knob: `NN_RVV_N_HARTS` (default `1`).

- `=1` — no threading runtime compiled in; `parallel_for` is a single
  inline call. Kernels are still fully vectorized.
- `≥2` — work-stealing scheduler installed as the secondary-hart entry
  point (`__main`). Requires the parent project's `chip-config` /
  `clint` / `rocketcore` targets for CLINT MSIP + `mhartid` access.

```c
#include "nn_rvv/threading.h"
nn_rvv_threading_init();   // once, on hart 0, before any kernel call
```

After that every kernel that exposes parallelism (all GEMMs, attention,
RoPE) fans work across harts automatically.

## Using nn-rvv (as a parent-build dependency)

```cmake
add_subdirectory(nn-rvv)
target_link_libraries(my_target PRIVATE nnrvv)
```

`nnrvv` is a static library; its `PUBLIC` include dir is set so consumers
can `#include "nn_rvv/layers.h"` and `#include "nn_rvv/threading.h"`
directly. From the Baremetal-IDE root:

```bash
make build CHIP=bearly25 TARGET=<your-target> RVV=1 \
           BUILD_NN_RVV=ON NN_RVV_N_HARTS=2
```

## Build options

- `NN_RVV_N_HARTS` — hart count (default 1).
- `NN_RVV_MAX_PERF` — default ON; adds
  `-O3 -funroll-loops -fno-math-errno -fno-trapping-math`.

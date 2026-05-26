# nn-rvv

**nn-rvv** is a lightweight deep‑learning framework for compiling high‑level neural networks into efficient, bare‑metal C code targeting the **RISC‑V Vector Extension (RVV) 1.0**. Built for and tested within the Chipyard environment, it currently targets the **Saturn Vector Unit**.

✅ **Quantization Support** – Full symmetric *and* asymmetric flows, including zero‑points, bias quantization and per‑layer clamping
✅ **Transformer-friendly path** – `int8_qgemm_fout` + `quant_fully_connected_int8_t` for transposed-weight int8 transformer inference
⚠️ **Roadmap Focus** – Ongoing work on a smoother Python → C compilation path and deeper RVV kernel optimizations (3×3 depthwise first, then 5×5 and 1×1 pointwise)

---

## 🚀 Features

| Layer / Op                       | f32 | int8 |
|----------------------------------|:---:|:----:|
| Fully Connected                  | ✅  | ✅   |
| Fully Connected (transposed weights) | —   | ✅   |
| Depthwise Conv2D (3×3)           | ✅  | ✅   |
| Depthwise Conv2D (5×5)           | ✅  | ✅   |
| Pointwise Conv (1×1)             | ✅  | ✅   |
| Max Pool (3×3, str 1/2/3)        | ✅  | ✅   |
| Softmax                          | ✅  | ❌   |
| Transpose                        | —   | ✅   |
| ReLU6                            | —   | ✅   |
| Residual Add                     | —   | ✅   |
| Quantize / Dequantize / Requantize | —   | ✅   |
| Padding (channel-wise)           | —   | ✅   |

---

## 📁 Layout

```
nn-rvv/
├── CMakeLists.txt          # Builds static library `nnrvv`
├── include/
│   └── nn_rvv/
│       └── layers.h        # Public API
├── src/
│   ├── layers/             # High-level layer dispatchers
│   └── ops/                # Kernel implementations
│       ├── matmul/         # f32/int8/quant GEMM kernels (vec-nn authoritative)
│       ├── conv2D/         # Depthwise + standard conv kernels (C and asm)
│       ├── pooling/        # Max-pool ukernels
│       ├── padding/        # Channel-wise input padding
│       └── ara/            # Exp implementation used by softmax
└── models/                 # Reference MNIST models (training scripts + sample C entry points)
```

---

## 🔧 Using nn-rvv

### As a CMake subdirectory

```cmake
add_subdirectory(nn-rvv)
target_link_libraries(my_target PRIVATE nnrvv)
```

`nnrvv` is a static library; its `PUBLIC` include directory is set up so you can immediately `#include "nn_rvv/layers.h"` in your code without any extra `target_include_directories` calls.

### Standalone

```bash
cmake -S nn-rvv -B build -DCMAKE_TOOLCHAIN_FILE=<your-riscv.cmake>
cmake --build build
```

This produces `build/libnnrvv.a`. The library does not assume a particular `march`/`mabi` — supply those via the toolchain file or `CMAKE_C_FLAGS`. For Saturn-class hardware use `-march=rv64gcv_zfh_zvfh -mabi=lp64d`.

`NN_RVV_MAX_PERF` (default ON) adds `-O3 -funroll-loops -fno-math-errno -fno-trapping-math` to the library sources.

---

## 🧵 Multi-core threading

nn-rvv ships a lightweight work-stealing runtime selected at **compile time**
via `NN_RVV_N_HARTS` (default `1`).

| `NN_RVV_N_HARTS` | Behavior |
|---|---|
| `1` | No threading runtime is compiled in. `parallel_for` is an inline call to the body; kernels run on the calling hart. |
| `>= 2` | Per-hart deques + CLINT MSIP wakeups. Hart 0 is the caller; harts 1..N-1 sit in the work-stealing scheduler installed as the linker's secondary-hart entry point (`__main`). |

When `NN_RVV_N_HARTS > 1`, the build links against `clint` + `rocketcore`
driver targets — those come from this parent project. Standalone builds
should stay at `NN_RVV_N_HARTS=1` unless the consumer provides equivalent
targets.

### Public API

```c
#include "nn_rvv/threading.h"

int main(void) {
    nn_rvv_threading_init();   // once, on hart 0
    // ... existing kernel calls — matmul variants now dispatch internally ...
}
```

The matmul kernels (`f32_gemm{,_relu,_nobias}`, `int8_qgemm{,_relu}`,
`int8_qgemm_int32bias{,_relu}`, `int8_qgemm_int32bias_conv1x1{,_relu}`,
`int8_qgemm_fout`) split the M (output-row) dimension across all
`NN_RVV_N_HARTS` harts. Other kernels remain single-hart but are
compatible — they just run on the calling hart, leaving the others idle.

If you need to parallelize your own outer loop, use `nn_rvv_parallel_for`:

```c
typedef struct { /* your context */ } my_ctx;
static void my_body(size_t begin, size_t end, void *ctx) {
    /* process indices [begin, end) */
}
my_ctx ctx = {...};
nn_rvv_parallel_for(/*n=*/work_items, my_body, &ctx);
```

### Building with threading

Parent build (the parent Makefile passes the value through):

```bash
make build CHIP=bearly25 TARGET=<your-target> RVV=1 BUILD_NN_RVV=ON NN_RVV_N_HARTS=2
```

Standalone build (only meaningful if you provide `clint` / `rocketcore`
targets):

```bash
cmake -S nn-rvv -B build -DCMAKE_TOOLCHAIN_FILE=<your-riscv.cmake> -DNN_RVV_N_HARTS=2
cmake --build build
```

⚠️ A binary cannot simultaneously link both nn-rvv's threading runtime and
the parent's `thread-lib` — both define `__main`. Pick one per binary
(either set `NN_RVV_N_HARTS=1` and use the parent's thread-lib, or use
nn-rvv's threading and drop the thread-lib link).

---

## 📫 Contact / Contributions

Early‑stage project — **Suggestions, discussions, and PRs are welcome!** 🙂

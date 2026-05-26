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

## 📫 Contact / Contributions

Early‑stage project — **Suggestions, discussions, and PRs are welcome!** 🙂

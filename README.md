# nn-rvv

**nn-rvv** is a lightweight deep‑learning library for converting high‑level PyTorch or TFLite models into highly optimized C code targeting RISC‑V Vector Extension (RVV) 1.0.  
Built and tested within the [Chipyard](https://github.com/ucb-bar/chipyard) framework, nn-rvv currently targets the [Saturn Vector Unit](https://github.com/ucb-bar/saturn-vectors).

> **⚠️ Work in Progress:** There is still a lot of work to be done—new layers, full quantization paths, and robust multithreading support are all on the roadmap!

> **TBD Documentation on each kernel and how to use them.**

---

## 🚀 Features

| Layer / Operation                    | f32 Support | int8 Support  |
|--------------------------------------|:-----------:|:-------------:|
| Fully‑Connected                      | ✅          | ✅            |
| Depthwise Conv2D (3×3)               | ✅          | ✅            |
| Depthwise Conv2D (5×5)               | ✅          | 🟨            |
| Conv 1×1                             | ✅          | ✅            |
| Max Pool                             | ✅          | ✅            |
| Softmax                              | ✅          | ❌            |
| Transpose                            | ✅          | ✅            |
| Quantize / Dequantize                |             ✅              |
| Multithreading                       |             ✅              |

---

## 📦 Getting Started

### Prerequisites

- A RISC‑V toolchain with RVV 1.0 support  
- Chipyard environment set up for bare‑metal builds  

### Building

```bash
# Single‑core MNIST CNN inference
make mnist_cnn

# Multi‑core (basic) MNIST CNN inference
make mnist_cnn_mc

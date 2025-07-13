# nn-rvv

**nn-rvv** is a lightweight deep‑learning framework for compiling high‑level neural networks into efficient, bare‑metal C code targeting the **RISC‑V Vector Extension (RVV) 1.0**. Built for and tested within the Chipyard environment, it currently targets the **Saturn Vector Unit**.

✅ **Quantization Support** – Full symmetric *and* asymmetric flows, including zero‑points, bias quantization and per‑layer clamping  
⚠️ **Roadmap Focus** – Ongoing work on a smoother Python → C compilation path and deeper RVV kernel optimizations (3 × 3 depthwise first, then 5 × 5 and 1 × 1 pointwise)

---

## 🚀 Features

| Layer / Op                       | f32 | int8 |
|----------------------------------|:---:|:----:|
| Fully Connected                  | ✅  | ✅   |
| Depthwise Conv2D (3 × 3)          | ✅  | ✅   |
| Depthwise Conv2D (5 × 5)          | ✅  | ✅   |
| Pointwise Conv (1 × 1)            | ✅  | ✅   |
| Max Pool                         | ✅  | ✅   |
| Softmax                          | ✅  | ❌   |
| Transpose                        | ✅  | ✅   |
| Quantize / Dequantize            | —   | ✅   |
| Multithreading (basic)           | ✅  | ✅   |

---

## 🔧 Getting Started

1. **Prerequisites**  
   • A RISC‑V toolchain with **RVV 1.0** support  
   • A Chipyard installation with the appropriate conda environment sourced (Spike, `riscv64-unknown-elf-gcc`, …)

---

## 🧠 Example Network

```python
class DWPWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw0   = nn.Conv2d(1, 1, 3, groups=1, bias=False)
        self.pw0   = nn.Conv2d(1, 16, 1, bias=True)
        self.pool0 = nn.MaxPool2d(3, 3)
        self.dw1   = nn.Conv2d(16, 16, 3, groups=16, bias=False)
        self.pw1   = nn.Conv2d(16, 32, 1, bias=True)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.fc0   = nn.Linear(32*2*2, 32)
        self.fc1   = nn.Linear(32, 10)
```

You’ll find the full file at `models/mnist_models/mnist_cnn/mnist_cnn.py`.

---

## 🧪 Training + Compilation Workflow

**Step 1 – Train (float32)**  
```bash
cd models/mnist_models/mnist_cnn
python mnist_cnn.py      # dumps model_params.h with learned weights
```

**Step 2 – Compile & Run (float32 inference)**  
```bash
make mnist_cnn
spike --isa=rv64gcv_zicntr mnist_cnn.riscv
```

These examples are using spike, but you can also run the binary in RTL simulation in Chipyard using a Saturn configuration. 
---

## 🧮 Quantized Workflow

Quantized inference is fully supported and stable. This example uses a standard symmetric flow and I didn't include bias in the layers, but asymmetric with zero‑points and int32 biases is also available.

```python
# Example forward snippet
x = torch.clamp(x * self.input_scale, min=-128, max=127).round()
x = self.dw0(x)
x = torch.clamp(x * self.dw0.output_scale, min=-128, max=127).round()
# ...
x = self.fc1(x)
x = torch.clamp(x * self.fc1.output_scale, min=-128, max=127).round()
```

**Step 1 – Quantize & Export**  
Run `export_quantized_to_header.py` to generate `model_params.h` (weights + scales).

**Step 2 – Compile & Run (int8 inference)**  
```bash
make mnist_cnn_quant
spike --isa=rv64gcv_zicntr mnist_cnn_quant.riscv
```

---

## 🧵 Multicore Support

Basic multicore inference targets are provided:

```bash
make mnist_cnn_mc
spike -p2 --isa=rv64gcv_zicntr mnist_cnn_mc.riscv   # run on 2 cores
```

---

## 📂 Example Models

```bash
make mnist
make mnist_cnn
make mnist_cnn_mc
make mnist_quant2
make mnist_quant2_mc
make mnist_cnn_quant
make mnist_quant
```

Run with:
```bash
spike --isa=rv64gcv_zicntr <binary>.riscv
```

---

## 🧪 Sample Output

You should ideally see something like this 
```
Sample 0 → 7   probs: 0 0 0 0 0 0 0 100 0 0
Sample 1 → 2   probs: 0 0 100 0 0 0 0 0 0 0
Sample 2 → 1   probs: 0 100 0 0 0 0 0 0 0 0
...
```

---

## 📌 Roadmap

- [x] **Full int8 quantization support** (asymmetric zero‑points + bias quantization)  
- [x] **Basic multithreaded inference** at the task level  
- [x] **Export weights from PyTorch to C headers**  
- [ ] **Improve Python → C compilation path** (streamlined front‑ends & code‑gen)  
- [ ] **Expand convolution kernels** (stride, dilation, padding variants)  
- [ ] **Optimize & multithread kernels at the RVV level  
- [ ] **Smarter tiling / parallelism** inside convolution kernels  
- [ ] **One‑click tooling** for `.py → .h → .c → .riscv` flow  
- [ ] **Better build‑time diagnostics** (shape checks, warnings, …)

---

## 📫 Contact / Contributions

Early‑stage project — **PRs and discussions welcome!** 🙂

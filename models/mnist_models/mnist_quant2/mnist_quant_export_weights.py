import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as qt
from torchvision import datasets, transforms
import numpy as np
import math

# -------------------- 1) Load MNIST & compute input quant params ----------
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# grab all training images at once
all_loader = torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
X_all, _ = next(iter(all_loader))
X_all = X_all.view(-1, 784).numpy()

def symmetric_quant_params(tensor: np.ndarray, num_bits=8):
    qmax = 2**(num_bits-1) - 1
    max_abs = np.max(np.abs(tensor))
    scale = max_abs / qmax if max_abs != 0 else 1.0
    return scale, 0

scale_input, zp_input = symmetric_quant_params(X_all)

# -------------------- 2) Define QAT-ready model ---------------------------
class QATMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = qt.QuantStub()
        self.fc0     = nn.Linear(784, 64)
        self.relu    = nn.ReLU()
        self.fc1     = nn.Linear(64, 10)
        self.dequant = qt.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.dequant(x)
        return x

model = QATMLP()

# pick supported backend
engines = torch.backends.quantized.supported_engines
if 'qnnpack' in engines:
    torch.backends.quantized.engine = 'qnnpack'
elif 'fbgemm' in engines:
    torch.backends.quantized.engine = 'fbgemm'
else:
    raise RuntimeError(f"No quant backend in {engines}")

# fuse & prepare for QAT
model.qconfig = qt.get_default_qat_qconfig(torch.backends.quantized.engine)
qt.fuse_modules(model, ['fc0', 'relu'], inplace=True)
qt.prepare_qat(model, inplace=True)

# -------------------- 3) Train QAT model -----------------------------------
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
model.train()
for epoch in range(7):
    total_loss = 0.0
    for X, y in train_loader:
        X = X.view(-1, 784)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    print(f"Epoch {epoch+1} loss: {total_loss/len(train_ds):.4f}")

# -------------------- 4) Compute per-tensor output scales -----------------
model.eval()
with torch.no_grad():
    # Layer0 activations after ReLU
    x0 = model.fc0(torch.from_numpy(X_all).float())
    x0 = torch.relu(x0)
    act0 = x0.numpy()
    scale_out0, zp_out0 = symmetric_quant_params(act0)
    # Layer1 logits
    x1 = model.fc1(x0)
    logits = x1.numpy()
    scale_out1, zp_out1 = symmetric_quant_params(logits)

# -------------------- 5) Extract & per-channel quant params  -------------
def per_channel_params(weight: np.ndarray):
    """Compute symmetric per-row quant params for W with shape (out, in)."""
    scales, zps = [], []
    for row in weight:
        s, z = symmetric_quant_params(row)
        scales.append(s)
        zps.append(z)
    return np.array(scales, dtype=np.float32), np.array(zps, dtype=np.int32)

# Layer0
w0 = model.fc0.weight.detach().cpu().numpy()  # (64,784)
b0 = model.fc0.bias.detach().cpu().numpy()    # (64,)
scale_w0, zp_w0 = per_channel_params(w0)
# quantize bias per-channel
b0_re = b0 - zp_input * w0.sum(axis=1)
b0_q  = np.round(b0_re / (scale_input * scale_w0)).astype(np.int32)
b0_q  = np.clip(b0_q, -128, 127).astype(np.int8)
# quantize weights per-channel
w0_q = np.zeros_like(w0, dtype=np.int8)
for i in range(w0.shape[0]):
    q = np.round(w0[i] / scale_w0[i]).astype(np.int32) + zp_w0[i]
    w0_q[i] = np.clip(q, -128, 127).astype(np.int8)
# flatten bias+weights
flat_w0 = w0_q.T.reshape(-1)
wb0_q   = np.concatenate([b0_q, flat_w0])

# compute requant scales per-channel for layer0
requant0 = (scale_input * scale_w0) / scale_out0

# Layer1
w1 = model.fc1.weight.detach().cpu().numpy()  # (10,64)
b1 = model.fc1.bias.detach().cpu().numpy()    # (10,)
scale_w1, zp_w1 = per_channel_params(w1)
b1_re = b1 - zp_out0 * w1.sum(axis=1)
b1_q  = np.round(b1_re / (scale_out0 * scale_w1)).astype(np.int32)
b1_q  = np.clip(b1_q, -128, 127).astype(np.int8)
w1_q = np.zeros_like(w1, dtype=np.int8)
for i in range(w1.shape[0]):
    q = np.round(w1[i] / scale_w1[i]).astype(np.int32) + zp_w1[i]
    w1_q[i] = np.clip(q, -128, 127).astype(np.int8)
flat_w1 = w1_q.T.reshape(-1)
wb1_q   = np.concatenate([b1_q, flat_w1])

requant1 = (scale_out0 * scale_w1) / scale_out1

# -------------------- 6) Write C source file -----------------------------
BATCHES = math.ceil(len(train_ds) / 128)

with open('model_params_quant.c', 'w') as f:
    f.write('// Auto-generated quantized model parameters and requant tables\n')
    f.write('#include <stdint.h>\n#include "lib_layers.h"\n\n')
    f.write(f'#define BATCHES {BATCHES}\n\n')
    # input QP
    f.write('// Input quant params\n')
    f.write(f'const quantization_params_t qp_input = {{ {scale_input:.6e}f, {zp_input} }};\n\n')
    # output QPs
    f.write('// Output quant params for layer0 & layer1\n')
    f.write(f'const quantization_params_t qp_out0  = {{ {scale_out0:.6e}f, {zp_out0} }};\n')
    f.write(f'const quantization_params_t qp_out1  = {{ {scale_out1:.6e}f, {zp_out1} }};\n\n')
    # requant arrays
    f.write('// Requant scales per channel\n')
    f.write(f'static const float requant0[{requant0.size}] = {{\n    ')
    for i, s in enumerate(requant0):
        f.write(f'{s:.6e}f, ')
        if (i+1) % 8 == 0:
            f.write('\n    ')
    f.write('\n};\n')
    f.write(f'static const requantization_params_t rq_layer0 = {{ requant0, {zp_out0} }};\n\n')

    f.write(f'static const float requant1[{requant1.size}] = {{\n    ')
    for i, s in enumerate(requant1):
        f.write(f'{s:.6e}f, ')
        if (i+1) % 8 == 0:
            f.write('\n    ')
    f.write('\n};\n')
    f.write(f'static const requantization_params_t rq_layer1 = {{ requant1, {zp_out1} }};\n\n')

    # layer0 weights+bias
    size0 = wb0_q.size
    f.write(f'static const int8_t layer0_wb_q[{size0}] = {{\n    ')
    for i, v in enumerate(wb0_q):
        f.write(f'{int(v)}, ')
        if (i+1) % 16 == 0:
            f.write('\n    ')
    f.write('\n};\n\n')

    # layer1 weights+bias
    size1 = wb1_q.size
    f.write(f'static const int8_t layer1_wb_q[{size1}] = {{\n    ')
    for i, v in enumerate(wb1_q):
        f.write(f'{int(v)}, ')
        if (i+1) % 16 == 0:
            f.write('\n    ')
    f.write('\n};\n')

print(f"// Final dequantization params for logits (qp_out1):")
print(f"//   scale = {scale_out1:.6e}f")
print(f"//   zero_point = {zp_out1}")

print("✅ model_params_quant.c written with per-channel requantization tables.")
#!/usr/bin/env python3
"""
export_model_params.py

… <same docstring as before> …
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

# -------------------- Helpers --------------------
INT8_MIN, INT8_MAX = -128, 127

def symmetric_qparams(x: np.ndarray, bits=8):
    qmax = 2**(bits-1)-1
    m = np.max(np.abs(x))
    scale = m / qmax if m != 0 else 1.0
    zp    = 0
    return np.float32(scale), zp

def quant_conv(W, b_fp, in_s, out_s):
    s_w = np.array([symmetric_qparams(row)[0] for row in W], np.float32)
    Wq  = np.clip(np.round(W / s_w[:,None]), INT8_MIN, INT8_MAX).astype(np.int8)
    scale_acc = in_s * s_w
    b_q       = np.round(b_fp / scale_acc).astype(np.int32)
    rq        = scale_acc / out_s
    return b_q, Wq, rq

# -------------------- QAT Model Definition --------------------
class QATDWPWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = tq.QuantStub()
        self.dw0     = nn.Conv2d(1,1,3,bias=True)
        self.pw0     = nn.Conv2d(1,16,1,bias=True)
        self.relu0   = nn.ReLU()
        self.pool0   = nn.MaxPool2d(3,3)
        self.dw1     = nn.Conv2d(16,16,3,groups=16,bias=True)
        self.pw1     = nn.Conv2d(16,32,1,bias=True)
        self.relu1   = nn.ReLU()
        self.pool1   = nn.MaxPool2d(3,3)
        self.fc0     = nn.Linear(32*2*2,32,bias=True)
        self.relu2   = nn.ReLU()
        self.fc1     = nn.Linear(32,10,bias=True)
        self.dequant = tq.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.dw0(x)
        x = self.relu0(self.pw0(x))
        x = self.pool0(x)
        x = self.dw1(x)
        x = self.relu1(self.pw1(x))
        x = self.pool1(x)
        x = x.flatten(1)
        x = self.relu2(self.fc0(x))
        x = self.fc1(x)
        return self.dequant(x)

# -------------------- QAT Training --------------------
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_qat  = QATDWPWNet().to(device)
tq.fuse_modules(model_qat,
    [['pw0','relu0'], ['pw1','relu1'], ['fc0','relu2']],
    inplace=True)

symm_qconfig = torch.ao.quantization.QConfig(
    activation=FakeQuantize.with_args(
        observer=MinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric
    ),
    weight=FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric
    )
)
model_qat.qconfig = symm_qconfig

tq.prepare_qat(model_qat, inplace=True)

transform    = transforms.ToTensor()
train_ds     = datasets.MNIST('.', train=True, download=True, transform=transform)
test_ds      = datasets.MNIST('.', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
optimizer    = torch.optim.Adam(model_qat.parameters(), lr=1e-3)
criterion    = nn.CrossEntropyLoss()

model_qat.train()
for epoch in range(5):
    tot = 0.0
    for xb,yb in train_loader:
        xb,yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model_qat(xb)
        loss= criterion(out,yb)
        loss.backward()
        optimizer.step()
        tot += loss.item()*xb.size(0)
    print(f"Epoch {epoch+1}/5 QAT loss: {tot/len(train_ds):.4f}")
model_qat.eval()

# -------------------- Convert to Quantized --------------------
torch.backends.quantized.engine = 'qnnpack'
model_int = tq.convert(model_qat, inplace=False)
model_int.eval()

# -------------------- Prepare float_model & scales --------------------
float_model = QATDWPWNet().to(device)
clean_sd    = {k:v for k,v in model_qat.state_dict().items()
               if 'activation_post_process' not in k and 'fake_quant' not in k}
float_model.load_state_dict(clean_sd, strict=False)
float_model.eval()

all_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
X_all,_    = next(iter(all_loader))
X_flat     = X_all.view(-1).numpy().astype(np.float32)
s_in, zp_in= symmetric_qparams(X_flat)

with torch.no_grad():
    x = float_model.dw0(torch.from_numpy(X_flat.reshape(-1,1,28,28)))
    x = F.relu(float_model.pw0(x))
    A0= x.numpy().ravel(); s_a0,_= symmetric_qparams(A0)
    x = float_model.pool0(x)
    x = float_model.dw1(x)
    x = F.relu(float_model.pw1(x))
    A1= x.numpy().ravel(); s_a1,_= symmetric_qparams(A1)
    x = float_model.pool1(x).view(x.size(0),-1)
    x = F.relu(float_model.fc0(x))
    A2= x.numpy().ravel(); s_a2,_= symmetric_qparams(A2)
    LOG= float_model.fc1(x).numpy().ravel(); s_out,_= symmetric_qparams(LOG)

# -------------------- Quantize & Pack weights/biases --------------------
# DW0
W = float_model.dw0.weight.detach().numpy().reshape(1,9)
b = float_model.dw0.bias.detach().numpy()
dw0_bq, dw0_wq, rq0 = quant_conv(W, b, s_in, s_a0)

# PW0
W = float_model.pw0.weight.detach().numpy().reshape(16,1)
b = float_model.pw0.bias.detach().numpy()
pw0_bq, pw0_wq, rq1 = quant_conv(W, b, s_in, s_a0)

# DW1
W = float_model.dw1.weight.detach().numpy().reshape(16,9)
b = float_model.dw1.bias.detach().numpy()
dw1_bq, dw1_wq, rq2 = quant_conv(W, b, s_a0, s_a1)

# PW1
W = float_model.pw1.weight.detach().numpy().reshape(32,16)
b = float_model.pw1.bias.detach().numpy()
pw1_bq, pw1_wq, rq3 = quant_conv(W, b, s_a1, s_a2)

# FC0
W = float_model.fc0.weight.detach().numpy()
b = float_model.fc0.bias.detach().numpy()
fc0_bq, fc0_wq, rq4 = quant_conv(W, b, s_a2, s_out)

# FC1
W = float_model.fc1.weight.detach().numpy()
b = float_model.fc1.bias.detach().numpy()
fc1_bq, fc1_wq, rq5 = quant_conv(W, b, s_out, s_out)

# -------------------- Write model_params.h --------------------
hdr = []
def A(line=""):
    hdr.append(line)

A("// Auto-generated symmetric-quant model_params.h")
A("#ifndef MODELPARAMS_H")
A("#define MODELPARAMS_H\n")
A('#include <stdint.h>')
A('#include "lib_layers.h"\n')
A("#define BATCHES 1\n")
A(f"const quantization_params_t qp_input  = {{ {s_in:.8e}f, {zp_in} }};")
A(f"const quantization_params_t qp_logits = {{ {s_out:.8e}f, 0 }};\n")

def write_rq(tag, scales):
    A(f"static const float {tag}_scale[{len(scales)}] = {{ " +
      ", ".join(f"{s:.8e}f" for s in scales) + " };")
    A(f"const quantization_params_t {tag} = {{ {tag}_scale, 0 }};\n")

write_rq("rq_conv0_dw", rq0)
write_rq("rq_conv0_pw", rq1)
write_rq("rq_conv1_dw", rq2)
write_rq("rq_conv1_pw", rq3)
write_rq("rq_fc0"     , rq4)
write_rq("rq_fc1"     , rq5)

def write_blob(name, bq, wq):
    data = bytearray()
    for bi in bq.flatten():
        data += int(bi).to_bytes(4,'little',signed=True)
    arr = wq if not name.startswith("fc") else wq
    if name in ("conv0_pw","conv1_pw","fc0","fc1"):
        arr = arr.T
    data += arr.flatten().astype(np.uint8).tobytes()
    A(f"static const uint8_t {name}_wb_q[] = {{")
    for i in range(0,len(data),16):
        chunk = ", ".join(str(b) for b in data[i:i+16])
        suffix = "," if i+16<len(data) else ""
        A("    "+chunk+suffix)
    A("};\n")

write_blob("conv0_dw", dw0_bq, dw0_wq)
write_blob("conv0_pw", pw0_bq, pw0_wq)
write_blob("conv1_dw", dw1_bq, dw1_wq)
write_blob("conv1_pw", pw1_bq, pw1_wq)
write_blob("fc0"     , fc0_bq, fc0_wq)
write_blob("fc1"     , fc1_bq, fc1_wq)

A("#endif // MODELPARAMS_H")
Path("model_params.h").write_text("\n".join(hdr))
print("✓ model_params.h written")

# -------------------- DEBUG: Quantized Inference --------------------
# Uncomment to print each activation row‑by‑row, channel‑by‑channel

# sample, label = test_ds[0]
# x = sample.unsqueeze(0).to(device)  # (1,1,28,28)
# with torch.no_grad():
#     # Use **quint8** for activations + QNNPACK
#     x_q = torch.quantize_per_tensor(x, s_in, 0, torch.qint8)

#     def print_q(t_q, title):
#         arr = t_q.int_repr().cpu().numpy()   # now signed -128…127
#         _,C,H,W = arr.shape
#         print(f"\n{title} (int8):")
#         for c in range(C):
#             print(f"Ch {c}:")
#             for h in range(H):
#                 print(" ".join(f"{arr[0,c,h,w]:4d}" for w in range(W)))
#         farr = t_q.dequantize().cpu().numpy()
#         print(f"{title} (dequantized):")
#         for c in range(C):
#             print(f"Ch {c}:")
#             for h in range(H):
#                 print(" ".join(f"{farr[0,c,h,w]:.4f}" for w in range(W)))

#     print_q(x_q, "Input")
#     out = model_int.dw0(x_q);                              print_q(out, "After conv0_dw")
#     out = model_int.relu0(model_int.pw0(out));            print_q(out, "After conv0_pw+ReLU")
#     p   = F.max_pool2d(out.dequantize(),3,3)
#     p_q = torch.quantize_per_tensor(p, s_a0, 128, torch.quint8)
#     print_q(p_q, "After pool0")
#     out = model_int.dw1(p_q);                             print_q(out, "After conv1_dw")
#     out = model_int.relu1(model_int.pw1(out));            print_q(out, "After conv1_pw+ReLU")
#     p   = F.max_pool2d(out.dequantize(),3,3)
#     p_q = torch.quantize_per_tensor(p, s_a1, 128, torch.quint8)
#     print_q(p_q, "After pool1")
#     flat = p_q.dequantize().flatten(1)
#     f_q  = torch.quantize_per_tensor(flat, s_a2, 128, torch.quint8)
#     out = model_int.relu2(model_int.fc0(f_q.dequantize()))
#     out_q= torch.quantize_per_tensor(out, s_a2, 128, torch.quint8)
#     print_q(out_q, "After fc0+ReLU")
#     out = model_int.fc1(out_q.dequantize())
#     out_q= torch.quantize_per_tensor(out, s_out,128,torch.quint8)
#     print_q(out_q, "After fc1 (logits)")
#     print("True label:", label)
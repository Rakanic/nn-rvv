#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# --- Hyperparameters --------------------------------------------------------
BATCH_SIZE = 128
EPOCHS     = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data -------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(".", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(".", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# --- Model ------------------------------------------------------------------
class DWPWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw0 = nn.Conv2d(1, 1, 3, groups=1, bias=False)
        self.pw0 = nn.Conv2d(1, 16, 1, bias=True)
        self.pool0 = nn.MaxPool2d(3, 3)
        self.dw1 = nn.Conv2d(16, 16, 3, groups=16, bias=False)
        self.pw1 = nn.Conv2d(16, 32, 1, bias=True)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.fc0 = nn.Linear(32*2*2, 32, bias=True)
        self.fc1 = nn.Linear(32, 10, bias=True)

    def forward(self, x):
        x = self.dw0(x)
        x = F.relu(self.pw0(x))
        x = self.pool0(x)
        x = self.dw1(x)
        x = F.relu(self.pw1(x))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        return self.fc1(x)

model = DWPWNet().to(DEVICE)

# --- Training ---------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} done")

# --- Evaluate & print test accuracy ----------------------------------------
model.eval()
correct = total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
accuracy = 100.0 * correct / total
print(f"\nTest accuracy: {accuracy:.2f}%\n")

# --- Export helper ----------------------------------------------------------
def dump_layer(f, name, weight, bias):
    out_ch = weight.shape[0]
    flat = []
    # bias first (real bias or zeros-per-output-channel)
    if bias is None:
        flat += [0.0] * out_ch
    else:
        flat += bias.cpu().numpy().reshape(-1).tolist()
    # then weights
    w = weight.cpu().numpy()
    if w.ndim == 2:
        # Fully‑connected layer: export as input‑major (in × out)
        flat += w.T.reshape(-1).tolist()
    else:
        # Conv layers: keep (out,in,h,w) in C‑order
        flat += w.reshape(-1).tolist()
    f.write(f"// {name}: len={len(flat)}\n")
    f.write(f"static float {name}[{len(flat)}] = {{\n    ")
    f.write(", ".join(f"{v:.8e}" for v in flat))
    f.write("\n};\n\n")

# --- Export model_params.h -------------------------------------------------
hdr = Path("model_params.h")
with hdr.open("w") as f:
    f.write("// Auto-generated C header from PyTorch model parameters\n")
    f.write("#ifndef MODELPARAMS_H\n")
    f.write("#define MODELPARAMS_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write("#define BATCHES 1\n\n")
    dump_layer(f, "dw0", model.dw0.weight.data, model.dw0.bias)
    dump_layer(f, "pw0", model.pw0.weight.data, model.pw0.bias.data)
    dump_layer(f, "dw1", model.dw1.weight.data, model.dw1.bias)
    dump_layer(f, "pw1", model.pw1.weight.data, model.pw1.bias.data)
    dump_layer(f, "fc0", model.fc0.weight.data, model.fc0.bias.data)
    dump_layer(f, "fc1", model.fc1.weight.data, model.fc1.bias.data)
    f.write("#endif  // MODELPARAMS_H\n")
print(f"Wrote {hdr.resolve()}")

# --- Export input_data.h ---------------------------------------------------
N = 20
X = torch.stack([test_ds[i][0] for i in range(N)])  # (N,1,28,28)
flat_in = X.numpy().reshape(-1)
inp_hdr = Path("input_data.h")
with inp_hdr.open("w") as f:
    f.write("#ifndef INPUT_DATA_H\n#define INPUT_DATA_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write("// Auto‑generated C header from MNIST test inputs (flattened)\n")
    f.write(f"// Contains {N} test inputs; each is 28×28 = 784 floats.\n\n")
    f.write(f"static float input[{N*784}] = {{\n    ")
    f.write(", ".join(f"{v:.8e}" for v in flat_in))
    f.write("\n};\n\n#endif  // INPUT_DATA_H\n")
print(f"Wrote {inp_hdr.resolve()}")

# --- Inference with layer-by-layer prints -------------------------------
model.eval()
# sample, label = test_ds[0]
# x = sample.unsqueeze(0).to(DEVICE)  # shape (1,1,28,28)

# with torch.no_grad():
#     print("\n=== Inference on one sample ===")

#     # Input
#     print("Input (1×1×28×28):")
#     arr = x.cpu().numpy()
#     N,C,H,W = arr.shape
#     for n in range(N):
#         for c in range(C):
#             for h in range(H):
#                 for w in range(W):
#                     print(f"input[{n},{c},{h},{w}] = {arr[n,c,h,w]:.8e}")

#     # dw0
#     x1 = model.dw0(x)
#     print("\nAfter dw0 (1×1×26×26):")
#     arr = x1.cpu().numpy()
#     N,C,H,W = arr.shape
#     for n in range(N):
#         for c in range(C):
#             for h in range(H):
#                 for w in range(W):
#                     print(f"dw0[{n},{c},{h},{w}] = {arr[n,c,h,w]:.8e}")

#     # pw0 + ReLU
#     x2 = model.pw0(x1)
#     x2 = F.relu(x2)
#     print("\nAfter pw0+ReLU (1×16×26×26):")
#     arr = x2.cpu().numpy()
#     N,C,H,W = arr.shape
#     for n in range(N):
#         for c in range(C):
#             for h in range(H):
#                 for w in range(W):
#                     print(f"pw0_relu[{n},{c},{h},{w}] = {arr[n,c,h,w]:.8e}")

#     # pool0
#     x3 = model.pool0(x2)
#     print("\nAfter pool0 (1×16×8×8):")
#     arr = x3.cpu().numpy()
#     N,C,H,W = arr.shape
#     for n in range(N):
#         for c in range(C):
#             for h in range(H):
#                 for w in range(W):
#                     print(f"pool0[{n},{c},{h},{w}] = {arr[n,c,h,w]:.8e}")

#     # dw1
#     x4 = model.dw1(x3)
#     print("\nAfter dw1 (1×16×6×6):")
#     arr = x4.cpu().numpy()
#     N,C,H,W = arr.shape
#     for n in range(N):
#         for c in range(C):
#             for h in range(H):
#                 for w in range(W):
#                     print(f"dw1[{n},{c},{h},{w}] = {arr[n,c,h,w]:.8e}")

#     # pw1 + ReLU
#     x5 = model.pw1(x4)
#     x5 = F.relu(x5)
#     print("\nAfter pw1+ReLU (1×32×6×6):")
#     arr = x5.cpu().numpy()
#     N,C,H,W = arr.shape
#     for n in range(N):
#         for c in range(C):
#             for h in range(H):
#                 for w in range(W):
#                     print(f"pw1_relu[{n},{c},{h},{w}] = {arr[n,c,h,w]:.8e}")

#     # pool1
#     x6 = model.pool1(x5)
#     print("\nAfter pool1 (1×32×2×2):")
#     arr = x6.cpu().numpy()
#     N,C,H,W = arr.shape
#     for n in range(N):
#         for c in range(C):
#             for h in range(H):
#                 for w in range(W):
#                     print(f"pool1[{n},{c},{h},{w}] = {arr[n,c,h,w]:.8e}")

#     # flatten
#     x7 = x6.view(x6.size(0), -1)
#     print("\nAfter flatten (1×128):")
#     arr = x7.cpu().numpy()
#     N,D = arr.shape
#     for n in range(N):
#         for d in range(D):
#             print(f"flat[{n},{d}] = {arr[n,d]:.8e}")

#     # fc0 + ReLU
#     x8 = model.fc0(x7)
#     x8 = F.relu(x8)
#     print("\nAfter fc0+ReLU (1×32):")
#     arr = x8.cpu().numpy()
#     N,D = arr.shape
#     for n in range(N):
#         for d in range(D):
#             print(f"fc0_relu[{n},{d}] = {arr[n,d]:.8e}")

#     # fc1 logits
#     logits = model.fc1(x8)
#     print("\nAfter fc1 (logits, 1×10):")
#     arr = logits.cpu().numpy()
#     N,D = arr.shape
#     for n in range(N):
#         for d in range(D):
#             print(f"logits[{n},{d}] = {arr[n,d]:.8e}")

#     # softmax
#     probs = F.softmax(logits, dim=1)
#     print("\nAfter softmax (1×10):")
#     arr = probs.cpu().numpy()
#     N,D = arr.shape
#     for n in range(N):
#         for d in range(D):
#             print(f"probs[{n},{d}] = {arr[n,d]:.8e}")

#     print(f"\nPredicted class: {probs.argmax(dim=1).item()}, true label: {label}")
# 🧠 Synthetic Human Face Generation (DCGAN 128×128)

> **Synthesize photorealistic human face variations from latent random noise vectors using a Deep Convolutional Generative Adversarial Network (DCGAN) trained at 128×128 high resolution on the CelebA dataset.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CelebA%20(202k)-blue)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
[![Target Resolution](https://img.shields.io/badge/Target%20Resolution-128%C3%97128%20RGB-purple)]()
[![Hardware](https://img.shields.io/badge/Trained%20On-NVIDIA%20T4%20(Colab)-76B900?logo=nvidia&logoColor=white)](https://colab.research.google.com/)
[![Architecture](https://img.shields.io/badge/Architecture-DCGAN%20128-orange)]()

---

## 📌 Project Overview

Generative Adversarial Networks (GANs) formulate generative modeling as a two-player minimax game between a **Generator ($G$)** and a **Discriminator ($D$)**. In this project:
- **Objective:** Map a continuous 128-dimensional isotropic Gaussian latent space $z \sim \mathcal{N}(0, I_{128})$ to high-resolution synthetic **$128 \times 128 \times 3$** human face images.
- **Dataset:** **CelebA (CelebFaces Attributes Dataset)** containing **202,599** aligned celebrity face images downloaded via Kagglehub.
- **Framework:** Expanded **DCGAN 128×128** architecture with multi-stage convolutional scaling, TTUR (Two Time-Scale Update Rule), adaptive extra generator updates to counter discriminator dominance, gradient clipping under AMP, and equilibrium-based checkpoint tracking.
- **Historical Note:** The preliminary exploratory run (Run 1) was trained at **64×64** resolution for 20 epochs to establish baseline convergence behavior, identify discriminator saturation dynamics, and validate stability mechanisms (detailed in [`log.md`](log.md) and [`Training1_Epochs20.md`](Training1_Epochs20.md)). All production training from this point forward is standardized at **128×128**.

---

## 🏛️ Network Architecture (128×128)

### 1. Generator ($G$)
Maps latent noise $z \in \mathbb{R}^{128}$ through 5 transposed convolutional upsamplings to produce $x_{\text{fake}} \in [-1, 1]^{3 \times 128 \times 128}$:

```
z ~ N(0, I) [128, 1, 1]
   │
   ▼ ConvTranspose2d(128 -> 512, k=4, s=1, p=0) + BatchNorm2d + ReLU
[512, 4, 4]
   │
   ▼ ConvTranspose2d(512 -> 512, k=4, s=2, p=1) + BatchNorm2d + ReLU
[512, 8, 8]
   │
   ▼ ConvTranspose2d(512 -> 256, k=4, s=2, p=1) + BatchNorm2d + ReLU
[256, 16, 16]
   │
   ▼ ConvTranspose2d(256 -> 128, k=4, s=2, p=1) + BatchNorm2d + ReLU
[128, 32, 32]
   │
   ▼ ConvTranspose2d(128 -> 64,  k=4, s=2, p=1) + BatchNorm2d + ReLU
[64, 64, 64]
   │
   ▼ ConvTranspose2d(64  -> 3,   k=4, s=2, p=1) + Tanh
[3, 128, 128] Synthetic Face
```

*Total Generator Parameters: **8.00 Million**.*

### 2. Discriminator ($D$)
Classifies an input image $x \in [-1, 1]^{3 \times 128 \times 128}$ as real ($0.9$) or synthetic ($0.0$) through 5 strided convolutional downsamplings:

```
x [3, 128, 128]
   │
   ▼ Conv2d(3 -> 64, k=4, s=2, p=1) + LeakyReLU(0.2)  [No BatchNorm on input layer]
[64, 64, 64]
   │
   ▼ Conv2d(64 -> 128, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2)
[128, 32, 32]
   │
   ▼ Conv2d(128 -> 256, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2)
[256, 16, 16]
   │
   ▼ Conv2d(256 -> 512, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2)
[512, 8, 8]
   │
   ▼ Conv2d(512 -> 512, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2)
[512, 4, 4]
   │
   ▼ Conv2d(512 -> 1, k=4, s=1, p=0)  [Unconstrained scalar logit]
[1, 1, 1] -> Scalar Score
```

*Total Discriminator Parameters: **6.96 Million** (Combined: **14.96 Million**).*

### 3. Weight Initialization
Following Radford et al. (2015):
- Convolution & Transposed Convolution weights: $\mathcal{N}(0.0, 0.02)$
- Batch Normalization weights: $\mathcal{N}(1.0, 0.02)$, biases initialized to $0.0$.

---

## 🔬 Dataset & Preprocessing Pipeline (128×128)

The CelebA dataset consists of 202,599 cropped face images ($178 \times 218$). To optimize 128×128 high-resolution generation:

1. **Center Crop (160×160):** Tightly frames primary facial morphology (eyes, nose, mouth, hairline) while eliminating background distractions and apparel boundaries.
2. **Bilinear Resize (128×128):** Resamples to the target $128 \times 128$ resolution with bilinear filtering.
3. **Random Horizontal Flip ($p=0.5$):** Enhances pose invariance and doubles effective training diversity.
4. **Range Normalization:** Pixels scaled from $[0, 255] \to [-1.0, 1.0]$ via `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])` to match the Generator's `Tanh` activation function.

---

## ⚙️ Tuned Hyperparameters & Stability Engineering (Profile A)

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Target Resolution** | **128×128** | Enhanced facial micro-details and feature clarity |
| **Batch Size** | **64** | Provides **3,165 updates/epoch** for fine-grained gradient updates |
| **Latent Vector ($z$)** | **128** | +28% expressive capacity for complex facial morphology |
| **Generator LR ($LR_G$)** | **$2.0 \times 10^{-4}$** | Standard DCGAN learning rate |
| **Discriminator LR ($LR_D$)** | **$1.5 \times 10^{-4}$** | **TTUR:** Slightly slower $D$ updates prevent discriminator runaway |
| **Adam Betas** | $\beta_1=0.5, \beta_2=0.999$ | Low momentum $\beta_1$ prevents limit-cycle oscillations |
| **Loss Function** | `BCEWithLogitsLoss` | Numerically stable log-sum-exp sigmoid computation |
| **Label Smoothing** | Real = 0.9, Fake = 0.0 | One-sided smoothing prevents discriminator overconfidence |
| **LR Scheduler** | Cosine Annealing ($\eta_{\min}=5\times 10^{-5}$) | Preserves gradient flow in late epochs |
| **Adaptive G Boosting** | Active when $D(\text{fake}) < 0.20$ | Triggers extra G step when discriminator begins dominating |
| **Gradient Clipping** | Max norm 1.0 on $G$ | Stabilizes high-resolution feature synthesis under AMP |
| **Mixed Precision (AMP)** | FP16 with `GradScaler` | Fast execution (~1.5–1.8 min/epoch on Colab T4) |

---

## 📁 Repository Structure

```
Synthetic Human Face Generator/
├── face-genrator.ipynb        # All-in-one Google Colab 128x128 training notebook
├── README.md                  # Comprehensive project documentation (128x128 standard)
├── requirements.md            # Hardware & software dependency specification
├── log.md                     # Chronological experiment log (64x64 baseline + 128x128 tuning)
├── Training1_Epochs20.md      # Full metrics & embedded visual progression of Run 1 (64x64)
└── DCGAN_64_Epochs_20/        # Preserved historical assets from 64x64 baseline run
    ├── DCGAN_64.pth           # Full 64x64 checkpoint
    ├── DCGAN_64.csv           # 64x64 metrics history
    └── DCGAN_64 (2)/          # Generated 64-face grids (epoch_01.png -> epoch_20.png)
```

---

## 🚀 How to Train & Run (128×128)

### Running in Google Colab (Recommended)
1. Open [`face-genrator.ipynb`](face-genrator.ipynb) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells:
   - Cell 1: Environment & GPU verification.
   - Cell 2: Automated CelebA dataset download via `kagglehub`.
   - Cells 3–6: Tuned 128×128 Model definition, loss, optimizers, and DataLoader.
   - Cell 7: High-speed training loop with adaptive equilibrium & live 128×128 face visualizer.
   - Cell 8: Checkpoint export and download of `dcgan128_export.zip`.

### Generating 128×128 Faces Locally (Python / PyTorch)
```python
import torch
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Instantiate 128x128 Generator (z_dim=128)
# (Use DCGAN_Generator class from face-genrator.ipynb)
G = DCGAN_Generator(z_dim=128, ngf=64, nc=3).to(device)

# 2. Load Weights
ckpt = torch.load('outputs/checkpoints/best_generator.pth', map_location=device)
G.load_state_dict(ckpt)
G.eval()

# 3. Generate 36 High-Res 128x128 Faces
with torch.no_grad():
    z = torch.randn(36, 128, device=device)
    fake_faces = G(z).cpu()
    grid = vutils.make_grid((fake_faces * 0.5 + 0.5).clamp(0, 1), nrow=6)
    vutils.save_image(grid, 'generated_faces_128.png')
```

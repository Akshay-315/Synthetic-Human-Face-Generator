# Dependencies & System Requirements (128×128 DCGAN)

This document outlines the software dependencies, system prerequisites, and hardware specifications required to train and run the **Synthetic Human Face Generator (DCGAN 128×128)**.

---

## 💻 Hardware Prerequisites

| Resource | Minimum | Recommended (Used in Training) |
|---|---|---|
| **GPU** | NVIDIA GPU with 6–8 GB VRAM (RTX 2060 / 3060+) | NVIDIA T4 / V100 / A100 (15–16+ GB VRAM on Colab) |
| **System RAM** | 8 GB | 12–16 GB |
| **Storage** | ~8 GB free space | ~15–20 GB free space (to store uncompressed CelebA images) |
| **CUDA Compute** | CUDA 11.8+ / 12.x | CUDA 12.x with cuDNN Benchmark enabled |
| **Environment** | Local Linux / Windows PC or Google Colab | Google Colab Free / Pro (T4 GPU runtime) |

*Note: With 128×128 resolution at `batch_size = 64` and FP16 Automatic Mixed Precision (AMP), GPU VRAM consumption is ~1.2–1.8 GB during training, allowing it to train smoothly within free-tier Colab GPU limits.*

---

## 🐍 Python Environment

- **Python Version:** `3.10` or `3.11` (Tested on Python 3.11)
- **Package Manager:** `pip` or `conda`

---

## 📦 Required Packages

### Core Deep Learning & Computer Vision

```bash
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
```

*For GPU acceleration, install PyTorch with matching CUDA drivers (e.g., `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).*

### Dataset Acquisition & Image Processing

```bash
kagglehub>=0.2.0    # Automated high-speed download of CelebA from Kaggle
pillow>=9.5.0       # Image decoding, center cropping (160x160), and resizing (128x128)
numpy>=1.23.0       # Numerical operations
```

### Telemetry, Visualization & Utilities

```bash
matplotlib>=3.7.0   # Training loss curves, probability plots, and sample grids
ipywidgets>=8.0.0   # Colab interactive widget displays
ipython>=8.0.0      # In-notebook display clearing and dynamic telemetry
```

### Optional (Inference & Web Demo)

```bash
gradio>=3.40.0      # Interactive web UI for latent space exploration (app.py)
```

---

## 🚀 Quick Setup & Installation

### Option 1: Google Colab (Zero-Setup)
In Google Colab, PyTorch, Torchvision, and CUDA are pre-installed. You only need:
```bash
!pip install -q kagglehub
```

### Option 2: Local Installation (Virtual Environment)

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 3. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install supporting libraries
pip install kagglehub pillow numpy matplotlib ipython gradio
```

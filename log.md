# 📜 Project Execution Log: Synthetic Human Face Generation

This document provides a comprehensive, chronological record of all engineering steps, architectural decisions, training runs, telemetry metrics, and diagnostic evaluations performed for the **Synthetic Human Face Generation** project.

---

## 📋 Task & Topic Specification

- **Topic:** Synthetic Human Face Generation
- **Core Task:** Synthesize photorealistic face variations from latent random noise vectors.
- **Dataset:** CelebA (CelebFaces Attributes Dataset) — 202,599 aligned images from Kaggle (`jessicali9530/celeba-dataset`) / Hugging Face.
- **Model Family:** Deep Convolutional Generative Adversarial Network (DCGAN).
- **Current Target Resolution:** **$128 \times 128$ RGB** (Scaled from the exploratory $64 \times 64$ baseline).

---

## 🛠️ Step 1: Baseline Architecture Design & Historical Diagnosis

### Initial Problem Context & Previous Failure Modes
In early experiments before baseline training, generative face models frequently exhibited catastrophic degradation ("got worse every epoch"):
1. **Critic Overpowering:** The discriminator was updated at an asymmetric ratio ($N_{\text{critic}} = 5$) with $LR_D = 4 \times 10^{-4}$ vs $LR_G = 1 \times 10^{-4}$. Within half an epoch, the discriminator overpowered the generator by over $20\times$, starving $G$ of informative gradients.
2. **Incompatible Regularization:** Mixing Spectral Normalization with WGAN-GP introduced conflicting Lipschitz constraints that destabilized optimization.
3. **Confidence Saturation:** With hard binary cross-entropy targets ($1.0$ and $0.0$), the discriminator rapidly approached near-zero loss, causing vanishing gradients for the generator.

### Baseline Formulation (Radford et al., 2015)
The architecture was initially formulated with canonical DCGAN standards to establish baseline convergence:
- **Balanced 1:1 Updates:** $G$ and $D$ updated in lockstep with identical base learning rates ($LR_G = LR_D = 2 \times 10^{-4}$).
- **One-Sided Label Smoothing:** Real image targets set to $0.9$ (instead of $1.0$), while fake targets remained $0.0$.
- **BCEWithLogitsLoss:** Integrated Sigmoid layer with cross-entropy for maximal numerical stability in floating-point precision.
- **Weight Initialization:** Convolutions initialized with $\mathcal{N}(0.0, 0.02)$, Batch Normalization weights with $\mathcal{N}(1.0, 0.02)$, biases with $0.0$.
- **Cosine Annealing Learning Rate:** Decaying learning rate across 20 epochs to prevent parameter oscillation.

---

## 🔄 Step 2: CelebA Preprocessing Pipeline

A dedicated PyTorch Dataset (`FastCelebADataset`) was constructed to handle the 202,599 CelebA face images:
1. **Automated Acquisition:** Ingested directly using `kagglehub.dataset_download('jessicali9530/celeba-dataset')` inside Google Cloud network in under 60 seconds.
2. **Center Crop:** The original $178 \times 218$ images feature large proportions of background, hair margins, and clothing. A tight center crop isolates primary facial morphology (eyes, nose, mouth, chin).
3. **Bilinear Downsampling:** Resized to the target resolution with bilinear interpolation.
4. **Data Augmentation:** Random horizontal flipping ($p=0.5$) applied on-the-fly.
5. **Range Mapping:** Normalization with mean `[0.5, 0.5, 0.5]` and standard deviation `[0.5, 0.5, 0.5]` scales pixel intensities to $[-1.0, 1.0]$, precisely aligning with the Generator's `Tanh` activation function.

---

## 📊 Step 3: Record of Training Run 1 (64×64 Resolution Baseline)

> **Important Resolution Note:** **Training Run 1 was conducted exclusively at $64 \times 64$ resolution** across 20 epochs as an exploratory baseline. Its primary objectives were to verify end-to-end throughput on Colab T4, evaluate training stability, measure loss trajectory, and identify failure points under standard DCGAN conditions.

### Session Telemetry (Run 1 — 64×64)
- **Resolution:** $64 \times 64 \times 3$
- **Hardware:** NVIDIA Tesla T4 (Google Colab)
- **Batch Size:** 128 (1,582 batches/epoch)
- **VRAM Utilization:** ~0.4 GB allocated
- **Iteration Throughput:** ~22–24 iterations/second (~2,800–3,000 faces/second)
- **Time Per Epoch:** ~1.1 to 1.3 minutes
- **Total Duration:** ~25 minutes for all 20 epochs

### Epoch-by-Epoch Metrics Table (Run 1 — 64×64 Baseline)

| Epoch | Loss D | Loss G | $D(\text{real})$ Prob | $D(\text{fake})$ Prob | Learning Rate | Observations & Key Milestones |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | 0.8047 | 4.6283 | 0.7388 | 0.1619 | 2.000e-4 | Coarse facial oval contours emerge from Gaussian noise |
| **2** | 0.8352 | 3.1522 | 0.7026 | 0.1962 | 1.988e-4 | Generator loss drops sharply by 32% |
| **3** | 0.8924 | 2.4686 | 0.6734 | 0.2262 | 1.954e-4 | Loss G converges below 2.5; healthy adversarial competition |
| **4** | 0.8659 | 2.3908 | 0.6772 | 0.2221 | 1.896e-4 | **Equilibrium zone entered**: facial symmetry established |
| **5** | 0.8709 | 2.3368 | 0.6748 | 0.2246 | 1.819e-4 | Clear eye sockets, lip definition, skin tones, and hair color |
| **6** | 0.8416 | 2.3013 | 0.6836 | 0.2158 | 1.722e-4 | Expression diversity and head angle variations emerge |
| **7** | 0.8218 | 2.3535 | 0.6981 | 0.2020 | 1.608e-4 | **Optimal balance point**: Loss D ~0.82, Loss G ~2.35 |
| **8** | 0.7427 | 2.4929 | 0.7246 | 0.1749 | 1.481e-4 | Discriminator begins learning subtle artifacts; D loss declines |
| **9** | 0.6793 | 2.6681 | 0.7514 | 0.1485 | 1.344e-4 | $D(\text{fake})$ drops below 0.15; D gaining statistical advantage |
| **10** | 0.6111 | 2.8499 | 0.7783 | 0.1224 | 1.199e-4 | **Peak visual fidelity**: Sharp features, realistic illumination |
| **11** | 0.5808 | 3.0221 | 0.7972 | 0.1025 | 1.050e-4 | Fine details distinct; eyebrow and hair boundaries crisp |
| **12** | 0.5365 | 3.1098 | 0.8094 | 0.0903 | 9.014e-5 | D confidence crosses 0.80 on real images |
| **13** | 0.5194 | 3.2402 | 0.8224 | 0.0787 | 7.564e-5 | Generator loss rises above 3.2 |
| **14** | 0.4813 | 3.4071 | 0.8350 | 0.0648 | 6.187e-5 | D rejection rate on fake images exceeds 93% |
| **15** | 0.4395 | 3.5715 | 0.8505 | 0.0492 | 4.916e-5 | Visual quality remains high, but D dominance is pronounced |
| **16** | 0.4208 | 3.7083 | 0.8594 | 0.0404 | 3.782e-5 | Loss D falls to 0.42; gradient magnitude to G starts tapering |
| **17** | 0.4087 | 3.8536 | 0.8654 | 0.0346 | 2.814e-5 | Slower visual evolution due to diminished learning rate |
| **18** | 0.3996 | 3.9781 | 0.8700 | 0.0299 | 2.035e-5 | Generator struggles to find novel gradients against strong D |
| **19** | 0.3921 | 4.0584 | 0.8729 | 0.0269 | 1.465e-5 | Loss G exceeds 4.0; $D(\text{fake})$ drops to 0.027 |
| **20** | 0.3876 | 4.1300 | 0.8750 | 0.0248 | 1.117e-5 | Training concluded; final weights and sample grids saved |

### Preserved 64×64 Artifacts
All outputs from this initial 64×64 run have been archived:
- `DCGAN_64_Epochs_20/DCGAN_64.pth` (76.8 MB) — Complete 64×64 checkpoint.
- `DCGAN_64_Epochs_20/DCGAN_64.csv` — Full numerical metrics log.
- `DCGAN_64_Epochs_20/DCGAN_64 (2)/` — Generated 64-face grids (`epoch_01.png` to `epoch_20.png`).
- `Training1_Epochs20.md` — Self-contained markdown report featuring Base64-embedded face grids.

---

## 🔍 Step 4: Diagnostic Findings from Run 1

Analysis of the 64×64 run revealed a classic GAN instability pattern: **late-stage discriminator saturation**:
1. **The Divergence Point (Epoch 8):** In Epochs 4–7, $D(\text{real}) \approx 0.68$ and $D(\text{fake}) \approx 0.22$, which represented an ideal competitive balance. By Epoch 8, the discriminator began mastering subtle textural artifacts, driving $D(\text{fake})$ steadily downward.
2. **Discriminator Saturation (Epochs 16–20):** By Epoch 20, $D(\text{fake})$ had collapsed to **0.025** (a 97.5% rejection rate). Because the learning rate simultaneously decayed from $2 \times 10^{-4} \to 1.1 \times 10^{-5}$, the generator lacked sufficient update velocity to counter the discriminator's sharpening decision surface.
3. **Checkpoint Assessment:** The model produced its best facial diversity and sharpest features between **Epochs 10 and 13**, rather than the final Epoch 20 checkpoint.

---

## 🚀 Step 5: Transition to 128×128 High Resolution

Based on the diagnostic findings and the requirement for photorealistic synthesis, **the project resolution has been permanently upgraded from 64×64 to 128×128**.

### Architecture Scaling (64×64 → 128×128)
Both networks were expanded with an additional convolutional processing stage:
- **Generator ($128 \times 128$):** $z \to 4\times 4\,(512) \to 8\times 8\,(512) \to 16\times 16\,(256) \to 32\times 32\,(128) \to 64\times 64\,(64) \to 128\times 128\times 3$ with Tanh. Total parameters: **8.00M** (with $Z_{\text{DIM}}=128$).
- **Discriminator ($128 \times 128$):** $128\times 128\times 3 \to 64\times 64\,(64) \to 32\times 32\,(128) \to 16\times 16\,(256) \to 8\times 8\,(512) \to 4\times 4\,(512) \to 1\times 1$ scalar logit. Total parameters: **6.96M** (Combined: **14.96M**).

### Preprocessing Adjustments for 128×128
- **Center Crop:** Upgraded from $140 \times 140$ to **$160 \times 160$** to capture fuller facial contours, ear-to-ear proportions, and natural hairline geometry.
- **Resampling:** Target bilinear resizing scaled to **$128 \times 128$**.

---

## 🎛️ Step 6: Hyperparameter Tuning for Run 2 (Profile A — Balanced & Safe)

Following evaluation of the 64×64 baseline and the higher pixel dimensionality of 128×128 ($4\times$ more spatial information), **Profile A (Balanced & Safe)** has been selected and logged as the official configuration for all upcoming training runs.

### Detailed Hyperparameter Tuning Matrix

| Parameter | Baseline (Run 1) | Tuned Value (Profile A) | Engineering Rationale |
|---|---|---|---|
| **Target Resolution** | $64 \times 64$ | **$128 \times 128$** | Standardized production target for facial photorealism. |
| **Latent Space ($Z_{\text{DIM}}$)** | $100$ | **$128$** | **+28% capacity expansion.** An isotropic 128-dim Gaussian provides sufficient geometric degrees of freedom to independently disentangle hair, pose, expression, lighting, and accessories. |
| **Generator LR ($LR_G$)** | $2.0 \times 10^{-4}$ | **$2.0 \times 10^{-4}$** | Optimal Adam learning rate for transposed convolutions. |
| **Discriminator LR ($LR_D$)** | $2.0 \times 10^{-4}$ | **$1.5 \times 10^{-4}$** | **Two Time-Scale Update Rule (TTUR):** At 128×128, the discriminator's classification problem is fundamentally easier than the generator's synthesis problem. Lowering $LR_D$ by 25% slows down discriminator convergence, preventing the late-epoch divergence observed in Run 1. |
| **Batch Size** | $128$ | **$64$** | At 128×128, batch size 64 provides **3,165 updates per epoch** (vs 1,582 previously). More frequent gradient updates smooth the generator's optimization trajectory on complex facial features. |
| **Dominance Threshold** | $0.15$ | **$0.20$** | **Earlier Intervention:** Fires an extra generator step whenever batch $D(\text{fake}) < 0.20$, catching discriminator runaway before confidence saturation sets in. |
| **Cosine LR Floor (`eta_min`)** | $1.0 \times 10^{-5}$ | **$5.0 \times 10^{-5}$** | Prevents the generator's learning rate from falling to near-zero in Epochs 15–20, preserving active gradient updates for fine facial textures. |
| **Gradient Clipping on G** | None | **$1.0$** | Enforces `torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)` under AMP to suppress floating-point instability. |
| **Number of Epochs** | $20$ | **$20$** | At batch size 64, 20 epochs yields **63,300 total generator updates**, providing ample convergence without overfitting. |
| **Checkpoint Strategy** | Last Epoch | **Equilibrium Tracker** | Computes $\lvert D(\text{real}) - 0.5 \rvert + \lvert D(\text{fake}) - 0.5 \rvert$ each epoch and automatically exports `best_generator.pth` at the point of minimal adversarial divergence. |

---

## 📦 Step 7: Summary of Files & Status

1. [`face-genrator.ipynb`](face-genrator.ipynb): All-in-one Google Colab notebook updated with Profile A hyperparameters, 128×128 architecture, and single-click export cell.
2. [`README.md`](README.md): Production project documentation detailing the 128×128 architecture and Profile A settings.
3. [`requirements.md`](requirements.md): Hardware and software specifications calibrated for 128×128 training.
4. [`Training1_Epochs20.md`](Training1_Epochs20.md): Visual and numerical record of the 64×64 baseline run.
5. [`DCGAN_64_Epochs_20/`](DCGAN_64_Epochs_20/): Archived checkpoints and sample grids from Run 1.

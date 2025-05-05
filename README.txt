# AMR: Advanced Modulation Recognition with Diffusion Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://www.pytorchlightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.1-blue.svg)](https://hydra.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="https://via.placeholder.com/800x200/0077B5/FFFFFF?text=Advanced+Modulation+Recognition" alt="AMR Logo">
</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configurations](#-configurations)
- [Visualization Capabilities](#-visualization-capabilities)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Weights & Biases Integration](#-weights--biases-integration)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

AMR (Advanced Modulation Recognition) is a deep learning project that implements latent diffusion models for radio frequency signal processing and modulation recognition. The project combines variational autoencoders (VAE) with diffusion models and classification capabilities to analyze, generate, and classify radio frequency signals.

---

## 📁 Project Structure

```
AMR/
├── configs/               # Configuration files
│   ├── hydra-config.yaml  # Main configuration
│   └── sweeps.yaml        # Hyperparameter tuning configs
├── src/
│   ├── callbacks/         # Visualization callbacks
│   ├── data/              # Data loading utilities
│   ├── models/            # Model definitions
│   │   ├── classifier/    # Classification models
│   │   ├── diffusion/     # Diffusion model components
│   │   └── latent_encoder_models/ # VAE encoder/decoder
│   ├── utils/             # Utility functions
│   └── main.py            # Main entry point
```

---

## ✨ Features

- **Latent Diffusion Models**: Implementation of diffusion-based generative models in latent space
- **Radio Signal Encoding**: VAE encoder/decoder for compressing RF signals to latent space
- **Conditional Generation**: Class-conditional diffusion model for generating signals
- **Visualization Tools**: Rich visualization callbacks for diffusion process and latent space analysis
- **Modulation Recognition**: Classification model using latent representations

---

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AMR.git
cd AMR
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training the VAE Encoder

```bash
python src/main.py train_encoder=True train_diffusion=False train_classifier=False
```

### Training the Diffusion Model

```bash
python src/main.py train_encoder=False train_diffusion=True train_classifier=False
```

### Training the Classifier

```bash
python src/main.py train_encoder=False train_diffusion=False train_classifier=True
```

### Visualizing Diffusion Process

```bash
python src/main.py train_encoder=False train_diffusion=False train_classifier=False vis_diffusion=True
```

---

## ⚙️ Configurations

The project uses Hydra for configuration management. The main configuration file is `configs/hydra-config.yaml`. You can override configurations using command-line arguments:

```bash
python src/main.py hyperparams.learning_rate=0.0005 hyperparams.batch_size=128
```

<details>
<summary><b>Example Configuration</b></summary>

```yaml
project_name: Diffusion_Vis
run_name: Conditioned_Vis
train_encoder: False
train_diffusion: False
conditon: True
vis_diffusion: True
train_classifier: False
encoder_checkpoint_name: vae_phaseepoch=96_val_loss=0.0005596.ckpt
diffusion_checkpoint_name: diffusion_condition_3MParams_epoch=181_val_loss=0.0757.ckpt
mode: online
debug: False
log_model: True
seed: 42

hyperparams:
    epochs: 200
    classifier_epochs: 10
    learning_rate: .001
    batch_size: 256
    dropout: 0.25
    hidden_dim: 32
    feature_dim: 16
    num_classes: 11
```
</details>

---

## 📊 Visualization Capabilities

The project includes advanced visualization tools:

<div align="center">
  <img src="https://via.placeholder.com/800x400/003366/FFFFFF?text=Diffusion+Process+Visualization" alt="Diffusion Process Visualization">
</div>

1. **Diffusion Process Animation**: Visualize how signals are gradually denoised
2. **t-SNE Latent Space Visualization**: See how different signal classes are distributed in latent space
3. **UNet Bottleneck Visualization**: Analyze internal representations of the diffusion model

<details>
<summary><b>Visualization Examples</b></summary>

```python
# Visualization callbacks are automatically triggered during validation
trainer = L.Trainer(
    max_epochs=cfg.hyperparams.epochs,
    logger=wandb_logger,
    callbacks=[
        DiffusionVisualizationCallback(every_n_epochs=1, create_animation=True),
        DiffusionTSNEVisualizationCallback(every_n_epochs=1, create_animation=True)
    ],
)
```
</details>

---

## 📊 Dataset

The project uses the RadioML 2016.10a dataset for training and evaluation. This dataset contains various modulation types across different signal-to-noise ratios (SNRs).

- **11 Modulation Types**: Including AM, FM, PSK, QAM, and others
- **20 SNR Levels**: From -20dB to 18dB in 2dB steps
- **1000 Examples**: Per modulation and SNR combination

<div align="center">
  <img src="https://via.placeholder.com/600x300/4CAF50/FFFFFF?text=RadioML+Dataset+Distribution" alt="Dataset Distribution">
</div>

---

## 🧠 Model Architecture

<div align="center">
  <img src="https://via.placeholder.com/800x400/2196F3/FFFFFF?text=Model+Architecture" alt="Model Architecture">
</div>

### Encoder
- ResNet-based 1D convolutional network for encoding RF signals to latent space
- Feature pyramid with adaptive pooling for dimension reduction
- Multiple residual blocks for improved feature extraction

### Diffusion Model
- UNet-based architecture with self and cross attention mechanisms
- Timestep embedding with Fourier features
- Conditional generation through cross-attention layers

### Classifier
- Convolutional model that operates on denoised latent representations
- Leverages latent features from diffusion model for improved classification

<details>
<summary><b>UNet Architecture Details</b></summary>

```
UNet1DModel(
  (time_proj): Timesteps(...)
  (time_mlp): TimestepEmbedding(...)
  (down_blocks): ModuleList(
    (0): DownResnetBlock1D(...)
    (1): AttnDownBlock1D(...)
    (2): AttnDownBlock1D(...)
  )
  (mid_block): UNetMidBlock1D(...)
  (up_blocks): ModuleList(
    (0): AttnUpBlock1D(...)
    (1): AttnUpBlock1D(...)
    (2): UpResnetBlock1D(...)
  )
)
```
</details>

---

## 📈 Weights & Biases Integration

The project uses Weights & Biases (wandb) for experiment tracking. Training runs, visualizations, and metrics are logged automatically.

<div align="center">
  <img src="https://via.placeholder.com/800x400/FF5722/FFFFFF?text=Weights+%26+Biases+Dashboard" alt="Weights & Biases Dashboard">
</div>

---

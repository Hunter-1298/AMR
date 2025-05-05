# 🌀 AMR: Advanced Modulation Recognition with Diffusion Models

A research framework for applying **latent diffusion models** to **radio signal modulation recognition**, built with PyTorch Lightning, Hydra, and the RadioML 2016.10a dataset.

---

## 📌 Features

- 🎯 **Latent Diffusion Modeling** of RF signals  
- 🔁 **Variational Autoencoder (VAE)** for signal compression  
- 🧠 **Latent Classifier** for modulation recognition  
- 🎨 **Visualizations**: t-SNE plots, diffusion sampling, bottleneck features  
- 🧪 Designed for **research**, extensibility, and experiment tracking  

---

## 📂 Project Structure

AMR/
├── configs/ # Hydra configs
├── src/
│ ├── callbacks/ # Visualization callbacks
│ ├── data/ # Data modules for RadioML
│ ├── models/ # Encoder, diffusion, classifier
│ └── main.py # Hydra entrypoint
├── requirements.txt
└── README.md


---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/AMR.git
cd AMR
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

➤ Train Encoder (VAE)
python src/main.py train_encoder=True

➤ Train Diffusion Model
python src/main.py train_diffusion=True

➤ Train Classifier
python src/main.py train_classifier=True

➤ Visualize Diffusion
python src/main.py vis_diffusion=True

⚙️ Configuration (via Hydra)
Edit configs/hydra-config.yaml or override from CLI:
python src/main.py hyperparams.learning_rate=0.0005 hyperparams.batch_size=128
Example
project_name: AMR
run_name: experiment_v1
hyperparams:
  learning_rate: 0.001
  batch_size: 256
  epochs: 200
  dropout: 0.25
  feature_dim: 16
  num_classes: 11


🧠 Model Overview
Encoder (VAE)
ResNet-style 1D CNN encoder/decoder

Adaptive pooling & residual blocks

Diffusion Model
1D UNet with attention

Timestep embeddings (Fourier)

Conditional/noise-aware generation

Classifier
Latent-space CNN classifier

Uses encoder or sampled latents

📊 Dataset
Uses RadioML 2016.10a

11 modulation types

SNR range: -20 dB to +18 dB

220,000 examples total

Shape: (1024, complex) → real-valued representation

Download via DeepSig Datasets

📈 Visualizations
✅ Diffusion Sampling Steps

✅ t-SNE of Latent Space

✅ UNet Bottleneck Projections

✅ Modulation-wise Latent Clustering

All visuals can be optionally logged to Weights & Biases (W&B).

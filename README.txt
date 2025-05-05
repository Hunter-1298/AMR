Thought for a couple of seconds


# AMR: Advanced Modulation Recognition with Diffusion Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://www.pytorchlightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.1-blue.svg)](https://hydra.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Usage](#usage)
* [Configuration](#configuration)
* [Model Architecture](#model-architecture)
* [Dataset](#dataset)
* [Visualization](#visualization)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Overview

AMR (Advanced Modulation Recognition) is a deep learning project that implements latent diffusion models for radio frequency signal processing and modulation recognition. It combines variational autoencoders (VAE) with diffusion models and classification capabilities to analyze, generate, and classify radio frequency signals.

## Features

* **Latent Diffusion Models**: Generative modeling in latent space
* **Radio Signal Encoding**: VAE-based encoder/decoder for RF signal compression
* **Conditional Generation**: Class-conditional signal generation
* **Visualization Tools**: Visualize diffusion steps, t-SNE projections, and more
* **Modulation Recognition**: Classification from latent representations

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<yourusername>/AMR.git
   cd AMR
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Project Structure

```
AMR/
├── configs/               # Configuration files
│   ├── hydra-config.yaml  # Main configuration
│   └── sweeps.yaml        # Hyperparameter tuning
├── src/
│   ├── callbacks/         # Visualization callbacks
│   ├── data/              # Data loaders and transforms
│   ├── models/            # Model definitions
│   │   ├── classifier/    # Classification models
│   │   ├── diffusion/     # Diffusion model components
│   │   └── latent_encoder_models/  # VAE encoder/decoder
│   ├── utils/             # Helper functions
│   └── main.py            # Main training script
└── README.md
```

## Usage

### Train VAE Encoder

```bash
python src/main.py train_encoder=True train_diffusion=False train_classifier=False
```

### Train Diffusion Model

```bash
python src/main.py train_encoder=False train_diffusion=True train_classifier=False
```

### Train Classifier

```bash
python src/main.py train_encoder=False train_diffusion=False train_classifier=True
```

### Visualize Diffusion

```bash
python src/main.py train_encoder=False train_diffusion=False train_classifier=False vis_diffusion=True
```

## Configuration

Hydra is used for configuration management. Main file: `configs/hydra-config.yaml`.

Override configs via CLI:

```bash
python src/main.py hyperparams.learning_rate=0.0005 hyperparams.batch_size=128
```

<details>
<summary>Default Configuration</summary>

```yaml
project_name: Diffusion_Vis
run_name: Conditioned_Vis
hyperparams:
  epochs: 200
  classifier_epochs: 10
  learning_rate: 0.001
  batch_size: 256
  dropout: 0.25
  hidden_dim: 32
  feature_dim: 16
  num_classes: 11
```

</details>

## Model Architecture

### Encoder

* ResNet-style 1D conv
* Feature pyramids + adaptive pooling
* Residual blocks

### Diffusion

* UNet1D with attention
* Fourier timestep embeddings
* Conditional sampling support

### Classifier

* Latent-space conv model
* Incorporates diffusion embeddings

<details>
<summary>UNet Architecture Details</summary>

```python
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

## Dataset

Uses [RadioML 2016.10a](https://www.deepsig.io/datasets):

* 11 modulation classes
* 20 SNR levels from -20dB to +18dB
* 1000 examples per (mod, SNR) pair

## Visualization

* t-SNE plots of latent space
* Diffusion process frames
* UNet bottleneck inspection

All visualizations are optionally logged to [Weights & Biases](https://wandb.ai/).

## Contributing

1. Fork the repo
2. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make changes and commit:

   ```bash
   git commit -am "Add new feature"
   ```
4. Push to your branch:

   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

* DeepSig for the RadioML dataset
* PyTorch Lightning team
* Hugging Face Diffusers library


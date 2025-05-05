# AMR: Advanced Modulation Recognition with Diffusion Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://www.pytorchlightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.1-blue.svg)](https://hydra.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview

AMR (Advanced Modulation Recognition) is a deep learning project that implements latent diffusion models for radio frequency signal processing and modulation recognition. The project combines variational autoencoders (VAE) with diffusion models and classification capabilities to analyze, generate, and classify radio frequency signals.

## Features

- **Latent Diffusion Models**: Implementation of diffusion-based generative models in latent space
- **Radio Signal Encoding**: VAE encoder/decoder for compressing RF signals to latent space
- **Conditional Generation**: Class-conditional diffusion model for generating signals
- **Visualization Tools**: Rich visualization callbacks for diffusion process and latent space analysis
- **Modulation Recognition**: Classification model using latent representations

## Installation

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

## Project Structure

```
AMR/
├── configs/               # Configuration files
│   ├── hydra-config.yaml # Main configuration
│   └── sweeps.yaml       # Hyperparameter tuning configs
├── src/
│   ├── callbacks/        # Visualization callbacks
│   ├── data/            # Data loading utilities
│   ├── models/          # Model definitions
│   │   ├── classifier/  # Classification models
│   │   ├── diffusion/  # Diffusion model components
│   │   └── latent_encoder_models/ # VAE encoder/decoder
│   ├── utils/          # Utility functions
│   └── main.py         # Main entry point
└── README.md
```

## Usage

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

## Configuration

The project uses Hydra for configuration management. Main configurations can be found in `configs/hydra-config.yaml`.

Example configuration override:
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
    learning_rate: .001
    batch_size: 256
    dropout: 0.25
    hidden_dim: 32
    feature_dim: 16
    num_classes: 11
```
</details>

## Model Architecture

### Encoder
- ResNet-based 1D convolutional network
- Feature pyramid with adaptive pooling
- Multiple residual blocks

### Diffusion Model
- UNet-based architecture with attention
- Timestep embedding with Fourier features
- Conditional generation support

### Classifier
- Convolutional model operating on latent space
- Leverages diffusion features

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

Uses the RadioML 2016.10a dataset:
- 11 modulation types
- 20 SNR levels (-20dB to 18dB)
- 1000 examples per modulation/SNR combination

## Visualization

The project includes several visualization tools:
- Diffusion process animations
- t-SNE latent space visualization
- UNet bottleneck analysis

Visualizations are automatically logged to Weights & Biases during training.

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes and commit (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The RadioML dataset creators
- PyTorch Lightning team
- Hugging Face Diffusers library

---

**Note**: To use this project, you'll need to:
1. Set up a Weights & Biases account for logging
2. Have access to the RadioML dataset
3. Have a CUDA-capable GPU for training

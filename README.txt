# AMR: Advanced Modulation Recognition with Diffusion Models

## Overview

AMR (Advanced Modulation Recognition) is a deep learning project that implements latent diffusion models for radio frequency signal processing and modulation recognition. The project combines variational autoencoders (VAE) with diffusion models and classification capabilities to analyze, generate, and classify radio frequency signals.

## Project Structure

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

## Configurations

The project uses Hydra for configuration management. The main configuration file is `configs/hydra-config.yaml`. You can override configurations using command-line arguments:

```bash
python src/main.py hyperparams.learning_rate=0.0005 hyperparams.batch_size=128
```

## Visualization Capabilities

The project includes advanced visualization tools:

1. **Diffusion Process Animation**: Visualize how signals are gradually denoised
2. **t-SNE Latent Space Visualization**: See how different signal classes are distributed in latent space
3. **UNet Bottleneck Visualization**: Analyze internal representations of the diffusion model

## Dataset

The project uses the RadioML 2016.10a dataset for training and evaluation. This dataset contains various modulation types across different signal-to-noise ratios (SNRs).

## Model Architecture

- **Encoder**: ResNet-based 1D convolutional network for encoding RF signals to latent space
- **Diffusion Model**: UNet-based architecture with self and cross attention mechanisms
- **Classifier**: Convolutional model that operates on denoised latent representations

## Weights & Biases Integration

The project uses Weights & Biases (wandb) for experiment tracking. Training runs, visualizations, and metrics are logged automatically.

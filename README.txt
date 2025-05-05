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
- [Acknowledgements](#acknowledgements)

## Overview

AMR (Advanced Modulation Recognition) is a deep learning project that implements latent diffusion models for radio frequency signal processing and modulation recognition. It combines variational autoencoders (VAE) with diffusion models and classification capabilities to analyze, generate, and classify radio frequency signals.

## Features

- **Latent Diffusion Models**: Generative modeling in latent space  
- **Radio Signal Encoding**: VAE-based encoder/decoder for RF signal compression  
- **Conditional Generation**: Class-conditional signal generation  
- **Visualization Tools**: Visualize diffusion steps, t-SNE projections, and more  
- **Modulation Recognition**: Classification from latent representations  

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/<yourusername>/AMR.git
   cd AMR


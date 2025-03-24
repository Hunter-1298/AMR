# ML Project with W&B and Hydra

This project uses Weights & Biases (wandb) for experiment tracking and Hydra for configuration management.

## Project Structure

```
.
├── configs/          # Hydra configuration files
├── src/              # Source code
│   ├── data/         # Data loading and processing
│   ├── models/       # Model definitions
│   ├── training/     # Training logic
│   └── utils/        # Utility functions
├── tests/            # Test files
└── notebooks/        # Jupyter notebooks
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up wandb:

```bash
wandb login
```

## Running Experiments

To run an experiment:

```bash
python src/main.py
```

## Configuration

The project uses Hydra for configuration management. Main configuration files are in the `configs/` directory:

- `config.yaml`: Main configuration file
- `model/`: Model-specific configurations
- `data/`: Data-specific configurations
- `training/`: Training-specific configurations

Automatic Modulation Recognition (AMR)

Research and experimentation repository for iterative PSK synchronization and latent diffusion-assisted classification of radio signals.
Designed to flexibly explore how synchronization, latent modeling, and classifier design affect modulation recognition performance.

🚀 Highlights

Iterative PSK Synchronization → improves signal alignment and classification accuracy

Hydra Configurations → control datasets, models, encoders, hyperparameters, and options like pretrained encoders

Latent Diffusion Models → 1D diffusion in learned latent space

Weights & Biases Sweeps → for systematic hyperparameter exploration

Modular design for research flexibility

📂 Repository Structure
AMR/ (root)
├── configs/                          # Hydra experiment configs
├── multirun/                         # Sweep output overrides
├── src/
│   ├── callbacks/                   # Visualization & logging callbacks
│   ├── data/                        # DeepSig & TorchSig data loaders
│   ├── models/
│   │   ├── classifiers/             # Classifier model definitions
│   │   ├── diffusion/               # Diffusion model implementation
│   │   └── latent_encoder_models/   # Encoders + PSK models
│   │       └── psk.py               # End-to-end PSK sync + classification
│   ├── architectures/               # Architecture specifications
│   └── main.py                      # Hydra entry-point orchestrator
├── best_checkpoints/                # Saved model checkpoints
├── requirements.txt
├── README.md
└── research_notes.txt

🧠 Core Components
🧩 PSK Sync & Classification

The primary PSK synchronization and classification pipeline is implemented in:

src/models/latent_encoder_models/psk.py


This module provides a self-conditioning diffusion model with classifier that performs:

Iterative PSK phase and timing synchronization

Latent encoding (optional pretrained encoder)

Diffusion refinement in latent space

Classification using the aligned features

Hydra config determines which model class is instantiated.

Example Hydra selection:

Encoder:
  _target_: models.latent_encoder_models.psk.SelfConditioningDiffusionWithClassifier


This makes the PSK model the active encoder/classifier in your training workflow.

🧠 Model Selection via Hydra

All model choices—including whether to use pretrained encoders or diffusion settings—are controlled via your Hydra config files.

Typical config keys include:

model:
  Encoder:
    _target_: models.latent_encoder_models.psk.SelfConditioningDiffusionWithClassifier
  diffusion:
    _target_: models.diffusion.YourDiffusionClass
  classifier:
    ...


Hydra allows overrides so you can swap models and architectures without changing code.

🧪 Configs & Experiment Management
🎛 Hydra Configs

Configs are stored in:

configs/


They define:

Dataset parameters

Model architectures

Optimization hyperparameters

Logging & checkpoint behavior

Pretrained encoder options

You can override any value at run time:

python src/main.py \
    model.Encoder._target_=models.latent_encoder_models.psk.SelfConditioningDiffusionWithClassifier \
    training.batch_size=128

📊 Sweeps (Weights & Biases)

Sweep configurations reside in:

configs/sweeps/


Launch a sweep:

wandb sweep configs/sweeps/sweep.yaml
wandb agent <SWEEP_ID>

▶️ Typical Workflows
🔧 Training
python src/main.py training=true

📈 Training with Overridden Hyperparameters
python src/main.py \
    training.lr=5e-4 \
    model.diffusion.num_steps=1000 \
    model.use_pretrained_encoder=true

📊 Data Support

We support multiple radio ML datasets:

src/data/


Current loaders are implemented for:

DeepSig

TorchSig

Data paths and preprocessing settings are entirely Hydra-configurable.

🧠 Diffusion and Encoder Options

Your run configuration determines:

Whether you include a pretrained encoder before diffusion

What diffusion architecture to use

Whether to fine-tune or freeze encoder weights

Example snippet:

model:
  use_pretrained_encoder: true
  Encoder:
    _target_: models.latent_encoder_models.psk.SelfConditioningDiffusionWithClassifier

🧩 Visualization & Callbacks

Visualizations like reconstructions, latent space inspection, and metric logging are handled by callbacks in:

src/callbacks/


Use Hydra or override flags to turn visualizations on/off.

🛠 Installation

Install dependencies:

git clone https://github.com/Hunter-1298/AMR.git --branch bit_error_rate
cd AMR
pip install -r requirements.txt

🧪 Experiment Philosophy

Designed for:

Config-driven experimentation

Reproducible results via Hydra

Sweep-level exploration via W&B

Iterative algorithm research

🎯 Goals

Improve PSK classification accuracy by aligning signal features

Modular research codebase for diffusion + classification

Hydra-centric experimentation for flexible model combos

Sweep support for large-scale hyperparameter optimization

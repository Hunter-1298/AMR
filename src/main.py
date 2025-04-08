import hydra
from hydra.utils import get_original_cwd
import wandb
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from data.rfml_dataset_2016 import get_dataloaders, get_tokenized_dataloaders
import os
import torch


@hydra.main(config_path="../configs", config_name="hydra-config", version_base="1.1")
def main(cfg: DictConfig):
    # Initialize wandb first
    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
        mode="offline" if cfg.debug else "online",
    )

    # Initialize WandbLogger after config is updated
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.run_name,
        offline=cfg.debug,
        log_model=cfg.log_model,
    )

    # Convert entire config to dict and log it
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger.experiment.config.update(config_dict)
    wandb_logger.log_hyperparams(dict(cfg.hyperparams))

    # Get original dataloaders
    train_loader, val_loader, label_names = get_dataloaders(cfg.dataset)

    # Create and train MoE model
    model = hydra.utils.instantiate(cfg.Model, label_names=label_names)

    # Set up trainer for MoE
    trainer = L.Trainer(
        max_epochs=cfg.hyperparams.epochs,
        logger=wandb_logger,
        default_root_dir=".",
        log_every_n_steps=10,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    # Train MoE
    trainer.fit(model, train_loader, val_loader)
    print("Model Done Training")


if __name__ == "__main__":
    main()

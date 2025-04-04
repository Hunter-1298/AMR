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

    if cfg.train_tokenizer:
        print("Training VQVAE encoder...")

        # Create VQVAE model
        vqvae = hydra.utils.instantiate(cfg.Tokenizer)

        # Make sure checkpoint directory exists
        checkpoint_dir = os.path.join(get_original_cwd(), "checkpoints", "vqvae")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up trainer for VQVAE
        vqvae_trainer = L.Trainer(
            max_epochs=cfg.hyperparams.epochs,
            logger=wandb_logger,
            default_root_dir=".",
            log_every_n_steps=50,
            callbacks=[
                ModelCheckpoint(
                    monitor="recon_loss",
                    filename="vqvae_{epoch:02d}_{recon_loss:.4f}",
                    dirpath=checkpoint_dir,
                    save_top_k=3,
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="step"),
            ],
        )

        # Train VQVAE
        vqvae_trainer.fit(vqvae, train_loader, val_loader)
        # Save final model
        print("VQVAE training complete! Model saved to checkpoints/vqvae/")
    else:
        # Load and freeze VQVAE
        vqvae = hydra.utils.instantiate(cfg.Tokenizer)
        checkpoint_dir = os.path.join(get_original_cwd(), "checkpoints", "vqvae")
        checkpoint = torch.load(
            checkpoint_dir + "/vqvae_epoch=98_recon_loss=2.4166.ckpt"
        )
        vqvae.load_state_dict(checkpoint["state_dict"])
        vqvae.eval()
        for param in vqvae.parameters():
            param.requires_grad = False

    #######################################################################################
    # VQ-VAE Training Complete
    # Now testing downstream architecture using the trained VQ-VAE as a feature extractor
    #######################################################################################

    # Get tokenized dataloaders
    train_loader, val_loader = get_tokenized_dataloaders(
        cfg.dataset, vqvae, train_loader, val_loader
    )

    # Create and train MoE model
    model = hydra.utils.instantiate(cfg.Transformer, label_names=label_names)
    # model = torch.compile(hydra.utils.instantiate(cfg.Transformer, label_names=label_names))

    # Make sure checkpoint directory exists
    checkpoint_dir = os.path.join(get_original_cwd(), "checkpoints", "Transformer")
    os.makedirs("checkpoints/Transformer", exist_ok=True)

    # Set up trainer for MoE
    trainer = L.Trainer(
        max_epochs=cfg.hyperparams.epochs,
        logger=wandb_logger,
        default_root_dir=".",
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                filename="moe_{epoch:02d}_{val_loss:.4f}",
                dirpath=checkpoint_dir,
                save_top_k=3,
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    # Train MoE
    trainer.fit(model, train_loader, val_loader)

    print("Transformer training complete!")


if __name__ == "__main__":
    main()

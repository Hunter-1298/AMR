import hydra
import torch
from hydra.utils import get_original_cwd
import wandb
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from data.rfml_dataset_2016 import get_dataloaders
from callbacks import DiffusionVisualizationCallback
from utils.latent_scaling import calculate_latent_scaling_factor
import os


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

    # If we need to train the encoder
    if cfg.train_encoder:
        print("Training VAE Encoder...")

        # Create and train VAE model
        encoder = hydra.utils.instantiate(cfg.Encoder, label_names=label_names)

        # Create checkpoint dir
        checkpoint_dir = os.path.join(get_original_cwd(), "checkpoints", "encoder")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up trainer for MoE
        encoder_trainer = L.Trainer(
            max_epochs=cfg.hyperparams.epochs,
            logger=wandb_logger,
            default_root_dir=".",
            log_every_n_steps=10,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    filename="vae_phase{epoch:02d}_{val_loss:.7f}",
                    dirpath=checkpoint_dir,
                    save_top_k=3,
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="step"),
            ],
        )

        # Train Encoder
        encoder_trainer.fit(encoder, train_loader, val_loader)
        print("Encoder finished training")

    else:
        # Load and freeze the encoder
        encoder = hydra.utils.instantiate(cfg.Encoder, label_names=label_names)
        checkpoint_dir = "/home/hshayde/Projects/MIT/AMR/checkpoints/encoder/"
        checkpoint_name = cfg.encoder_checkpoint_name
        checkpoint = torch.load(checkpoint_dir + checkpoint_name)
        encoder.load_state_dict(checkpoint["state_dict"])
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    # Calculate scaling factor
    if not cfg.Diffusion.latent_scaling:
        cfg.Diffusion.latent_scaling = calculate_latent_scaling_factor(encoder, val_loader)
        print(f"Calculated latent scaling factor: {cfg.Diffusion.latent_scaling:.5f}")

    if cfg.train_diffusion:
        # Train Diffusion Model
        model = hydra.utils.instantiate(cfg.Diffusion, encoder=encoder)
        model = torch.compile(model)

        # Create checkpoint dir
        checkpoint_dir = os.path.join(get_original_cwd(), "checkpoints", "diffusion")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up trainer for MoE
        trainer = L.Trainer(
            max_epochs=cfg.hyperparams.epochs,
            logger=wandb_logger,
            default_root_dir=".",
            log_every_n_steps=10,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    filename="diffusion_{epoch:02d}_{val_loss:.4f}",
                    dirpath=checkpoint_dir,
                    save_top_k=3,
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="step"),
                DiffusionVisualizationCallback(every_n_epochs=5)
            ],
        )

        trainer.fit(model, train_loader, val_loader) #pyright: ignore
        print("Diffusion Model Finished Training")

    else:
        # Load and freeze the diffusion model
        diffusion = hydra.utils.instantiate(cfg.Diffusion, encoder=encoder)
        checkpoint_dir = "/home/hshayde/Projects/MIT/AMR/checkpoints/diffusion/"
        checkpoint_name = cfg.diffusion_checkpoint_name
        checkpoint = torch.load(checkpoint_dir + checkpoint_name, weights_only=False)
        diffusion.load_state_dict(checkpoint["state_dict"])
        diffusion.eval()
        for param in diffusion.parameters():
            param.requires_grad = False


        # Set up trainer for MoE
        trainer = L.Trainer(
            max_epochs=cfg.hyperparams.epochs,
            logger=wandb_logger,
            callbacks=[
                DiffusionVisualizationCallback(every_n_epochs=1, create_animation=True)
            ],
        )

        trainer.validate(model=diffusion, dataloaders=val_loader) #pyright: ignore
        print("Diffusion Model Finished Training")


if __name__ == "__main__":
    main()

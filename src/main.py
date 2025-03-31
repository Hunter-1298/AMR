import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as L  # type: ignore
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from data.rfml_dataset_2016 import RFMLDataset, get_dataloaders


@hydra.main(config_path="../configs", config_name="hydra-config", version_base="1.1")
def main(cfg: DictConfig):
    # Initialize wandb first
    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="offline" if cfg.debug else "online"
    )

    if cfg.sweeps:
        # Dynamically update any hyperparameters that exist in wandb.config
        for param_name, param_value in wandb.config.hyperparams.items():
            if hasattr(cfg.hyperparams, param_name):
                cfg.hyperparams[param_name] = param_value
    
    # Initialize WandbLogger after config is updated
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.run_name,
        offline=cfg.debug,
        log_model=True,
    )

    # Convert entire config to dict and log it
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger.experiment.config.update(config_dict)
    # Log hyperparameters from config
    wandb_logger.log_hyperparams(dict(cfg.hyperparams))

    # get our data loaders (batch_size, 2, 128)
    train_loader, val_loader, label_names = get_dataloaders(cfg.dataset)

    # Instantiate objects directly from config
    model = hydra.utils.instantiate(cfg.Model, label_names)

    # We save a checkpoint every time we train, we can specify this in the config if we want to load one
    ckpt_path = None
    if hasattr(cfg, "checkpoint") and cfg.checkpoint_path:
        ckpt_path = cfg.checkpoint_path

    trainer = L.Trainer(
        max_epochs=cfg.hyperparams.epochs,
        logger=wandb_logger,
        default_root_dir=".",
        log_every_n_steps=10,
        # save best three state dicts based off of val loss
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",  # or whatever metric you want to track
                filename="model:{cfg.model}--{val_loss:.2f}",
                dirpath="checkpoints",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()

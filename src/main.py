import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from data.rfml_dataset_2016 import get_dataloaders

@hydra.main(config_path="../configs", config_name="hydra-config")
def main(cfg: DictConfig):
    # Hydra automatically creates outputs directory and changes working directory
    
    # Initialize wandb logger with additional best practices
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.run_name,
        offline=cfg.debug,
        log_model=True  # Automatically upload model checkpoints
    )
    
    # Log hyperparameters from config
    wandb_logger.log_hyperparams(dict(cfg.hyperparams))

    # get our data loaders (batch_size, 2, 128)
    train_loader, val_loader = get_dataloaders(cfg.dataset)
    
    # Instantiate objects directly from config
    model = hydra.utils.instantiate(cfg.model, optimizer=cfg.optimizer)

    # We save a checkpoint every time we train, we can specify this in the config if we want to load one 
    ckpt_path = None
    if hasattr(cfg, 'checkpoint') and cfg.checkpoint_path:
        ckpt_path = cfg.checkpoint_path
    

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        logger=wandb_logger,
        default_root_dir=".",
        # save best three state dicts based off of val loss
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor='val_loss',  # or whatever metric you want to track
                mode='min',
                save_top_k=1,
                filename='model:{cfg.model}--{val_loss:.2f}',
                dirpath='checkpoints'
            ),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        max_epochs=cfg.hyperparams.epochs,
        ckpt_path=ckpt_path
    )


if __name__ == "__main__":
    main()
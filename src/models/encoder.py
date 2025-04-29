import torch
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
import wandb


class LatentEncoder(L.LightningModule):
    def __init__(self, label_names, encoder, decoder, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.label_names = label_names

        # Initialize encoder and decoder with proper parameters
        self.encoder = encoder

        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        # x is shape [64,2,128]
        # Define the forward pass
        encoded = self.encoder(x)
        # encoded(x) is shape [64,32,64]
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, labels, snr = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # x is originally [batch_size, 2, 128], labels = [batch_size,1]
        x, labels, snr = batch
        x_hat = self(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        # Store indices for epoch-end histogram - create new list at start of epoch
        if batch_idx == 0:
            self.original_signal_real = x[0][0].detach().cpu()
            self.original_signal_imag = x[0][1].detach().cpu()
            self.reconstructed_signal_real = x_hat[0][0].detach().cpu()
            self.reconstructed_signal_imag = x_hat[0][1].detach().cpu()

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):  # pyright: ignore
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-2
        )
        # OneCycleLR automatically determines warm-up and decay schedules
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,  # Peak LR at warmup end
            total_steps=self.trainer.estimated_stepping_batches,  # pyright: ignore
            pct_start=0.05,  # 10% of training is used for warm-up
            anneal_strategy="cos",  # Cosine decay after warmup
            final_div_factor=100,  # Reduce LR by 10x at the end
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self):
        # Create signal plots
        amp_fig = self._create_comparison_plot()

        # Log everything to wandb - remove np_histogram parameter
        self.logger.experiment.log(  # pyright: ignore
            {"train/signla_reconsruction": amp_fig, "epoch": self.current_epoch}
        )

    def _create_comparison_plot(self):
        """Helper function to create comparison plots"""
        fig, ax = plt.subplots(figsize=(10, 4))

        # Access class variables
        original_real = self.original_signal_real.flatten().numpy()
        original_imag = self.original_signal_imag.flatten().numpy()
        reconstructed_real = self.reconstructed_signal_real.flatten().numpy()
        reconstructed_imag = self.reconstructed_signal_imag.flatten().numpy()

        x = range(len(original_real))  # Assuming all signals have the same length

        # Plotting
        ax.plot(x, original_real, label="Original Real", alpha=0.7)
        ax.plot(x, original_imag, label="Original Imaginary", alpha=0.7)
        ax.plot(x, reconstructed_real, label="Reconstructed Real", alpha=0.7)
        ax.plot(x, reconstructed_imag, label="Reconstructed Imaginary", alpha=0.7)

        ax.set_xlabel("Index")
        ax.set_ylabel("Amplitude")
        ax.set_title("Comparison of Original and Reconstructed Signals")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        return wandb.Image(fig)

import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Dict, Any, Tuple
from .unet_1d import UNet1DModel
from ..latent_encoder_models import ResNet1D, Decoder1D


class LatentDiffusion(L.LightningModule):
    def __init__(
        self,
        unet,
        encoder,
        n_steps: int = 1000,
        linear_start: float = 0.0001,  # starting noise value for beta schedule
        linear_end: float = 0.02,  # ending value for the beta schedule
        latent_scaling_factor: float = 0.18215,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize UNet for latent space diffusion and the encoder for sampling
        self.unet = unet
        self.encoder = encoder

        # Diffusion parameters
        self.n_steps = n_steps  # total noising steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Setup noise schedule
        # Create linearly spaced array - squared to get the beta values
        # beta - variance of noise at each step
        # alpha = 1-beta: proportion of original signal retained at each step
        # alpha_bar = cumulative products of alpha which is the total retention up to step t
        # sqrt_alpha_bar = precomputed square root of alpha for efficiency
        # sqrt_one_minus_alpha_bar = precomputed square root for efficiency
        beta = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_steps, dtype=torch.float64
            )
            ** 2
        )
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha = nn.Parameter(alpha.to(torch.float32), requires_grad=False)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

        # Precompute values for sampling
        self.sqrt_alpha_bar = nn.Parameter(
            torch.sqrt(alpha_bar).to(torch.float32), requires_grad=False
        )
        self.sqrt_one_minus_alpha_bar = nn.Parameter(
            torch.sqrt(1.0 - alpha_bar).to(torch.float32), requires_grad=False
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space and scale"""
        z = self.encoder.encode(x)
        return z

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to x_start according to noise schedule at timestep t

        Args:
            x_start: Starting clean data [B, C, T]
            t: Timesteps [B]
            noise: Optional predetermined noise [B, C, T]

        Returns:
            x_t: Noised data
            noise: The noise added
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Extract appropriate alphas for this timestep
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        # Generate x_t from x_0 and noise
        x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise using the UNet model

        Args:
            x: Noisy data at timestep t [B, C, T]
            t: Timesteps [B]
            context: Optional conditioning [B, context_dim]

        Returns:
            Predicted noise
        """
        return self.unet(x, t, context)

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate diffusion loss for denoising score matching

        Args:
            x_start: Starting clean data [B, C, T]
            t: Timesteps [B]
            context: Optional conditioning [B, context_dim]

        Returns:
            MSE loss between predicted and actual noise
        """
        # Add noise to input
        x_noisy, noise = self.q_sample(x_start, t)

        # Predict noise
        predicted_noise = self.forward(x_noisy, t, context)

        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise, predicted_noise)

        return loss, predicted_noise

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lightning training step

        Args:
            batch: Batch of data containing (x, context, _)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Unpack batch
        x, context, snr = batch

        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (x.shape[0],), device=self.device).long()

        # Encode input to latent space
        z = self.encode(x)

        # Calculate diffusion loss
        noise_loss, predicted_noise = self.p_losses(z, t, context)

        # Log loss
        self.log("train_loss", noise_loss, prog_bar=True)

        return noise_loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lightning validation step

        Args:
            batch: Batch of data containing (x, context, _)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Unpack batch
        x, context, snr = batch

        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (x.shape[0],), device=self.device).long()

        # Encode input to latent space
        z = self.encode(x)

        # Calculate diffusion loss
        noise_loss, predicted_noise = self.p_losses(z, t, context)

        # Log loss
        self.log("val_loss", noise_loss, prog_bar=True)


        return noise_loss

    def configure_optimizers(self): #pyright: ignore
        """Setup optimizer and learning rate scheduler"""
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.trainer.estimated_stepping_batches),
            eta_min=self.learning_rate / 10,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

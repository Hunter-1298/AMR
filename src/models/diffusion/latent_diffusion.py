import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Dict, Any, Tuple
from .unet_1d import UNet1DModel
from ..latent_encoder_models import ResNet1D, Decoder1D
from matplotlib import pyplot as plt


class LatentDiffusion(L.LightningModule):
    def __init__(
        self,
        unet,
        encoder,
        n_steps: int = 1000,
        linear_start: float = 0.0001,  # starting noise value for beta schedule
        linear_end: float = 0.02,  # ending value for the beta schedule
        latent_scaling: float = 0.18215,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        # save hyperparms to lightning and wandb
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.latent_scaling = latent_scaling

        # Initialize UNet for latent space diffusion and the encoder for sampling
        self.unet = unet
        self.encoder = encoder

        # Create noise schedule
        beta = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_steps, dtype=torch.float64
            )
            ** 2
        )

        # Convert to float32 and create self attributes first
        # 1. Beta (β) - Amount of noise added at each step
        beta = beta.to(torch.float32)  # [0.0001 -> 0.02]
        # - Controls how much noise is added at each timestep
        # - Starts small (0.0001) and increases to final value (0.02)
        # - Like a schedule of how much noise to add at each step

        # 2. Alpha (α) - Amount of original signal preserved
        alpha = 1 - beta  # [0.9999 -> 0.98]
        # - Opposite of beta
        # - How much of original signal remains at each step
        # - Starts high (0.9999) and decreases

        # 3. Alpha_bar (ᾱ) - Cumulative signal preservation
        alpha_bar = torch.cumprod(alpha, dim=0)  # [0.9999 -> ~0.02]
        # - Cumulative product of alphas
        # - Total amount of original signal remaining after t steps
        # - Decreases more rapidly than alpha

        # 4. sqrt_alpha_bar (√ᾱ) - For scaling original signal
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        # - Used to scale the original signal in noise addition
        # - Square root for numerical stability

        # 5. sqrt_one_minus_alpha_bar (√(1-ᾱ)) - For scaling noise
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        # - Used to scale the noise in noise addition
        # - Complement to sqrt_alpha_bar

        # Now register them as buffers
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space and scale"""
        z = self.encoder.encode(x) * self.latent_scaling
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space after removing scaling"""
        z_unscaled = z / self.latent_scaling  # First remove scaling
        x_recon = self.encoder.decoder(z_unscaled)  # Then decode
        return x_recon

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

        # Get appropriate alphas for timestep t
        # Forward diffusion formula:
        # x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)  # pyright: ignore
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)  # pyright: ignore

        # Add noise according to schedule
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

        # Predict noise - remove context
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

        # we have to assign context in our modle to a value of 0-19 for snr ranges
        snr_context = {x:idx for idx,x in enumerate(range(-20,20,2))}
        context_cpu = snr.cpu().tolist()
        context = torch.tensor([snr_context[x] for x in context_cpu], device=self.device)

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

        snr_context = {x:idx for idx,x in enumerate(range(-20,20,2))}
        context_cpu = snr.cpu().tolist()
        context = torch.tensor([snr_context[x] for x in context_cpu], device=self.device)

        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (x.shape[0],), device=self.device).long()

        # Encode input to latent space
        z = self.encode(x)

        # Calculate diffusion loss
        noise_loss, predicted_noise = self.p_losses(z, t, context)

        # Log loss
        self.log("val_loss", noise_loss, prog_bar=True)

        # save first batch of visualizations
        if batch_idx == 0:
            self.example_batch = batch

        return noise_loss

    def configure_optimizers(self):  # pyright: ignore
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        # OneCycleLR automatically determines warm-up and decay schedules
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,  # Peak LR at warmup end
            total_steps=int(
                self.trainer.estimated_stepping_batches
            ),  # Total training steps
            pct_start=0.05,  # 5% of training is used for warm-up
            anneal_strategy="cos",  # Cosine decay after warmup
            final_div_factor=100,  # Reduce LR by 10x at the end
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

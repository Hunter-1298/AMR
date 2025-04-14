import torch
import torch.nn as nn
import lightning as L
from .unet_1d import UNet1DModel
from ..latent_encoder_models import ResNet1D, Decoder1D


class LatentDiffusion(L.LightningModule):
    def __init__(
        self,
        unet_config: dict,
        vae_config: dict,
        n_steps: int = 1000,
        linear_start: float = 0.0001,
        linear_end: float = 0.02,
        latent_scaling_factor: float = 0.18215,
    ):
        super().__init__()
        
        # Initialize VAE
        self.encoder = ResNet1D(**vae_config['encoder'])
        self.decoder = Decoder1D(**vae_config['decoder'])
        
        # Initialize UNet for latent space
        self.unet = UNet1DModel(**unet_config)
        
        # Diffusion parameters
        self.latent_scaling_factor = latent_scaling_factor
        self.n_steps = n_steps
        
        # Noise schedule
        beta = torch.linspace(linear_start**0.5, linear_end**0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x) * self.latent_scaling_factor
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to data space"""
        return self.decoder(z / self.latent_scaling_factor)
        
    def get_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Get noise prediction from UNet"""
        return self.unet(x, t)
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Add noise to input at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise
        
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Calculate loss for training"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.get_noise(x_noisy, t)
        
        return torch.nn.functional.mse_loss(noise, predicted_noise)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        # Encode to latent space
        z = self.encode(x)
        
        # Get noise prediction
        noise_pred = self.get_noise(z, t)
        
        return noise_pred
        
    def training_step(self, batch, batch_idx):
        x, labels, snr = batch
        
        # Sample timesteps
        t = torch.randint(0, self.n_steps, (x.shape[0],), device=x.device).long()
        
        # Encode to latent space
        z = self.encode(x)
        
        # Calculate loss
        loss = self.p_losses(z, t)
        
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, labels, snr = batch
        
        # Sample timesteps
        t = torch.randint(0, self.n_steps, (x.shape[0],), device=x.device).long()
        
        # Encode to latent space
        z = self.encode(x)
        
        # Calculate loss
        loss = self.p_losses(z, t)
        
        self.log("val_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-2
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-4,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=10
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
        
    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from the model"""
        # Start with random noise
        z = torch.randn(batch_size, self.unet.in_channels, self.unet.sample_size, device=device)
        
        # Denoise step by step
        for t in range(self.n_steps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = self.get_noise(z, t_batch)
            
            alpha_t = 1 - self.beta[t]
            alpha_bar_t = self.alpha_bar[t]
            
            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = 0
                
            z = 1 / torch.sqrt(alpha_t) * (
                z - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred
            ) + torch.sqrt(self.beta[t]) * noise
            
        # Decode to data space
        return self.decode(z)

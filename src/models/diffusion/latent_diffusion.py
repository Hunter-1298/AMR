import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Optional, Dict, Any, Tuple
from .unet_1d import UNet1DModel
from ..latent_encoder_models import ResNet1D, Decoder1D
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
from PIL import Image
import wandb

class EmbeddingConditioner(nn.Module):
    def __init__(self, num_classes, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.num_classes = num_classes

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # MLP classifier with residual connection
        self.proj1 = nn.Linear(latent_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, latent):
        # Global pooling
        x = self.pool(latent).squeeze(-1)  # [B, latent_dim]

        # First projection
        h = F.relu(self.norm1(self.proj1(x)))

        # Second projection with residual
        h = h + F.relu(self.norm2(self.proj2(h)))

        # Classification
        class_logits = self.classifier(h)
        pred_class = torch.argmax(class_logits, dim=1)

        return pred_class, class_logits

class LatentDiffusion(L.LightningModule):
    def __init__(
        self,
        unet,
        encoder,
        label_names,
        n_steps: int = 1000,
        linear_start: float = 0.0001,  # starting noise value for beta schedule
        linear_end: float = 0.02,  # ending value for the beta schedule
        latent_scaling: float = 0.18215,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        num_classes: int = 11,
        diffusion_contrastive: str = "bottleneck",
    ):
        super().__init__()
        # save hyperparms to lightning and wandb
        self.save_hyperparameters()
        self.automatic_optimization = False  # Important: disable automatic optimization

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.latent_scaling = latent_scaling
        self.diffusion_contrastive = diffusion_contrastive
        self.num_classes = num_classes
        self.label_names = label_names

        # Initialize UNet for latent space diffusion and the encoder for sampling
        self.unet = unet
        self.encoder = encoder
        self.contrastive_projector = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 128)
        )

        # Add embedding conditioner
        self.embedding_conditioner = EmbeddingConditioner(
            num_classes=num_classes,
            latent_dim=32,  # Adjust based on your encoder's output channels
            hidden_dim=64
        )

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

    def info_nce_loss(self, z1, z2, temperature=0.07):
        """
        Computes InfoNCE loss between two batches of embeddings.
        z1: (N, D) or (N, T, D)
        z2: (N, D) or (N, T, D)
        """
        # If input is 3D, apply global average pooling over time dimension
        if z1.dim() == 3:
            z1 = F.adaptive_avg_pool1d(z1, 1).squeeze(-1)
        if z2.dim() == 3:
            z2 = F.adaptive_avg_pool1d(z2, 1).squeeze(-1)

        # Normalize to unit vectors
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        N = z1.shape[0]

        # Compute cosine similarity matrix: (N, N)
        logits = torch.matmul(z1, z2.T) / temperature  # shape: [N, N]

        # Labels: diagonal entries are positives
        labels = torch.arange(N, device=z1.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        # Get optimizers and schedulers
        diff_opt, cls_opt = self.optimizers()
        diff_sch, cls_sch = self.lr_schedulers()

        # Unpack batch
        x, context, snr = batch

        # Encode input to latent space
        z = self.encode(x)

        # STEP 1: Update embedding conditioner
        # ===================================
        # Forward pass through embedding conditioner
        pred_class, class_logits = self.embedding_conditioner(z)
        cls_loss = F.cross_entropy(class_logits, context)

        # Backward and update embedding conditioner
        cls_opt.zero_grad()
        self.manual_backward(cls_loss)
        cls_opt.step()
        cls_sch.step()

        self.log("train/class_embed_loss", cls_loss, prog_bar=True)
        with torch.no_grad():
            acc = (torch.argmax(class_logits, dim=1) == context).float().mean()
            self.log("train/cls_acc", acc, prog_bar=True)

        # STEP 2: Update diffusion and contrastive components
        # ==================================================
        # Let the class predictor learn for the first 10 epochs by itself first
        if self.current_epoch > 10:
            # Sample random timesteps
            t1 = torch.randint(0, self.n_steps, (x.shape[0],), device=self.device).long()
            t2 = torch.randint(0, self.n_steps, (x.shape[0],), device=self.device).long()

            # Add noise to input at different timesteps
            z_noisy1, noise1 = self.q_sample(z, t1)
            z_noisy2, noise2 = self.q_sample(z, t2)

            # Predict noise
            predicted_noise1 = self.forward(z_noisy1, t1, pred_class)
            predicted_noise2 = self.forward(z_noisy2, t2, pred_class)

            # Calculate noise prediction loss
            noise_loss1 = F.mse_loss(noise1, predicted_noise1)
            noise_loss2 = F.mse_loss(noise2, predicted_noise2)
            noise_loss = (noise_loss1 + noise_loss2) / 2

            # Get denoised representations
            sqrt_alpha_bar_t1 = self.sqrt_alpha_bar[t1].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t1 = self.sqrt_one_minus_alpha_bar[t1].view(-1, 1, 1)
            z_denoised1 = (z_noisy1 - sqrt_one_minus_alpha_bar_t1 * predicted_noise1) / sqrt_alpha_bar_t1

            sqrt_alpha_bar_t2 = self.sqrt_alpha_bar[t2].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t2 = self.sqrt_one_minus_alpha_bar[t2].view(-1, 1, 1)
            z_denoised2 = (z_noisy2 - sqrt_one_minus_alpha_bar_t2 * predicted_noise2) / sqrt_alpha_bar_t2

            # Apply global pooling to get vector representations
            z1_pooled = F.adaptive_avg_pool1d(z_denoised1, 1).squeeze(-1)
            z2_pooled = F.adaptive_avg_pool1d(z_denoised2, 1).squeeze(-1)

            # Project to contrastive space
            z1_proj = self.contrastive_projector(z1_pooled)
            z2_proj = self.contrastive_projector(z2_pooled)

            # Compute InfoNCE loss
            contrastive_loss = self.info_nce_loss(z1_proj, z2_proj)

            # Combined diffusion loss
            diffusion_loss = noise_loss + 0.05 * contrastive_loss

            # Update diffusion model
            diff_opt.zero_grad()
            self.manual_backward(diffusion_loss)
            diff_opt.step()

            # Update learning rate schedulers
            diff_sch.step()

        # Log metrics
            self.log("train/noise_loss", noise_loss, prog_bar=True)
            self.log("train/contrastive_loss", contrastive_loss, prog_bar=True)
            self.log("train/total_loss", diffusion_loss + cls_loss, prog_bar=True)


        return cls_loss  # Return value not used with manual optimization

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lightning validation step using denoised representations for contrastive learning

        Args:
            batch: Batch of data containing (x, context, snr)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Unpack batch
        x, context, snr = batch

        # Encode input to latent space
        z = self.encode(x)

        # Get predicted class and logits from embedding conditioner
        pred_class, class_logits = self.embedding_conditioner(z)

        # Store predictions for confusion matrix
        if not hasattr(self, 'val_preds'):
            self.val_preds = []
            self.val_labels = []

        self.val_preds.append(pred_class.detach().cpu())
        self.val_labels.append(context.detach().cpu())

        # Classification loss
        cls_loss = F.cross_entropy(class_logits, context)

        # Log classification accuracy
        with torch.no_grad():
            acc = (torch.argmax(class_logits, dim=1) == context).float().mean()
            self.log("val/class_embed_acc", acc, prog_bar=True)
            self.log("val/class_embed_loss", cls_loss, prog_bar=True)

        # Only compute diffusion and contrastive losses after the first 10 epochs
        if self.current_epoch > 10:
            # Sample random timesteps
            t1 = torch.randint(0, self.n_steps, (x.shape[0],), device=self.device).long()
            t2 = torch.randint(0, self.n_steps, (x.shape[0],), device=self.device).long()

            # Add noise to input at different timesteps
            z_noisy1, noise1 = self.q_sample(z, t1)
            z_noisy2, noise2 = self.q_sample(z, t2)

            # Predict noise
            predicted_noise1 = self.forward(z_noisy1, t1, pred_class)
            predicted_noise2 = self.forward(z_noisy2, t2, pred_class)

            # Calculate noise prediction loss
            noise_loss1 = F.mse_loss(noise1, predicted_noise1)
            noise_loss2 = F.mse_loss(noise2, predicted_noise2)
            noise_loss = (noise_loss1 + noise_loss2) / 2

            # Get denoised representations from each prediction
            # Using the predicted noise to reconstruct the clean signal
            sqrt_alpha_bar_t1 = self.sqrt_alpha_bar[t1].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t1 = self.sqrt_one_minus_alpha_bar[t1].view(-1, 1, 1)
            z_denoised1 = (z_noisy1 - sqrt_one_minus_alpha_bar_t1 * predicted_noise1) / sqrt_alpha_bar_t1

            sqrt_alpha_bar_t2 = self.sqrt_alpha_bar[t2].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t2 = self.sqrt_one_minus_alpha_bar[t2].view(-1, 1, 1)
            z_denoised2 = (z_noisy2 - sqrt_one_minus_alpha_bar_t2 * predicted_noise2) / sqrt_alpha_bar_t2

            # Apply global pooling to get vector representations
            z1_pooled = F.adaptive_avg_pool1d(z_denoised1, 1).squeeze(-1)
            z2_pooled = F.adaptive_avg_pool1d(z_denoised2, 1).squeeze(-1)

            # Project to contrastive space
            z1_proj = self.contrastive_projector(z1_pooled)
            z2_proj = self.contrastive_projector(z2_pooled)

            # Compute InfoNCE loss
            contrastive_loss = self.info_nce_loss(z1_proj, z2_proj)

            # Total loss
            total_loss = noise_loss + 0.05 * contrastive_loss + cls_loss

            # Log losses
            self.log("val/noise_loss", noise_loss, prog_bar=True)
            self.log("val/contrastive_loss", contrastive_loss, prog_bar=True)
            self.log("val_loss", total_loss, prog_bar=True)
        else:
            # During the first 10 epochs, only use classification loss
            total_loss = cls_loss
            self.log("val_loss", total_loss, prog_bar=True)

        # Save example batch for visualization
        if batch_idx == 0:
            self.example_batch = batch

        return total_loss

    def configure_optimizers(self):
        # Separate optimizers for each component
        diff_params = [p for n, p in self.named_parameters() if "embedding_conditioner" not in n]
        cls_params = list(self.embedding_conditioner.parameters())

        diff_optimizer = torch.optim.AdamW(diff_params, lr=self.learning_rate, weight_decay=1e-4)
        cls_optimizer = torch.optim.AdamW(cls_params, lr=self.learning_rate, weight_decay=1e-4)

        diff_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            diff_optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.05,
            anneal_strategy="cos",
            final_div_factor=100,
        )

        cls_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            cls_optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.05,
            anneal_strategy="cos",
            final_div_factor=100,
        )

        return [diff_optimizer, cls_optimizer], [
            {"scheduler": diff_scheduler, "interval": "step"},
            {"scheduler": cls_scheduler, "interval": "step"}
        ]

    def on_validation_epoch_end(self):
        if hasattr(self, 'val_preds') and len(self.val_preds) > 0:
            # Concatenate predictions and labels
            all_preds = torch.cat(self.val_preds).cpu().numpy()
            all_labels = torch.cat(self.val_labels).cpu().numpy()
            label_names = [self.label_names[x] for x in range(self.num_classes)]

            # Compute confusion matrix with fixed class order
            cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))

            # Normalize by row (true labels)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names,
                ax=ax
            )

            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.title('Confusion Matrix (Normalized by Row)', fontsize=14)
            plt.tight_layout()

            # Log to Weights & Biases
            if self.logger:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                img = Image.open(buf)
                self.logger.experiment.log({"confusion_matrix": wandb.Image(img)})

            plt.close()

            # Clear predictions
            self.val_preds.clear()
            self.val_labels.clear()

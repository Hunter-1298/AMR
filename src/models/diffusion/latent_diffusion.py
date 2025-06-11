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


class SoftEmbeddingConditioner(nn.Module):
    """Soft embedding conditioner that outputs probability-weighted embeddings"""
    def __init__(self, num_classes, latent_dim=32, hidden_dim=64, embedding_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Class embeddings
        self.class_embeddings = nn.Embedding(num_classes, embedding_dim)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # MLP classifier
        self.proj1 = nn.Linear(latent_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, latent, temperature=1.0):
        # Global pooling
        x = self.pool(latent).squeeze(-1)  # [B, latent_dim]

        # First projection
        h = F.relu(self.norm1(self.proj1(x)))

        # Second projection with residual
        h = h + F.relu(self.norm2(self.proj2(h)))

        # Classification
        class_logits = self.classifier(h)

        # Soft probabilities with temperature
        class_probs = F.softmax(class_logits / temperature, dim=1)

        # Get all class embeddings
        all_embeddings = self.class_embeddings.weight  # [num_classes, embedding_dim]

        # Compute weighted embedding
        soft_embedding = torch.matmul(class_probs, all_embeddings)  # [B, embedding_dim]

        # Also return hard prediction for logging
        pred_class = torch.argmax(class_logits, dim=1)

        # Confidence score (max probability)
        confidence = class_probs.max(dim=1)[0]

        return soft_embedding, class_logits, pred_class, confidence


class LatentDiffusion(L.LightningModule):
    def __init__(
        self,
        unet,
        encoder,
        label_names,
        n_steps: int = 1000,
        linear_start: float = 0.0001,
        linear_end: float = 0.02,
        latent_scaling: float = 0.18215,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        num_classes: int = 11,
        diffusion_contrastive: str = "bottleneck",
        # New parameters for hybrid approach
        embedding_dim: int = 128,
        temperature_start: float = 2.0,
        temperature_end: float = 0.5,
        teacher_forcing_start: float = 0.8,
        teacher_forcing_end: float = 0.0,
        teacher_forcing_epochs: int = 100,
        snr_curriculum_epochs: int = 150,
        min_snr_start: float = 0.0,
        min_snr_end: float = -20.0,
        # Contrastive loss parameters
        intra_class_weight: float = 0.1,
        inter_class_weight: float = 0.05,
        contrastive_margin: float = 2.0,
        # Progressive denoising
        num_progressive_stages: int = 2,
        confidence_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['unet', 'encoder'])
        self.automatic_optimization = False

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.latent_scaling = latent_scaling
        self.diffusion_contrastive = diffusion_contrastive
        self.num_classes = num_classes
        self.label_names = label_names

        # Hybrid approach parameters
        self.embedding_dim = embedding_dim
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.teacher_forcing_start = teacher_forcing_start
        self.teacher_forcing_end = teacher_forcing_end
        self.teacher_forcing_epochs = teacher_forcing_epochs
        self.snr_curriculum_epochs = snr_curriculum_epochs
        self.min_snr_start = min_snr_start
        self.min_snr_end = min_snr_end

        # Contrastive parameters
        self.intra_class_weight = intra_class_weight
        self.inter_class_weight = inter_class_weight
        self.contrastive_margin = contrastive_margin

        # Progressive denoising
        self.num_progressive_stages = num_progressive_stages
        self.confidence_threshold = confidence_threshold

        # Initialize models
        self.unet = unet
        self.encoder = encoder
        self.contrastive_projector = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Use soft embedding conditioner instead
        self.embedding_conditioner = SoftEmbeddingConditioner(
            num_classes=num_classes,
            latent_dim=32,
            hidden_dim=64,
            embedding_dim=embedding_dim
        )

        # Create true class embeddings for teacher forcing
        self.true_class_embeddings = nn.Embedding(num_classes, embedding_dim)

        # Create noise schedule
        beta = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_steps, dtype=torch.float64
            )
            ** 2
        )

        beta = beta.to(torch.float32)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        # Register buffers
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar)

    def get_current_temperature(self):
        """Get temperature for current epoch (for soft probabilities)"""
        progress = min(self.current_epoch / self.teacher_forcing_epochs, 1.0)
        temp = self.temperature_start + (self.temperature_end - self.temperature_start) * progress
        return temp

    def get_teacher_forcing_prob(self):
        """Get teacher forcing probability for current epoch"""
        if self.current_epoch >= self.teacher_forcing_epochs:
            return self.teacher_forcing_end

        progress = self.current_epoch / self.teacher_forcing_epochs
        prob = self.teacher_forcing_start + (self.teacher_forcing_end - self.teacher_forcing_start) * progress
        return prob

    def get_min_snr_threshold(self):
        """Get minimum SNR for curriculum learning"""
        if self.current_epoch >= self.snr_curriculum_epochs:
            return self.min_snr_end

        progress = self.current_epoch / self.snr_curriculum_epochs
        min_snr = self.min_snr_start + (self.min_snr_end - self.min_snr_start) * progress
        return min_snr

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space and scale"""
        # Check which method the encoder has
        if hasattr(self.encoder, 'encode'):
            z = self.encoder.encode(x) * self.latent_scaling
        elif hasattr(self.encoder, 'get_spatial_features'):
            z = self.encoder.get_spatial_features(x) * self.latent_scaling
        else:
            # If neither method exists, try calling the encoder directly
            z = self.encoder(x) * self.latent_scaling
        return z

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    def estimate_snr_from_noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """Estimate SNR in dB from diffusion timestep"""
        # Higher timestep = more noise = lower SNR
        # Map timestep [0, n_steps] to SNR [20, -20] dB
        snr_db = 20 - (40 * t.float() / self.n_steps)
        return snr_db

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context_embedding: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with soft embeddings and confidence weighting"""
        # If confidence is provided, weight the embedding
        if confidence is not None:
            # Estimate SNR from timestep
            snr_db = self.estimate_snr_from_noise_level(t)

            # SNR-adaptive weighting (sigmoid centered at -5dB)
            snr_weight = torch.sigmoid((snr_db + 5) / 5).view(-1, 1)

            # Combined weighting
            embedding_weight = confidence.view(-1, 1) * snr_weight
            weighted_context = context_embedding * embedding_weight
        else:
            weighted_context = context_embedding

        return self.unet(x, t, weighted_context)

    def progressive_denoise(
        self,
        z_noisy: torch.Tensor,
        t: torch.Tensor,
        true_class: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Progressive denoising with iterative refinement"""
        current_z = z_noisy
        final_class_logits = None
        final_pred_class = None

        for stage in range(self.num_progressive_stages):
            # For training, we need to be careful about gradients
            # We only need gradients through the UNet, not the embedding conditioner
            with torch.no_grad():
                # Get soft embedding from current (partially) denoised signal
                temperature = self.get_current_temperature()
                soft_embed, class_logits, pred_class, confidence = self.embedding_conditioner(
                    current_z.detach(), temperature
                )

            # Save final predictions
            if stage == 0:
                final_class_logits = class_logits
                final_pred_class = pred_class

            # Teacher forcing during training
            if is_training and true_class is not None:
                teacher_prob = self.get_teacher_forcing_prob()
                if torch.rand(1).item() < teacher_prob:
                    # Use true class embedding
                    context_embedding = self.true_class_embeddings(true_class)
                else:
                    # Use predicted soft embedding
                    context_embedding = soft_embed
            else:
                # Use predicted soft embedding with confidence weighting
                context_embedding = soft_embed

            # Predict noise with confidence-weighted embedding
            predicted_noise = self.forward(current_z, t, context_embedding, confidence)

            # Partially denoise for next stage (except last stage)
            if stage < self.num_progressive_stages - 1:
                # Partial denoising step
                sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)
                sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

                # Move halfway towards predicted clean signal
                partial_factor = 0.5
                current_z = current_z - partial_factor * sqrt_one_minus_alpha_bar_t * predicted_noise

        return predicted_noise, final_class_logits, final_pred_class
    def compute_class_contrastive_loss(
        self,
        z_denoised: torch.Tensor,
        true_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute intra-class and inter-class contrastive losses"""
        batch_size = z_denoised.shape[0]

        # Pool to get feature vectors
        z_pooled = F.adaptive_avg_pool1d(z_denoised, 1).squeeze(-1)

        # Group by class
        class_groups = {}
        for i, label in enumerate(true_labels):
            label_item = label.item()
            if label_item not in class_groups:
                class_groups[label_item] = []
            class_groups[label_item].append(z_pooled[i])

        # Intra-class loss (pull same class together)
        intra_loss = torch.tensor(0.0, device=z_denoised.device)
        intra_count = 0

        for class_id, samples in class_groups.items():
            if len(samples) > 1:
                samples_tensor = torch.stack(samples)
                centroid = samples_tensor.mean(0, keepdim=True)

                # Distance from each sample to centroid
                distances = F.mse_loss(samples_tensor, centroid.expand_as(samples_tensor), reduction='none')
                intra_loss += distances.mean()
                intra_count += 1

        if intra_count > 0:
            intra_loss = intra_loss / intra_count

        # Inter-class loss (push different classes apart)
        inter_loss = torch.tensor(0.0, device=z_denoised.device)
        inter_count = 0

        # Compute class centroids
        centroids = []
        class_ids = []
        for class_id, samples in class_groups.items():
            if len(samples) > 0:
                centroid = torch.stack(samples).mean(0)
                centroids.append(centroid)
                class_ids.append(class_id)

        if len(centroids) > 1:
            centroids_tensor = torch.stack(centroids)

            # Pairwise distances between centroids
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = F.mse_loss(centroids_tensor[i], centroids_tensor[j], reduction='none').sum()
                    # Hinge loss: we want distance > margin
                    hinge_loss = torch.clamp(self.contrastive_margin - dist, min=0)
                    inter_loss += hinge_loss
                    inter_count += 1

        if inter_count > 0:
            inter_loss = inter_loss / inter_count

        return intra_loss, inter_loss

    def training_step(self, batch, batch_idx):
        # Get optimizers and schedulers
        diff_opt, cls_opt = self.optimizers()
        diff_sch, cls_sch = self.lr_schedulers()

        # Unpack batch
        x, context, snr = batch

        # Apply SNR curriculum - filter out samples below threshold
        min_snr_threshold = self.get_min_snr_threshold()
        mask = snr >= min_snr_threshold

        if mask.sum() == 0:
            # Skip batch if all samples are below threshold
            return None

        # Filter batch
        x = x[mask]
        context = context[mask]
        snr = snr[mask]

        # Encode input to latent space
        z = self.encode(x)

        # STEP 1: Update embedding conditioner
        # Get soft embeddings and predictions
        temperature = self.get_current_temperature()
        soft_embed, class_logits, pred_class, confidence = self.embedding_conditioner(z, temperature)
        cls_loss = F.cross_entropy(class_logits, context)

        # Backward and update embedding conditioner
        cls_opt.zero_grad()
        self.manual_backward(cls_loss)
        cls_opt.step()
        cls_sch.step()

        self.log("train/class_embed_loss", cls_loss, prog_bar=True)
        self.log("train/temperature", temperature)
        self.log("train/min_snr_threshold", min_snr_threshold)

        with torch.no_grad():
            acc = (pred_class == context).float().mean()
            avg_confidence = confidence.mean()
            self.log("train/cls_acc", acc, prog_bar=True)
            self.log("train/avg_confidence", avg_confidence)

        # STEP 2: Update diffusion model
        if self.current_epoch > 10:
            # IMPORTANT: Detach z to prevent backprop through the encoder again
            z_detached = z.detach()

            # Sample random timesteps
            t = torch.randint(0, self.n_steps, (x.shape[0],), device=self.device).long()

            # Add noise to input
            z_noisy, noise = self.q_sample(z_detached, t)

            # Progressive denoising with teacher forcing
            predicted_noise, _, _ = self.progressive_denoise(
                z_noisy, t, context, is_training=True
            )

            # Noise prediction loss
            noise_loss = F.mse_loss(noise, predicted_noise)

            # Get denoised representation for contrastive loss
            sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
            z_denoised = (z_noisy - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

            # Class-aware contrastive losses
            intra_loss, inter_loss = self.compute_class_contrastive_loss(z_denoised, context)

            # Original contrastive loss (instance-level)
            z_pooled = F.adaptive_avg_pool1d(z_denoised, 1).squeeze(-1)
            z_proj = self.contrastive_projector(z_pooled)

            # Create positive pairs by adding different noise to same input
            t2 = torch.randint(0, self.n_steps, (x.shape[0],), device=self.device).long()
            z_noisy2, noise2 = self.q_sample(z_detached, t2)
            predicted_noise2, _, _ = self.progressive_denoise(
                z_noisy2, t2, context, is_training=True
            )

            sqrt_alpha_bar_t2 = self.sqrt_alpha_bar[t2].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t2 = self.sqrt_one_minus_alpha_bar[t2].view(-1, 1, 1)
            z_denoised2 = (z_noisy2 - sqrt_one_minus_alpha_bar_t2 * predicted_noise2) / sqrt_alpha_bar_t2

            z2_pooled = F.adaptive_avg_pool1d(z_denoised2, 1).squeeze(-1)
            z2_proj = self.contrastive_projector(z2_pooled)

            instance_contrastive_loss = self.info_nce_loss(z_proj, z2_proj)

            # Combined loss
            diffusion_loss = (
                noise_loss +
                0.05 * instance_contrastive_loss +
                self.intra_class_weight * intra_loss +
                self.inter_class_weight * inter_loss
            )

            # Update diffusion model
            diff_opt.zero_grad()
            self.manual_backward(diffusion_loss)
            diff_opt.step()
            diff_sch.step()

            # Log metrics
            self.log("train/noise_loss", noise_loss, prog_bar=True)
            self.log("train/instance_contrastive_loss", instance_contrastive_loss)
            self.log("train/intra_class_loss", intra_loss)
            self.log("train/inter_class_loss", inter_loss)
            self.log("train/total_loss", diffusion_loss + cls_loss, prog_bar=True)
            self.log("train/teacher_forcing_prob", self.get_teacher_forcing_prob())

        return cls_loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        # Unpack batch
        x, context, snr = batch

        # Encode input to latent space
        z = self.encode(x)

        # Get soft embeddings and predictions
        temperature = self.get_current_temperature()
        soft_embed, class_logits, pred_class, confidence = self.embedding_conditioner(z, temperature)

        # Store predictions for confusion matrix
        if not hasattr(self, "val_preds"):
            self.val_preds = []
            self.val_labels = []
            self.val_snrs = []
            self.val_confidences = []

        self.val_preds.append(pred_class.detach().cpu())
        self.val_labels.append(context.detach().cpu())
        self.val_snrs.append(snr.detach().cpu())
        self.val_confidences.append(confidence.detach().cpu())

        # Classification loss
        cls_loss = F.cross_entropy(class_logits, context)

        # Log classification accuracy
        with torch.no_grad():
            acc = (pred_class == context).float().mean()
            self.log("val/class_embed_acc", acc, prog_bar=True)
            self.log("val/class_embed_loss", cls_loss, prog_bar=True)
            self.log("val/avg_confidence", confidence.mean())

        # Only compute diffusion losses after the first 10 epochs
        if self.current_epoch > 10:
            # Test on multiple noise levels
            noise_levels = [100, 500, 900]  # Early, middle, late timesteps
            total_noise_loss = 0

            for t_val in noise_levels:
                t = torch.full((x.shape[0],), t_val, device=self.device).long()

                # Add noise
                z_noisy, noise = self.q_sample(z, t)

                # Progressive denoising without teacher forcing
                predicted_noise, _, _ = self.progressive_denoise(
                    z_noisy, t, None, is_training=False
                )

                # Noise loss
                noise_loss = F.mse_loss(noise, predicted_noise)
                total_noise_loss += noise_loss

                self.log(f"val/noise_loss_t{t_val}", noise_loss)

            avg_noise_loss = total_noise_loss / len(noise_levels)

            # Get denoised representation for final timestep
            sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
            z_denoised = (z_noisy - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

            # Contrastive losses
            intra_loss, inter_loss = self.compute_class_contrastive_loss(z_denoised, context)

            total_loss = avg_noise_loss + cls_loss + self.intra_class_weight * intra_loss + self.inter_class_weight * inter_loss

            self.log("val/noise_loss", avg_noise_loss, prog_bar=True)
            self.log("val/intra_class_loss", intra_loss)
            self.log("val/inter_class_loss", inter_loss)
            self.log("val_loss", total_loss, prog_bar=True)
        else:
            total_loss = cls_loss
            self.log("val_loss", 10.0, prog_bar=True)  # Dummy loss for checkpointing

        # Save example batch for visualization
        if batch_idx == 0:
            self.example_batch = batch

        return total_loss

    def on_validation_epoch_end(self):
        if hasattr(self, "val_preds") and len(self.val_preds) > 0:
            # Concatenate all predictions
            all_preds = torch.cat(self.val_preds).cpu().numpy()
            all_labels = torch.cat(self.val_labels).cpu().numpy()
            all_snrs = torch.cat(self.val_snrs).cpu().numpy()
            all_confidences = torch.cat(self.val_confidences).cpu().numpy()

            label_names = [self.label_names[x] for x in range(self.num_classes)]

            # Compute confusion matrix
            cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

            # Plot confusion matrix
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Standard confusion matrix
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=label_names,
                yticklabels=label_names,
                ax=ax1
            )
            ax1.set_xlabel("Predicted Label", fontsize=12)
            ax1.set_ylabel("True Label", fontsize=12)
            ax1.set_title("Confusion Matrix (Normalized by Row)", fontsize=14)

            # Accuracy vs SNR plot
            snr_bins = np.arange(-20, 25, 5)
            accuracies = []
            avg_confidences = []

            for i in range(len(snr_bins) - 1):
                mask = (all_snrs >= snr_bins[i]) & (all_snrs < snr_bins[i + 1])
                if mask.sum() > 0:
                    acc = (all_preds[mask] == all_labels[mask]).mean()
                    conf = all_confidences[mask].mean()
                    accuracies.append(acc)
                    avg_confidences.append(conf)
                else:
                    accuracies.append(0)
                    avg_confidences.append(0)

            ax2.plot(snr_bins[:-1] + 2.5, accuracies, 'b-o', label='Accuracy')
            ax2.plot(snr_bins[:-1] + 2.5, avg_confidences, 'r--s', label='Avg Confidence')
            ax2.set_xlabel("SNR (dB)", fontsize=12)
            ax2.set_ylabel("Accuracy / Confidence", fontsize=12)
            ax2.set_title("Classification Performance vs SNR", fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(0, 1.05)

            plt.tight_layout()

            # Log to Weights & Biases
            if self.logger:
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150)
                buf.seek(0)
                img = Image.open(buf)
                self.logger.experiment.log({"validation_analysis": wandb.Image(img)})

            plt.close()

            # Clear predictions
            self.val_preds.clear()
            self.val_labels.clear()
            self.val_snrs.clear()
            self.val_confidences.clear()

    def configure_optimizers(self):
        # Separate optimizers for each component
        diff_params = [
            p for n, p in self.named_parameters()
            if "embedding_conditioner" not in n and "true_class_embeddings" not in n
        ]

        # Include both embedding conditioner and true class embeddings
        cls_params = list(self.embedding_conditioner.parameters()) + list(self.true_class_embeddings.parameters())

        diff_optimizer = torch.optim.AdamW(
            diff_params, lr=self.learning_rate, weight_decay=1e-4
        )
        cls_optimizer = torch.optim.AdamW(
            cls_params, lr=self.learning_rate * 2, weight_decay=1e-4  # Higher LR for classifier
        )

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
            max_lr=self.learning_rate * 2,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.05,
            anneal_strategy="cos",
            final_div_factor=100,
        )

        return [diff_optimizer, cls_optimizer], [
            {"scheduler": diff_scheduler, "interval": "step"},
            {"scheduler": cls_scheduler, "interval": "step"},
        ]
    def info_nce_loss(self, z1, z2, temperature=0.07):
        """InfoNCE loss implementation"""
        if z1.dim() == 3:
            z1 = F.adaptive_avg_pool1d(z1, 1).squeeze(-1)
        if z2.dim() == 3:
            z2 = F.adaptive_avg_pool1d(z2, 1).squeeze(-1)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        N = z1.shape[0]
        logits = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(N, device=z1.device)
        loss = F.cross_entropy(logits, labels)

        return loss

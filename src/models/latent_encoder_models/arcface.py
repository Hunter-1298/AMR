import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
from torch.optim import AdamW
from typing import Dict, Tuple, Optional, List
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import math


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 1D sequences.
    """
    def __init__(self, d_model: int, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model

        # Create frequency tensor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]
        return self._cos_cached[:, :seq_len, :], self._sin_cached[:, :seq_len, :]

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embedding."""
        seq_len = x.shape[1]
        cos, sin = self._update_cos_sin_cache(seq_len, x.device, x.dtype)

        # Ensure cos and sin have the right shape
        if cos.shape[-1] != x.shape[-1]:
            # Repeat or truncate to match x dimensions
            cos = cos.repeat(1, 1, x.shape[-1] // cos.shape[-1])[:, :, :x.shape[-1]]
            sin = sin.repeat(1, 1, x.shape[-1] // sin.shape[-1])[:, :, :x.shape[-1]]

        return x * cos + self.rotate_half(x) * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.apply_rotary_pos_emb(x)


class RFViCMAEEncoder(nn.Module):
    """
    Vision Contrastive MAE (ViC-MAE) style encoder for RF signals.
    Combines masked autoencoding with contrastive learning for RF waveforms.
    """

    def __init__(
        self,
        signal_length: int = 128,
        patch_size: int = 16,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1536,
        mask_ratio: float = 0.75,
        projection_dim: int = 256,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.signal_length = signal_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.temperature = temperature

        # Number of patches
        self.num_patches = signal_length // patch_size
        assert signal_length % patch_size == 0, f"Signal length {signal_length} must be divisible by patch size {patch_size}"

        # Patch embedding: Conv1d to create tokens from IQ signals
        self.patch_embed = nn.Conv1d(2, d_model, kernel_size=patch_size, stride=patch_size, bias=False)

        # Layer norm after patch embedding (applied to the correct dimension)
        self.patch_norm = nn.LayerNorm(d_model)

        # RoPE positional encoding
        self.rope = RoPEPositionalEncoding(d_model, max_seq_len=self.num_patches)

        # Mask token for decoder
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Shared transformer encoder (processes only visible tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Lightweight decoder for reconstruction
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward // 2,  # Smaller decoder
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # Improved decoder prediction head
        self.decoder_pred = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, patch_size * 2),
        )

        # Projection head for contrastive learning (3-layer MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False),  # No affine for L2 norm
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RF signal to patches - embedding version."""
        # x: [batch, 2, signal_length]
        patches = self.patch_embed(x)  # [batch, d_model, num_patches]
        patches = patches.transpose(1, 2)  # [batch, num_patches, d_model]
        patches = self.patch_norm(patches)  # Apply layer norm
        return patches

    def create_patch_targets(self, x: torch.Tensor) -> torch.Tensor:
        """Create patch targets without going through embedding."""
        batch_size, channels, signal_length = x.shape
        # Reshape to patches: [batch, num_patches, patch_size * channels]
        x_patches = x.reshape(batch_size, channels, self.num_patches, self.patch_size)
        x_patches = x_patches.permute(0, 2, 3, 1)  # [batch, num_patches, patch_size, 2]
        x_patches = x_patches.reshape(batch_size, self.num_patches, -1)  # [batch, num_patches, patch_size * 2]
        return x_patches

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to RF signal format."""
        batch_size, num_patches, patch_dim = x.shape
        # Reshape: [batch, num_patches, patch_size * 2] -> [batch, 2, signal_length]
        x = x.reshape(batch_size, num_patches, self.patch_size, 2)
        x = x.permute(0, 3, 1, 2)  # [batch, 2, num_patches, patch_size]
        x = x.reshape(batch_size, 2, -1)  # [batch, 2, signal_length]
        return x

    def forward_single_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode from a single latent vector (for diffusion sampling).

        Args:
            latent: [batch, d_model] latent representation

        Returns:
            reconstructed: [batch, 2, signal_length] RF signal
        """
        batch_size = latent.shape[0]

        # Expand latent to sequence
        latent_seq = latent.unsqueeze(1).repeat(1, self.num_patches, 1)  # [batch, num_patches, d_model]

        # Apply RoPE
        latent_seq = self.rope(latent_seq)

        # Decode
        decoded = self.decoder(latent_seq)

        # Project to IQ
        pred = self.decoder_pred(decoded)  # [batch, num_patches, patch_size * 2]

        # Unpatchify
        reconstructed = self.unpatchify(pred)

        return reconstructed

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking on input tokens.

        Args:
            x: [batch, num_patches, d_model]

        Returns:
            x_masked: visible tokens [batch, num_visible, d_model]
            mask: binary mask [batch, num_patches] (1 = masked, 0 = visible)
            ids_restore: indices to restore original order [batch, num_patches]
        """
        batch_size, num_patches, D = x.shape
        len_keep = int(num_patches * (1 - self.mask_ratio))

        # Generate random noise for each sample
        noise = torch.rand(batch_size, num_patches, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask: 0 is keep, 1 is remove (for loss computation)
        mask = torch.ones([batch_size, num_patches], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder with RoPE.

        Args:
            x: input tokens [batch, num_patches, d_model] (only visible if masked)

        Returns:
            latent: encoder output [batch, num_patches, d_model]
        """
        # Apply RoPE positional encoding
        x = self.rope(x)

        # Apply transformer encoder
        latent = self.encoder(x)

        return latent

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder with proper RoPE handling.

        Args:
            x: visible tokens from encoder [batch, num_visible, d_model]
            ids_restore: indices to restore full sequence [batch, num_patches]

        Returns:
            pred: reconstructed patches [batch, num_patches, patch_size * 2]
        """
        batch_size, num_visible, D = x.shape

        # Create mask tokens
        num_mask = ids_restore.shape[1] - num_visible
        mask_tokens = self.mask_token.repeat(batch_size, num_mask, 1)

        # Concatenate visible and mask tokens
        x_full = torch.cat([x, mask_tokens], dim=1)  # [batch, num_patches, d_model]

        # Unshuffle to restore original order
        x_full = torch.gather(x_full, dim=1,
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        # Apply RoPE to full sequence
        x_full = self.rope(x_full)

        # Apply decoder
        decoded = self.decoder(x_full)

        # Predict patches
        pred = self.decoder_pred(decoded)

        return pred

    def compute_reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Enhanced reconstruction loss for RF signals.

        Args:
            pred: predicted patches [batch, num_patches, patch_size * 2]
            target: original patches [batch, num_patches, patch_size * 2]
            mask: binary mask (1 = masked, 0 = visible) [batch, num_patches]

        Returns:
            loss: scalar reconstruction loss
        """
        # 1. MSE on IQ components
        mse_loss = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # [batch, num_patches]

        # 2. Complex magnitude loss
        pred_reshaped = pred.reshape(pred.shape[0], pred.shape[1], self.patch_size, 2)
        target_reshaped = target.reshape(target.shape[0], target.shape[1], self.patch_size, 2)

        pred_mag = torch.sqrt(pred_reshaped[..., 0]**2 + pred_reshaped[..., 1]**2 + 1e-8)
        target_mag = torch.sqrt(target_reshaped[..., 0]**2 + target_reshaped[..., 1]**2 + 1e-8)
        mag_loss = F.mse_loss(pred_mag, target_mag, reduction='none').mean(dim=-1)  # [batch, num_patches]

        # 3. Phase consistency loss
        pred_phase = torch.atan2(pred_reshaped[..., 1], pred_reshaped[..., 0] + 1e-8)
        target_phase = torch.atan2(target_reshaped[..., 1], target_reshaped[..., 0] + 1e-8)

        # Handle phase wrapping
        phase_diff = pred_phase - target_phase
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        phase_loss = (phase_diff**2).mean(dim=-1)  # [batch, num_patches]

        # Combine losses
        total_loss = mse_loss + 0.5 * mag_loss + 0.3 * phase_loss

        # Apply mask (only compute loss on masked patches)
        masked_loss = (total_loss * mask).sum() / (mask.sum() + 1e-8)

        return masked_loss

    def compute_contrastive_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss between two views.

        Args:
            z_i: L2-normalized embeddings from view i [batch, projection_dim]
            z_j: L2-normalized embeddings from view j [batch, projection_dim]

        Returns:
            loss: InfoNCE loss
        """
        batch_size = z_i.shape[0]

        # Compute similarity matrix
        sim_i_j = torch.mm(z_i, z_j.t()) / self.temperature  # [batch, batch]
        sim_j_i = torch.mm(z_j, z_i.t()) / self.temperature  # [batch, batch]

        # Create positive mask (diagonal elements)
        pos_mask = torch.eye(batch_size, device=z_i.device, dtype=torch.bool)

        # InfoNCE loss for i->j
        exp_sim_i_j = torch.exp(sim_i_j)
        log_prob_i_j = sim_i_j[pos_mask] - torch.log(exp_sim_i_j.sum(dim=1))

        # InfoNCE loss for j->i
        exp_sim_j_i = torch.exp(sim_j_i)
        log_prob_j_i = sim_j_i[pos_mask] - torch.log(exp_sim_j_i.sum(dim=1))

        # Average both directions
        loss = -(log_prob_i_j + log_prob_j_i).mean() / 2

        return loss

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ViC-MAE.

        Args:
            x_i: first view [batch, 2, signal_length]
            x_j: second view (same modulation, different SNR) [batch, 2, signal_length]

        Returns:
            Dictionary containing reconstruction and contrastive outputs
        """
        # Patchify inputs (embedding version)
        patches_i = self.patchify(x_i)  # [batch, num_patches, d_model]
        patches_j = self.patchify(x_j)  # [batch, num_patches, d_model]

        # Create proper targets (raw patch version)
        target_i = self.create_patch_targets(x_i)  # [batch, num_patches, patch_size * 2]
        target_j = self.create_patch_targets(x_j)

        # Random masking
        patches_i_vis, mask_i, ids_restore_i = self.random_masking(patches_i)
        patches_j_vis, mask_j, ids_restore_j = self.random_masking(patches_j)

        # Encode visible patches
        latent_i_vis = self.forward_encoder(patches_i_vis)  # [batch, num_visible, d_model]
        latent_j_vis = self.forward_encoder(patches_j_vis)

        # Decode to reconstruct full signal
        pred_i = self.forward_decoder(latent_i_vis, ids_restore_i)  # [batch, num_patches, patch_size * 2]
        pred_j = self.forward_decoder(latent_j_vis, ids_restore_j)

        # Compute reconstruction loss on masked patches only
        loss_recon_i = self.compute_reconstruction_loss(pred_i, target_i, mask_i)
        loss_recon_j = self.compute_reconstruction_loss(pred_j, target_j, mask_j)
        loss_recon = (loss_recon_i + loss_recon_j) / 2

        # Unpatchify for visualization
        recon_xi = self.unpatchify(pred_i)  # [batch, 2, signal_length]
        recon_xj = self.unpatchify(pred_j)

        # Global pooling over visible tokens for contrastive learning
        global_i = latent_i_vis.mean(dim=1)  # [batch, d_model]
        global_j = latent_j_vis.mean(dim=1)  # [batch, d_model]

        # Project to contrastive space
        z_i = self.projection_head(global_i)  # [batch, projection_dim]
        z_j = self.projection_head(global_j)

        # L2 normalize
        emb_i = F.normalize(z_i, p=2, dim=1)
        emb_j = F.normalize(z_j, p=2, dim=1)

        # Compute contrastive loss
        loss_contrast = self.compute_contrastive_loss(emb_i, emb_j)

        return {
            'recon_xi': recon_xi,
            'recon_xj': recon_xj,
            'emb_i': emb_i,
            'emb_j': emb_j,
            'latent_i': global_i,  # For downstream tasks
            'latent_j': global_j,
            'loss_recon': loss_recon,
            'loss_contrast': loss_contrast,
            'mask_i': mask_i,  # For visualization
            'mask_j': mask_j,
        }

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single input (inference mode).

        Args:
            x: input signal [batch, 2, signal_length]

        Returns:
            latent: global latent representation [batch, d_model]
        """
        patches = self.patchify(x)
        patches = self.rope(patches)
        latent = self.encoder(patches)
        global_feat = latent.mean(dim=1)
        return global_feat


class RFSpecificReconstructionLoss(nn.Module):
    """Enhanced reconstruction loss for RF signals"""

    def __init__(self, signal_length=128):
        super().__init__()
        self.signal_length = signal_length

    def complex_mse(self, pred, target):
        """MSE on complex representation"""
        pred_complex = torch.complex(pred[:, 0], pred[:, 1])
        target_complex = torch.complex(target[:, 0], target[:, 1])
        return torch.mean(torch.abs(pred_complex - target_complex) ** 2)

    def magnitude_loss(self, pred, target):
        """Loss on signal magnitude"""
        pred_mag = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + 1e-8)
        target_mag = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + 1e-8)
        return F.mse_loss(pred_mag, target_mag)

    def phase_loss(self, pred, target):
        """Loss on signal phase"""
        pred_phase = torch.atan2(pred[:, 1], pred[:, 0] + 1e-8)
        target_phase = torch.atan2(target[:, 1], target[:, 0] + 1e-8)

        # Handle phase wrapping
        phase_diff = pred_phase - target_phase
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        return torch.mean(phase_diff**2)

    def spectral_loss(self, pred, target):
        """Loss on frequency domain"""
        pred_complex = torch.complex(pred[:, 0], pred[:, 1])
        target_complex = torch.complex(target[:, 0], target[:, 1])

        pred_fft = torch.fft.fft(pred_complex, dim=-1)
        target_fft = torch.fft.fft(target_complex, dim=-1)

        # Magnitude spectrum loss
        pred_mag_spec = torch.abs(pred_fft)
        target_mag_spec = torch.abs(target_fft)

        return F.mse_loss(pred_mag_spec, target_mag_spec)

    def forward(self, pred, target, mask=None):
        """Combined RF reconstruction loss"""
        # Component losses
        complex_loss = self.complex_mse(pred, target)
        mag_loss = self.magnitude_loss(pred, target)
        phase_loss = self.phase_loss(pred, target)
        spectral_loss = self.spectral_loss(pred, target)

        # Weighted combination
        total_loss = (
            1.0 * complex_loss +      # Primary reconstruction
            0.5 * mag_loss +          # Magnitude preservation
            0.3 * phase_loss +        # Phase preservation
            0.2 * spectral_loss       # Spectral characteristics
        )

        return total_loss, {
            'complex_mse': complex_loss,
            'magnitude_loss': mag_loss,
            'phase_loss': phase_loss,
            'spectral_loss': spectral_loss
        }


class RFEncoderDecoder(L.LightningModule):
    """
    Lightning module for ViC-MAE training on RF signals.
    """

    def __init__(
        self,
        label_names: List[str],
        signal_length: int = 1024,
        patch_size: int = 32,
        d_model: int = 384,
        mask_ratio: float = 0.50,
        nhead: int = 12,
        temperature: float = 0.07,
        learning_rate: float = 3e-4,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        reconstruction_weight: float = 3.0,
        contrastive_weight: float = 1.0,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.label_names = label_names
        self.num_classes = num_classes or len(label_names)

        # ViC-MAE encoder
        self.encoder = RFViCMAEEncoder(
            signal_length=signal_length,
            patch_size=patch_size,
            d_model=d_model,
            mask_ratio=mask_ratio,
            temperature=temperature,
            nhead = nhead,
        )

        # Optional classification head for finetuning
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, self.num_classes),
        ) if num_classes else None

    def training_step(self, batch, batch_idx):
        # Unpack batch - assuming dataloader provides (x_i, x_j, labels, snrs)
        if len(batch) == 4:
            x_i, x_j, labels, snrs = batch
        elif len(batch) == 3:
            (x_i, x_j), labels, snrs = batch
        else:
            raise ValueError(f"Expected batch with 3 or 4 elements, got {len(batch)}")

        # Forward pass through ViC-MAE
        outputs = self.encoder(x_i, x_j)

        # Compute total loss
        loss_recon = outputs['loss_recon']
        loss_contrast = outputs['loss_contrast']

        # Scale reconstruction loss if it's too small
        if loss_recon < 0.1:
            loss_recon = loss_recon * 10

        total_loss = (
            self.hparams.reconstruction_weight * loss_recon +
            self.hparams.contrastive_weight * loss_contrast
        )

        # Optional classification loss if classifier exists
        if self.classifier is not None:
            logits_i = self.classifier(outputs['latent_i'])
            logits_j = self.classifier(outputs['latent_j'])

            # Handle labels - make sure they're the right shape
            target_labels = labels.squeeze() if labels.dim() > 1 else labels

            class_loss_i = F.cross_entropy(logits_i, target_labels)
            class_loss_j = F.cross_entropy(logits_j, target_labels)
            class_loss = (class_loss_i + class_loss_j) / 2

            total_loss = total_loss + 0.1 * class_loss
            self.log('train_class_loss', class_loss, prog_bar=True)

            # Log accuracy
            acc_i = (logits_i.argmax(dim=1) == target_labels).float().mean()
            acc_j = (logits_j.argmax(dim=1) == target_labels).float().mean()
            acc = (acc_i + acc_j) / 2
            self.log('train_acc', acc, prog_bar=True)

        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_recon_loss', loss_recon)
        self.log('train_contrast_loss', loss_contrast)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        if len(batch) == 4:
            x_i, x_j, labels, snrs = batch
        elif len(batch) == 3:
            (x_i, x_j), labels, snrs = batch
        else:
            raise ValueError(f"Expected batch with 3 or 4 elements, got {len(batch)}")

        # Forward pass
        outputs = self.encoder(x_i, x_j)

        # Compute losses
        loss_recon = outputs['loss_recon']
        loss_contrast = outputs['loss_contrast']

        # Scale reconstruction loss if it's too small
        if loss_recon < 0.1:
            loss_recon = loss_recon * 10

        total_loss = (
            self.hparams.reconstruction_weight * loss_recon +
            self.hparams.contrastive_weight * loss_contrast
        )

        # Optional classification
        if self.classifier is not None:
            logits_i = self.classifier(outputs['latent_i'])
            logits_j = self.classifier(outputs['latent_j'])

            target_labels = labels.squeeze() if labels.dim() > 1 else labels

            class_loss_i = F.cross_entropy(logits_i, target_labels)
            class_loss_j = F.cross_entropy(logits_j, target_labels)
            class_loss = (class_loss_i + class_loss_j) / 2

            # Accuracy
            acc_i = (logits_i.argmax(dim=1) == target_labels).float().mean()
            acc_j = (logits_j.argmax(dim=1) == target_labels).float().mean()
            acc = (acc_i + acc_j) / 2

            total_loss = total_loss + 0.1 * class_loss
            self.log('val_class_loss', class_loss)
            self.log('val_acc', acc, prog_bar=True)

        # Logging
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_recon_loss', loss_recon)
        self.log('val_contrast_loss', loss_contrast)

        # Store for visualization (first batch only)
        if batch_idx == 0:
            self.val_outputs = outputs
            self.val_x_i = x_i.detach().cpu()
            self.val_x_j = x_j.detach().cpu()
            self.val_labels = labels.detach().cpu()
            if len(batch) == 4:
                self.val_snrs = snrs.detach().cpu()

        return total_loss

    def on_validation_epoch_end(self):
        """Visualize reconstruction and embeddings."""
        if hasattr(self, 'val_outputs'):
            self._visualize_reconstruction()
            self._visualize_embeddings()
            self._visualize_rf_metrics()
            self._visualize_frequency_domain()

    def _visualize_reconstruction(self):
        """Visualize masked reconstruction."""
        if not hasattr(self, 'val_outputs'):
            return

        try:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            # Select first sample
            idx = 0

            # Original signals
            orig_i = self.val_x_i[idx].numpy()
            orig_j = self.val_x_j[idx].numpy()

            # Reconstructed signals
            recon_i = self.val_outputs['recon_xi'][idx].detach().cpu().numpy()
            recon_j = self.val_outputs['recon_xj'][idx].detach().cpu().numpy()

            # Plot view i
            time = np.arange(orig_i.shape[1])

            axes[0, 0].plot(time, orig_i[0], label='Original I', alpha=0.8)
            axes[0, 0].plot(time, recon_i[0], label='Reconstructed I', alpha=0.8)
            axes[0, 0].set_title('View i - I channel')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(time, orig_i[1], label='Original Q', alpha=0.8)
            axes[0, 1].plot(time, recon_i[1], label='Reconstructed Q', alpha=0.8)
            axes[0, 1].set_title('View i - Q channel')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot view j
            axes[1, 0].plot(time, orig_j[0], label='Original I', alpha=0.8)
            axes[1, 0].plot(time, recon_j[0], label='Reconstructed I', alpha=0.8)
            axes[1, 0].set_title('View j - I channel')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(time, orig_j[1], label='Original Q', alpha=0.8)
            axes[1, 1].plot(time, recon_j[1], label='Reconstructed Q', alpha=0.8)
            axes[1, 1].set_title('View j - Q channel')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # Constellation diagrams
            axes[0, 2].scatter(orig_i[0], orig_i[1], alpha=0.5, s=10, label='Original')
            axes[0, 2].scatter(recon_i[0], recon_i[1], alpha=0.5, s=10, label='Reconstructed')
            axes[0, 2].set_title('View i - Constellation')
            axes[0, 2].set_xlabel('I')
            axes[0, 2].set_ylabel('Q')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            axes[1, 2].scatter(orig_j[0], orig_j[1], alpha=0.5, s=10, label='Original')
            axes[1, 2].scatter(recon_j[0], recon_j[1], alpha=0.5, s=10, label='Reconstructed')
            axes[1, 2].set_title('View j - Constellation')
            axes[1, 2].set_xlabel('I')
            axes[1, 2].set_ylabel('Q')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

            # Mask visualization
            mask_i = self.val_outputs['mask_i'][idx].detach().cpu().numpy()
            mask_j = self.val_outputs['mask_j'][idx].detach().cpu().numpy()

            axes[0, 3].imshow(mask_i.reshape(1, -1), aspect='auto', cmap='binary')
            axes[0, 3].set_title(f'Mask i (ratio: {mask_i.mean():.2f})')
            axes[0, 3].set_yticks([])

            axes[1, 3].imshow(mask_j.reshape(1, -1), aspect='auto', cmap='binary')
            axes[1, 3].set_title(f'Mask j (ratio: {mask_j.mean():.2f})')
            axes[1, 3].set_yticks([])

            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"reconstruction_quality": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in reconstruction visualization: {e}")
            plt.close('all')

    def _visualize_embeddings(self):
        """Visualize contrastive embeddings."""
        if not hasattr(self, 'val_outputs'):
            return

        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Get embeddings
            emb_i = self.val_outputs['emb_i'].detach().cpu().numpy()
            emb_j = self.val_outputs['emb_j'].detach().cpu().numpy()
            labels = self.val_labels.numpy()

            # t-SNE visualization
            embeddings_all = np.vstack([emb_i, emb_j])
            labels_all = np.hstack([labels, labels])

            if len(embeddings_all) > 2:
                perplexity = min(30, len(embeddings_all) - 1)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings_all)

                # Plot t-SNE
                unique_labels = np.unique(labels_all)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

                for i, label in enumerate(unique_labels):
                    mask = labels_all == label
                    axes[0].scatter(
                        embeddings_2d[mask, 0],
                        embeddings_2d[mask, 1],
                        c=[colors[i]],
                        label=self.label_names[int(label)] if int(label) < len(self.label_names) else f'Class {int(label)}',
                        alpha=0.7,
                        s=20
                    )

                axes[0].set_title('t-SNE of Contrastive Embeddings')
                axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0].grid(True, alpha=0.3)

            # Similarity matrix
            similarity = np.dot(emb_i, emb_j.T)
            im = axes[1].imshow(similarity, cmap='viridis', aspect='auto')
            axes[1].set_title('Cosine Similarity Matrix')
            axes[1].set_xlabel('View j samples')
            axes[1].set_ylabel('View i samples')
            plt.colorbar(im, ax=axes[1])

            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"contrastive_embeddings": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in embedding visualization: {e}")
            plt.close('all')

    def _visualize_rf_metrics(self):
        """Visualize RF-specific metrics: magnitude, phase, and error analysis."""
        if not hasattr(self, 'val_outputs'):
            return

        try:
            fig, axes = plt.subplots(3, 4, figsize=(20, 12))

            # Select first few samples for visualization
            n_samples = min(4, self.val_x_i.shape[0])

            for sample_idx in range(n_samples):
                # Get original and reconstructed signals
                orig_i = self.val_x_i[sample_idx].numpy()
                recon_i = self.val_outputs['recon_xi'][sample_idx].detach().cpu().numpy()

                # Convert to complex
                orig_complex = orig_i[0] + 1j * orig_i[1]
                recon_complex = recon_i[0] + 1j * recon_i[1]

                # Compute magnitude and phase
                orig_mag = np.abs(orig_complex)
                recon_mag = np.abs(recon_complex)
                orig_phase = np.angle(orig_complex)
                recon_phase = np.angle(recon_complex)

                time = np.arange(len(orig_complex))

                # Row 1: Magnitude comparison
                axes[0, sample_idx].plot(time, orig_mag, label='Original', alpha=0.8, linewidth=2)
                axes[0, sample_idx].plot(time, recon_mag, label='Reconstructed', alpha=0.8, linewidth=2)
                axes[0, sample_idx].set_title(f'Sample {sample_idx+1} - Magnitude')
                axes[0, sample_idx].set_ylabel('Magnitude')
                axes[0, sample_idx].legend()
                axes[0, sample_idx].grid(True, alpha=0.3)

                # Row 2: Phase comparison
                axes[1, sample_idx].plot(time, orig_phase, label='Original', alpha=0.8, linewidth=2)
                axes[1, sample_idx].plot(time, recon_phase, label='Reconstructed', alpha=0.8, linewidth=2)
                axes[1, sample_idx].set_title(f'Sample {sample_idx+1} - Phase')
                axes[1, sample_idx].set_ylabel('Phase (radians)')
                axes[1, sample_idx].legend()
                axes[1, sample_idx].grid(True, alpha=0.3)

                # Row 3: Error analysis
                mag_error = np.abs(orig_mag - recon_mag)
                phase_error = np.abs(np.angle(np.exp(1j * (orig_phase - recon_phase))))  # Wrapped phase error

                ax_error = axes[2, sample_idx]
                ax_error2 = ax_error.twinx()

                line1 = ax_error.plot(time, mag_error, 'r-', label='Magnitude Error', alpha=0.8)
                line2 = ax_error2.plot(time, phase_error, 'b-', label='Phase Error', alpha=0.8)

                ax_error.set_xlabel('Time')
                ax_error.set_ylabel('Magnitude Error', color='r')
                ax_error2.set_ylabel('Phase Error (rad)', color='b')
                ax_error.set_title(f'Sample {sample_idx+1} - Reconstruction Errors')

                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax_error.legend(lines, labels, loc='upper right')

                ax_error.grid(True, alpha=0.3)

                # Add error statistics as text
                mag_mse = np.mean(mag_error**2)
                phase_mse = np.mean(phase_error**2)
                ax_error.text(0.02, 0.98, f'Mag MSE: {mag_mse:.4f}\nPhase MSE: {phase_mse:.4f}',
                             transform=ax_error.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"rf_metrics_analysis": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in RF metrics visualization: {e}")
            plt.close('all')

    def _visualize_frequency_domain(self):
        """Visualize frequency domain characteristics and spectral reconstruction quality."""
        if not hasattr(self, 'val_outputs'):
            return

        try:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))

            # Select first few samples
            n_samples = min(4, self.val_x_i.shape[0])

            for sample_idx in range(n_samples):
                # Get signals
                orig_i = self.val_x_i[sample_idx].numpy()
                recon_i = self.val_outputs['recon_xi'][sample_idx].detach().cpu().numpy()

                # Convert to complex
                orig_complex = orig_i[0] + 1j * orig_i[1]
                recon_complex = recon_i[0] + 1j * recon_i[1]

                # Compute FFT
                orig_fft = np.fft.fft(orig_complex)
                recon_fft = np.fft.fft(recon_complex)

                # Frequency bins
                freqs = np.fft.fftfreq(len(orig_complex), d=1.0)
                freqs_shifted = np.fft.fftshift(freqs)

                # Shift FFT for visualization
                orig_fft_shifted = np.fft.fftshift(orig_fft)
                recon_fft_shifted = np.fft.fftshift(recon_fft)

                # Row 1: Magnitude spectrum
                axes[0, sample_idx].plot(freqs_shifted, np.abs(orig_fft_shifted),
                                       label='Original', alpha=0.8, linewidth=2)
                axes[0, sample_idx].plot(freqs_shifted, np.abs(recon_fft_shifted),
                                       label='Reconstructed', alpha=0.8, linewidth=2)
                axes[0, sample_idx].set_title(f'Sample {sample_idx+1} - Magnitude Spectrum')
                axes[0, sample_idx].set_xlabel('Normalized Frequency')
                axes[0, sample_idx].set_ylabel('Magnitude')
                axes[0, sample_idx].legend()
                axes[0, sample_idx].grid(True, alpha=0.3)
                axes[0, sample_idx].set_yscale('log')

                # Row 2: Phase spectrum
                orig_phase_spectrum = np.angle(orig_fft_shifted)
                recon_phase_spectrum = np.angle(recon_fft_shifted)

                axes[1, sample_idx].plot(freqs_shifted, orig_phase_spectrum,
                                       label='Original', alpha=0.8, linewidth=2)
                axes[1, sample_idx].plot(freqs_shifted, recon_phase_spectrum,
                                       label='Reconstructed', alpha=0.8, linewidth=2)
                axes[1, sample_idx].set_title(f'Sample {sample_idx+1} - Phase Spectrum')
                axes[1, sample_idx].set_xlabel('Normalized Frequency')
                axes[1, sample_idx].set_ylabel('Phase (radians)')
                axes[1, sample_idx].legend()
                axes[1, sample_idx].grid(True, alpha=0.3)

                # Add spectral error metrics
                spectral_mag_error = np.mean(np.abs(np.abs(orig_fft) - np.abs(recon_fft))**2)
                spectral_phase_error = np.mean(np.abs(np.angle(orig_fft) - np.angle(recon_fft))**2)

                axes[0, sample_idx].text(0.02, 0.98, f'Spectral Mag MSE: {spectral_mag_error:.4f}',
                                       transform=axes[0, sample_idx].transAxes, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                axes[1, sample_idx].text(0.02, 0.98, f'Spectral Phase MSE: {spectral_phase_error:.4f}',
                                       transform=axes[1, sample_idx].transAxes, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"frequency_domain_analysis": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in frequency domain visualization: {e}")
            plt.close('all')

    def configure_optimizers(self):
        """Configure optimizer with warmup and cosine annealing."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.95),  # Common for vision transformers
        )

        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                # Linear warmup
                return epoch / self.hparams.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.hparams.warmup_epochs) / (self.hparams.max_epochs - self.hparams.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping for stable training."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for downstream tasks."""
        with torch.no_grad():
            return self.encoder.forward_single(x)

    def get_contrastive_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized contrastive embeddings."""
        with torch.no_grad():
            latent = self.encoder.forward_single(x)
            z = self.encoder.projection_head(latent)
            return F.normalize(z, p=2, dim=1)

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
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


# class EMAVectorQuantizer(nn.Module):
#     """
#     Exponential Moving Average Vector Quantization layer for more stable training.
#     """
#     def __init__(
#         self,
#         num_embeddings: int,
#         embedding_dim: int,
#         commitment_cost: float = 0.25,
#         decay: float = 0.99,
#         epsilon: float = 1e-5
#     ):
#         super().__init__()
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.commitment_cost = commitment_cost

#         # Simple learnable embeddings (no EMA)
#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)
#         self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

#     def forward(self, inputs):
#         input_shape = inputs.shape
#         flat_input = inputs.view(-1, self.embedding_dim)

#         # Calculate distances
#         distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
#                     + torch.sum(self.embedding.weight**2, dim=1)
#                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

#         # Find closest codes
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         quantized = self.embedding(encoding_indices).view(input_shape)

#         # Standard VQ loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())
#         vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

#         # Straight-through estimator
#         quantized = inputs + (quantized - inputs).detach()

#         return quantized, vq_loss, encoding_indices.view(input_shape[:-1])

class EMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        restart_unused_codes: bool = True,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.restart_unused_codes = restart_unused_codes

        # Initialize embeddings
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.embeddings.data.normal_(0, 0.02)  # Better initialization

        # EMA parameters (registered as buffers)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embeddings', self.embeddings.data.clone())
        self.register_buffer('_ema_initialized', torch.tensor(False))

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embeddings**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings)
        quantized = quantized.view(input_shape)

        # Update EMA during training
        if self.training:
            if not self._ema_initialized:
                self.ema_cluster_size.data = encodings.sum(0)
                self.ema_embeddings.data = torch.matmul(encodings.t(), flat_input)
                self._ema_initialized.data = torch.tensor(True)
            else:
                self.ema_cluster_size.data = self.decay * self.ema_cluster_size + (1 - self.decay) * encodings.sum(0)
                self.ema_embeddings.data = self.decay * self.ema_embeddings + (1 - self.decay) * torch.matmul(encodings.t(), flat_input)

            # Laplace smoothing
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )

            # Update embeddings
            self.embeddings.data = self.ema_embeddings / self.ema_cluster_size.unsqueeze(1)

            # Restart unused codes
            if self.restart_unused_codes:
                usage = (self.ema_cluster_size > 1.0).float()
                num_unused = (1 - usage).sum()
                if num_unused > 0:
                    # Sample from input batch for restart
                    random_indices = torch.randperm(flat_input.size(0))[:int(num_unused)]
                    for i, unused_idx in enumerate((usage == 0).nonzero(as_tuple=True)[0]):
                        self.embeddings.data[unused_idx] = flat_input[random_indices[i % len(random_indices)]]
                        self.ema_embeddings.data[unused_idx] = flat_input[random_indices[i % len(random_indices)]]
                        self.ema_cluster_size.data[unused_idx] = 1.0

        # Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Add perplexity calculation
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, encoding_indices.view(input_shape[:-1]), perplexity

class VQ_RFViCMAEEncoder(nn.Module):
    """
    Pure Vector Quantized RF ViC-MAE Encoder.
    Uses only quantized features for reconstruction - no hybrid approach.
    """

    def __init__(
        self,
        signal_length: int = 1024,
        patch_size: int = 32,
        d_model: int = 384,
        nhead: int = 8,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1536,
        mask_ratio: float = 0.5,
        projection_dim: int = 256,
        temperature: float = 0.07,
        dropout: float = 0.1,
        num_embeddings: int = 512,
        vq_commitment_cost: float = 0.25,
        vq_decay: float = 0.99,
    ):
        super().__init__()

        self.signal_length = signal_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.num_embeddings = num_embeddings

        # Number of patches
        self.num_patches = signal_length // patch_size
        assert signal_length % patch_size == 0, f"Signal length {signal_length} must be divisible by patch size {patch_size}"

        # Patch embedding: Conv1d to create tokens from IQ signals
        self.patch_embed = nn.Conv1d(2, d_model, kernel_size=patch_size, stride=patch_size, bias=False)

        # Layer norm after patch embedding
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

        # EMA Vector Quantization layer
        self.vq_layer = EMAVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=d_model,
            commitment_cost=vq_commitment_cost,
            decay=vq_decay
        )

        # Lightweight decoder for reconstruction (works with quantized features only)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward // 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # Decoder prediction head
        self.decoder_pred = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, patch_size * 2),
        )

        # Projection head for contrastive learning (works with quantized features)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False),
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

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking on input tokens.
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
        """Forward pass through encoder with RoPE."""
        # Apply RoPE positional encoding
        x = self.rope(x)

        # Apply transformer encoder
        latent = self.encoder(x)

        return latent

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder - PURE VQ VERSION.
        Uses only quantized features, no mixing with original features.
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
        """Enhanced reconstruction loss for RF signals."""
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
        """Compute InfoNCE contrastive loss between two views."""
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

    def _compute_codebook_usage(self, codes: torch.Tensor) -> float:
        """Compute what fraction of the codebook is being used."""
        unique_codes = torch.unique(codes)
        usage_fraction = len(unique_codes) / self.num_embeddings
        return usage_fraction

    def compute_simple_reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Simplified reconstruction loss for better VQ learning."""
        # Simple MSE loss
        loss = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # [batch, num_patches]

        # Apply mask (only compute loss on masked patches)
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return masked_loss
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Revised Forward pass for VQ ViC-MAE with better learning dynamics.
        """
        batch_size = x_i.shape[0]

        # ===== PATCH CREATION =====
        patches_i = self.patchify(x_i)  # [batch, num_patches, d_model]
        patches_j = self.patchify(x_j)  # [batch, num_patches, d_model]

        # Create proper targets (raw patch version)
        target_i = self.create_patch_targets(x_i)  # [batch, num_patches, patch_size * 2]
        target_j = self.create_patch_targets(x_j)

        # ===== ENCODE ALL PATCHES FIRST =====
        # Apply RoPE and encode ALL patches (not just visible ones)
        patches_i_encoded = self.rope(patches_i)
        patches_j_encoded = self.rope(patches_j)

        latent_i_full = self.encoder(patches_i_encoded)  # [batch, num_patches, d_model]
        latent_j_full = self.encoder(patches_j_encoded)  # [batch, num_patches, d_model]

        # ===== VECTOR QUANTIZATION ON ALL PATCHES =====
        # Updated to handle 4 return values (including perplexity)
        quantized_i_full, vq_loss_i, codes_i_full, perplexity_i = self.vq_layer(latent_i_full)
        quantized_j_full, vq_loss_j, codes_j_full, perplexity_j = self.vq_layer(latent_j_full)

        vq_loss = (vq_loss_i + vq_loss_j) / 2
        avg_perplexity = (perplexity_i + perplexity_j) / 2

        # ===== MASKING AFTER QUANTIZATION =====
        # Now apply masking to quantized features for reconstruction task
        quantized_i_vis, mask_i, ids_restore_i = self.random_masking(quantized_i_full)
        quantized_j_vis, mask_j, ids_restore_j = self.random_masking(quantized_j_full)

        # ===== RECONSTRUCTION =====
        # Decoder uses quantized visible patches to reconstruct full sequence
        pred_i = self.forward_decoder(quantized_i_vis, ids_restore_i)
        pred_j = self.forward_decoder(quantized_j_vis, ids_restore_j)

        # ===== SIMPLIFIED RECONSTRUCTION LOSS =====
        # Use simple MSE loss on masked patches only
        loss_recon_i = self.compute_simple_reconstruction_loss(pred_i, target_i, mask_i)
        loss_recon_j = self.compute_simple_reconstruction_loss(pred_j, target_j, mask_j)
        loss_recon = (loss_recon_i + loss_recon_j) / 2

        # Unpatchify for visualization
        recon_xi = self.unpatchify(pred_i)
        recon_xj = self.unpatchify(pred_j)

        # ===== CONTRASTIVE LEARNING =====
        # Use quantized features for contrastive learning
        # Global pooling over ALL quantized patches (not just visible)
        global_i = quantized_i_full.mean(dim=1)  # [batch, d_model]
        global_j = quantized_j_full.mean(dim=1)  # [batch, d_model]

        # Project to contrastive space
        z_i = self.projection_head(global_i)
        z_j = self.projection_head(global_j)

        # L2 normalize
        emb_i = F.normalize(z_i, p=2, dim=1)
        emb_j = F.normalize(z_j, p=2, dim=1)

        # Compute contrastive loss
        loss_contrast = self.compute_contrastive_loss(emb_i, emb_j)

        # ===== ANALYSIS =====
        codes_i_flat = codes_i_full.view(-1)
        codes_j_flat = codes_j_full.view(-1)
        all_codes = torch.cat([codes_i_flat, codes_j_flat])

        codebook_usage = self._compute_codebook_usage(all_codes)

        # Quantization error (how much information is lost)
        quantization_error_i = F.mse_loss(quantized_i_full, latent_i_full.detach())
        quantization_error_j = F.mse_loss(quantized_j_full, latent_j_full.detach())
        avg_quantization_error = (quantization_error_i + quantization_error_j) / 2

        return {
            # Standard outputs
            'recon_xi': recon_xi,
            'recon_xj': recon_xj,
            'emb_i': emb_i,
            'emb_j': emb_j,
            'loss_recon': loss_recon,
            'loss_contrast': loss_contrast,
            'mask_i': mask_i,
            'mask_j': mask_j,

            # VQ-specific outputs
            'quantized_i': global_i,  # Global quantized features
            'quantized_j': global_j,
            'vq_loss': vq_loss,
            'codes_i': codes_i_flat,
            'codes_j': codes_j_flat,

            # Analysis outputs
            'codebook_usage': codebook_usage,
            'quantization_error': avg_quantization_error,
            'perplexity': avg_perplexity,  # Add perplexity to outputs
            'latent_i': global_i,  # Use quantized features for downstream
            'latent_j': global_j,
        }

    def forward_single_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from a single quantized latent vector."""
        batch_size = latent.shape[0]

        # Expand latent to sequence
        latent_seq = latent.unsqueeze(1).repeat(1, self.num_patches, 1)

        # Apply RoPE
        latent_seq = self.rope(latent_seq)

        # Decode
        decoded = self.decoder(latent_seq)

        # Project to IQ
        pred = self.decoder_pred(decoded)

        # Unpatchify
        reconstructed = self.unpatchify(pred)

        return reconstructed

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single input - returns quantized features."""
        patches = self.patchify(x)
        patches = self.rope(patches)
        latent = self.encoder(patches)

        # Quantize all patches - handle 4 return values
        quantized, _, _, _ = self.vq_layer(latent)

        # Global pooling of quantized features
        global_feat = quantized.mean(dim=1)

        return global_feat

    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Get discrete codes for input signal."""
        patches = self.patchify(x)
        patches = self.rope(patches)
        latent = self.encoder(patches)

        # Handle 4 return values
        _, _, codes, _ = self.vq_layer(latent)

        return codes


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
    Lightning module for Pure VQ ViC-MAE training on RF signals.
    """

    def __init__(
        self,
        label_names: List[str],
        signal_length: int = 1024,
        patch_size: int = 32,
        d_model: int = 384,
        mask_ratio: float = 0.5,
        temperature: float = 0.07,
        learning_rate: float = 1e-4,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        reconstruction_weight: float = 1.0,
        contrastive_weight: float = 2.0,
        vq_weight: float = 0.1,
        classification_weight: float = 1.0,
        num_classes: Optional[int] = None,
        num_embeddings: int = 512,
        vq_commitment_cost: float = 1,
        vq_decay: float = 0.95,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.label_names = label_names
        self.num_classes = num_classes or len(label_names)

        # Pure VQ ViC-MAE encoder
        self.encoder = VQ_RFViCMAEEncoder(
            signal_length=signal_length,
            patch_size=patch_size,
            d_model=d_model,
            mask_ratio=mask_ratio,
            temperature=temperature,
            num_embeddings=num_embeddings,
            vq_commitment_cost=vq_commitment_cost,
            vq_decay=vq_decay,
        )

        # Optional classification head for finetuning
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, self.num_classes),
        ) if num_classes else None

        # Track codebook statistics
        self.codebook_stats = defaultdict(int)
        self.epoch_codes = []

    def training_step(self, batch, batch_idx):
        # Unpack batch
        if len(batch) == 4:
            x_i, x_j, labels, snrs = batch
        elif len(batch) == 3:
            (x_i, x_j), labels, snrs = batch
        else:
            raise ValueError(f"Expected batch with 3 or 4 elements, got {len(batch)}")

        # Forward pass through Pure VQ ViC-MAE
        outputs = self.encoder(x_i, x_j)

        # Extract losses
        loss_recon = outputs['loss_recon']
        loss_contrast = outputs['loss_contrast']
        vq_loss = outputs['vq_loss']

        # Compute total loss
        total_loss = (
            self.hparams.reconstruction_weight * loss_recon +
            self.hparams.contrastive_weight * loss_contrast +
            self.hparams.vq_weight * vq_loss
        )

        # Optional classification loss if classifier exists
        if self.classifier is not None:
            # Use quantized features for classification
            logits_i = self.classifier(outputs['quantized_i'])
            logits_j = self.classifier(outputs['quantized_j'])

            target_labels = labels.squeeze() if labels.dim() > 1 else labels

            class_loss_i = F.cross_entropy(logits_i, target_labels)
            class_loss_j = F.cross_entropy(logits_j, target_labels)
            class_loss = (class_loss_i + class_loss_j) / 2

            total_loss = total_loss + self.hparams.classification_weight * class_loss
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
        self.log('train_vq_loss', vq_loss, prog_bar=True)
        self.log('train_codebook_usage', outputs['codebook_usage'], prog_bar=True)
        self.log('train_quantization_error', outputs['quantization_error'])

        # Collect codes for analysis (sample to avoid memory issues)
        if batch_idx % 10 == 0:
            self.epoch_codes.extend(outputs['codes_i'].detach().cpu().numpy().tolist()[:100])  # Sample first 100
            self.epoch_codes.extend(outputs['codes_j'].detach().cpu().numpy().tolist()[:100])

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
        vq_loss = outputs['vq_loss']

        total_loss = (
            self.hparams.reconstruction_weight * loss_recon +
            self.hparams.contrastive_weight * loss_contrast +
            self.hparams.vq_weight * vq_loss
        )

        # Optional classification
        if self.classifier is not None:
            logits_i = self.classifier(outputs['quantized_i'])
            logits_j = self.classifier(outputs['quantized_j'])

            target_labels = labels.squeeze() if labels.dim() > 1 else labels

            class_loss_i = F.cross_entropy(logits_i, target_labels)
            class_loss_j = F.cross_entropy(logits_j, target_labels)
            class_loss = (class_loss_i + class_loss_j) / 2

            # Accuracy
            acc_i = (logits_i.argmax(dim=1) == target_labels).float().mean()
            acc_j = (logits_j.argmax(dim=1) == target_labels).float().mean()
            acc = (acc_i + acc_j) / 2

            total_loss = total_loss + self.hparams.classification_weight * class_loss
            self.log('val_class_loss', class_loss)
            self.log('val_acc', acc, prog_bar=True)

        # Logging
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_recon_loss', loss_recon)
        self.log('val_contrast_loss', loss_contrast)
        self.log('val_vq_loss', vq_loss)
        self.log('val_codebook_usage', outputs['codebook_usage'])
        self.log('val_quantization_error', outputs['quantization_error'])

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
        """Visualize reconstruction, embeddings, and codebook usage."""
        if hasattr(self, 'val_outputs'):
            self._visualize_reconstruction()
            self._visualize_embeddings()
            self._visualize_codebook_analysis()

        # Reset epoch codes
        self.epoch_codes = []

    def _visualize_reconstruction(self):
        """Visualize pure VQ reconstruction."""
        if not hasattr(self, 'val_outputs'):
            return

        try:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            # Select first sample
            idx = 0

            # Original signals
            orig_i = self.val_x_i[idx].numpy()
            orig_j = self.val_x_j[idx].numpy()

            # Pure VQ reconstructed signals (these exist in pure VQ version)
            recon_i = self.val_outputs['recon_xi'][idx].detach().cpu().numpy()
            recon_j = self.val_outputs['recon_xj'][idx].detach().cpu().numpy()

            # Plot view i
            time = np.arange(orig_i.shape[1])

            axes[0, 0].plot(time, orig_i[0], label='Original I', alpha=0.8, linewidth=2, color='blue')
            axes[0, 0].plot(time, recon_i[0], label='VQ Reconstructed I', alpha=0.8, linewidth=1.5, color='red')
            axes[0, 0].set_title('View i - I channel')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(time, orig_i[1], label='Original Q', alpha=0.8, linewidth=2, color='blue')
            axes[0, 1].plot(time, recon_i[1], label='VQ Reconstructed Q', alpha=0.8, linewidth=1.5, color='red')
            axes[0, 1].set_title('View i - Q channel')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot view j
            axes[1, 0].plot(time, orig_j[0], label='Original I', alpha=0.8, linewidth=2, color='blue')
            axes[1, 0].plot(time, recon_j[0], label='VQ Reconstructed I', alpha=0.8, linewidth=1.5, color='red')
            axes[1, 0].set_title('View j - I channel')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(time, orig_j[1], label='Original Q', alpha=0.8, linewidth=2, color='blue')
            axes[1, 1].plot(time, recon_j[1], label='VQ Reconstructed Q', alpha=0.8, linewidth=1.5, color='red')
            axes[1, 1].set_title('View j - Q channel')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # Constellation diagrams
            axes[0, 2].scatter(orig_i[0], orig_i[1], alpha=0.6, s=15, label='Original', c='blue')
            axes[0, 2].scatter(recon_i[0], recon_i[1], alpha=0.6, s=10, label='VQ Reconstructed', c='red', marker='x')
            axes[0, 2].set_title('View i - Constellation')
            axes[0, 2].set_xlabel('I')
            axes[0, 2].set_ylabel('Q')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            axes[1, 2].scatter(orig_j[0], orig_j[1], alpha=0.6, s=15, label='Original', c='blue')
            axes[1, 2].scatter(recon_j[0], recon_j[1], alpha=0.6, s=10, label='VQ Reconstructed', c='red', marker='x')
            axes[1, 2].set_title('View j - Constellation')
            axes[1, 2].set_xlabel('I')
            axes[1, 2].set_ylabel('Q')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

            # Mask visualization with sample codes
            mask_i = self.val_outputs['mask_i'][idx].detach().cpu().numpy()
            mask_j = self.val_outputs['mask_j'][idx].detach().cpu().numpy()

            # Get sample codes (first few from each view)
            codes_i_sample = self.val_outputs['codes_i'][:5].detach().cpu().numpy()
            codes_j_sample = self.val_outputs['codes_j'][:5].detach().cpu().numpy()

            axes[0, 3].imshow(mask_i.reshape(1, -1), aspect='auto', cmap='binary')
            axes[0, 3].set_title(f'Mask i (ratio: {mask_i.mean():.2f})\nSample codes: {codes_i_sample}')
            axes[0, 3].set_yticks([])

            axes[1, 3].imshow(mask_j.reshape(1, -1), aspect='auto', cmap='binary')
            axes[1, 3].set_title(f'Mask j (ratio: {mask_j.mean():.2f})\nSample codes: {codes_j_sample}')
            axes[1, 3].set_yticks([])

            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"pure_vq_reconstruction": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in reconstruction visualization: {e}")
            plt.close('all')

    def _visualize_embeddings(self):
        """Visualize contrastive embeddings."""
        if not hasattr(self, 'val_outputs'):
            return

        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Get embeddings (quantized)
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
                        s=30
                    )

                axes[0].set_title('t-SNE of Pure VQ Embeddings')
                axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0].grid(True, alpha=0.3)

            # Similarity matrix
            similarity = np.dot(emb_i, emb_j.T)
            im1 = axes[1].imshow(similarity, cmap='viridis', aspect='auto')
            axes[1].set_title('Cosine Similarity Matrix\n(Quantized Features)')
            axes[1].set_xlabel('View j samples')
            axes[1].set_ylabel('View i samples')
            plt.colorbar(im1, ax=axes[1])

            # Quantization error distribution
            quant_error = self.val_outputs['quantization_error']
            if hasattr(quant_error, 'item'):
                quant_error = quant_error.item()

            # Create a simple bar plot for quantization error
            axes[2].bar(['Quantization Error'], [quant_error], color='red', alpha=0.7)
            axes[2].set_title('Average Quantization Error')
            axes[2].set_ylabel('MSE Error')
            axes[2].grid(True, alpha=0.3)

            # Add text with exact value
            axes[2].text(0, quant_error + quant_error*0.1, f'{quant_error:.4f}',
                        ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"pure_vq_embeddings": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in embedding visualization: {e}")
            plt.close('all')

    def _visualize_codebook_analysis(self):
        """Visualize codebook usage and analysis."""
        if not hasattr(self, 'val_outputs'):
            return

        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # Get codebook embeddings - FIXED: use 'embeddings' instead of 'embedding.weight'
            codebook_embeddings = self.encoder.vq_layer.embeddings.detach().cpu().numpy()

            # Get codes from validation batch
            codes_i = self.val_outputs['codes_i'].detach().cpu().numpy()
            codes_j = self.val_outputs['codes_j'].detach().cpu().numpy()
            all_val_codes = np.concatenate([codes_i, codes_j])

            # 1. Codebook usage histogram
            axes[0, 0].hist(all_val_codes, bins=min(50, self.hparams.num_embeddings//20),
                        alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Codebook Usage (Current Batch)')
            axes[0, 0].set_xlabel('Code Index')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Usage statistics
            unique_codes_val = len(np.unique(all_val_codes))
            usage_ratio_val = unique_codes_val / self.hparams.num_embeddings

            # Calculate additional statistics
            code_counts = np.bincount(all_val_codes, minlength=self.hparams.num_embeddings)
            unused_codes = np.sum(code_counts == 0)
            most_used_code = np.argmax(code_counts)
            most_used_count = np.max(code_counts)

            # Get EMA statistics
            ema_cluster_size = self.encoder.vq_layer.ema_cluster_size.detach().cpu().numpy()
            avg_cluster_size = np.mean(ema_cluster_size)
            active_codes_ema = np.sum(ema_cluster_size > 1.0)

            stats_text = f"""Validation Batch Statistics:

    Total codes available: {self.hparams.num_embeddings}
    Unique codes used: {unique_codes_val}
    Unused codes: {unused_codes}
    Usage ratio: {usage_ratio_val:.3f}
    Most used code: {most_used_code} ({most_used_count} times)
    Codebook usage: {self.val_outputs['codebook_usage']:.3f}
    Quantization error: {self.val_outputs['quantization_error']:.4f}
    Perplexity: {self.val_outputs['perplexity']:.2f}

    EMA Statistics:
    Active codes (EMA > 1.0): {active_codes_ema}
    Avg cluster size: {avg_cluster_size:.2f}
    VQ Type: EMA-based
    Commitment cost: {self.encoder.vq_layer.commitment_cost}
    Decay: {self.encoder.vq_layer.decay}
            """

            axes[0, 1].text(0.05, 0.95, stats_text, transform=axes[0, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].axis('off')
            axes[0, 1].set_title('Codebook Statistics')

            # 3. Code distribution by class (if available)
            if hasattr(self, 'val_labels'):
                labels = self.val_labels.numpy()
                unique_labels = np.unique(labels)

                # Get batch size to properly index codes
                batch_size = len(labels)
                codes_i_batch = codes_i[:batch_size] if len(codes_i) >= batch_size else codes_i
                codes_j_batch = codes_j[:batch_size] if len(codes_j) >= batch_size else codes_j

                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    if np.any(mask):
                        # Get codes for this class
                        if len(codes_i_batch) > 0:
                            label_codes_i = codes_i_batch[mask] if len(mask) <= len(codes_i_batch) else codes_i_batch[:np.sum(mask)]
                        else:
                            label_codes_i = []

                        if len(codes_j_batch) > 0:
                            label_codes_j = codes_j_batch[mask] if len(mask) <= len(codes_j_batch) else codes_j_batch[:np.sum(mask)]
                        else:
                            label_codes_j = []

                        # Combine codes
                        label_codes = []
                        if len(label_codes_i) > 0:
                            label_codes.extend(label_codes_i)
                        if len(label_codes_j) > 0:
                            label_codes.extend(label_codes_j)

                        if len(label_codes) > 0:
                            axes[0, 2].hist(label_codes, bins=20, alpha=0.6, color=colors[i],
                                        label=self.label_names[int(label)] if int(label) < len(self.label_names) else f'Class {int(label)}',
                                        density=True)

                axes[0, 2].set_title('Code Distribution by Class')
                axes[0, 2].set_xlabel('Code Index')
                axes[0, 2].set_ylabel('Density')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            else:
                axes[0, 2].text(0.5, 0.5, 'No class labels available',
                            transform=axes[0, 2].transAxes, ha='center', va='center')
                axes[0, 2].set_title('Code Distribution by Class')

            # 4. Epoch-wise code usage (if available)
            if hasattr(self, 'epoch_codes') and len(self.epoch_codes) > 0:
                epoch_codes_array = np.array(self.epoch_codes)
                axes[1, 0].hist(epoch_codes_array, bins=min(50, self.hparams.num_embeddings//20),
                            alpha=0.7, edgecolor='black', color='orange')
                axes[1, 0].set_title('Epoch Code Usage')
                axes[1, 0].set_xlabel('Code Index')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)

                # Add usage statistics
                unique_codes_epoch = len(np.unique(epoch_codes_array))
                usage_ratio_epoch = unique_codes_epoch / self.hparams.num_embeddings
                axes[1, 0].text(0.02, 0.98, f'Epoch unique codes: {unique_codes_epoch}/{self.hparams.num_embeddings}\nEpoch usage ratio: {usage_ratio_epoch:.3f}',
                            transform=axes[1, 0].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[1, 0].text(0.5, 0.5, 'No epoch codes collected yet',
                            transform=axes[1, 0].transAxes, ha='center', va='center')
                axes[1, 0].set_title('Epoch Code Usage')
                axes[1, 0].axis('off')

            # 5. EMA Cluster Size Analysis
            if len(all_val_codes) > 0:
                # Plot EMA cluster sizes
                axes[1, 1].bar(range(len(ema_cluster_size)), ema_cluster_size, alpha=0.7, color='purple')
                axes[1, 1].set_title('EMA Cluster Sizes')
                axes[1, 1].set_xlabel('Code Index')
                axes[1, 1].set_ylabel('EMA Cluster Size')
                axes[1, 1].grid(True, alpha=0.3)

                # Add threshold line for active codes
                axes[1, 1].axhline(y=1.0, color='red', linestyle='--',
                                label='Active threshold (1.0)')
                axes[1, 1].legend()

                # Set reasonable y-axis limit
                max_cluster_size = np.max(ema_cluster_size)
                if max_cluster_size > 0:
                    axes[1, 1].set_ylim(0, max_cluster_size * 1.1)
            else:
                axes[1, 1].text(0.5, 0.5, 'No codes available',
                            transform=axes[1, 1].transAxes, ha='center', va='center')
                axes[1, 1].set_title('EMA Cluster Sizes')

            # 6. Codebook embedding analysis
            if codebook_embeddings.shape[0] > 1:
                # Calculate pairwise distances between codebook entries
                sample_size = min(100, codebook_embeddings.shape[0])
                sample_indices = np.random.choice(codebook_embeddings.shape[0], sample_size, replace=False)
                sample_embeddings = codebook_embeddings[sample_indices]

                # Calculate distances
                distances = np.linalg.norm(
                    sample_embeddings[:, None, :] - sample_embeddings[None, :, :], axis=2
                )

                # Plot distance distribution
                upper_tri_distances = distances[np.triu_indices(distances.shape[0], k=1)]
                axes[1, 2].hist(upper_tri_distances, bins=30, alpha=0.7, color='green', edgecolor='black')
                axes[1, 2].set_title(f'Codebook Distance Distribution\n(Sample of {sample_size} codes)')
                axes[1, 2].set_xlabel('L2 Distance')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].grid(True, alpha=0.3)

                # Add statistics
                mean_dist = np.mean(upper_tri_distances)
                std_dist = np.std(upper_tri_distances)
                axes[1, 2].axvline(mean_dist, color='red', linestyle='--',
                                label=f'Mean: {mean_dist:.3f}')
                axes[1, 2].text(0.02, 0.98, f'Mean dist: {mean_dist:.3f}\nStd dist: {std_dist:.3f}',
                            transform=axes[1, 2].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                axes[1, 2].legend()
            else:
                axes[1, 2].text(0.5, 0.5, 'Insufficient codebook entries',
                            transform=axes[1, 2].transAxes, ha='center', va='center')
                axes[1, 2].set_title('Codebook Distance Distribution')

            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"codebook_analysis": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in codebook visualization: {e}")
            import traceback
            traceback.print_exc()
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

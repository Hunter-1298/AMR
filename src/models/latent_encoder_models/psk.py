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
import math

class ComplexLosses(nn.Module):
    """Complex signal losses for PSK synchronization"""

    def __init__(self):
        super().__init__()

    def complex_mse_loss(self, pred_signal, target_signal):
        """MSE loss in complex domain"""
        # Convert to complex
        pred = torch.complex(pred_signal[:, 0], pred_signal[:, 1])
        target = torch.complex(target_signal[:, 0], target_signal[:, 1])

        pred = pred / (torch.abs(pred) + 1e-8)
        target = target / (torch.abs(target) + 1e-8)

        return torch.mean(torch.abs(pred - target) ** 2)

    def circular_phase_loss(self, pred_signal, target_signal):
        """Cosine-based circular phase difference loss"""
        # Convert to complex
        pred_complex = torch.complex(pred_signal[:, 0], pred_signal[:, 1])
        target_complex = torch.complex(target_signal[:, 0], target_signal[:, 1])

        # Get phases
        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)

        # Cosine loss
        phase_diff = pred_phase - target_phase
        phase_loss = torch.mean(1 - torch.cos(phase_diff))

        return phase_loss

    def magnitude_loss(self, pred_signal, target_signal):
        """Magnitude preservation loss"""
        # Convert to complex
        pred_complex = torch.complex(pred_signal[:, 0], pred_signal[:, 1])
        target_complex = torch.complex(target_signal[:, 0], target_signal[:, 1])

        # Get magnitudes
        pred_mag = torch.abs(pred_complex)
        target_mag = torch.abs(target_complex)

        # MSE on magnitudes
        mag_loss = F.mse_loss(pred_mag, target_mag)

        return mag_loss
class ContrastiveLoss(nn.Module):
    """Contrastive loss for PSK signal pairs"""

    def __init__(self, temperature=0.5, feature_dim=128):  # Increased temperature
        super().__init__()
        self.temperature = temperature

        # Learnable projection head for better features
        self.projection = nn.Sequential(
            nn.Linear(9, 64),  # 9 input features
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )

    def forward(self, synchronized_pairs, labels):
        """
        Args:
            synchronized_pairs: tuple of (sync_signals_i, sync_signals_j)
            labels: modulation type labels
        """
        sync_i, sync_j = synchronized_pairs
        batch_size = sync_i.shape[0]
        device = sync_i.device

        # Extract better features
        features_i = self.extract_psk_specific_features(sync_i, labels)
        features_j = self.extract_psk_specific_features(sync_j, labels)

        # Project features
        features_i = self.projection(features_i)
        features_j = self.projection(features_j)

        # L2 normalize
        features_i = F.normalize(features_i, dim=1)
        features_j = F.normalize(features_j, dim=1)

        # Simple NT-Xent loss (SimCLR style)
        features = torch.cat([features_i, features_j], dim=0)  # [2*batch, dim]
        labels_doubled = torch.cat([labels, labels], dim=0)    # [2*batch]

        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.t()) / self.temperature

        # Mask to remove self-similarities
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))

        # For each anchor, find positive and negative pairs
        loss = 0
        valid_samples = 0

        for i in range(2 * batch_size):
            # Positive mask: same label, different sample
            pos_mask = (labels_doubled == labels_doubled[i]) & ~mask[i]
            # Negative mask: different label
            neg_mask = (labels_doubled != labels_doubled[i])

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                # Get positive similarities
                pos_sim = sim_matrix[i][pos_mask]

                # Get negative similarities
                neg_sim = sim_matrix[i][neg_mask]

                # Compute loss for this anchor
                # log(sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))))
                pos_exp_sum = torch.exp(pos_sim).sum()
                neg_exp_sum = torch.exp(neg_sim).sum()

                loss_i = -torch.log(pos_exp_sum / (pos_exp_sum + neg_exp_sum + 1e-8))
                loss += loss_i
                valid_samples += 1

        if valid_samples > 0:
            loss = loss / valid_samples
        else:
            # If no valid samples, return small loss to avoid NaN
            loss = torch.tensor(0.01, device=device, requires_grad=True)

        return loss

    def extract_psk_specific_features(self, signals, labels):
        """Extract features that specifically distinguish PSK types"""
        batch_size = signals.shape[0]
        device = signals.device

        # Convert to complex
        complex_signals = torch.complex(signals[:, 0], signals[:, 1])

        features = []

        # 1. Phase histogram features (different for each PSK)
        phases = torch.angle(complex_signals)

        # QPSK should have 4 phase clusters, 8PSK has 8, 16PSK has 16
        # Compute phase histogram in different bins
        phase_bins_4 = torch.histc(phases.view(batch_size, -1), bins=4, min=-np.pi, max=np.pi)
        phase_bins_8 = torch.histc(phases.view(batch_size, -1), bins=8, min=-np.pi, max=np.pi)
        phase_bins_16 = torch.histc(phases.view(batch_size, -1), bins=16, min=-np.pi, max=np.pi)

        # Normalize histograms
        phase_entropy_4 = -torch.sum(phase_bins_4 * torch.log(phase_bins_4 + 1e-8), dim=1)
        phase_entropy_8 = -torch.sum(phase_bins_8 * torch.log(phase_bins_8 + 1e-8), dim=1)
        phase_entropy_16 = -torch.sum(phase_bins_16 * torch.log(phase_bins_16 + 1e-8), dim=1)

        features.extend([
            phase_entropy_4.unsqueeze(1),
            phase_entropy_8.unsqueeze(1),
            phase_entropy_16.unsqueeze(1)
        ])

        # 2. Distance to ideal constellations (should be smallest for correct type)
        avg_distances = []
        for n_points in [4, 8, 16]:
            angles = torch.linspace(0, 2*np.pi, n_points+1, device=device)[:-1]

            if n_points == 4:  # QPSK
                ideal_points = torch.exp(1j * (angles + np.pi/4))  # 45° offset
                ideal_points = ideal_points / np.sqrt(2)  # Normalize
            else:
                ideal_points = torch.exp(1j * angles)

            # Compute minimum distance to ideal points
            ideal_points = ideal_points.unsqueeze(0).unsqueeze(0)  # [1, 1, n_points]
            signal_points = complex_signals.unsqueeze(2)  # [batch, length, 1]

            distances = torch.abs(signal_points - ideal_points)  # [batch, length, n_points]
            min_distances = torch.min(distances, dim=2)[0]  # [batch, length]
            avg_distance = torch.mean(min_distances, dim=1)  # [batch]

            avg_distances.append(avg_distance.unsqueeze(1))

        features.extend(avg_distances)

        # 3. Phase transition statistics (different patterns for different PSK)
        phase_diff = torch.diff(phases, dim=1)
        # Wrap phase differences to [-pi, pi]
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

        features.extend([
            torch.mean(torch.abs(phase_diff), dim=1).unsqueeze(1),
            torch.std(phase_diff, dim=1).unsqueeze(1)
        ])

        # 4. Magnitude variance (should be low for good PSK)
        magnitudes = torch.abs(complex_signals)
        features.append(torch.std(magnitudes, dim=1).unsqueeze(1))

        # Stack all features
        feature_tensor = torch.cat(features, dim=1)  # [batch, 9]

        return feature_tensor
class TimeSyncScheduler(nn.Module):
    """Scheduler for time synchronization errors only"""

    def __init__(self, n_steps: int = 1000, max_timing_shift: int = 16, max_phase_shift_deg: float = 45.0):
        super().__init__()
        self.n_steps = n_steps
        self.max_timing_shift = max_timing_shift
        self.max_phase_shift_deg = max_phase_shift_deg

        # Create noise schedule
        betas = torch.linspace(0.0001, 0.02, n_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.n_steps, (batch_size,), device=device)

    def add_sync_errors(self, clean_signal: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add both timing and phase errors.
        Returns: (corrupted_signal, clean_signal, timing_offsets, phase_offsets)
        """
        batch_size = clean_signal.shape[0]
        device = clean_signal.device

        # Scale by noise level
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timestep]

        # --- Timing Offsets ---
        max_shift_t = (self.max_timing_shift * sqrt_one_minus_alpha_t).int()
        timing_offsets = torch.zeros(batch_size, device=device, dtype=torch.int)

        for i in range(batch_size):
            if max_shift_t[i] > 0:
                timing_offsets[i] = torch.randint(-max_shift_t[i], max_shift_t[i] + 1, (1,), device=device)

        # --- Phase Offsets ---
        max_phase_rad = self.max_phase_shift_deg * (3.14159265 / 180.0)
        phase_offsets = (2 * torch.rand(batch_size, device=device) - 1) * max_phase_rad * sqrt_one_minus_alpha_t

        # Apply both corruptions
        corrupted_signal = self.apply_timing_shifts(clean_signal, timing_offsets)
        corrupted_signal = self.apply_phase_shifts(corrupted_signal, phase_offsets)

        return corrupted_signal, clean_signal, timing_offsets, phase_offsets

    def apply_timing_shifts(self, signal: torch.Tensor, timing_offsets: torch.Tensor) -> torch.Tensor:
        batch_size = signal.shape[0]
        shifted_signal = signal.clone()
        for i in range(batch_size):
            shift = timing_offsets[i].item()
            if shift != 0:
                shifted_signal[i] = torch.roll(signal[i], shifts=shift, dims=-1)
        return shifted_signal

    def apply_phase_shifts(self, signal: torch.Tensor, phase_offsets: torch.Tensor) -> torch.Tensor:
        """
        Rotate I/Q pairs by a phase offset: z = x + j y → z' = z * exp(jθ)
        """
        i, q = signal[:, 0], signal[:, 1]
        complex_signal = torch.complex(i, q)

        phase_rotations = torch.exp(1j * phase_offsets).unsqueeze(-1)  # [B, 1]
        rotated_signal = complex_signal * phase_rotations  # [B, L]

        return torch.stack([rotated_signal.real, rotated_signal.imag], dim=1)  # [B, 2,

class TimeSyncDiffusion(nn.Module):
    """Time synchronization using diffusion model"""

    def __init__(self, unet: nn.Module, signal_length: int = 1024, num_diffusion_steps: int = 500):
        super().__init__()
        self.signal_length = signal_length
        self.num_diffusion_steps = num_diffusion_steps

        # UNet for signal-to-signal correction
        self.unet = unet

        # Time sync scheduler
        self.scheduler = TimeSyncScheduler(num_diffusion_steps)

    def forward(self, shifted_signal: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Predict the time-synchronized signal"""
        return self.unet(shifted_signal, timestep)

    def get_latent_features(self, signal: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Extract latent features from UNet for visualization"""
        # This assumes your UNet has a method to get intermediate features
        # You may need to modify based on your UNet architecture
        return self.unet.get_features(signal, timestep)

class PSKDiscriminator(L.LightningModule):
    """Simplified PSK discriminator focusing only on time synchronization"""

    def __init__(
        self,
        unet,
        signal_length: int = 1024,
        learning_rate: float = 1e-3,
        num_diffusion_steps: int = 1000,
        complex_loss_weight: float = 1.0,
        phase_loss_weight: float = 2.0,
        contrastive_weight: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['unet'])
        self.automatic_optimization = False

        self.label_names = ['QPSK', '8PSK', '16PSK']
        self.num_classes = 3

        # Time sync diffusion model
        self.sync_model = TimeSyncDiffusion(unet, signal_length, num_diffusion_steps)

        # Simple classifier for PSK type (for visualization only)
        self.psk_classifier = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=32, stride=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 3)
        )

        # Loss functions
        self.complex_losses = ComplexLosses()
        self.contrastive_loss_fn = ContrastiveLoss(temperature=0.1)

        # Loss weights
        self.complex_loss_weight = complex_loss_weight
        self.phase_loss_weight = phase_loss_weight
        self.contrastive_weight = contrastive_weight
    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        """Forward pass for time synchronization"""
        # Progressive synchronization
        current_signal = x
        intermediates = [] if return_intermediates else None

        # Reverse diffusion process
        sync_steps = list(reversed(range(0, self.sync_model.num_diffusion_steps, 100)))

        for i, t in enumerate(sync_steps):
            t_tensor = torch.full((current_signal.shape[0],), t, device=current_signal.device)

            # Predict synchronized signal
            predicted_signal = self.sync_model(current_signal, t_tensor)

            if return_intermediates and i % 2 == 0:  # Save every other step
                intermediates.append({
                    'signal': predicted_signal.clone(),
                    'timestep': t,
                    'step': i
                })

            current_signal = predicted_signal

        return current_signal, intermediates

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # Extract BOTH signals from the pair
        if len(batch) == 4:
            x_i, x_j, labels, snrs = batch
        elif len(batch) == 3:
            (x_i, x_j), labels, snrs = batch
        elif len(batch) == 2:
            x_i, labels = batch
            x_j = x_i  # Fallback if no pairs
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        labels = labels.long()
        batch_size = x_i.shape[0]
        device = x_i.device

        # Sample timesteps
        timesteps = self.sync_model.scheduler.sample_timesteps(batch_size, device)

        # Add timing errors to BOTH signals in the pair
        shifted_i, clean_i, timing_offsets_i, phase_offsets_i = self.sync_model.scheduler.add_sync_errors(x_i, timesteps)
        shifted_j, clean_j, timing_offsets_j, phase_offsets_j = self.sync_model.scheduler.add_sync_errors(x_j, timesteps)

        # Predict synchronized signals for BOTH
        predicted_i = self.sync_model(shifted_i, timesteps)
        predicted_j = self.sync_model(shifted_j, timesteps)

        # Complex MSE loss
        complex_loss_i = self.complex_losses.complex_mse_loss(predicted_i, clean_i)
        complex_loss_j = self.complex_losses.complex_mse_loss(predicted_j, clean_j)
        complex_loss = (complex_loss_i + complex_loss_j) / 2

        # Phase loss
        phase_loss_i = self.complex_losses.circular_phase_loss(predicted_i, clean_i)
        phase_loss_j = self.complex_losses.circular_phase_loss(predicted_j, clean_j)
        phase_loss = (phase_loss_i + phase_loss_j) / 2

        # Magnitude loss (optional, helps maintain signal power)
        mag_loss_i = self.complex_losses.magnitude_loss(predicted_i, clean_i)
        mag_loss_j = self.complex_losses.magnitude_loss(predicted_j, clean_j)
        mag_loss = (mag_loss_i + mag_loss_j) / 2

        # Contrastive loss on synchronized outputs
        # contrastive_loss = self.contrastive_loss_fn((predicted_i, predicted_j), labels)

        # Combined loss
        # total_loss = (
        #     self.complex_loss_weight * complex_loss +
        #     self.phase_loss_weight * phase_loss +
        #     0.1 * mag_loss +  # Small weight for magnitude
        #     self.contrastive_weight * contrastive_loss
        # )
        total_loss = (
            self.complex_loss_weight * complex_loss +
            self.phase_loss_weight * phase_loss +
            0.1 * mag_loss  # Small weight for magnitude
        )

        # Backprop
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Logging
        self.log('train_complex_loss', complex_loss, prog_bar=True)
        self.log('train_phase_loss', phase_loss, prog_bar=True)
        self.log('train_mag_loss', mag_loss)
        # self.log('train_contrastive_loss', contrastive_loss, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)

        # Log timing offset statistics
        avg_offset = (torch.abs(timing_offsets_i).float().mean() + torch.abs(timing_offsets_j).float().mean()) / 2
        self.log('train_avg_timing_offset', avg_offset)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Extract BOTH signals from the pair
        if len(batch) == 4:
            x_i, x_j, labels, snrs = batch
        elif len(batch) == 3:
            (x_i, x_j), labels, snrs = batch
        elif len(batch) == 2:
            x_i, labels = batch
            x_j = x_i  # Fallback if no pairs
            snrs = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        labels = labels.long()
        batch_size = x_i.shape[0]
        device = x_i.device

        # Sample timesteps
        timesteps = self.sync_model.scheduler.sample_timesteps(batch_size, device)

        # Add timing errors to BOTH signals in the pair
        shifted_i, clean_i, timing_offsets_i, phase_offsets_i = self.sync_model.scheduler.add_sync_errors(x_i, timesteps)
        shifted_j, clean_j, timing_offsets_j, phase_offsets_j = self.sync_model.scheduler.add_sync_errors(x_j, timesteps)

        # Predict synchronized signals for BOTH
        predicted_i = self.sync_model(shifted_i, timesteps)
        predicted_j = self.sync_model(shifted_j, timesteps)

        # Complex MSE loss
        complex_loss_i = self.complex_losses.complex_mse_loss(predicted_i, clean_i)
        complex_loss_j = self.complex_losses.complex_mse_loss(predicted_j, clean_j)
        complex_loss = (complex_loss_i + complex_loss_j) / 2

        # Phase loss
        phase_loss_i = self.complex_losses.circular_phase_loss(predicted_i, clean_i)
        phase_loss_j = self.complex_losses.circular_phase_loss(predicted_j, clean_j)
        phase_loss = (phase_loss_i + phase_loss_j) / 2

        # Magnitude loss
        mag_loss_i = self.complex_losses.magnitude_loss(predicted_i, clean_i)
        mag_loss_j = self.complex_losses.magnitude_loss(predicted_j, clean_j)
        mag_loss = (mag_loss_i + mag_loss_j) / 2

        # Combined loss
        total_loss = (
            self.complex_loss_weight * complex_loss +
            self.phase_loss_weight * phase_loss +
            0.1 * mag_loss
        )

        # Logging
        self.log('val_complex_loss', complex_loss, prog_bar=True)
        self.log('val_phase_loss', phase_loss, prog_bar=True)
        self.log('val_mag_loss', mag_loss)
        self.log('val_loss', total_loss, prog_bar=True)

        # Log timing offset statistics for monitoring
        avg_offset_i = torch.abs(timing_offsets_i).float().mean()
        avg_offset_j = torch.abs(timing_offsets_j).float().mean()
        self.log('val_avg_timing_offset_i', avg_offset_i)
        self.log('val_avg_timing_offset_j', avg_offset_j)

        # Store for visualization (first batch only)
        if batch_idx == 0:
            # Try to get a diverse set of samples with different PSK types
            diverse_indices = []
            labels_np = labels.cpu().numpy()

            # Try to get at least one sample of each PSK type
            for psk_type in [0, 1, 2]:
                type_indices = np.where(labels_np == psk_type)[0]
                if len(type_indices) > 0:
                    diverse_indices.append(type_indices[0])

            # Fill remaining slots with any available samples
            while len(diverse_indices) < 4 and len(diverse_indices) < len(labels):
                for i in range(len(labels)):
                    if i not in diverse_indices:
                        diverse_indices.append(i)
                        if len(diverse_indices) >= 4:
                            break

            # Ensure we have at least some samples
            if len(diverse_indices) == 0:
                diverse_indices = [0, 1, 2, 3] if len(labels) >= 4 else list(range(len(labels)))

            # Take up to 4 diverse samples
            diverse_indices = diverse_indices[:4]

            # Perform full synchronization on diverse samples
            synchronized_i, intermediates_i = self.forward(shifted_i[diverse_indices], return_intermediates=True)
            synchronized_j, intermediates_j = self.forward(shifted_j[diverse_indices], return_intermediates=True)

            # NEW: Also run synchronization on ORIGINAL (undistorted) signals
            with torch.no_grad():
                # Take the original clean signals (no timing errors added)
                clean_originals_i = x_i[diverse_indices]
                clean_originals_j = x_j[diverse_indices]

                # Run through the synchronization process
                sync_from_clean_i, clean_intermediates_i = self.forward(clean_originals_i, return_intermediates=True)
                sync_from_clean_j, clean_intermediates_j = self.forward(clean_originals_j, return_intermediates=True)

            self.val_data = {
                # Store both signal pairs for visualization
                'original_signals_i': x_i[diverse_indices].detach().cpu(),
                'original_signals_j': x_j[diverse_indices].detach().cpu(),
                'shifted_signals_i': shifted_i[diverse_indices].detach().cpu(),
                'shifted_signals_j': shifted_j[diverse_indices].detach().cpu(),
                'synchronized_signals_i': synchronized_i.detach().cpu(),
                'synchronized_signals_j': synchronized_j.detach().cpu(),
                'intermediates_i': intermediates_i,
                'intermediates_j': intermediates_j,
                'labels': labels[diverse_indices].detach().cpu(),
                'timing_offsets_i': timing_offsets_i[diverse_indices].detach().cpu(),
                'timing_offsets_j': timing_offsets_j[diverse_indices].detach().cpu(),
                'phase_offsets_i': phase_offsets_i[diverse_indices].detach().cpu(),
                'phase_offsets_j': phase_offsets_j[diverse_indices].detach().cpu(),
                'timesteps': timesteps[diverse_indices].detach().cpu(),
                'diverse_indices': diverse_indices,
                # NEW: Add synchronization results from clean signals
                'sync_from_clean_i': sync_from_clean_i.detach().cpu(),
                'sync_from_clean_j': sync_from_clean_j.detach().cpu(),
                'clean_intermediates_i': clean_intermediates_i,
                'clean_intermediates_j': clean_intermediates_j
            }

        return total_loss

    def on_validation_epoch_end(self):
        """Create visualization"""
        self._create_sync_visualization()

    def _create_sync_visualization(self):
        """Create simplified visualization: clean sync test + 3x3 grid for PSK types"""
        try:
            fig = plt.figure(figsize=(15, 20))  # Adjusted size for new layout

            # Check if we have validation data
            if not hasattr(self, 'val_data') or self.val_data is None:
                plt.suptitle(f'No validation data available - Epoch {self.current_epoch}', fontsize=16)
                plt.tight_layout()
                if self.logger and hasattr(self.logger, 'experiment'):
                    self.logger.experiment.log({'time_sync_analysis': wandb.Image(fig)})
                plt.close(fig)
                return

            # 1. Clean Signal Synchronization Test (Row 1 - 2 columns only)
            # Column 1: Original clean signal
            ax_clean_orig = plt.subplot(5, 3, 1)
            ax_clean_orig.text(0.5, 0.9, 'Clean Signal Test',
                            ha='center', va='center', transform=ax_clean_orig.transAxes,
                            fontsize=14, fontweight='bold')

            # Show first sample's clean signal
            try:
                sample_idx = 0
                label_idx = self.val_data['labels'][sample_idx].item()
                label_name = self.label_names[label_idx]

                # Get ideal constellation
                if label_idx == 0:  # QPSK
                    ideal_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
                elif label_idx == 1:  # 8PSK
                    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                    ideal_points = np.exp(1j * angles)
                else:  # 16PSK
                    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
                    ideal_points = np.exp(1j * angles)

                # Original clean signal
                orig_signal = self.val_data['original_signals_i'][sample_idx]
                orig_complex = torch.complex(orig_signal[0], orig_signal[1])
                # subsample_orig = orig_complex[::20]
                subsample_orig = orig_complex

                ax_clean_orig.scatter(subsample_orig.real, subsample_orig.imag,
                                    alpha=0.8, s=30, c='blue', label='Clean Original')
                ax_clean_orig.scatter(ideal_points.real, ideal_points.imag,
                                    c='black', s=120, marker='x', linewidth=4, alpha=0.9)

                ax_clean_orig.set_title(f'{label_name} - Clean', fontsize=12, fontweight='bold')
                ax_clean_orig.grid(True, alpha=0.3)
                ax_clean_orig.set_aspect('equal')
                ax_clean_orig.set_xlim(-1.5, 1.5)
                ax_clean_orig.set_ylim(-1.5, 1.5)

            except Exception as e:
                ax_clean_orig.text(0.5, 0.5, f'Error: {str(e)[:20]}',
                                ha='center', va='center', transform=ax_clean_orig.transAxes)

            # Column 2: Synchronized from clean
            ax_clean_sync = plt.subplot(5, 3, 2)
            try:
                # Synchronized from clean signal
                sync_from_clean = self.val_data['sync_from_clean_i'][sample_idx]
                sync_complex = torch.complex(sync_from_clean[0], sync_from_clean[1])
                subsample_sync = sync_complex
                # subsample_sync = sync_complex[::20]

                ax_clean_sync.scatter(subsample_sync.real, subsample_sync.imag,
                                    alpha=0.8, s=30, c='red', label='After Sync')
                ax_clean_sync.scatter(ideal_points.real, ideal_points.imag,
                                    c='black', s=120, marker='x', linewidth=4, alpha=0.9)

                # Compute MSE between original and "synchronized" clean
                orig_clean = self.val_data['original_signals_i'][sample_idx].numpy()
                sync_clean = self.val_data['sync_from_clean_i'][sample_idx].numpy()
                mse = np.mean((orig_clean - sync_clean) ** 2)

                ax_clean_sync.text(0.02, 0.98, f'MSE: {mse:.5f}',
                                transform=ax_clean_sync.transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                fontsize=10)

                ax_clean_sync.set_title(f'{label_name} - After Sync', fontsize=12, fontweight='bold')
                ax_clean_sync.grid(True, alpha=0.3)
                ax_clean_sync.set_aspect('equal')
                ax_clean_sync.set_xlim(-1.5, 1.5)
                ax_clean_sync.set_ylim(-1.5, 1.5)

            except Exception as e:
                ax_clean_sync.text(0.5, 0.5, f'Error: {str(e)[:20]}',
                                ha='center', va='center', transform=ax_clean_sync.transAxes)

            # 2. Corrupted Signal Synchronization (Rows 2-4, 3 columns each)
            # Create 3x3 grid for each PSK type showing: Original → Corrupted → Synchronized
            psk_sample_map = {}
            for i, label in enumerate(self.val_data['labels']):
                label_val = label.item()
                if label_val not in psk_sample_map and label_val in [0, 1, 2]:
                    psk_sample_map[label_val] = i

            # Ensure we have all PSK types represented
            for psk_type in [0, 1, 2]:
                if psk_type not in psk_sample_map:
                    if len(psk_sample_map) > 0:
                        psk_sample_map[psk_type] = list(psk_sample_map.values())[0]
                    else:
                        psk_sample_map[psk_type] = 0

            for row_idx, psk_type in enumerate([0, 1, 2]):  # QPSK, 8PSK, 16PSK
                sample_idx = psk_sample_map[psk_type]
                label_name = self.label_names[psk_type]

                # Get timing and phase offsets for display
                timing_offset = self.val_data['timing_offsets_i'][sample_idx].item()
                phase_offset = self.val_data['phase_offsets_i'][sample_idx].item()
                phase_offset_deg = phase_offset * 180 / np.pi

                # Get ideal constellation points
                if psk_type == 0:  # QPSK
                    ideal_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
                elif psk_type == 1:  # 8PSK
                    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                    ideal_points = np.exp(1j * angles)
                else:  # 16PSK
                    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
                    ideal_points = np.exp(1j * angles)

                # Row for this PSK type (rows 2, 3, 4)
                current_row = row_idx + 2

                # Column 1: Original signal
                ax_orig = plt.subplot(5, 3, (current_row - 1) * 3 + 1)
                try:
                    orig_signal = self.val_data['original_signals_i'][sample_idx]
                    orig_complex = torch.complex(orig_signal[0], orig_signal[1])
                    subsample = orig_complex
                    # subsample = orig_complex[::20]

                    ax_orig.scatter(subsample.real, subsample.imag, alpha=0.7, s=25, c='blue')
                    ax_orig.scatter(ideal_points.real, ideal_points.imag,
                                c='black', s=120, marker='x', linewidth=4, alpha=0.9)
                except Exception as e:
                    ax_orig.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_orig.transAxes)

                if row_idx == 0:
                    ax_orig.set_title('Original', fontsize=12, fontweight='bold')

                ax_orig.set_ylabel(f'{label_name}', fontsize=12, fontweight='bold')
                ax_orig.grid(True, alpha=0.3)
                ax_orig.set_aspect('equal')
                ax_orig.set_xlim(-1.5, 1.5)
                ax_orig.set_ylim(-1.5, 1.5)

                # Column 2: Corrupted signal (timing + phase errors)
                ax_corrupted = plt.subplot(5, 3, (current_row - 1) * 3 + 2)
                try:
                    corrupted_signal = self.val_data['shifted_signals_i'][sample_idx]
                    corrupted_complex = torch.complex(corrupted_signal[0], corrupted_signal[1])
                    subsample_corrupted = corrupted_complex
                    # subsample_corrupted = corrupted_complex[::20]

                    ax_corrupted.scatter(subsample_corrupted.real, subsample_corrupted.imag,
                                    alpha=0.7, s=25, c='red')
                    ax_corrupted.scatter(ideal_points.real, ideal_points.imag,
                                    c='black', s=120, marker='x', linewidth=4, alpha=0.9)
                except Exception as e:
                    ax_corrupted.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_corrupted.transAxes)

                if row_idx == 0:
                    ax_corrupted.set_title('Corrupted', fontsize=12, fontweight='bold')

                # Add corruption info
                ax_corrupted.text(0.02, 0.98, f'Time: {timing_offset}\nPhase: {phase_offset_deg:.1f}°',
                                transform=ax_corrupted.transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                fontsize=8)

                ax_corrupted.grid(True, alpha=0.3)
                ax_corrupted.set_aspect('equal')
                ax_corrupted.set_xlim(-1.5, 1.5)
                ax_corrupted.set_ylim(-1.5, 1.5)

                # Column 3: Synchronized signal
                ax_sync = plt.subplot(5, 3, (current_row - 1) * 3 + 3)
                try:
                    sync_signal = self.val_data['synchronized_signals_i'][sample_idx]
                    sync_complex = torch.complex(sync_signal[0], sync_signal[1])
                    # subsample_sync = sync_complex[::20]
                    subsample_sync = sync_complex

                    ax_sync.scatter(subsample_sync.real, subsample_sync.imag,
                                alpha=0.7, s=25, c='green')
                    ax_sync.scatter(ideal_points.real, ideal_points.imag,
                                c='black', s=120, marker='x', linewidth=4, alpha=0.9)

                    # Compute reconstruction quality
                    orig_np = self.val_data['original_signals_i'][sample_idx].numpy()
                    sync_np = sync_signal.numpy()
                    recon_mse = np.mean((orig_np - sync_np) ** 2)

                    ax_sync.text(0.02, 0.98, f'Recon MSE:\n{recon_mse:.4f}',
                            transform=ax_sync.transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=8)

                except Exception as e:
                    ax_sync.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax_sync.transAxes)

                if row_idx == 0:
                    ax_sync.set_title('Synchronized', fontsize=12, fontweight='bold')

                ax_sync.grid(True, alpha=0.3)
                ax_sync.set_aspect('equal')
                ax_sync.set_xlim(-1.5, 1.5)
                ax_sync.set_ylim(-1.5, 1.5)
            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({'time_sync_analysis': wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()

            # Create a simple error plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f'Visualization Error: {str(e)}',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Visualization Error - Epoch {self.current_epoch}')

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({'time_sync_analysis': wandb.Image(fig)})

            plt.close(fig)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

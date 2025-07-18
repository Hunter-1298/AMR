import torch
import umap
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import tempfile  # Add this missing import
import pytorch_lightning as pl
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
from torch.optim import AdamW
from typing import Dict, Tuple, Optional, List
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import math
import seaborn as sns

class DDPMScheduler(nn.Module):
    """Standard DDPM scheduler for AWGN noise"""

    def __init__(
        self,
        n_steps: int = 10,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        scale_factor = 0.3,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.n_steps = n_steps

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule == "cosine":
            # Cosine schedule (typically better)
            s = 0.008
            steps = n_steps + 1
            x = torch.linspace(0, n_steps, steps)
            alphas_cumprod = (
                torch.cos(((x / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = scale_factor * betas
            betas = torch.clamp(betas, 1e-8, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Store as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod)
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.n_steps, (batch_size,), device=device)

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add AWGN noise according to DDPM schedule"""
        noise = torch.randn_like(x)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1
        )

        noisy_signal = (
            sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        )

        return noisy_signal, noise

class SelfConditioningDiffusionWithClassifier(L.LightningModule):
    """
    Complete PSK diffusion denoiser with integrated classification
    JOINT TRAINING with step-wise corrections and all improvements
    """
    def __init__(
        self,
        unet: nn.Module,
        label_names: Optional[List[str]] = None,
        learning_rate: float = 1e-4,
        num_train_timesteps: int = 10,

        # Classifier parameters
        num_classes: int = 3,
        conv_hidden: tuple = (32,64,128),
        trans_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        mlp_hidden: int = 256,

        # Loss weights for joint training
        # lambda_sync: float = 5.0,      # Synchronization loss weight
        lambda_sync: float = 0.0,      # Synchronization loss weight
        lambda_class: float = 1.0,     # Classification loss weight

        # Training schedule
        classification_start_epoch: int = 0,  # Start classification after sync has some progress

        # Normalization parameters
        sps: int = 8,
        max_freq_offset: float = 1e-3,
        normalize_params: bool = True,

        # Visualization settings
        log_every_n_epochs: int = 1,
        vis_batch_size: int = 5,

        # Other settings
        use_scheduler: bool = True,
        interpolation_type: str = "cosine",  # Keep cosine for better step distribution
        noise_regularization: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['unet'])

        # Store the pre-instantiated UNet
        self.model = unet
        if label_names is not None:
            # Create ordered list: [label_names[0], label_names[1], label_names[2], ...]
            max_class = max(label_names.keys())
            self.label_names = [label_names.get(i, f"Class_{i}") for i in range(max_class + 1)]
        else:
            self.label_names = [f"Class_{i}" for i in range(num_classes)]

        # === INSTANTIATE CLASSIFIER ===
        self.classifier = HybridConvTransformer(
            in_ch=2,  # I/Q channels
            num_classes=num_classes,
            conv_hidden=conv_hidden,
            trans_dim=trans_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_hidden=mlp_hidden,
        )

        # Classification loss function
        self.classification_criterion = nn.CrossEntropyLoss()

        # Store validation samples for visualization
        self.val_samples_stored = False
        self.stored_val_data = None

        # Store validation outputs for SNR analysis
        self.validation_step_outputs = []

    def normalize_sync_params(self, timing, freq, phase):
        """Normalize synchronization parameters for training"""
        if not self.hparams.normalize_params:
            return timing, freq, phase

        norm_timing = timing / self.hparams.sps
        norm_freq = freq / self.hparams.max_freq_offset
        norm_phase = phase / torch.pi

        return norm_timing, norm_freq, norm_phase

    def denormalize_sync_params(self, norm_timing, norm_freq, norm_phase):
        """Convert normalized parameters back to physical units"""
        if not self.hparams.normalize_params:
            return norm_timing, norm_freq, norm_phase

        timing = norm_timing * self.hparams.sps
        freq = norm_freq * self.hparams.max_freq_offset
        phase = norm_phase * torch.pi

        return timing, freq, phase

    def get_interpolation_alpha(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get interpolation weights based on timesteps"""
        alpha = timesteps.float() / (self.hparams.num_train_timesteps - 1)

        if self.hparams.interpolation_type == "linear":
            return alpha
        elif self.hparams.interpolation_type == "cosine":
            return 0.5 * (1 - torch.cos(alpha * torch.pi))
        else:
            return alpha

    def create_sync_interpolation(
        self,
        sync_signals: torch.Tensor,
        unsync_signals: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Create interpolation between synchronized and unsynchronized signals"""
        sync_signals = sync_signals.float()
        unsync_signals = unsync_signals.float()

        alpha = self.get_interpolation_alpha(timesteps)
        alpha = alpha.view(-1, 1, 1).float()

        # At t=0: fully synchronized (alpha=0)
        # At t=19: fully unsynchronized (alpha=1)
        interpolated = (1 - alpha) * sync_signals + alpha * unsync_signals

        if self.training and self.hparams.noise_regularization > 0:
            noise_scale = self.hparams.noise_regularization * (1 - alpha + 0.1)
            noise = torch.randn_like(interpolated) * noise_scale
            interpolated = interpolated + noise

        return interpolated.float()

    def should_do_classification(self) -> bool:
        """Determine if we should include classification loss"""
        return self.current_epoch >= self.hparams.classification_start_epoch

    def training_step(self, batch, batch_idx):
        """Joint training with step-wise sync corrections + classification on fully synchronized signals"""
        sync_signals, unsync_signals, labels, snrs, sync_params = batch
        batch_size = sync_signals.shape[0]

        # === NORMALIZE SYNC PARAMETERS ===
        true_timing, true_freq, true_phase = sync_params
        norm_true_timing, norm_true_freq, norm_true_phase = self.normalize_sync_params(
            true_timing, true_freq, true_phase
        )

        # === SYNCHRONIZATION LOSS (STEP-WISE) ===
        timesteps = torch.randint(0, self.hparams.num_train_timesteps, (batch_size,), device=self.device)
        interpolated_signals = self.create_sync_interpolation(sync_signals, unsync_signals, timesteps)

        sync_output = self.model(interpolated_signals, timesteps, labels, return_dict=True)

        # STEP-WISE TARGETS: How much to correct to get to the next step
        alpha_current = self.get_interpolation_alpha(timesteps).to(self.device)
        next_timesteps = torch.clamp(timesteps - 1, 0, self.hparams.num_train_timesteps - 1)
        alpha_next = self.get_interpolation_alpha(next_timesteps).to(self.device)

        # For t=0, step should be zero (already at target)
        step_alpha = torch.where(timesteps == 0,
                                torch.zeros_like(alpha_current),
                                alpha_current - alpha_next)

        target_timing = norm_true_timing * step_alpha
        target_freq = norm_true_freq * step_alpha
        target_phase = norm_true_phase * step_alpha

        # Synchronization loss
        timing_mse = F.mse_loss(sync_output['timing_offset'].squeeze(), target_timing)
        freq_mse = F.mse_loss(sync_output['freq_offset'].squeeze(), target_freq)
        phase_mse = F.mse_loss(sync_output['phase_offset'].squeeze(), target_phase)
        sync_loss = timing_mse + freq_mse + phase_mse

        # === CLASSIFICATION LOSS (FULLY SYNCHRONIZED SIGNALS) ===
        classification_loss = torch.tensor(0.0, device=self.device)
        classification_acc = torch.tensor(0.0, device=self.device)

        # Run FULL iterative synchronization for classification training
        fully_synchronized_signals, _ = self.iterative_synchronization_for_gif(
            unsync_signals, labels, num_steps=10
        )

        # Train classifier on FULLY synchronized signals
        class_logits = self.classifier(fully_synchronized_signals)
        # class_logits = self.classifier(unsync_signals)
        classification_loss = self.classification_criterion(class_logits, labels)

        # Calculate accuracy
        class_preds = torch.argmax(class_logits, dim=1)
        classification_acc = (class_preds == labels).float().mean()

        # === TOTAL LOSS ===
        total_loss = (
            self.hparams.lambda_sync * sync_loss +
            self.hparams.lambda_class * classification_loss
        )

        # === ENHANCED LOGGING ===
        self.log('train/sync_loss', sync_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/timing_mse', timing_mse, on_step=True)
        self.log('train/freq_mse', freq_mse, on_step=True)
        self.log('train/phase_mse', phase_mse, on_step=True)
        self.log('train/class_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train/class_acc', classification_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True)

        # Log step-wise behavior
        avg_step_timing = torch.mean(torch.abs(target_timing))
        avg_step_freq = torch.mean(torch.abs(target_freq))
        avg_step_phase = torch.mean(torch.abs(target_phase))

        self.log('train/avg_step_timing', avg_step_timing)
        self.log('train/avg_step_freq', avg_step_freq)
        self.log('train/avg_step_phase', avg_step_phase)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation with step-wise corrections"""
        sync_signals, unsync_signals, labels, snrs, sync_params = batch
        batch_size = sync_signals.shape[0]

        # === NORMALIZE SYNC PARAMETERS ===
        true_timing, true_freq, true_phase = sync_params
        norm_true_timing, norm_true_freq, norm_true_phase = self.normalize_sync_params(
            true_timing, true_freq, true_phase
        )

        # # Store first batch for visualization
        if batch_idx == 0 and not self.val_samples_stored:
            self.stored_val_data = {
                'sync_signals': sync_signals[:self.hparams.vis_batch_size].cpu(),
                'unsync_signals': unsync_signals[:self.hparams.vis_batch_size].cpu(),
                'snrs': snrs[:self.hparams.vis_batch_size].cpu(),
                'labels': labels[:self.hparams.vis_batch_size].cpu(),
                'sync_params': (
                    norm_true_timing[:self.hparams.vis_batch_size].cpu(),
                    norm_true_freq[:self.hparams.vis_batch_size].cpu(),
                    norm_true_phase[:self.hparams.vis_batch_size].cpu(),
                )
            }
            self.val_samples_stored = True

        with torch.no_grad():
            # === SYNCHRONIZATION LOSS (at t=0 for best case) ===
            timesteps = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
            interpolated_signals = self.create_sync_interpolation(sync_signals, unsync_signals, timesteps)

            sync_output = self.model(interpolated_signals, timesteps, labels, return_dict=True)

            # Step-wise sync loss (for t=0, step_alpha should be 0)
            alpha_current = self.get_interpolation_alpha(timesteps).to(self.device)
            next_timesteps = torch.clamp(timesteps - 1, 0, self.hparams.num_train_timesteps - 1)
            alpha_next = self.get_interpolation_alpha(next_timesteps).to(self.device)
            step_alpha = torch.where(timesteps == 0, torch.zeros_like(alpha_current), alpha_current - alpha_next)

            target_timing = norm_true_timing * step_alpha
            target_freq = norm_true_freq * step_alpha
            target_phase = norm_true_phase * step_alpha

            sync_loss = (
                F.mse_loss(sync_output['timing_offset'].squeeze(), target_timing) +
                F.mse_loss(sync_output['freq_offset'].squeeze(), target_freq) +
                F.mse_loss(sync_output['phase_offset'].squeeze(), target_phase)
            )

            # === CLASSIFICATION ON FULLY SYNCHRONIZED SIGNALS ===
            fully_synchronized_signals, progression = self.iterative_synchronization_for_gif(
                unsync_signals, labels, num_steps=10
            )

            class_logits = self.classifier(fully_synchronized_signals)
            class_loss = self.classification_criterion(class_logits, labels)
            class_preds = torch.argmax(class_logits, dim=1)
            class_acc = (class_preds == labels).float().mean()
            class_probs = F.softmax(class_logits, dim=-1)

            # Store outputs for SNR analysis
            self.validation_step_outputs.append({
                'predictions': class_preds.cpu(),
                'labels': labels.cpu(),
                'snrs': snrs.cpu(),
                'probabilities': class_probs.cpu(),
                'sync_loss': sync_loss.cpu(),
            })

        # Log metrics
        self.log('val/sync_loss', sync_loss, on_epoch=True, prog_bar=True)
        self.log('val/class_loss', class_loss, on_epoch=True)
        self.log('val/class_acc', class_acc, on_epoch=True, prog_bar=True)

        # Log per-class accuracy
        for i, label_name in enumerate(self.label_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_acc_per_class = (class_preds[class_mask] == labels[class_mask]).float().mean()
                self.log(f'val/class_acc_{label_name}', class_acc_per_class)

        # Debug timestep response for first batch
        if batch_idx == 0:
            self.debug_timestep_conditioning_stepwise(unsync_signals[:1], labels[:1])

        return sync_loss + class_loss

    @torch.no_grad()
    def iterative_synchronization_for_gif(
        self,
        unsync_signal: torch.Tensor,
        modulation: torch.Tensor,
        num_steps: int = 10
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Iterative synchronization with cumulative timing tracking for visualization"""
        device = unsync_signal.device
        current_signal = unsync_signal.clone()

        progression = []
        cumulative_timing = torch.zeros(current_signal.shape[0], device=device)  # Track cumulative timing

        # # Reverse diffusion schedule
        timesteps = torch.linspace(
            self.hparams.num_train_timesteps - 1, 0, num_steps
        ).long().to(device)

        for step, t in enumerate(timesteps):
            t_batch = torch.full((current_signal.shape[0],), t, device=device)

            # Get model prediction
            output = self.model(current_signal, t_batch, modulation, return_dict=True)

            # Apply step-wise corrections to current signal
            corrected_signal = self.apply_predicted_sync(
                current_signal,
                output['timing_offset'],
                output['freq_offset'],
                output['phase_offset']
            )

            # Update cumulative timing (denormalized for tracking)
            timing_physical, freq_physical, phase_physical = self.denormalize_sync_params(
                output['timing_offset'].squeeze(-1),
                output['freq_offset'].squeeze(-1),
                output['phase_offset'].squeeze(-1)
            )
            cumulative_timing += timing_physical

            # Get classification for this step
            if hasattr(self, 'classifier'):
                class_logits = self.classifier(corrected_signal)
                class_probs = F.softmax(class_logits, dim=-1)
                class_preds = torch.argmax(class_logits, dim=1)
            else:
                class_probs = torch.zeros((corrected_signal.shape[0], self.hparams.num_classes))
                class_preds = torch.zeros((corrected_signal.shape[0],), dtype=torch.long)

            # Store step info with cumulative timing tracking
            step_info = {
                'step': step,
                'timestep': t.item(),
                'signal': corrected_signal.cpu().clone(),
                'input_signal': current_signal.cpu().clone(),
                'timing_offset': output['timing_offset'].cpu().clone(),
                'freq_offset': output['freq_offset'].cpu().clone(),
                'phase_offset': output['phase_offset'].cpu().clone(),
                'cumulative_timing_offset': cumulative_timing.cpu().clone(),  # Track cumulative timing
                'cumulative_freq_offset': torch.zeros_like(cumulative_timing.cpu()),  # Placeholder for future
                'cumulative_phase_offset': torch.zeros_like(cumulative_timing.cpu()),  # Placeholder for future
                'class_probs': class_probs.cpu().clone(),
                'class_preds': class_preds.cpu().clone(),
            }

            progression.append(step_info)

            # Update for next iteration
            current_signal = corrected_signal.detach()

        return current_signal, progression

    def apply_predicted_sync(
        self,
        signal: torch.Tensor,
        timing_offset: torch.Tensor,
        freq_offset: torch.Tensor,
        phase_offset: torch.Tensor,
    ) -> torch.Tensor:
        """Apply predicted synchronization with fractional timing correction"""
        batch_size, channels, length = signal.shape
        device = signal.device

        assert channels == 2, f"Expected 2 channels (I/Q), got {channels}"

        # Denormalize parameters to physical units
        timing_physical, freq_physical, phase_physical = self.denormalize_sync_params(
            timing_offset.squeeze(-1), freq_offset.squeeze(-1), phase_offset.squeeze(-1)
        )

        # Convert I/Q to complex
        complex_signal = signal[:, 0] + 1j * signal[:, 1]

        # Apply fractional timing correction
        timing_corrected = self.apply_fractional_timing_correction_freq_domain(
            complex_signal, timing_physical
        )

        # Apply frequency and phase correction (FIXED: removed /length)
        t = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(0)
        freq_phase_correction = (
            -2 * torch.pi * freq_physical.unsqueeze(-1) * t -  # Removed /length division
            phase_physical.unsqueeze(-1)
        )
        correction_phasor = torch.exp(1j * freq_phase_correction)

        fully_corrected = timing_corrected * correction_phasor

        # Convert back to I/Q
        output_signal = torch.stack([
            fully_corrected.real,
            fully_corrected.imag
        ], dim=1)

        return output_signal.float()

    def apply_fractional_timing_correction_freq_domain(
        self,
        complex_signal: torch.Tensor,
        timing_offset_samples: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized frequency domain fractional timing correction"""
        batch_size, length = complex_signal.shape
        device = complex_signal.device

        # FFT
        signal_fft = torch.fft.fft(complex_signal, dim=-1)

        # Frequency vector
        freqs = torch.fft.fftfreq(length, device=device).unsqueeze(0)

        # Phase shifts for fractional delays
        phase_shifts = torch.exp(
            -1j * 2 * torch.pi * freqs * timing_offset_samples.unsqueeze(-1)
        )

        # Apply phase shifts in frequency domain
        delayed_fft = signal_fft * phase_shifts

        # IFFT back to time domain
        delayed_signals = torch.fft.ifft(delayed_fft, dim=-1)

        return delayed_signals

    @torch.no_grad()
    def debug_timestep_conditioning_stepwise(self, unsync_signal: torch.Tensor, modulation: torch.Tensor):
        """Debug function to check step-wise timestep response"""
        device = unsync_signal.device

        print("\nStep-wise Timestep Response Debug:")
        print("Expected: step sizes should decrease as timestep decreases")
        print("-" * 80)

        for t in [9, 7, 5 ,2,0]:
            t_batch = torch.full((1,), t, device=device)

            output = self.model(unsync_signal, t_batch, modulation, return_dict=True)

            alpha_current = self.get_interpolation_alpha(torch.tensor([t])).item()
            next_t = max(0, t - 1)
            alpha_next = self.get_interpolation_alpha(torch.tensor([next_t])).item()
            step_alpha = alpha_current - alpha_next if t > 0 else 0.0

            print(f"t={t:2d} (α_curr={alpha_current:.3f}, α_next={alpha_next:.3f}, step={step_alpha:.3f}):")
            print(f"    pred_norm: timing={output['timing_offset'][0,0].item():7.4f}, "
                  f"freq={output['freq_offset'][0,0].item():7.4f}, "
                  f"phase={output['phase_offset'][0,0].item():7.4f}")

            # Show denormalized values
            timing_phys, freq_phys, phase_phys = self.denormalize_sync_params(
                output['timing_offset'][0,0],
                output['freq_offset'][0,0],
                output['phase_offset'][0,0]
            )
            print(f"    pred_phys: timing={timing_phys.item():7.4f}, "
                  f"freq={freq_phys.item():9.6f}, "
                  f"phase={phase_phys.item():7.4f}")

        print("-" * 80)


    def create_snr_vs_accuracy_plot(self):
        """Create SNR vs Classification Accuracy plot"""
        if not self.validation_step_outputs:
            print("No validation outputs available for SNR analysis")
            return

        try:
            # Aggregate all validation outputs
            all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
            all_snrs = torch.cat([x['snrs'] for x in self.validation_step_outputs])

            # Define SNR bins - UPDATED to cover -20dB to 30dB range
            snr_bins = [(-20, -18), (-18, -16), (-16, -14), (-14, -12), (-12, -10),
                    (-10, -8), (-8, -6), (-6, -4), (-4, -2), (-2, 0),
                    (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12),
                    (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
                    (22, 24), (24, 26), (26, 28), (28, 30)]
            snr_centers = [(low + high) / 2 for low, high in snr_bins]

            # Calculate overall accuracy per SNR bin
            overall_accuracies = []
            class_accuracies = {label_name: [] for label_name in self.label_names}

            for snr_min, snr_max in snr_bins:
                snr_mask = (all_snrs >= snr_min) & (all_snrs < snr_max)

                if snr_mask.sum() > 0:
                    # Overall accuracy
                    overall_acc = (all_preds[snr_mask] == all_labels[snr_mask]).float().mean().item()
                    overall_accuracies.append(overall_acc)

                    # Per-class accuracy
                    for class_idx, label_name in enumerate(self.label_names):
                        class_mask = snr_mask & (all_labels == class_idx)
                        if class_mask.sum() > 0:
                            class_acc = (all_preds[class_mask] == all_labels[class_mask]).float().mean().item()
                            class_accuracies[label_name].append(class_acc)
                        else:
                            class_accuracies[label_name].append(0.0)
                else:
                    overall_accuracies.append(0.0)
                    for label_name in self.label_names:
                        class_accuracies[label_name].append(0.0)

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot overall accuracy
            ax.plot(snr_centers, overall_accuracies, 'ko-', linewidth=3, markersize=8,
                label='Overall Accuracy', zorder=3)

            colors = ['red', 'blue', 'green', 'orange', 'purple']
            markers = ['s', '^', 'D', 'v', '<']

            for idx, (label_name, accuracies) in enumerate(class_accuracies.items()):
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                ax.plot(snr_centers, accuracies, color=color, marker=marker,
                    linewidth=2, markersize=6, label=str(label_name), alpha=0.8)

            # Add only the random guess baseline reference line
            num_classes = len(self.label_names)
            ax.axhline(y=1/num_classes, color='gray', linestyle='--', alpha=0.5,
                    label=f'Random Guess ({1/num_classes:.3f})')

            # Formatting with updated axis limits
            ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Classification Accuracy', fontsize=14, fontweight='bold')
            ax.set_title(f'Classification Accuracy vs SNR - Epoch {self.current_epoch}\n'
                        f'Synchronized Signals (Joint Training)', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='lower right', frameon=True, fancybox=True, shadow=True)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(-22, 32)  # UPDATED range

            # Add sample count annotations
            for i, (snr_min, snr_max) in enumerate(snr_bins):
                snr_mask = (all_snrs >= snr_min) & (all_snrs < snr_max)
                count = snr_mask.sum().item()
                if count > 0:
                    ax.annotate(f'n={count}', (snr_centers[i], 0.02),
                            ha='center', fontsize=8, alpha=0.7)

            # Add performance statistics text box
            max_acc = max(overall_accuracies)
            max_snr_idx = overall_accuracies.index(max_acc)
            max_snr = snr_centers[max_snr_idx]

            stats_text = f'Peak Accuracy: {max_acc:.3f} @ {max_snr:.1f}dB\n'
            stats_text += f'Samples: {len(all_preds)} total\n'
            stats_text += f'Classes: {num_classes}'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()

            # Log to wandb
            if hasattr(self, 'logger') and self.logger is not None:
                logger_class_name = self.logger.__class__.__name__
                if 'WandbLogger' in logger_class_name:
                    self.logger.experiment.log({
                        "snr_vs_accuracy": wandb.Image(fig),
                        "epoch": self.current_epoch
                    })
                    print("SNR vs Accuracy plot logged to WandB")

            # Save locally
            plot_filename = f'snr_vs_accuracy_epoch_{self.current_epoch}.png'
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"SNR vs Accuracy plot saved as {plot_filename}")

            plt.close(fig)

        except Exception as e:
            print(f"Error creating SNR vs Accuracy plot: {e}")
            import traceback
            traceback.print_exc()

    def create_sync_animation_with_classification(
        self,
        unsync_signals: torch.Tensor,
        progression: List[Dict],
        labels: torch.Tensor,
        snrs: torch.Tensor,
        num_samples: int = 4
    ):
        """Enhanced animation showing synchronization progression + classification confidence with corrected symbol tracking"""
        try:
            num_samples = min(num_samples, unsync_signals.shape[0])
            num_steps = len(progression)

            if num_steps == 0:
                print("Error: No progression data available")
                return None, None

            # Set up figure: constellation plots, parameter evolution, and classification confidence
            fig = plt.figure(figsize=(6*num_samples, 14))
            gs = fig.add_gridspec(3, num_samples, height_ratios=[1, 1, 1],
                                hspace=0.4, wspace=0.3)

            # Store plots for animation
            scatters = []
            param_lines = []

            # Base symbol stride and calculate corrected indices for each step
            symbol_stride = 8
            signal_length = unsync_signals.shape[2]
            base_symbol_indices = torch.arange(0, signal_length, symbol_stride)

            # Pre-calculate corrected symbol indices for each step and sample
            corrected_indices_per_step = []
            for step_idx in range(num_steps):
                step_indices = []
                for sample_idx in range(num_samples):
                    # Get cumulative timing correction for this step and sample
                    if 'cumulative_timing_offset' in progression[step_idx]:
                        cumulative_timing = progression[step_idx]['cumulative_timing_offset'][sample_idx].item()
                    else:
                        # Fallback: calculate cumulative from step values up to this point
                        cumulative_timing = 0.0
                        for prev_step in range(step_idx + 1):
                            step_timing_norm = progression[prev_step]['timing_offset'][sample_idx].item()
                            step_timing_phys = step_timing_norm * getattr(self.hparams, 'sps', 8)
                            cumulative_timing += step_timing_phys

                    # Adjust symbol indices by cumulative timing correction
                    adjusted_indices = base_symbol_indices + int(round(cumulative_timing))

                    # Clamp to valid range
                    adjusted_indices = torch.clamp(adjusted_indices, 0, signal_length - 1)
                    step_indices.append(adjusted_indices)

                corrected_indices_per_step.append(step_indices)

            # Initialize constellation and parameter plots for each sample
            for sample_idx in range(num_samples):
                label = labels[sample_idx].item()
                snr = snrs[sample_idx].item()

                # Convert label to class name
                if isinstance(self.label_names, dict):
                    class_name = self.label_names.get(label, f"Class_{label}")
                elif isinstance(self.label_names, list) and label < len(self.label_names):
                    class_name = self.label_names[label]
                else:
                    class_name = f"Class_{label}"

                # Top row: Constellation plots
                ax_const = fig.add_subplot(gs[0, sample_idx])
                ax_const.set_xlim(-2, 2)
                ax_const.set_ylim(-2, 2)
                ax_const.set_xlabel('I Channel', fontsize=10)
                ax_const.set_ylabel('Q Channel', fontsize=10)
                ax_const.set_title(f'Sample {sample_idx}: {class_name}, SNR {snr:.1f}dB',
                                fontsize=11, pad=10)
                ax_const.grid(True, alpha=0.3)
                ax_const.set_aspect('equal')
                ax_const.tick_params(labelsize=9)

                # Plot initial unsynchronized signal (using original indices)
                unsync_sig = unsync_signals[sample_idx]
                original_indices = corrected_indices_per_step[0][sample_idx]

                ax_const.scatter(
                    unsync_sig[0, original_indices], unsync_sig[1, original_indices],
                    c='red', alpha=0.5, s=15, label='Initial (Unsync)', zorder=1, marker='x'
                )

                # Initialize animated scatter for progression
                initial_signal = progression[0]['signal'][sample_idx]
                initial_indices = corrected_indices_per_step[0][sample_idx]
                scatter = ax_const.scatter(
                    initial_signal[0, initial_indices], initial_signal[1, initial_indices],
                    c='blue', alpha=0.8, s=25, label='Synchronizing', zorder=3, marker='o'
                )
                scatters.append((scatter, sample_idx))
                ax_const.legend(fontsize=9)

                # Middle row: Parameter evolution (WITHOUT cumulative timing)
                ax_params = fig.add_subplot(gs[1, sample_idx])
                ax_params.set_xlim(0, num_steps-1)
                ax_params.set_xlabel('Iteration Step', fontsize=10)
                ax_params.set_ylabel('Parameter Value (Normalized)', fontsize=10)
                ax_params.set_title('Parameter Evolution', fontsize=11, pad=10)
                ax_params.grid(True, alpha=0.3)
                ax_params.tick_params(labelsize=9)

                # Initialize parameter plots (NO cumulative line)
                timing_line, = ax_params.plot([], [], 'o-', label='Timing Offset', color='blue',
                                            linewidth=2, markersize=3)
                freq_line, = ax_params.plot([], [], 's-', label='Freq Offset', color='orange',
                                        linewidth=2, markersize=3)
                phase_line, = ax_params.plot([], [], '^-', label='Phase Offset', color='purple',
                                        linewidth=2, markersize=3)

                param_lines.append((timing_line, freq_line, phase_line))

                # Set parameter plot limits (without cumulative values)
                try:
                    all_timing = [progression[i]['timing_offset'][sample_idx].item() for i in range(num_steps)]
                    all_freq = [progression[i]['freq_offset'][sample_idx].item() for i in range(num_steps)]
                    all_phase = [progression[i]['phase_offset'][sample_idx].item() for i in range(num_steps)]

                    all_values = all_timing + all_freq + all_phase
                    y_min = min(all_values) - 0.1
                    y_max = max(all_values) + 0.1
                    if y_min != y_max:
                        ax_params.set_ylim(y_min, y_max)
                    else:
                        ax_params.set_ylim(y_min - 0.1, y_max + 0.1)
                except:
                    ax_params.set_ylim(-1, 1)

                ax_params.legend(fontsize=8)

                # Bottom row: Classification confidence over steps
                ax_class = fig.add_subplot(gs[2, sample_idx])
                ax_class.set_xlim(0, num_steps-1)
                ax_class.set_xlabel('Iteration Step', fontsize=10)
                ax_class.set_ylabel('Classification Probability', fontsize=10)
                ax_class.set_title('Classification Confidence', fontsize=11, pad=10)
                ax_class.grid(True, alpha=0.3)
                ax_class.set_ylim(0, 1.05)
                ax_class.tick_params(labelsize=9)

                # Initialize classification confidence plots
                class_lines = []
                colors = ['red', 'blue', 'green', 'orange', 'purple']

                # Handle both dict and list label_names
                if isinstance(self.label_names, dict):
                    label_names_list = [self.label_names[i] for i in sorted(self.label_names.keys())]
                else:
                    label_names_list = self.label_names

                for class_idx, label_name in enumerate(label_names_list):
                    color = colors[class_idx % len(colors)]
                    line, = ax_class.plot([], [], label=str(label_name), color=color, linewidth=2)
                    class_lines.append(line)

                param_lines[sample_idx] = param_lines[sample_idx] + (class_lines,)
                ax_class.legend(fontsize=9, loc='upper right')

            # Animation function
            def animate(frame):
                try:
                    updates = []

                    for sample_idx in range(num_samples):
                        # Update constellation plot with CORRECTED indices
                        scatter, _ = scatters[sample_idx]
                        current_signal = progression[frame]['signal'][sample_idx]

                        # Use pre-calculated corrected indices for this frame and sample
                        corrected_indices = corrected_indices_per_step[frame][sample_idx]

                        new_offsets = np.column_stack([
                            current_signal[0, corrected_indices].numpy(),
                            current_signal[1, corrected_indices].numpy()
                        ])
                        scatter.set_offsets(new_offsets)

                        # Update parameter plots (WITHOUT cumulative)
                        timing_line, freq_line, phase_line, class_lines = param_lines[sample_idx]

                        steps_so_far = list(range(frame + 1))
                        timing_vals = [progression[i]['timing_offset'][sample_idx].item() for i in steps_so_far]
                        freq_vals = [progression[i]['freq_offset'][sample_idx].item() for i in steps_so_far]
                        phase_vals = [progression[i]['phase_offset'][sample_idx].item() for i in steps_so_far]

                        timing_line.set_data(steps_so_far, timing_vals)
                        freq_line.set_data(steps_so_far, freq_vals)
                        phase_line.set_data(steps_so_far, phase_vals)

                        # Update classification confidence
                        for class_idx, class_line in enumerate(class_lines):
                            if class_idx < progression[frame]['class_probs'].shape[1]:
                                class_probs = [progression[i]['class_probs'][sample_idx][class_idx].item()
                                            for i in steps_so_far]
                                class_line.set_data(steps_so_far, class_probs)

                        updates.extend([scatter, timing_line, freq_line, phase_line] + class_lines)

                    # Update main title (remove cumulative timing info)
                    timestep_val = progression[frame]["timestep"]
                    current_predictions = progression[frame]['class_preds']
                    current_labels = labels.cpu()
                    current_accuracy = (current_predictions == current_labels).float().mean().item()

                    fig.suptitle(f'Diffusion Synchronization + Classification - Step {frame}/{num_steps-1} (t={timestep_val:.0f})\n'
                                f'Current Accuracy: {current_accuracy:.3f}',
                                fontsize=14, fontweight='bold', y=0.96)

                    return updates

                except Exception as e:
                    print(f"Error in animation frame {frame}: {e}")
                    return []

            # Create animation
            anim = animation.FuncAnimation(
                fig, animate, frames=num_steps,
                interval=1000, blit=False, repeat=True
            )

            return fig, anim

        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def visualize_sync_progression(self):
        """Create enhanced synchronization visualization with classification"""
        if not self.val_samples_stored or self.stored_val_data is None:
            print("No validation data stored for visualization")
            return

        try:
            # Get stored validation data
            sync_signals = self.stored_val_data['sync_signals'].to(self.device)
            unsync_signals = self.stored_val_data['unsync_signals'].to(self.device)
            labels = self.stored_val_data['labels'].to(self.device)
            snrs = self.stored_val_data['snrs']

            print(f"Creating animation with shapes: sync={sync_signals.shape}, unsync={unsync_signals.shape}")

            # Perform iterative synchronization
            final_signals, progression = self.iterative_synchronization_for_gif(
                unsync_signals, labels, num_steps=10
            )

            if not progression:
                print("Error: No progression data generated")
                return

            print(f"Generated {len(progression)} progression steps")


            # Create enhanced animation with classification
            result = self.create_sync_animation_with_classification(
                unsync_signals.cpu(),
                progression,
                labels.cpu(),
                snrs,
                num_samples=4
            )

            # BETTER ERROR HANDLING
            if result is None:
                print("Error: create_sync_animation_with_classification returned None")
                return

            if not isinstance(result, tuple) or len(result) != 2:
                print(f"Error: Expected tuple of length 2, got {type(result)} of length {len(result) if hasattr(result, '__len__') else 'unknown'}")
                return

            fig, anim = result

            if fig is None or anim is None:
                print("Error: Animation creation failed - fig or anim is None")
                return

            # Save and log animation
            if hasattr(self, 'logger') and self.logger is not None:
                logger_class_name = self.logger.__class__.__name__
                if 'WandbLogger' in logger_class_name:
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_file:
                            writer = PillowWriter(fps=1.25)
                            anim.save(tmp_file.name, writer=writer)

                            self.logger.experiment.log({
                                "synchronization_classification_animation": wandb.Video(tmp_file.name, fps=1.25, format="gif"),
                                "global_step": self.global_step,
                                "epoch": self.current_epoch
                            })

                        os.unlink(tmp_file.name)
                        print("Successfully logged enhanced animation to WandB")

                    except Exception as e:
                        print(f"Failed to log to WandB: {e}")
                        local_filename = f'sync_class_animation_epoch_{self.current_epoch}.gif'
                        anim.save(local_filename, writer=PillowWriter(fps=1.25))
                        print(f"Saved enhanced animation locally as {local_filename}")
                else:
                    local_filename = f'sync_class_animation_epoch_{self.current_epoch}.gif'
                    anim.save(local_filename, writer=PillowWriter(fps=1.25))
                    print(f"Saved enhanced animation locally as {local_filename}")
            else:
                local_filename = f'sync_class_animation_epoch_{self.current_epoch}.gif'
                anim.save(local_filename, writer=PillowWriter(fps=1.25))
                print(f"Saved enhanced animation locally as {local_filename}")

            plt.close(fig)
            print("Enhanced animation creation completed successfully")

        except Exception as e:
            print(f"Enhanced animation creation failed: {e}")
            import traceback
            traceback.print_exc()

    def create_umap_visualization(self):
        """Create UMAP visualization of bottleneck features"""
        if not self.val_samples_stored or self.stored_val_data is None:
            print("No validation data stored for UMAP visualization")
            return

        try:
            # Get stored validation data
            unsync_signals = self.stored_val_data['unsync_signals'].to(self.device)
            labels = self.stored_val_data['labels'].to(self.device)
            snrs = self.stored_val_data['snrs']

            # Use more samples for better UMAP (up to 100)
            max_samples = min(100, len(unsync_signals))
            unsync_signals = unsync_signals[:max_samples]
            labels = labels[:max_samples]
            snrs = snrs[:max_samples]

            print(f"Creating UMAP visualization with {max_samples} samples...")

            # Collect bottleneck features from different timesteps
            all_features = []
            all_labels = []
            all_snrs = []
            all_timesteps = []

            # Test multiple timesteps to see how bottleneck space changes
            test_timesteps = [0, 2, 5, 7, 9]  # From synchronized to unsynchronized

            with torch.no_grad():
                for t in test_timesteps:
                    t_batch = torch.full((max_samples,), t, device=self.device)

                    # Get bottleneck features by extracting from UNet
                    bottleneck_features = self.extract_bottleneck_features(
                        unsync_signals, t_batch, labels
                    )

                    # Store features and metadata
                    all_features.append(bottleneck_features.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_snrs.extend(snrs.numpy())
                    all_timesteps.extend([t] * max_samples)

            # Concatenate all features
            all_features = np.vstack(all_features)  # Shape: [num_samples * num_timesteps, feature_dim]
            all_labels = np.array(all_labels)
            all_snrs = np.array(all_snrs)
            all_timesteps = np.array(all_timesteps)

            print(f"UMAP input shape: {all_features.shape}")

            # Apply UMAP
            umap_reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                metric='euclidean',
                random_state=42
            )

            umap_embedding = umap_reducer.fit_transform(all_features)

            # Create visualizations
            self.plot_umap_by_class(umap_embedding, all_labels, all_timesteps, all_snrs)
            self.plot_umap_by_timestep(umap_embedding, all_labels, all_timesteps, all_snrs)

            print("UMAP visualization completed successfully")

        except Exception as e:
            print(f"Error creating UMAP visualization: {e}")
            import traceback
            traceback.print_exc()

    def extract_bottleneck_features(self, sample, timestep, modulation):
        """Extract bottleneck features from UNet without parameter prediction"""
        # Ensure correct dtypes
        sample = sample.float()
        timestep = timestep.long()
        modulation = modulation.long()

        # Time embedding
        t_emb = self.model.time_proj(timestep)
        t_emb = self.model.time_embedding(t_emb)

        # Modulation conditioning
        context = None
        if self.model.mod_embedding is not None:
            mod_emb = self.model.mod_embedding(modulation)
            t_emb = t_emb + mod_emb

        # Down path
        x = sample
        for down_block in self.model.down_blocks:
            x, _ = down_block(hidden_states=x, temb=t_emb, context=context)

        # Bottleneck
        bottleneck_features = self.model.mid_block(hidden_states=x, temb=t_emb, context=context)

        # Global average pooling to get fixed-size features
        pooled_features = F.adaptive_avg_pool1d(bottleneck_features, 1).squeeze(-1)

        return pooled_features

    def plot_umap_by_class(self, umap_embedding, labels, timesteps, snrs):
        """Plot UMAP colored by modulation class"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Colored by class
        ax1 = axes[0]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        class_names = self.label_names if hasattr(self, 'label_names') else [f'Class_{i}' for i in range(3)]

        for class_idx, class_name in enumerate(class_names):
            mask = labels == class_idx
            if mask.sum() > 0:
                ax1.scatter(
                    umap_embedding[mask, 0],
                    umap_embedding[mask, 1],
                    c=colors[class_idx % len(colors)],
                    label=class_name,
                    alpha=0.7,
                    s=30
                )

        ax1.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax1.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax1.set_title('UNet Bottleneck Features by Modulation Class', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Colored by SNR
        ax2 = axes[1]
        scatter = ax2.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c=snrs,
            cmap='viridis',
            alpha=0.7,
            s=30
        )

        ax2.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax2.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax2.set_title('UNet Bottleneck Features by SNR', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label='SNR (dB)')

        plt.tight_layout()

        # Log to wandb
        if hasattr(self, 'logger') and self.logger is not None:
            logger_class_name = self.logger.__class__.__name__
            if 'WandbLogger' in logger_class_name:
                self.logger.experiment.log({
                    "umap_bottleneck_by_class": wandb.Image(fig),
                    "epoch": self.current_epoch
                })
                print("UMAP by class plot logged to WandB")

        # Save locally
        plot_filename = f'umap_by_class_epoch_{self.current_epoch}.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"UMAP by class plot saved as {plot_filename}")

        plt.close(fig)

    def plot_umap_by_timestep(self, umap_embedding, labels, timesteps, snrs):
        """Plot UMAP colored by timestep"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Colored by timestep
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c=timesteps,
            cmap='plasma',
            alpha=0.7,
            s=30
        )

        ax1.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax1.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax1.set_title('UNet Bottleneck Features by Timestep', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Timestep')

        # Plot 2: Separate by timestep with different markers
        ax2 = axes[1]
        timestep_colors = ['purple', 'blue', 'green', 'orange', 'red']
        markers = ['o', 's', '^', 'D', 'v']

        unique_timesteps = np.unique(timesteps)
        for i, t in enumerate(unique_timesteps):
            mask = timesteps == t
            ax2.scatter(
                umap_embedding[mask, 0],
                umap_embedding[mask, 1],
                c=timestep_colors[i % len(timestep_colors)],
                marker=markers[i % len(markers)],
                label=f't={t}',
                alpha=0.7,
                s=30
            )

        ax2.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax2.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax2.set_title('UNet Bottleneck Features by Timestep (Detailed)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Log to wandb
        if hasattr(self, 'logger') and self.logger is not None:
            logger_class_name = self.logger.__class__.__name__
            if 'WandbLogger' in logger_class_name:
                self.logger.experiment.log({
                    "umap_bottleneck_by_timestep": wandb.Image(fig),
                    "epoch": self.current_epoch
                })
                print("UMAP by timestep plot logged to WandB")

        # Save locally
        plot_filename = f'umap_by_timestep_epoch_{self.current_epoch}.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"UMAP by timestep plot saved as {plot_filename}")

        plt.close(fig)

    def calculate_ber_analysis(self):
        """Calculate BER for synchronized vs unsynchronized signals"""
        if not self.val_samples_stored or self.stored_val_data is None:
            print("No validation data stored for BER analysis")
            return

        try:
            # Get stored validation data
            sync_signals = self.stored_val_data['sync_signals'].to(self.device)
            unsync_signals = self.stored_val_data['unsync_signals'].to(self.device)
            labels = self.stored_val_data['labels'].to(self.device)
            snrs = self.stored_val_data['snrs']

            # Use more samples for better BER statistics
            max_samples = min(200, len(unsync_signals))
            sync_signals = sync_signals[:max_samples]
            unsync_signals = unsync_signals[:max_samples]
            labels = labels[:max_samples]
            snrs = snrs[:max_samples]

            print(f"Calculating BER for {max_samples} samples...")

            with torch.no_grad():
                # Get fully synchronized signals from diffusion model
                diffusion_sync_signals, _ = self.iterative_synchronization_for_gif(
                    unsync_signals, labels, num_steps=10
                )

                # Calculate BER for each condition
                ber_results = {
                    'snrs': snrs.numpy(),
                    'labels': labels.cpu().numpy(),
                    'unsync_ber': [],
                    'perfect_sync_ber': [],
                    'diffusion_sync_ber': [],
                }

                for i in range(max_samples):
                    # Get signals for this sample
                    unsync_sig = unsync_signals[i].cpu().numpy()
                    perfect_sync_sig = sync_signals[i].cpu().numpy()
                    diffusion_sync_sig = diffusion_sync_signals[i].cpu().numpy()

                    modulation_type = labels[i].item()
                    snr_db = snrs[i].item()

                    # Generate reference bits for this sample
                    ref_bits = self.generate_reference_bits(perfect_sync_sig, modulation_type)

                    # Calculate BER for each condition
                    unsync_ber = self.calculate_ber_for_signal(unsync_sig, ref_bits, modulation_type)
                    perfect_ber = self.calculate_ber_for_signal(perfect_sync_sig, ref_bits, modulation_type)
                    diffusion_ber = self.calculate_ber_for_signal(diffusion_sync_sig, ref_bits, modulation_type)

                    ber_results['unsync_ber'].append(unsync_ber)
                    ber_results['perfect_sync_ber'].append(perfect_ber)
                    ber_results['diffusion_sync_ber'].append(diffusion_ber)

                # Convert to numpy arrays
                for key in ['unsync_ber', 'perfect_sync_ber', 'diffusion_sync_ber']:
                    ber_results[key] = np.array(ber_results[key])

                # Create BER plots
                self.plot_ber_vs_snr(ber_results)
                self.plot_ber_by_modulation(ber_results)
                self.plot_ber_improvement(ber_results)

                print("BER analysis completed successfully")

        except Exception as e:
            print(f"Error in BER analysis: {e}")
            import traceback
            traceback.print_exc()

    def generate_reference_bits(self, perfect_sync_signal: np.ndarray, modulation_type: int) -> np.ndarray:
        """Generate reference bits from the perfectly synchronized signal"""
        # Convert to complex
        complex_signal = perfect_sync_signal[0] + 1j * perfect_sync_signal[1]

        # Simple symbol detection (downsample to symbol rate)
        # Assuming 8 samples per symbol
        sps = 8
        symbols = complex_signal[::sps]

        # Demodulate based on modulation type
        if modulation_type == 0:  # QPSK
            bits = self.demodulate_qpsk(symbols)
        elif modulation_type == 1:  # 8PSK
            bits = self.demodulate_8psk(symbols)
        elif modulation_type == 2:  # 16PSK
            bits = self.demodulate_16psk(symbols)
        else:
            bits = np.array([])

        return bits

    def calculate_ber_for_signal(self, signal: np.ndarray, ref_bits: np.ndarray, modulation_type: int) -> float:
        """Calculate BER for a given signal compared to reference bits"""
        if len(ref_bits) == 0:
            return 1.0  # Maximum error rate

        try:
            # Convert to complex
            complex_signal = signal[0] + 1j * signal[1]

            # Simple symbol detection
            sps = 8
            symbols = complex_signal[::sps]

            # Ensure same length as reference
            min_len = min(len(symbols), len(ref_bits) // self.bits_per_symbol(modulation_type))
            symbols = symbols[:min_len]
            ref_bits = ref_bits[:min_len * self.bits_per_symbol(modulation_type)]

            # Demodulate
            if modulation_type == 0:  # QPSK
                demod_bits = self.demodulate_qpsk(symbols)
            elif modulation_type == 1:  # 8PSK
                demod_bits = self.demodulate_8psk(symbols)
            elif modulation_type == 2:  # 16PSK
                demod_bits = self.demodulate_16psk(symbols)
            else:
                return 1.0

            # Calculate BER
            if len(demod_bits) == 0 or len(ref_bits) == 0:
                return 1.0

            min_bits = min(len(demod_bits), len(ref_bits))
            errors = np.sum(demod_bits[:min_bits] != ref_bits[:min_bits])
            ber = errors / min_bits

            return ber

        except Exception as e:
            print(f"Error calculating BER: {e}")
            return 1.0

    def bits_per_symbol(self, modulation_type: int) -> int:
        """Return bits per symbol for each modulation type"""
        if modulation_type == 0:  # QPSK
            return 2
        elif modulation_type == 1:  # 8PSK
            return 3
        elif modulation_type == 2:  # 16PSK
            return 4
        else:
            return 1

    def demodulate_qpsk(self, symbols: np.ndarray) -> np.ndarray:
        """Simple QPSK demodulation"""
        # Normalize symbols
        symbols = symbols / (np.abs(symbols) + 1e-8)

        # Decision regions
        bits = []
        for symbol in symbols:
            if symbol.real > 0 and symbol.imag > 0:  # Q1
                bits.extend([0, 0])
            elif symbol.real < 0 and symbol.imag > 0:  # Q2
                bits.extend([0, 1])
            elif symbol.real < 0 and symbol.imag < 0:  # Q3
                bits.extend([1, 1])
            else:  # Q4
                bits.extend([1, 0])

        return np.array(bits)

    def demodulate_8psk(self, symbols: np.ndarray) -> np.ndarray:
        """Simple 8PSK demodulation"""
        # Normalize symbols
        symbols = symbols / (np.abs(symbols) + 1e-8)

        # Calculate angles
        angles = np.angle(symbols)

        # 8PSK decision regions (0 to 2π divided into 8 regions)
        bits = []
        for angle in angles:
            # Convert to 0-2π range
            if angle < 0:
                angle += 2 * np.pi

            # Determine symbol (0-7)
            symbol_idx = int(np.round(angle / (2 * np.pi / 8))) % 8

            # Convert to 3 bits
            bit_pattern = [
                (symbol_idx >> 2) & 1,
                (symbol_idx >> 1) & 1,
                symbol_idx & 1
            ]
            bits.extend(bit_pattern)

        return np.array(bits)

    def demodulate_16psk(self, symbols: np.ndarray) -> np.ndarray:
        """Simple 16PSK demodulation"""
        # Normalize symbols
        symbols = symbols / (np.abs(symbols) + 1e-8)

        # Calculate angles
        angles = np.angle(symbols)

        # 16PSK decision regions
        bits = []
        for angle in angles:
            # Convert to 0-2π range
            if angle < 0:
                angle += 2 * np.pi

            # Determine symbol (0-15)
            symbol_idx = int(np.round(angle / (2 * np.pi / 16))) % 16

            # Convert to 4 bits
            bit_pattern = [
                (symbol_idx >> 3) & 1,
                (symbol_idx >> 2) & 1,
                (symbol_idx >> 1) & 1,
                symbol_idx & 1
            ]
            bits.extend(bit_pattern)

        return np.array(bits)
    def plot_ber_vs_snr(self, ber_results: Dict):
        """Plot BER vs SNR for different synchronization conditions"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Define SNR bins
        snr_bins = [(-20, -15), (-15, -10), (-10, -5), (-5, 0), (0, 5), (5, 10),
                    (10, 15), (15, 20), (20, 25), (25, 30)]
        snr_centers = [(low + high) / 2 for low, high in snr_bins]

        # Calculate average BER per SNR bin
        avg_ber = {condition: [] for condition in ['unsync_ber', 'perfect_sync_ber', 'diffusion_sync_ber']}

        for snr_min, snr_max in snr_bins:
            mask = (ber_results['snrs'] >= snr_min) & (ber_results['snrs'] < snr_max)

            for condition in avg_ber.keys():
                if mask.sum() > 0:
                    avg_ber[condition].append(np.mean(ber_results[condition][mask]))
                else:
                    avg_ber[condition].append(np.nan)

        # Plot overall BER comparison
        ax1 = axes[0]
        ax1.semilogy(snr_centers, avg_ber['unsync_ber'], 'r-o', label='Unsynchronized', linewidth=2, markersize=6)
        ax1.semilogy(snr_centers, avg_ber['perfect_sync_ber'], 'g-s', label='Perfect Sync', linewidth=2, markersize=6)
        ax1.semilogy(snr_centers, avg_ber['diffusion_sync_ber'], 'b-^', label='Diffusion Sync', linewidth=2, markersize=6)

        ax1.set_xlabel('SNR (dB)', fontsize=12)
        ax1.set_ylabel('Bit Error Rate', fontsize=12)
        ax1.set_title('BER vs SNR Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xlim(-22, 32)

        # Plot BER improvement (diffusion vs unsync)
        ax2 = axes[1]
        improvement = np.array(avg_ber['unsync_ber']) / (np.array(avg_ber['diffusion_sync_ber']) + 1e-10)
        ax2.semilogy(snr_centers, improvement, 'purple', linewidth=3, marker='o', markersize=8)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No Improvement')

        ax2.set_xlabel('SNR (dB)', fontsize=12)
        ax2.set_ylabel('BER Improvement Factor', fontsize=12)
        ax2.set_title('BER Improvement: Unsync/Diffusion', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_xlim(-22, 32)

        # Plot synchronization efficiency
        ax3 = axes[2]
        perfect_improvement = np.array(avg_ber['unsync_ber']) / (np.array(avg_ber['perfect_sync_ber']) + 1e-10)
        diffusion_improvement = np.array(avg_ber['unsync_ber']) / (np.array(avg_ber['diffusion_sync_ber']) + 1e-10)
        efficiency = diffusion_improvement / (perfect_improvement + 1e-10)

        ax3.plot(snr_centers, efficiency, 'orange', linewidth=3, marker='s', markersize=8)
        ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Perfect Efficiency')
        ax3.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='50% Efficiency')

        ax3.set_xlabel('SNR (dB)', fontsize=12)
        ax3.set_ylabel('Sync Efficiency', fontsize=12)
        ax3.set_title('Synchronization Efficiency', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        ax3.set_xlim(-22, 32)
        ax3.set_ylim(0, 1.2)

        plt.tight_layout()

        # Log to wandb
        if hasattr(self, 'logger') and self.logger is not None:
            logger_class_name = self.logger.__class__.__name__
            if 'WandbLogger' in logger_class_name:
                self.logger.experiment.log({
                    "ber_vs_snr": wandb.Image(fig),
                    "epoch": self.current_epoch
                })
                print("BER vs SNR plot logged to WandB")

        # Save locally
        plot_filename = f'ber_vs_snr_epoch_{self.current_epoch}.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"BER vs SNR plot saved as {plot_filename}")

        plt.close(fig)

    def plot_ber_by_modulation(self, ber_results: Dict):
        """Plot BER by modulation type"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        modulation_names = ['QPSK', '8PSK', '16PSK']
        colors = ['red', 'blue', 'green']

        for mod_idx, (mod_name, color) in enumerate(zip(modulation_names, colors)):
            ax = axes[mod_idx]

            # Filter by modulation type
            mask = ber_results['labels'] == mod_idx

            if mask.sum() > 0:
                snrs = ber_results['snrs'][mask]
                unsync_ber = ber_results['unsync_ber'][mask]
                perfect_ber = ber_results['perfect_sync_ber'][mask]
                diffusion_ber = ber_results['diffusion_sync_ber'][mask]

                # Sort by SNR for better plotting
                sort_idx = np.argsort(snrs)
                snrs = snrs[sort_idx]
                unsync_ber = unsync_ber[sort_idx]
                perfect_ber = perfect_ber[sort_idx]
                diffusion_ber = diffusion_ber[sort_idx]

                # Plot with some smoothing
                ax.semilogy(snrs, unsync_ber, 'r-o', alpha=0.7, label='Unsynchronized', markersize=4)
                ax.semilogy(snrs, perfect_ber, 'g-s', alpha=0.7, label='Perfect Sync', markersize=4)
                ax.semilogy(snrs, diffusion_ber, 'b-^', alpha=0.7, label='Diffusion Sync', markersize=4)

            ax.set_xlabel('SNR (dB)', fontsize=12)
            ax.set_ylabel('Bit Error Rate', fontsize=12)
            ax.set_title(f'{mod_name} BER Performance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_xlim(-22, 32)

        plt.tight_layout()

        # Log to wandb
        if hasattr(self, 'logger') and self.logger is not None:
            logger_class_name = self.logger.__class__.__name__
            if 'WandbLogger' in logger_class_name:
                self.logger.experiment.log({
                    "ber_by_modulation": wandb.Image(fig),
                    "epoch": self.current_epoch
                })
                print("BER by modulation plot logged to WandB")

        # Save locally
        plot_filename = f'ber_by_modulation_epoch_{self.current_epoch}.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"BER by modulation plot saved as {plot_filename}")

        plt.close(fig)

    def plot_ber_improvement(self, ber_results: Dict):
        """Plot BER improvement statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Calculate improvements
        ber_improvement = ber_results['unsync_ber'] / (ber_results['diffusion_sync_ber'] + 1e-10)
        perfect_improvement = ber_results['unsync_ber'] / (ber_results['perfect_sync_ber'] + 1e-10)

        # Plot 1: Improvement vs SNR scatter
        ax1 = axes[0, 0]
        scatter = ax1.scatter(ber_results['snrs'], ber_improvement,
                                c=ber_results['labels'], cmap='tab10', alpha=0.6)
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('BER Improvement Factor')
        ax1.set_title('BER Improvement vs SNR')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Modulation Type')

        # Plot 2: Improvement histogram
        ax2 = axes[0, 1]
        ax2.hist(ber_improvement, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, label='No Improvement')
        ax2.axvline(x=np.median(ber_improvement), color='green', linestyle='-', linewidth=2,
                    label=f'Median: {np.median(ber_improvement):.2f}')
        ax2.set_xlabel('BER Improvement Factor')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of BER Improvements')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Success rate by SNR
        ax3 = axes[1, 0]
        snr_bins = np.arange(-20, 31, 5)
        success_rates = []
        snr_centers = []

        for i in range(len(snr_bins) - 1):
            mask = (ber_results['snrs'] >= snr_bins[i]) & (ber_results['snrs'] < snr_bins[i+1])
            if mask.sum() > 0:
                success_rate = np.mean(ber_improvement[mask] > 1.0)
                success_rates.append(success_rate)
                snr_centers.append((snr_bins[i] + snr_bins[i+1]) / 2)

        ax3.plot(snr_centers, success_rates, 'o-', linewidth=2, markersize=8, color='purple')
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('SNR (dB)')
        ax3.set_ylabel('Success Rate (BER Improvement > 1)')
        ax3.set_title('Synchronization Success Rate vs SNR')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Calculate statistics
        mean_improvement = np.mean(ber_improvement)
        median_improvement = np.median(ber_improvement)
        success_rate_overall = np.mean(ber_improvement > 1.0)
        best_improvement = np.max(ber_improvement)
        worst_improvement = np.min(ber_improvement)

        stats_text = f"""
        BER Improvement Statistics:

        Mean Improvement: {mean_improvement:.2f}x
        Median Improvement: {median_improvement:.2f}x
        Success Rate: {success_rate_overall:.1%}
        Best Improvement: {best_improvement:.2f}x
        Worst Case: {worst_improvement:.2f}x

        Total Samples: {len(ber_improvement)}
        Samples Improved: {np.sum(ber_improvement > 1.0)}
        Samples Degraded: {np.sum(ber_improvement < 1.0)}
        """

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        # Log to wandb
        if hasattr(self, 'logger') and self.logger is not None:
            logger_class_name = self.logger.__class__.__name__
            if 'WandbLogger' in logger_class_name:
                self.logger.experiment.log({
                    "ber_improvement_analysis": wandb.Image(fig),
                    "epoch": self.current_epoch,
                    "mean_ber_improvement": mean_improvement,
                    "median_ber_improvement": median_improvement,
                    "ber_success_rate": success_rate_overall,
                })
                print("BER improvement analysis logged to WandB")

        # Save locally
        plot_filename = f'ber_improvement_epoch_{self.current_epoch}.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"BER improvement analysis saved as {plot_filename}")

        plt.close(fig)
    def on_train_epoch_start(self):
        """Log joint training status"""
        print(f"\nEpoch {self.current_epoch}: Joint Training (Sync + Classification)")
        print("   - UNet training 🔥")
        print("   - Classifier training 🔥")

    def on_validation_epoch_end(self):
        """Create visualizations and SNR analysis for joint training"""
        if self.validation_step_outputs:
            # Create SNR vs accuracy plot
            self.create_snr_vs_accuracy_plot()
            self.validation_step_outputs.clear()

        # Create animations periodically
        if self.current_epoch % self.hparams.log_every_n_epochs == 0:
            self.visualize_sync_progression()
            self.create_umap_visualization()  # Add UMAP visualization
            self.calculate_ber_analysis()  # Add BER analysis

    def configure_optimizers(self):
        """Configure optimizer for joint training"""
        # Joint training: optimize both UNet and classifier parameters
        all_params = list(self.model.parameters()) + list(self.classifier.parameters())
        optimizer = AdamW(
            all_params,
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        if self.hparams.use_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]['lr'],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy='cos'
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'name': 'learning_rate'
                }
            }

        return optimizer

    def get_classifier_state_dict(self):
        """Get classifier state dict for saving/loading"""
        return self.classifier.state_dict()

    def load_classifier_state_dict(self, state_dict):
        """Load classifier state dict"""
        self.classifier.load_state_dict(state_dict)

    def freeze_classifier(self):
        """Freeze classifier parameters"""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        """Unfreeze classifier parameters"""
        for param in self.classifier.parameters():
            param.requires_grad = True

    def freeze_unet(self):
        """Freeze UNet parameters"""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_unet(self):
        """Unfreeze UNet parameters"""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_classification_accuracy(self, signals: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Get classification accuracy for given signals and labels"""
        with torch.no_grad():
            logits = self.classifier(signals)
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean()

        return accuracy

    def classify_signals(self, signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify signals and return predictions and probabilities"""
        with torch.no_grad():
            logits = self.classifier(signals)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=1)

        return preds, probs

    def get_sync_and_classification_loss(self, batch):
        """Get both synchronization and classification losses for analysis"""
        sync_signals, unsync_signals, labels, snrs, sync_params = batch
        batch_size = sync_signals.shape[0]

        # Normalize sync parameters
        true_timing, true_freq, true_phase = sync_params
        norm_true_timing, norm_true_freq, norm_true_phase = self.normalize_sync_params(
            true_timing, true_freq, true_phase
        )

        # Random timesteps for training
        timesteps = torch.randint(0, self.hparams.num_train_timesteps, (batch_size,), device=self.device)
        interpolated_signals = self.create_sync_interpolation(sync_signals, unsync_signals, timesteps)

        # Get sync predictions
        sync_output = self.model(interpolated_signals, timesteps, labels, return_dict=True)

        # Apply corrections
        synchronized_signals = self.apply_predicted_sync(
            interpolated_signals,
            sync_output['timing_offset'],
            sync_output['freq_offset'],
            sync_output['phase_offset']
        )

        # Calculate step-wise sync loss
        alpha_current = self.get_interpolation_alpha(timesteps).to(self.device)
        next_timesteps = torch.clamp(timesteps - 1, 0, self.hparams.num_train_timesteps - 1)
        alpha_next = self.get_interpolation_alpha(next_timesteps).to(self.device)
        step_alpha = torch.where(timesteps == 0, torch.zeros_like(alpha_current), alpha_current - alpha_next)

        target_timing = norm_true_timing * step_alpha
        target_freq = norm_true_freq * step_alpha
        target_phase = norm_true_phase * step_alpha

        sync_loss = (
            F.mse_loss(sync_output['timing_offset'].squeeze(), target_timing) +
            F.mse_loss(sync_output['freq_offset'].squeeze(), target_freq) +
            F.mse_loss(sync_output['phase_offset'].squeeze(), target_phase)
        )

        # Get classification loss
        class_logits = self.classifier(synchronized_signals)
        classification_loss = self.classification_criterion(class_logits, labels)

        return sync_loss, classification_loss, synchronized_signals, class_logits

    def evaluate_sync_quality(self, signals: torch.Tensor, reference_signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate synchronization quality metrics"""
        with torch.no_grad():
            # Convert to complex
            signals_complex = signals[:, 0] + 1j * signals[:, 1]
            reference_complex = reference_signals[:, 0] + 1j * reference_signals[:, 1]

            # Calculate EVM (Error Vector Magnitude)
            error_vectors = signals_complex - reference_complex
            evm = torch.mean(torch.abs(error_vectors) ** 2, dim=1)

            # Calculate constellation tightness (standard deviation of points)
            constellation_std = torch.std(signals_complex, dim=1)

            # Calculate phase coherence
            phase_diff = torch.angle(signals_complex[:, 1:]) - torch.angle(signals_complex[:, :-1])
            phase_coherence = torch.mean(torch.abs(phase_diff), dim=1)

            return {
                'evm': evm,
                'constellation_std': constellation_std,
                'phase_coherence': phase_coherence
            }

class BaselineClassifier(L.LightningModule):
    """
    Baseline classifier that operates directly on raw input signals
    for comparison against the denoising + classification approach
    """

    def __init__(
        self,
        classifier,
        learning_rate: float = 1e-3,
        num_classes: int = 3,
        label_names: List[str] = ["QPSK", "8PSK", "16PSK"],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classifier"])

        # Core components
        self.classifier = classifier
        self.num_classes = num_classes
        self.label_names = label_names

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # For tracking validation outputs
        self.validation_step_outputs = []

    def forward(self, x, return_features=False):
        return self.classifier(x, return_features=return_features)

    def get_feature_layers(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple layers for perceptual loss
        Delegates to the underlying classifier model
        """
        return self.classifier.get_feature_layers(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract rich features for perceptual loss
        Delegates to the underlying classifier model
        """
        return self.classifier.extract_features(x)
    def training_step(self, batch, batch_idx):
        # Unpack batch - use corrupted signals (raw noisy data)
        clean_signals, corrupted_signals, labels, snrs = batch

        # Classify the raw corrupted signals directly
        logits = self.classifier(corrupted_signals)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        _, preds = torch.max(logits, 1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch - use corrupted signals (raw noisy data)
        clean_signals, corrupted_signals, labels, snrs = batch

        # Classify the raw corrupted signals directly
        logits = self.classifier(corrupted_signals)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == labels).float().mean()

        # Calculate average confidence
        confidence = torch.max(probs, dim=1)[0].mean()

        # Log main metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_confidence", confidence)

        # Per-class accuracy
        for i, class_name in enumerate(self.label_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == labels[class_mask]).float().mean()
                self.log(f"val_acc_{class_name}", class_acc)

        # SNR-based accuracy analysis (key for baseline comparison)
        snr_ranges = [
            (-20, -15), (-15, -10), (-10, -5), (-5, 0),
            (0, 5), (5, 10), (10, 15), (15, 20), (20, 30)
        ]
        for snr_min, snr_max in snr_ranges:
            snr_mask = (snrs >= snr_min) & (snrs < snr_max)
            if snr_mask.sum() > 0:
                snr_acc = (preds[snr_mask] == labels[snr_mask]).float().mean()
                self.log(f"val_acc_snr_{snr_min}to{snr_max}dB", snr_acc)

        # Store for epoch-end analysis
        self.validation_step_outputs.append({
            "preds": preds.detach().cpu(),
            "labels": labels.detach().cpu(),
            "snrs": snrs.detach().cpu(),
            "loss": loss.detach().cpu(),
            "corrupted_signals": corrupted_signals[:4].detach().cpu() if batch_idx == 0 else None,
        })

        return loss

    def on_validation_epoch_end(self):
        """Calculate and log epoch-level metrics and visualizations"""
        if not self.validation_step_outputs:
            return

        # Aggregate all predictions and labels
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        all_snrs = torch.cat([x["snrs"] for x in self.validation_step_outputs])

        # Calculate confusion matrix
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        for t, p in zip(all_labels, all_preds):
            confusion_matrix[t.long(), p.long()] += 1

        # Normalize confusion matrix
        row_sums = confusion_matrix.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        confusion_matrix = confusion_matrix / row_sums

        # Log confusion matrix
        if self.logger and hasattr(self.logger, "experiment"):
            # self._create_confusion_matrix_plot(confusion_matrix)
            self._create_snr_performance_plot(all_preds, all_labels, all_snrs)
        # Clear outputs
        self.validation_step_outputs.clear()

    def _create_confusion_matrix_plot(self, confusion_matrix):
        """Create confusion matrix visualization"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(confusion_matrix.numpy(), cmap="Blues")

            # Add labels
            ax.set_xticks(range(self.num_classes))
            ax.set_yticks(range(self.num_classes))
            ax.set_xticklabels(self.label_names)
            ax.set_yticklabels(self.label_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Baseline Classifier Confusion Matrix - Epoch {self.current_epoch}")

            # Add text annotations
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    text = ax.text(
                        j, i, f"{confusion_matrix[i, j]:.2f}",
                        ha="center", va="center", color="black"
                    )

            plt.colorbar(im)
            plt.tight_layout()

            self.logger.experiment.log({"baseline_confusion_matrix": wandb.Image(fig)})
            plt.close(fig)

        except Exception as e:
            print(f"Error in confusion matrix plot: {e}")

    def _create_snr_performance_plot(self, preds, labels, snrs):
        """Create SNR vs accuracy plot"""
        try:
            snr_ranges = [(-20, -15), (-15, -10), (-10, -5), (-5, 0),
                         (0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
            snr_centers = [(low + high) / 2 for low, high in snr_ranges]
            accuracies = []

            for snr_min, snr_max in snr_ranges:
                mask = (snrs >= snr_min) & (snrs < snr_max)
                if mask.sum() > 0:
                    acc = (preds[mask] == labels[mask]).float().mean().item()
                    accuracies.append(acc)
                else:
                    accuracies.append(0.0)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(snr_centers, accuracies, 'b-o', linewidth=2, markersize=8,
                   label="Baseline (Raw Signal Classification)")
            ax.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5, label="Chance Level")

            ax.set_xlabel("SNR (dB)", fontsize=12)
            ax.set_ylabel("Classification Accuracy", fontsize=12)
            ax.set_title(f"Baseline Performance vs SNR - Epoch {self.current_epoch}", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(-22, 32)

            plt.tight_layout()
            self.logger.experiment.log({"baseline_snr_performance": wandb.Image(fig)})
            plt.close(fig)

        except Exception as e:
            print(f"Error in SNR performance plot: {e}")

    def _create_raw_signal_visualization(self, signals, true_labels, pred_labels, snrs):
        """Visualize raw signal constellations"""
        try:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            for i in range(4):
                signal = signals[i]
                true_label = true_labels[i].item()
                pred_label = pred_labels[i].item()
                snr = snrs[i].item()

                # Convert to complex for constellation plot
                signal_complex = torch.complex(signal[0], signal[1])

                axes[i].scatter(
                    signal_complex.real.numpy(),
                    signal_complex.imag.numpy(),
                    alpha=0.6, s=20, c="red"
                )

                correct = true_label == pred_label
                status = "✓" if correct else "✗"
                color = "green" if correct else "red"

                axes[i].set_title(
                    f"{status} True: {self.label_names[true_label]}\n"
                    f"Pred: {self.label_names[pred_label]}\n"
                    f"SNR: {snr:.1f} dB",
                    color=color
                )
                axes[i].set_xlim(-2, 2)
                axes[i].set_ylim(-2, 2)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_aspect("equal")

            plt.suptitle(f"Baseline: Raw Signal Classification Examples - Epoch {self.current_epoch}", fontsize=16)
            plt.tight_layout()

            self.logger.experiment.log({"baseline_raw_signals": wandb.Image(fig)})
            plt.close(fig)

        except Exception as e:
            print(f"Error in raw signal visualization: {e}")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.classifier.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=7988,  # Steps per epoch
            T_mult=1,
            eta_min=self.hparams.learning_rate * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class ConvFeatureExtractor(nn.Module):
    """Stage A: spatial feature extraction via 1D convolutions."""

    def __init__(self, in_ch=2, hidden_chs=(64, 128, 256)):
        super().__init__()
        layers = []
        prev = in_ch
        for h in hidden_chs:
            layers += [
                nn.Conv1d(prev, h, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 2, 1024] → [B, hidden_chs[-1], 1024/2^len(hidden_chs)]
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class HybridConvTransformer(nn.Module):
    def __init__(self, in_ch=2, num_classes=24, conv_hidden=(64, 128, 256),
                 trans_dim=256, n_heads=4, n_layers=3, mlp_hidden=128,
                 use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token

        # Conv Feature Extractor
        self.conv_extractor = ConvFeatureExtractor(in_ch, conv_hidden)

        # Class Token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))

        # Transformer Input Projection
        self.input_proj = nn.Linear(conv_hidden[-1], trans_dim)
        self.pos_enc = PositionalEncoding(trans_dim, max_len=129)  # +1 for CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim, nhead=n_heads,
            dim_feedforward=trans_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # LayerNorm before classifier
        self.norm = nn.LayerNorm(trans_dim)

        # Classifier Head with Residual MLP
        self.classifier = nn.Sequential(
            nn.Linear(trans_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_classes),
        )
    def forward(self, x):
        conv_features = self.conv_extractor(x).permute(0, 2, 1)  # [B, 128, C]
        y = self.input_proj(conv_features)

        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            y = torch.cat((cls, y), dim=1)  # [B, 129, trans_dim]

        y = self.pos_enc(y)
        y = self.transformer(y)

        if self.use_cls_token:
            y = y[:, 0]  # [CLS] token
        else:
            y = y.mean(dim=1)

        y = self.norm(y)
        logits = self.classifier(y)

        return logits

import torch
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
        n_layers: int = 3,
        mlp_hidden: int = 256,

        # Loss weights for joint training
        lambda_sync: float = 1.0,      # Synchronization loss weight
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

        # === SYNCHRONIZATION LOSS (STEP-WISE FOR EFFICIENCY) ===
        timesteps = torch.randint(0, self.hparams.num_train_timesteps, (batch_size,), device=self.device)
        interpolated_signals = self.create_sync_interpolation(sync_signals, unsync_signals, timesteps)

        # Get sync model predictions
        sync_output = self.model(interpolated_signals, timesteps, labels, return_dict=True)

        # Calculate step-wise targets
        alpha_current = self.get_interpolation_alpha(timesteps).to(self.device)
        next_timesteps = torch.clamp(timesteps - 1, 0, self.hparams.num_train_timesteps - 1)
        alpha_next = self.get_interpolation_alpha(next_timesteps).to(self.device)

        # For t=0, step should be zero
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
        # Use the same function as visualization but enable gradients
        with torch.enable_grad():  # Explicitly enable gradients
            fully_synchronized_signals, _ = self.iterative_synchronization_for_gif(
                unsync_signals, labels, num_steps=5
            )

        # Train classifier on FULLY synchronized signals
        class_logits = self.classifier(fully_synchronized_signals)
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

        # Log by timestep ranges
        high_t_mask = timesteps >= 15
        med_t_mask = (timesteps >= 5) & (timesteps < 15)
        low_t_mask = timesteps < 5

        if high_t_mask.any():
            self.log('train/step_size_high_t', torch.mean(step_alpha[high_t_mask]))
            self.log('train/pred_timing_high_t', torch.mean(torch.abs(sync_output['timing_offset'][high_t_mask])))
        if med_t_mask.any():
            self.log('train/step_size_med_t', torch.mean(step_alpha[med_t_mask]))
            self.log('train/pred_timing_med_t', torch.mean(torch.abs(sync_output['timing_offset'][med_t_mask])))
        if low_t_mask.any():
            self.log('train/step_size_low_t', torch.mean(step_alpha[low_t_mask]))
            self.log('train/pred_timing_low_t', torch.mean(torch.abs(sync_output['timing_offset'][low_t_mask])))

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation with full iterative synchronization for consistent evaluation"""
        sync_signals, unsync_signals, labels, snrs, sync_params = batch
        batch_size = sync_signals.shape[0]

        # === NORMALIZE SYNC PARAMETERS ===
        true_timing, true_freq, true_phase = sync_params
        norm_true_timing, norm_true_freq, norm_true_phase = self.normalize_sync_params(
            true_timing, true_freq, true_phase
        )

        # Store first batch for visualization (with normalized params)
        if batch_idx == 0 and not self.val_samples_stored:
            self.stored_val_data = {
                'sync_signals': sync_signals[:self.hparams.vis_batch_size].cpu(),
                'unsync_signals': unsync_signals[:self.hparams.vis_batch_size].cpu(),
                'labels': labels[:self.hparams.vis_batch_size].cpu(),
                'snrs': snrs[:self.hparams.vis_batch_size].cpu(),
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
            # Run FULL iterative synchronization - same as training
            fully_synchronized_signals, progression = self.iterative_synchronization_for_gif(
                unsync_signals, labels, num_steps=5
            )

            # Classify the fully synchronized signals
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

        # Log per-class accuracy - UPDATED TO USE ACTUAL LABEL NAMES
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

        # Reverse diffusion schedule
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

        for t in [19, 15, 10, 5, 0]:
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

    @torch.no_grad()
    def iterative_synchronization_for_gif(
        self,
        unsync_signal: torch.Tensor,
        modulation: torch.Tensor,
        num_steps: int = 10
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Iterative synchronization with step-wise corrections for visualization"""
        device = unsync_signal.device
        current_signal = unsync_signal.clone()

        progression = []

        # Reverse diffusion schedule
        timesteps = torch.linspace(
            self.hparams.num_train_timesteps - 1, 0, num_steps
        ).long().to(device)

        for step, t in enumerate(timesteps):
            t_batch = torch.full((current_signal.shape[0],), t, device=device)

            # Get model prediction (should be step-wise correction)
            output = self.model(current_signal, t_batch, modulation, return_dict=True)

            # Apply step-wise corrections to current signal
            corrected_signal = self.apply_predicted_sync(
                current_signal,
                output['timing_offset'],
                output['freq_offset'],
                output['phase_offset']
            )

            # Get classification for this step
            if self.should_do_classification():
                class_logits = self.classifier(corrected_signal)
                class_probs = F.softmax(class_logits, dim=-1)
                class_preds = torch.argmax(class_logits, dim=1)
            else:
                class_probs = torch.zeros((corrected_signal.shape[0], self.hparams.num_classes))
                class_preds = torch.zeros((corrected_signal.shape[0],), dtype=torch.long)

            # Store step info
            step_info = {
                'step': step,
                'timestep': t.item(),
                'signal': corrected_signal.cpu().clone(),
                'input_signal': current_signal.cpu().clone(),
                'timing_offset': output['timing_offset'].cpu().clone(),
                'freq_offset': output['freq_offset'].cpu().clone(),
                'phase_offset': output['phase_offset'].cpu().clone(),
                'class_probs': class_probs.cpu().clone(),
                'class_preds': class_preds.cpu().clone(),
            }

            progression.append(step_info)

            # Update for next iteration (key: step-wise application)
            current_signal = corrected_signal.detach()

        return current_signal, progression

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
            symbol_stride = 16
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

            # DEBUG: Check if cumulative data exists
            if progression and 'cumulative_timing_offset' in progression[0]:
                print("✓ Cumulative timing data found in progression")
            else:
                print("⚠ Cumulative timing data missing - using fallback calculation")

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
                nn.Conv1d(prev, h, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 2, 1024] → [B, hidden_chs[-1], 1024/2^len(hidden_chs)]
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encodings to the transformer input."""

    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, L, d_model]
        L = x.size(1)
        return x + self.pe[:, :L]


class HybridConvTransformer(nn.Module):
    """
    Hybrid Convolutional Transformer feature extractor + classifier
    with feature extraction capability for VGG/perceptual losses
    """

    def __init__(
        self,
        in_ch: int = 2,
        num_classes: int = 24,
        conv_hidden: tuple = (64, 128, 256),
        trans_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        mlp_hidden: int = 128,
    ):
        super().__init__()

        # Stage A: Conv1D feature extractor
        self.conv_extractor = ConvFeatureExtractor(in_ch, conv_hidden)
        # After 3 x MaxPool(stride=2): sequence length = 1024 / 2^3 = 128

        # Stage B: Transformer Encoder
        self.input_proj = nn.Linear(conv_hidden[-1], trans_dim)
        self.pos_enc = PositionalEncoding(trans_dim, max_len=128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=n_heads,
            dim_feedforward=trans_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Stage C: Pooling and Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(trans_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, 2, 1024]  real and imag as two channels
            return_features: if True, return both features and logits
        Returns:
            logits: [B, num_classes] or (features, logits) if return_features=True
        """
        # Conv feature extraction
        conv_features = self.conv_extractor(x)  # [B, C, 128]

        # Prepare for transformer: -> [B, 128, C]
        y = conv_features.permute(0, 2, 1)

        # Project to transformer dimension
        y = self.input_proj(y)  # [B, 128, trans_dim]

        # Add positional encodings
        y = self.pos_enc(y)

        # Transformer encoder - THIS IS OUR RICH FEATURE REPRESENTATION
        transformer_features = self.transformer(y)  # [B, 128, trans_dim]

        # For classification: pool and classify
        # Back to [B, trans_dim, 128] for pooling
        pooled_input = transformer_features.permute(0, 2, 1)
        pooled = self.pool(pooled_input).squeeze(-1)  # [B, trans_dim]
        logits = self.classifier(pooled)  # [B, num_classes]

        if return_features:
            # Return rich transformer features before pooling
            return transformer_features, logits  # [B, 128, trans_dim], [B, num_classes]
        else:
            return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract rich features for perceptual/VGG loss

        Args:
            x: [B, 2, 1024] input signal
        Returns:
            features: [B, 128, trans_dim] rich spatial features
        """
        with torch.no_grad():
            features, _ = self.forward(x, return_features=True)
        return features

    def get_feature_layers(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple layers for multi-scale perceptual loss

        Args:
            x: [B, 2, 1024] input signal
        Returns:
            Dictionary of features from different layers
        """
        features = {}

        # Conv features (early spatial features)
        conv_features = self.conv_extractor(x)  # [B, 256, 128]
        features['conv'] = conv_features

        # Transformer input features
        y = conv_features.permute(0, 2, 1)
        y = self.input_proj(y)
        y = self.pos_enc(y)
        features['transformer_input'] = y  # [B, 128, trans_dim]

        # Transformer output features (richest representation)
        transformer_out = self.transformer(y)
        features['transformer_output'] = transformer_out  # [B, 128, trans_dim]

        return features

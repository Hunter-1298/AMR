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
    PSK diffusion denoiser with self-conditioning and baseline classifier

    Training Schedule:
    - Epochs 0-1: Diffusion only (reconstruction loss)
    - Epochs 2+: Diffusion + Classification
    - Training: Only SNR >= 0 dB
    - Validation: All SNR values
    """
    def __init__(
        self,
        unet: nn.Module,
        classifier: Optional[nn.Module] = None,
        label_names: Optional[List[str]] = None,
        learning_rate: float = 1e-4,
        num_train_timesteps: int = 20,  # Reduced from 1000 to 20
        # Loss weights
        lambda_t: float = 1.0,
        lambda_f: float = 1.0,
        lambda_phi: float = 1.0,
        use_phase_loss: bool = True,
        # Timestep-aware training
        use_timestep_aware_loss: bool = True,
        # Visualization settings
        log_every_n_epochs: int = 5,
        num_vis_steps: int = 20,  # Use all 20 steps for visualization
        vis_batch_size: int = 4,
        # Scheduler settings
        use_scheduler: bool = True,
        warmup_steps: int = 1000,
        # Interpolation strategy
        interpolation_type: str = "linear",
        noise_regularization: float = 0.01,
        sps: int = 8,  # samples per symbol
        max_freq_offset: float = 1e-3,  # max expected frequency offset in cycles/sample
        normalize_params: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'classifier'])
        self.sps = sps
        self.max_freq_offset = max_freq_offset
        self.normalize_params = normalize_params

        # Store the pre-instantiated model and classifier
        self.model = unet
        self.classifier = classifier
        if self.classifier is not None:
            self.classifier.eval()

        # Loss weights
        self.lambda_t = lambda_t
        self.lambda_f = lambda_f
        self.lambda_phi = lambda_phi
        self.use_phase_loss = use_phase_loss
        self.use_timestep_aware_loss = use_timestep_aware_loss

        # Metrics tracking
        self.train_losses = {
            'timing': [], 'freq': [], 'phase': [], 'total': []
        }
        self.val_losses = {
            'timing': [], 'freq': [], 'phase': [], 'total': []
        }

        # Store validation samples for visualization
        self.val_samples_stored = False
        self.stored_val_data = None

    def normalize_sync_params(self, timing, freq, phase):
        """Normalize synchronization parameters for training"""
        if not self.normalize_params:
            return timing, freq, phase

        norm_timing = timing / self.sps
        norm_freq = freq / self.max_freq_offset
        norm_phase = phase / torch.pi

        return norm_timing, norm_freq, norm_phase

    def denormalize_sync_params(self, norm_timing, norm_freq, norm_phase):
        """Convert normalized parameters back to physical units"""
        if not self.normalize_params:
            return norm_timing, norm_freq, norm_phase

        timing = norm_timing * self.sps
        freq = norm_freq * self.max_freq_offset
        phase = norm_phase * torch.pi

        return timing, freq, phase
    def get_interpolation_alpha(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get interpolation weights based on timesteps and interpolation strategy"""
        alpha = timesteps.float() / (self.hparams.num_train_timesteps - 1)  # Normalize to [0,1]

        if self.hparams.interpolation_type == "linear":
            return alpha
        elif self.hparams.interpolation_type == "cosine":
            return 0.5 * (1 - torch.cos(alpha * torch.pi))
        elif self.hparams.interpolation_type == "quadratic":
            return alpha ** 2
        else:
            return alpha

    def create_sync_interpolation(
        self,
        sync_signals: torch.Tensor,
        unsync_signals: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Create interpolation between synchronized and unsynchronized signals"""
        # Ensure all signals are float32
        sync_signals = sync_signals.float()
        unsync_signals = unsync_signals.float()

        alpha = self.get_interpolation_alpha(timesteps)
        alpha = alpha.view(-1, 1, 1).float()

        # At t=0: fully synchronized (alpha=0)
        # At t=19: fully unsynchronized (alpha=1)
        interpolated = (1 - alpha) * sync_signals + alpha * unsync_signals

        # Optional: Add small amount of noise for regularization
        if self.hparams.noise_regularization > 0:
            noise = torch.randn_like(interpolated) * self.hparams.noise_regularization
            interpolated = interpolated + noise * alpha

        # Debug: print some alpha values occasionally
        if torch.rand(1) < 0.001:  # Print very rarely
            print(f"Debug interpolation: timesteps={timesteps[:3]}, alphas={alpha[:3,0,0]}")

        return interpolated.float()

    def phase_loss(self, pred_phase: torch.Tensor, true_phase: torch.Tensor) -> torch.Tensor:
        """Special phase loss with proper dtype handling"""
        pred_phase = pred_phase.float()
        true_phase = true_phase.float()

        diff = pred_phase - true_phase
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return torch.mean(diff ** 2)

    def compute_timestep_aware_losses(
        self,
        output: Dict,
        true_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        timesteps: torch.Tensor,
        sync_signals: torch.Tensor,
        unsync_signals: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute losses with normalized parameters"""
        true_timing, true_freq, true_phase = true_params

        # Normalize true parameters
        norm_true_timing, norm_true_freq, norm_true_phase = self.normalize_sync_params(
            true_timing, true_freq, true_phase
        )

        # Ensure all tensors are float32
        norm_true_timing = norm_true_timing.float()
        norm_true_freq = norm_true_freq.float()
        norm_true_phase = norm_true_phase.float()

        if self.use_timestep_aware_loss:
            # Compute timestep-aware targets
            alpha = self.get_interpolation_alpha(timesteps).to(self.device)

            # Scale normalized target parameters based on timestep
            target_timing = norm_true_timing * alpha
            target_freq = norm_true_freq * alpha
            target_phase = norm_true_phase * alpha
        else:
            # Standard loss - always predict full correction
            target_timing = norm_true_timing
            target_freq = norm_true_freq
            target_phase = norm_true_phase

        losses = {}

        # Compute losses on normalized parameters
        losses['timing_loss'] = F.mse_loss(
            output['timing_offset'].squeeze().float(),
            target_timing
        )

        losses['freq_loss'] = F.mse_loss(
            output['freq_offset'].squeeze().float(),
            target_freq
        )

        if self.use_phase_loss:
            losses['phase_loss'] = self.phase_loss(
                output['phase_offset'].squeeze().float(),
                target_phase
            )
        else:
            losses['phase_loss'] = F.mse_loss(
                output['phase_offset'].squeeze().float(),
                target_phase
            )

        # Total offset loss
        losses['offset_loss'] = (
            self.lambda_t * losses['timing_loss'] +
            self.lambda_f * losses['freq_loss'] +
            self.lambda_phi * losses['phase_loss']
        )

        return losses

    def compute_standard_losses(
        self,
        output: Dict,
        true_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Standard loss computation for validation (always full correction)"""
        true_timing, true_freq, true_phase = true_params

        # Ensure all tensors are float32
        true_timing = true_timing.float()
        true_freq = true_freq.float()
        true_phase = true_phase.float()

        losses = {}

        # Individual parameter losses
        losses['timing_loss'] = F.mse_loss(
            output['timing_offset'].squeeze().float(),
            true_timing
        )

        losses['freq_loss'] = F.mse_loss(
            output['freq_offset'].squeeze().float(),
            true_freq
        )

        if self.use_phase_loss:
            losses['phase_loss'] = self.phase_loss(
                output['phase_offset'].squeeze().float(),
                true_phase
            )
        else:
            losses['phase_loss'] = F.mse_loss(
                output['phase_offset'].squeeze().float(),
                true_phase
            )

        # Total offset loss
        losses['offset_loss'] = (
            self.lambda_t * losses['timing_loss'] +
            self.lambda_f * losses['freq_loss'] +
            self.lambda_phi * losses['phase_loss']
        )

        return losses

    @torch.no_grad()
    def debug_timestep_conditioning(
        self,
        unsync_signal: torch.Tensor,
        modulation: torch.Tensor,
    ):
        """Debug function to check if model is using timestep information"""
        device = unsync_signal.device

        # Test all timesteps since we only have 20
        timesteps_to_test = [0, 5, 10, 15, 19]  # From synchronized to corrupted

        print("\nDebugging timestep conditioning (20 timesteps):")
        print("Expected: parameters should increase with timestep")
        print("t=0 (sync): small corrections needed")
        print("t=19 (corrupted): large corrections needed")
        print("-" * 60)

        for t in timesteps_to_test:
            t_batch = torch.full((1,), t, device=device)

            # Get model prediction
            output = self.model(unsync_signal[:1], t_batch, modulation[:1], return_dict=True)

            alpha = self.get_interpolation_alpha(torch.tensor([t])).item()

            print(f"Timestep {t:2d} (α={alpha:.2f}): "
                  f"timing={output['timing_offset'][0,0].item():7.4f}, "
                  f"freq={output['freq_offset'][0,0].item():7.4f}, "
                  f"phase={output['phase_offset'][0,0].item():7.4f}")

        print("-" * 60)

    def training_step(self, batch, batch_idx):
        """Enhanced training step with timestep-aware targets"""
        sync_signals, unsync_signals, labels, snrs, sync_params = batch
        # import pdb; pdb.set_trace()
        batch_size = sync_signals.shape[0]

        # Sample random timesteps from 0 to 19
        timesteps = torch.randint(
            0, self.hparams.num_train_timesteps,
            (batch_size,), device=self.device
        )

        # Create interpolated signals between sync and unsync
        interpolated_signals = self.create_sync_interpolation(
            sync_signals, unsync_signals, timesteps
        )

        # Forward pass
        output = self.model(interpolated_signals, timesteps, labels, return_dict=True)

        # Compute timestep-aware losses
        losses = self.compute_timestep_aware_losses(
            output, sync_params, timesteps, sync_signals, unsync_signals
        )

        # Logging with reduced frequency to avoid spam
        if batch_idx % 100 == 0:
            for loss_name, loss_value in losses.items():
                self.log(f'train/{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)

            # Log timestep statistics
            self.log('train/avg_timestep', timesteps.float().mean(), on_step=True)
            self.log('train/max_timestep', timesteps.float().max(), on_step=True)
            self.log('train/min_timestep', timesteps.float().min(), on_step=True)

            # Log interpolation alpha statistics
            alphas = self.get_interpolation_alpha(timesteps)
            self.log('train/avg_alpha', alphas.mean(), on_step=True)

        # Track for epoch-end statistics
        self.train_losses['timing'].append(losses['timing_loss'].item())
        self.train_losses['freq'].append(losses['freq_loss'].item())
        self.train_losses['phase'].append(losses['phase_loss'].item())
        self.train_losses['total'].append(losses['offset_loss'].item())

        return losses['offset_loss']

    def validation_step(self, batch, batch_idx):
        """Validation step with comprehensive evaluation"""
        sync_signals, unsync_signals, labels, snrs, sync_params = batch
        batch_size = sync_signals.shape[0]

        # Store first batch for visualization
        if batch_idx == 0 and not self.val_samples_stored:
            self.stored_val_data = {
                'sync_signals': sync_signals[:self.hparams.vis_batch_size].cpu(),
                'unsync_signals': unsync_signals[:self.hparams.vis_batch_size].cpu(),
                'labels': labels[:self.hparams.vis_batch_size].cpu(),
                'snrs': snrs[:self.hparams.vis_batch_size].cpu(),
                'sync_params': (
                    sync_params[0][:self.hparams.vis_batch_size].cpu(),
                    sync_params[1][:self.hparams.vis_batch_size].cpu(),
                    sync_params[2][:self.hparams.vis_batch_size].cpu(),
                )
            }
            self.val_samples_stored = True

        # Evaluate at different timesteps for comprehensive validation
        timestep_evaluations = [0, 5, 10, 15, 19]  # Adjusted for 20 timesteps
        all_losses = []
        all_outputs = []  # Store model outputs for parameter accuracy

        for t in timestep_evaluations:
            timesteps = torch.full((batch_size,), t, device=self.device)

            # Create interpolated signals for this timestep
            interpolated_signals = self.create_sync_interpolation(
                sync_signals, unsync_signals, timesteps
            )

            with torch.no_grad():
                output = self.model(interpolated_signals, timesteps, labels, return_dict=True)

            # Store model output
            all_outputs.append(output)

            # For validation, use standard loss (full correction expected)
            losses = self.compute_standard_losses(output, sync_params)
            all_losses.append(losses)

            # Log timestep-specific metrics for first batch only
            if batch_idx == 0:
                for loss_name, loss_value in losses.items():
                    if loss_name != 'offset_loss':
                        self.log(f'val/{loss_name}_t{t}', loss_value)

        # Average losses across timesteps
        avg_losses = {}
        for key in all_losses[0].keys():
            avg_losses[key] = torch.stack([loss[key] for loss in all_losses]).mean()

        # Log averaged validation metrics
        for loss_name, loss_value in avg_losses.items():
            self.log(f'val/{loss_name}', loss_value, on_epoch=True)

        self.val_losses['timing'].append(avg_losses['timing_loss'].item())
        self.val_losses['freq'].append(avg_losses['freq_loss'].item())
        self.val_losses['phase'].append(avg_losses['phase_loss'].item())
        self.val_losses['total'].append(avg_losses['offset_loss'].item())

        # Debug timestep conditioning and parameter accuracy for first batch
        if batch_idx == 0:
            # Use the model output from worst case (t=19) for parameter accuracy
            worst_case_output = all_outputs[-1]
            self.log_parameter_accuracy(worst_case_output, sync_params, labels)
            self.debug_timestep_conditioning(unsync_signals[:1], labels[:1])

        return avg_losses['offset_loss']

    def log_parameter_accuracy(self, output, true_params, labels):
        """Log parameter prediction accuracy metrics"""
        true_timing, true_freq, true_phase = true_params

        # Check if output is a losses dict or model output dict
        if 'timing_offset' in output:
            # This is a model output dict
            pred_timing = output['timing_offset'].squeeze()
            pred_freq = output['freq_offset'].squeeze()
            pred_phase = output['phase_offset'].squeeze()
        else:
            # This is a losses dict, we can't compute accuracy from losses
            print("Warning: Cannot compute parameter accuracy from loss dict")
            return

        # Mean absolute errors
        timing_mae = F.l1_loss(pred_timing.float(), true_timing.float())
        freq_mae = F.l1_loss(pred_freq.float(), true_freq.float())
        phase_mae = F.l1_loss(pred_phase.float(), true_phase.float())

        self.log('val/timing_mae', timing_mae)
        self.log('val/freq_mae', freq_mae)
        self.log('val/phase_mae', phase_mae)

        # Log parameter ranges for monitoring
        self.log('val/timing_range', pred_timing.max() - pred_timing.min())
        self.log('val/freq_range', pred_freq.max() - pred_freq.min())
        self.log('val/phase_range', pred_phase.max() - pred_phase.min())

    def apply_predicted_sync(
        self,
        signal: torch.Tensor,
        timing_offset: torch.Tensor,
        freq_offset: torch.Tensor,
        phase_offset: torch.Tensor,
    ) -> torch.Tensor:
        """Apply predicted synchronization parameters to signal"""

        # Denormalize parameters to physical units
        timing_physical, freq_physical, phase_physical = self.denormalize_sync_params(
            timing_offset.squeeze(), freq_offset.squeeze(), phase_offset.squeeze()
        )

        batch_size, channels, length = signal.shape
        device = signal.device

        # Ensure correct input shape [B, 2, L]
        assert channels == 2, f"Expected 2 channels (I/Q), got {channels}"
        assert signal.dim() == 3, f"Expected 3D input [B, 2, L], got {signal.shape}"

        # Convert I/Q to complex
        complex_signal = signal[:, 0] + 1j * signal[:, 1]  # [B, L]

        # Apply timing offset (simple circular shift)
        timing_samples = timing_physical.round().long()  # [B]
        shifted_signals = []
        for i, shift in enumerate(timing_samples):
            shifted = torch.roll(complex_signal[i], -shift.item(), dims=0)
            shifted_signals.append(shifted)
        shifted_signal = torch.stack(shifted_signals, dim=0)  # [B, L]

        # Apply frequency and phase correction
        t = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(0)  # [1, L]

        # Create correction phasor using physical units
        freq_phase_correction = (
            -2 * torch.pi * freq_physical.unsqueeze(-1) * t / length -
            phase_physical.unsqueeze(-1)
        )
        correction_phasor = torch.exp(1j * freq_phase_correction)  # [B, L]

        # Apply correction
        corrected_signal = shifted_signal * correction_phasor  # [B, L]

        # Convert back to I/Q format
        output_signal = torch.stack([
            corrected_signal.real,
            corrected_signal.imag
        ], dim=1)  # [B, 2, L]

        return output_signal

    def apply_fractional_delay(self, signal: torch.Tensor, delay_samples: torch.Tensor) -> torch.Tensor:
        """
        Apply fractional delay using linear interpolation
        """
        batch_size, length = signal.shape
        device = signal.device

        # Create time indices
        t_indices = torch.arange(length, device=device, dtype=torch.float32)  # [L]

        # Apply delay: positive delay means signal arrives later, so we shift left (subtract)
        delayed_indices = t_indices.unsqueeze(0) - delay_samples.unsqueeze(-1)  # [B, L]

        # Wrap around using modulo for circular buffer effect
        delayed_indices = delayed_indices % length

        # Linear interpolation
        floor_indices = torch.floor(delayed_indices).long()
        ceil_indices = (floor_indices + 1) % length

        # Interpolation weights
        weights = delayed_indices - floor_indices.float()

        # Gather values and interpolate
        signal_floor = torch.gather(signal, 1, floor_indices)
        signal_ceil = torch.gather(signal, 1, ceil_indices)

        interpolated = signal_floor * (1 - weights) + signal_ceil * weights

        return interpolated

    @torch.no_grad()
    def iterative_synchronization_for_gif(
        self,
        unsync_signal: torch.Tensor,
        modulation: torch.Tensor,
        num_steps: int = 20
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Proper iterative synchronization - apply corrections to previous iteration's result
        """
        device = unsync_signal.device
        original_corrupted = unsync_signal.clone()  # Keep for reference

        # Start with the corrupted signal
        current_signal = unsync_signal.clone()

        # Store progression
        progression = []

        # Reverse diffusion schedule (from most corrupted to synchronized)
        timesteps = torch.linspace(
            self.hparams.num_train_timesteps - 1, 0, num_steps
        ).long().to(device)

        for step, t in enumerate(timesteps):
            t_batch = torch.full((current_signal.shape[0],), t, device=device)

            # Get model prediction for current signal state
            output = self.model(current_signal, t_batch, modulation, return_dict=True)

            # Apply corrections to the CURRENT signal (not original)
            corrected_signal = self.apply_predicted_sync(
                current_signal,  # Apply to current iteration's result
                output['timing_offset'],
                output['freq_offset'],
                output['phase_offset']
            )

            # Store step info
            step_info = {
                'step': step,
                'timestep': t.item(),
                'signal': corrected_signal.cpu().clone(),
                'input_signal': current_signal.cpu().clone(),  # What we fed to the model
                'timing_offset': output['timing_offset'].cpu().clone(),
                'freq_offset': output['freq_offset'].cpu().clone(),
                'phase_offset': output['phase_offset'].cpu().clone(),
            }

            progression.append(step_info)

            # Update current signal for next iteration
            current_signal = corrected_signal.detach()

            # Optional: Add some debug info
            if step % 5 == 0:  # Print every 5 steps
                print(f"Step {step}, t={t.item()}: "
                      f"timing={output['timing_offset'][0,0].item():.4f}, "
                      f"freq={output['freq_offset'][0,0].item():.4f}, "
                      f"phase={output['phase_offset'][0,0].item():.4f}")

        return current_signal, progression

    @torch.no_grad()
    def single_step_synchronization_for_gif(
        self,
        unsync_signal: torch.Tensor,
        modulation: torch.Tensor,
        num_steps: int = 20
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Alternative: Show different timestep predictions without iteration
        (Keep this for comparison purposes)
        """
        device = unsync_signal.device
        original_corrupted = unsync_signal.clone()

        progression = []
        timesteps = torch.linspace(self.hparams.num_train_timesteps - 1, 0, num_steps).long().to(device)

        for step, t in enumerate(timesteps):
            t_batch = torch.full((original_corrupted.shape[0],), t, device=device)

            # Get model prediction for this timestep
            output = self.model(original_corrupted, t_batch, modulation, return_dict=True)

            # Apply to original signal (non-iterative)
            sync_signal = self.apply_predicted_sync(
                original_corrupted,
                output['timing_offset'],
                output['freq_offset'],
                output['phase_offset']
            )

            step_info = {
                'step': step,
                'timestep': t.item(),
                'signal': sync_signal.cpu().clone(),
                'timing_offset': output['timing_offset'].cpu().clone(),
                'freq_offset': output['freq_offset'].cpu().clone(),
                'phase_offset': output['phase_offset'].cpu().clone(),
            }

            progression.append(step_info)

        return sync_signal, progression

    def create_sync_animation(
        self,
        unsync_signals: torch.Tensor,
        true_sync_signals: torch.Tensor,
        progression: List[Dict],
        labels: torch.Tensor,
        snrs: torch.Tensor,
        num_samples: int = 2
    ):
        """
        Enhanced animation creation with better visualization of iterative progress
        """
        try:
            num_samples = min(num_samples, unsync_signals.shape[0])
            num_steps = len(progression)

            if num_steps == 0:
                print("Error: No progression data available")
                return None, None

            # Set up the figure with additional subplot for convergence metrics
            fig = plt.figure(figsize=(6*num_samples, 12))

            # Create subplots: constellation plots, parameter evolution, and convergence metrics
            gs = fig.add_gridspec(3, num_samples, height_ratios=[2, 1.5, 1])

            constellation_axes = [fig.add_subplot(gs[0, i]) for i in range(num_samples)]
            param_axes = [fig.add_subplot(gs[1, i]) for i in range(num_samples)]
            convergence_ax = fig.add_subplot(gs[2, :])  # Span all columns

            # Store plots for animation
            scatters = []
            param_lines = []

            # Sample every 16th point for symbol visualization
            symbol_stride = 1

            # Initialize constellation and parameter plots (same as before)
            for sample_idx in range(num_samples):
                try:
                    label = labels[sample_idx].item()
                    snr = snrs[sample_idx].item()

                    # Constellation plot
                    ax_const = constellation_axes[sample_idx]
                    ax_const.set_xlim(-2, 2)
                    ax_const.set_ylim(-2, 2)
                    ax_const.set_xlabel('I Channel')
                    ax_const.set_ylabel('Q Channel')
                    ax_const.set_title(f'Sample {sample_idx}: Class {label}, SNR {snr:.1f}dB')
                    ax_const.grid(True, alpha=0.3)
                    ax_const.set_aspect('equal')

                    # Plot reference signals
                    unsync_sig = unsync_signals[sample_idx]
                    true_sync_sig = true_sync_signals[sample_idx]

                    signal_length = unsync_sig.shape[1]
                    symbol_indices = torch.arange(0, signal_length, symbol_stride)

                    ax_const.scatter(
                        unsync_sig[0, symbol_indices], unsync_sig[1, symbol_indices],
                        c='red', alpha=0.5, s=20, label='Unsync', zorder=1, marker='x'
                    )
                    ax_const.scatter(
                        true_sync_sig[0, symbol_indices], true_sync_sig[1, symbol_indices],
                        c='green', alpha=0.6, s=25, label='True Sync', zorder=2, marker='o'
                    )

                    # Initialize animated scatter
                    initial_signal = progression[0]['signal'][sample_idx]
                    scatter = ax_const.scatter(
                        initial_signal[0, symbol_indices], initial_signal[1, symbol_indices],
                        c='blue', alpha=0.8, s=30, label='Iterative Sync', zorder=3, marker='s'
                    )
                    scatters.append((scatter, symbol_indices))
                    ax_const.legend()

                    # Parameter evolution plot
                    ax_params = param_axes[sample_idx]
                    ax_params.set_xlim(0, num_steps-1)
                    ax_params.set_xlabel('Iteration Step')
                    ax_params.set_ylabel('Parameter Value')
                    ax_params.set_title('Parameter Evolution')
                    ax_params.grid(True, alpha=0.3)

                    # Initialize parameter plots
                    timing_line, = ax_params.plot([], [], 'o-', label='Timing', color='blue', linewidth=2, markersize=4)
                    freq_line, = ax_params.plot([], [], 's-', label='Frequency', color='orange', linewidth=2, markersize=4)
                    phase_line, = ax_params.plot([], [], '^-', label='Phase', color='purple', linewidth=2, markersize=4)

                    param_lines.append((timing_line, freq_line, phase_line))

                    # Set parameter plot limits
                    all_timing = [progression[i]['timing_offset'][sample_idx].item() for i in range(num_steps)]
                    all_freq = [progression[i]['freq_offset'][sample_idx].item() for i in range(num_steps)]
                    all_phase = [progression[i]['phase_offset'][sample_idx].item() for i in range(num_steps)]

                    y_min = min(min(all_timing), min(all_freq), min(all_phase)) - 0.1
                    y_max = max(max(all_timing), max(all_freq), max(all_phase)) + 0.1
                    if y_min != y_max:
                        ax_params.set_ylim(y_min, y_max)
                    else:
                        ax_params.set_ylim(y_min - 0.1, y_max + 0.1)

                    ax_params.legend()

                except Exception as e:
                    print(f"Error setting up sample {sample_idx}: {e}")
                    return None, None

            # Initialize convergence metrics plot
            convergence_ax.set_xlim(0, num_steps-1)
            convergence_ax.set_xlabel('Iteration Step')
            convergence_ax.set_ylabel('Parameter Magnitude')
            convergence_ax.set_title('Convergence: Parameter Magnitudes Over Iterations')
            convergence_ax.grid(True, alpha=0.3)

            # Convergence lines (average across samples)
            conv_timing_line, = convergence_ax.plot([], [], 'o-', label='Avg |Timing|', color='blue', linewidth=2)
            conv_freq_line, = convergence_ax.plot([], [], 's-', label='Avg |Frequency|', color='orange', linewidth=2)
            conv_phase_line, = convergence_ax.plot([], [], '^-', label='Avg |Phase|', color='purple', linewidth=2)
            convergence_ax.legend()

            # Enhanced animation function
            def animate(frame):
                try:
                    updates = []

                    # Update constellation and parameter plots for each sample
                    for sample_idx in range(num_samples):
                        # Update constellation plot
                        scatter, symbol_indices = scatters[sample_idx]
                        current_signal = progression[frame]['signal'][sample_idx]

                        new_offsets = np.column_stack([
                            current_signal[0, symbol_indices].numpy(),
                            current_signal[1, symbol_indices].numpy()
                        ])
                        scatter.set_offsets(new_offsets)

                        # Update parameter plots
                        timing_line, freq_line, phase_line = param_lines[sample_idx]

                        steps_so_far = list(range(frame + 1))
                        timing_vals = [progression[i]['timing_offset'][sample_idx].item() for i in steps_so_far]
                        freq_vals = [progression[i]['freq_offset'][sample_idx].item() for i in steps_so_far]
                        phase_vals = [progression[i]['phase_offset'][sample_idx].item() for i in steps_so_far]

                        timing_line.set_data(steps_so_far, timing_vals)
                        freq_line.set_data(steps_so_far, freq_vals)
                        phase_line.set_data(steps_so_far, phase_vals)

                        updates.extend([scatter, timing_line, freq_line, phase_line])

                    # Update convergence plot (average magnitudes across samples)
                    steps_so_far = list(range(frame + 1))
                    avg_timing_mags = []
                    avg_freq_mags = []
                    avg_phase_mags = []

                    for step in steps_so_far:
                        timing_mags = [abs(progression[step]['timing_offset'][i].item()) for i in range(num_samples)]
                        freq_mags = [abs(progression[step]['freq_offset'][i].item()) for i in range(num_samples)]
                        phase_mags = [abs(progression[step]['phase_offset'][i].item()) for i in range(num_samples)]

                        avg_timing_mags.append(sum(timing_mags) / len(timing_mags))
                        avg_freq_mags.append(sum(freq_mags) / len(freq_mags))
                        avg_phase_mags.append(sum(phase_mags) / len(phase_mags))

                    conv_timing_line.set_data(steps_so_far, avg_timing_mags)
                    conv_freq_line.set_data(steps_so_far, avg_freq_mags)
                    conv_phase_line.set_data(steps_so_far, avg_phase_mags)

                    updates.extend([conv_timing_line, conv_freq_line, conv_phase_line])

                    # Update title with iteration info
                    timestep_val = progression[frame]["timestep"]
                    iteration_type = "Iterative" if hasattr(progression[frame], 'input_signal') else "Single-step"
                    fig.suptitle(f'{iteration_type} Synchronization - Step {frame}/{num_steps-1} (t={timestep_val:.0f})',
                                fontsize=14, fontweight='bold')

                    return updates

                except Exception as e:
                    print(f"Error in animation frame {frame}: {e}")
                    return []

            # Create animation
            anim = animation.FuncAnimation(
                fig, animate, frames=num_steps,
                interval=800,  # 800ms between frames
                blit=False, repeat=True
            )

            return fig, anim

        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def visualize_sync_progression(self):
        """Create animated GIF visualization of synchronization progression"""
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

            # Use ITERATIVE synchronization for proper diffusion behavior
            final_signals, progression = self.iterative_synchronization_for_gif(
                unsync_signals, labels, num_steps=20
            )

            if not progression:
                print("Error: No progression data generated")
                return

            print(f"Generated {len(progression)} progression steps")
            print("Using iterative synchronization - each step builds on the previous result")

            # Create animation
            result = self.create_sync_animation(
                unsync_signals.cpu(),
                sync_signals.cpu(),
                progression,
                labels.cpu(),
                snrs,
                num_samples=2
            )

            if result[0] is None or result[1] is None:
                print("Error: Animation creation failed")
                return

            fig, anim = result

            # Save and log animation (same as before)
            if hasattr(self, 'logger') and self.logger is not None:
                logger_class_name = self.logger.__class__.__name__
                if 'WandbLogger' in logger_class_name:
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_file:
                            writer = PillowWriter(fps=1.25)
                            anim.save(tmp_file.name, writer=writer)

                            self.logger.experiment.log({
                                "iterative_synchronization_animation": wandb.Video(tmp_file.name, fps=1.25, format="gif"),
                                "global_step": self.global_step,
                                "epoch": self.current_epoch
                            })

                        os.unlink(tmp_file.name)
                        print("Successfully logged iterative animation to WandB")

                    except Exception as e:
                        print(f"Failed to log to WandB: {e}")
                        local_filename = f'iterative_sync_animation_epoch_{self.current_epoch}.gif'
                        anim.save(local_filename, writer=PillowWriter(fps=1.25))
                        print(f"Saved animation locally as {local_filename}")
                else:
                    local_filename = f'iterative_sync_animation_epoch_{self.current_epoch}.gif'
                    anim.save(local_filename, writer=PillowWriter(fps=1.25))
                    print(f"Saved animation locally as {local_filename}")
            else:
                local_filename = f'iterative_sync_animation_epoch_{self.current_epoch}.gif'
                anim.save(local_filename, writer=PillowWriter(fps=1.25))
                print(f"Saved animation locally as {local_filename}")

            plt.close(fig)
            print("Iterative animation creation completed successfully")

        except Exception as e:
            print(f"Animation creation failed: {e}")
            import traceback
            traceback.print_exc()

    def on_train_epoch_end(self):
        """Log training epoch metrics"""
        if self.train_losses['total']:
            for loss_type, losses in self.train_losses.items():
                avg_loss = sum(losses) / len(losses)
                self.log(f'train/epoch_{loss_type}_loss', avg_loss)
                losses.clear()

    def on_validation_epoch_end(self):
        """Log validation epoch metrics and create animations"""
        # Log averaged losses
        if self.val_losses['total']:
            for loss_type, losses in self.val_losses.items():
                avg_loss = sum(losses) / len(losses)
                self.log(f'val/epoch_{loss_type}_loss', avg_loss)
                losses.clear()

        # Create animation every N epochs
        self.visualize_sync_progression()

    def configure_optimizers(self):
        """Configure optimizer and scheduler with proper step order"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        if self.hparams.use_scheduler:
            # Use step-based scheduler to avoid the warning
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos'
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',  # Step-based scheduling
                    'frequency': 1,
                    'name': 'learning_rate'
                }
            }

        return optimizer



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
            activation="gelu",
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

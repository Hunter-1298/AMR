import torch
from torchmetrics import Accuracy
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import tempfile
import os
import wandb
from typing import Optional, List, Dict

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
        num_classes: int = 4,
        conv_hidden: tuple = (32,64,128),
        trans_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        mlp_hidden: int = 256,

        # Loss weights for joint training
        lambda_sync: float = 0.0,
        lambda_class: float = 0.05,
        lambda_ber: float = 1.0,

        # Training schedule
        classification_start_epoch: int = 0,

        # Normalization parameters
        sps: int = 11,
        max_freq_offset: float = 1e-3,
        normalize_params: bool = True,

        # Visualization settings
        log_every_n_epochs: int = 1,
        vis_batch_size: int = 5,

        # Other settings
        use_scheduler: bool = True,
        interpolation_type: str = "cosine",
        noise_regularization: float = 0.01,
    ):
        super().__init__()
        # ... (other initializations like model, classifier, loss functions) ...

        self.val_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.save_hyperparameters(ignore=['unet'])
        self.curriculum_epoch = 15
        self.snr_start = 0.0
        self.snr_end =  0.0
        if self.curriculum_epoch > 0:
            self.snr_decrease_rate = (self.snr_start - self.snr_end) / self.curriculum_epoch
        else:
            self.snr_decrease_rate = 0.0


        # Store the pre-instantiated UNet
        self.model = unet
        self.sps = sps
        
        if label_names is not None:
            max_class = max(label_names.keys())
            self.label_names = [label_names.get(i, f"Class_{i}") for i in range(max_class + 1)]
        else:
            self.label_names = [f"Class_{i}" for i in range(num_classes)]

        # === INSTANTIATE CLASSIFIER ===
        self.classifier = HybridConvTransformer(
            in_ch=2,
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
        self.label_to_order = {
            0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64,
        }

    def get_psk_constellation(self, labels):
        """
        Returns PSK constellation points for given labels.
        Returns list of tensors, each (M_i, 2)
        """
        M_list = [2, 4, 8, 16, 32, 64]

        # Single label
        if isinstance(labels, int) or (isinstance(labels, torch.Tensor) and labels.dim() == 0):
            l_val = labels.item() if isinstance(labels, torch.Tensor) else labels
            M_val = M_list[l_val]
            angles = torch.linspace(0, 2*math.pi, steps=M_val+1, 
                                   device=getattr(labels, 'device', 'cpu'))[:-1]
            return [torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)]

        # Batch labels
        labels = labels.flatten()
        batch_const = []
        for l in labels:
            M_val = M_list[l.item()]
            angles = torch.linspace(0, 2*math.pi, steps=M_val+1, device=labels.device)[:-1]
            batch_const.append(torch.stack([torch.cos(angles), torch.sin(angles)], dim=1))
        return batch_const

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

        # Apply frequency and phase correction
        t = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(0)
        freq_phase_correction = (
            -2 * torch.pi * freq_physical.unsqueeze(-1) * t -
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

    def extract_symbols(self, signals, cumulative_timing=None, sps=None):
        """
        Extract symbols from 2D signal after synchronization.
        
        Args:
            signals: (B, 2, T) or (2, T)
            cumulative_timing: (B,) or scalar - additional timing shift in samples
            sps: samples per symbol
            
        Returns:
            (B, num_symbols, 2) if batch, else (num_symbols, 2)
        """
        sps = sps or self.sps

        # Ensure 3D tensor
        if signals.dim() == 2:
            signals = signals.unsqueeze(0)

        B, _, T = signals.shape
        num_symbols = T // sps

        # Base symbol indices (starting from middle of first symbol period)
        indices = torch.arange(0, num_symbols * sps, sps, device=signals.device)

        # Apply cumulative timing shifts
        if cumulative_timing is not None:
            if torch.is_tensor(cumulative_timing):
                indices = indices.unsqueeze(0) + cumulative_timing.unsqueeze(1).long()
            else:
                indices = indices + int(round(cumulative_timing))
        else:
            indices = indices.unsqueeze(0).repeat(B, 1)

        indices = torch.clamp(indices, 0, T - 1)

        I = signals[:, 0, indices]
        Q = signals[:, 1, indices]

        symbols = torch.stack([I, Q], dim=-1)  # (B, num_symbols, 2)
        if B == 1:
            symbols = symbols.squeeze(0)

        return symbols
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
    def symbol_alignment_loss(self, synced, labels, cumulative_timing=None):
        """
        Computes symbol-alignment loss focusing on angular distance (Phase Error).
        Returns tensor of shape (B, num_symbols) for fine-grained supervision.
        """
        import torch.nn.functional as F

        B, _, T = synced.shape
        s = synced.permute(0, 2, 1)  # (B, T, 2)
        batch_const = self.get_psk_constellation(labels)

        all_symbol_losses = []
        
        # Determine the constant number of symbols (should be fixed if T and SPS are fixed)
        num_symbols = T // self.sps

        for i in range(B):
            c_points = batch_const[i]       # (M_i, 2) - Ideal points (cos, sin)
            s_i = s[i]                      # (T, 2) - Synchronized signal

            # --- 1. Extract Symbols using Timing Correction ---
            indices = torch.arange(0, T, self.sps, device=s_i.device)
            if cumulative_timing is not None:
                # Apply predicted timing offset
                timing_shift = cumulative_timing[i].long()
                indices = indices + timing_shift
                indices = torch.clamp(indices, 0, T-1)

            extracted = s_i[indices]        # (num_symbols, 2) - IQ points

            # --- 2. Convert Extracted and Ideal Points to Phase (Angle) ---
            
            # Phase of Extracted Symbols: (num_symbols,)
            # torch.atan2(Q, I) returns angle in radians (-pi to pi)
            theta_r = torch.atan2(extracted[:, 1], extracted[:, 0])

            # Phase of Ideal Constellation Points: (M_i,)
            theta_s = torch.atan2(c_points[:, 1], c_points[:, 0]) 

            # --- 3. Calculate Angular Distance to Nearest Ideal Point ---

            # Expand for broadcasting: (num_symbols, 1) and (1, M_i)
            theta_r_exp = theta_r.unsqueeze(1) 
            theta_s_exp = theta_s.unsqueeze(0)

            # Calculate raw phase difference: (num_symbols, M_i)
            # This difference will be between -2*pi and 2*pi
            phase_dists = theta_r_exp - theta_s_exp 

            # Normalize phase difference to the [-pi, pi] range (modulo 2*pi)
            # This finds the shortest angular distance.
            pi = torch.pi
            normalized_dists = torch.remainder(phase_dists + pi, 2 * pi) - pi
            
            # The loss is the absolute shortest angular distance
            angular_loss = torch.abs(normalized_dists) # (num_symbols, M_i)

            # Find the minimum angular loss for each symbol
            min_angular_loss, _ = angular_loss.min(dim=1) # (num_symbols,)

            all_symbol_losses.append(min_angular_loss)

        # Pad is used for consistency, though lengths should be equal here
        # Note: If lengths are truly equal, use torch.stack for efficiency
        return torch.nn.utils.rnn.pad_sequence(all_symbol_losses, batch_first=True, padding_value=0.0)

    def iterative_synchronization_for_training(
    self,
    unsync_signal: torch.Tensor,
    modulation: torch.Tensor,
    num_steps: int = 10,
    return_history: bool = False  # NEW PARAMETER
):
        """
        Iteratively apply UNet predictions to synchronize signals.
        Can optionally return full history for loss computation.
        """
        device = unsync_signal.device
        current_signal = unsync_signal
        timesteps = torch.linspace(
            self.hparams.num_train_timesteps - 1, 0, num_steps
        ).long().to(device)

        # Track cumulative timing offset
        cumulative_timing = torch.zeros(current_signal.shape[0], device=device)
        
        # Store history if requested
        history = [] if return_history else None

        for step, t in enumerate(timesteps):
            t_batch = torch.full((current_signal.shape[0],), t, device=device)

            # Get model predictions
            output = self.model(current_signal, t_batch, modulation, return_dict=True)

            # Apply synchronization corrections
            corrected_signal = self.apply_predicted_sync(
                current_signal,
                output['timing_offset'],
                output['freq_offset'],
                output['phase_offset']
            )

            # Accumulate timing offset (in physical units)
            timing_physical, _, _ = self.denormalize_sync_params(
                output['timing_offset'].squeeze(-1),
                output['freq_offset'].squeeze(-1),
                output['phase_offset'].squeeze(-1)
            )
            cumulative_timing += timing_physical

            # Store history if requested
            if return_history:
                history.append({
                    'signal': corrected_signal,
                    'cumulative_timing': cumulative_timing.clone()
                })

            # Detach intermediate steps, keep final step for backprop
            if step < num_steps - 1:
                current_signal = corrected_signal.detach()
            else:
                current_signal = corrected_signal

        if return_history:
            return current_signal, history
        return current_signal

    def iterative_synchronization_for_gif(self, unsync_signal, labels, num_steps=10):
        """
        Runs iterative sync specifically for visualization.
        Differences from training version:
        1. Runs classifier at EVERY step to track confidence evolution.
        2. Returns a list of dicts with specific keys expected by the animator.
        """
        device = unsync_signal.device
        current_signal = unsync_signal.clone()
        
        # Create timesteps
        timesteps = torch.linspace(
            self.hparams.num_train_timesteps - 1, 0, num_steps
        ).long().to(device)

        cumulative_timing = torch.zeros(current_signal.shape[0], device=device)
        progression = []

        # --- Initial State (Step 0) ---
        # We capture the state BEFORE any correction for the first frame
        with torch.no_grad():
            initial_logits = self.classifier(current_signal)
            initial_probs = torch.softmax(initial_logits, dim=1)
            
            progression.append({
                'signal': current_signal.detach().cpu(),
                'cumulative_timing_offset': cumulative_timing.clone().cpu(),
                'timing_offset': torch.zeros(current_signal.shape[0]).cpu(), # Delta is 0
                'freq_offset': torch.zeros(current_signal.shape[0]).cpu(),
                'phase_offset': torch.zeros(current_signal.shape[0]).cpu(),
                'class_probs': initial_probs.cpu()
            })

        # --- Iterative Loop ---
        for step, t in enumerate(timesteps):
            t_batch = torch.full((current_signal.shape[0],), t, device=device)

            # 1. Get Sync Corrections
            output = self.model(current_signal, t_batch, labels, return_dict=True)

            # 2. Apply Corrections
            corrected_signal = self.apply_predicted_sync(
                current_signal,
                output['timing_offset'],
                output['freq_offset'],
                output['phase_offset']
            )
            
            # 3. Denormalize parameters for logging/plotting
            timing_phys, freq_phys, phase_phys = self.denormalize_sync_params(
                output['timing_offset'].squeeze(-1),
                output['freq_offset'].squeeze(-1),
                output['phase_offset'].squeeze(-1)
            )
            
            # Update cumulative timing (needed for the Scatter Plot index calculation)
            cumulative_timing += timing_phys

            # 4. Run Classifier on the INTERMEDIATE signal
            # This is expensive, so we only do it in this GIF generation method
            class_logits = self.classifier(corrected_signal)
            class_probs = torch.softmax(class_logits, dim=1)

            # 5. Store Data
            progression.append({
                'signal': corrected_signal.detach().cpu(),
                'cumulative_timing_offset': cumulative_timing.clone().cpu(),
                # Store the *incremental* correction for the parameter plot lines
                'timing_offset': timing_phys.detach().cpu(), 
                'freq_offset': freq_phys.detach().cpu(),
                'phase_offset': phase_phys.detach().cpu(),
                'class_probs': class_probs.detach().cpu()
            })

            # Prepare for next step
            current_signal = corrected_signal

        return current_signal, progression

    def training_step(self, batch, batch_idx):
            """Training step with step-wise synchronization loss"""
            unsync_signals, (labels, snrs) = batch 
            if self.current_epoch < self.curriculum_epoch:
                # linealry decrease min_snr to allow harder signals
                curr_min_snr = max(self.snr_end, self.snr_start - self.current_epoch * self.snr_decrease_rate)
                self.log("min_snr_in_data", curr_min_snr, on_step=True, on_epoch=True, prog_bar=True)
            else:
                curr_min_snr = 0
            mask = snrs >= curr_min_snr
            unsync_signals = unsync_signals[mask]
            labels = labels[mask]
            snrs = snrs[mask]


            # Run iterative synchronization and get full history
            synced, history = self.iterative_synchronization_for_training(
                unsync_signals, labels, 
                num_steps=self.hparams.num_train_timesteps, 
                return_history=True 
            )

            # Classification Loss
            class_logits = self.classifier(synced)
            classification_loss = self.classification_criterion(class_logits, labels)
            class_preds = torch.argmax(class_logits, dim=1)
            class_acc = (class_preds == labels).float().mean()

            # Step-wise Synchronization Loss
            L_sync_total = 0.0
            num_steps = len(history)
            
            for step_idx, step_data in enumerate(history):
                # BER loss for this step: (B, num_symbols)
                ber_loss_per_symbol = self.symbol_alignment_loss(
                    step_data['signal'], 
                    labels, 
                    cumulative_timing=step_data['cumulative_timing']
                )
                
                # Weight: give more importance to later steps (closer to t=0)
                step_weight = (step_idx + 1) / num_steps
                L_sync_total += step_weight * ber_loss_per_symbol

            # Average step-wise loss (L_sync_total is (B, N_sym))
            L_sync_avg = L_sync_total / num_steps
            
            # 1. REDUCE across symbols (N_sym) to get loss per sample (B)
            L_sync_per_sample = L_sync_avg.mean(dim=1) 
            
            # 2. REDUCE across batch (B) to get the final scalar loss
            L_sync_avg_scalar = L_sync_per_sample.mean() # SCALAR loss for logging/total_loss

            # Dynamic weighting: decay sync loss over training
            lambda_sync = self.hparams.lambda_ber
            global_step = self.global_step
            max_steps = self.trainer.estimated_stepping_batches
            sync_weight = lambda_sync * (1.0 - global_step / max_steps)
            
            # Total loss
            # Use the SCALAR loss in the final calculation
            total_loss = classification_loss * self.hparams.lambda_class + (sync_weight * L_sync_avg_scalar)

            # Logging (Use the SCALAR value)
            self.log("train/class_loss", classification_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/class_acc", class_acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/sync_loss", L_sync_avg_scalar, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/sync_weight", sync_weight, on_step=True, on_epoch=True)
            self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)

            return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step with data collection for visualization"""
        unsync_signals, (labels, snrs) = batch
        
        # Store first batch for visualization
        if batch_idx == 0 and not self.val_samples_stored:
            self.stored_val_data = {
                'unsync_signals': unsync_signals[:self.hparams.vis_batch_size].cpu(),
                'labels': labels[:self.hparams.vis_batch_size].cpu(),
                'snrs': snrs[:self.hparams.vis_batch_size].cpu(),
            }
            self.val_samples_stored = True
        
        with torch.no_grad():
            # Synchronize signals
            fully_synchronized_signals = self.iterative_synchronization_for_training(
                unsync_signals, labels, 
                num_steps=self.hparams.num_train_timesteps, 
                return_history=False 
            )
            
            # Classification
            class_logits = self.classifier(fully_synchronized_signals)
            class_loss = self.classification_criterion(class_logits, labels)
            class_preds = torch.argmax(class_logits, dim=1)
            class_acc = (class_preds == labels).float().mean()
            
            # Store outputs for SNR analysis
            self.validation_step_outputs.append({
                'predictions': class_preds.cpu(),
                'labels': labels.cpu(),
                'snrs': snrs.cpu(),
                'class_loss': class_loss.cpu(),
            })
        
        # Logging
        self.log('val/class_loss', class_loss, on_epoch=True, prog_bar=True)
        self.log('val/class_acc', class_acc, on_epoch=True, prog_bar=True)
        
        # Per-class accuracy
        for i, label_name in enumerate(self.label_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_acc_per_class = (class_preds[class_mask] == labels[class_mask]).float().mean()
                self.log(f'val/class_acc_{label_name}', class_acc_per_class, on_epoch=True)
        
        return class_loss


    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        
        # Skip during sanity check
        if self.trainer.sanity_checking:
            print("Skipping visualization during sanity check.")
            self.validation_step_outputs.clear()
            return
        
        # Create SNR vs accuracy plot
        if self.validation_step_outputs:
            self.create_snr_vs_accuracy_plot()
        
        # Create synchronization animation (every N epochs)
        if self.current_epoch % self.hparams.log_every_n_epochs == 0:
            self.visualize_sync_progression()
        
        # Clear outputs
        self.validation_step_outputs.clear()


    def visualize_sync_progression(self):
        """Generate synchronization animation using stored validation data"""
        if not self.val_samples_stored or self.stored_val_data is None:
            print("No validation data stored for visualization")
            return
        
        # --- Configuration ---
        NUM_PLOTS = 4  # The number of samples we want to visualize
        WANTED_LABELS = {0, 1, 2, 3}
        
        try:
            # Get stored data
            unsync_signals_all = self.stored_val_data['unsync_signals'].to(self.device)
            labels_all = self.stored_val_data['labels'].to(self.device)
            snrs_all = self.stored_val_data['snrs']
            
            # --- 1. Select Samples Based on Class ---
            
            # Find indices for wanted classes
            selected_indices = []
            found_labels = set()
            
            # Convert labels to a list/tensor we can iterate over easily
            labels_list = labels_all.cpu().tolist()
            
            # First, prioritize one sample from each wanted class
            for idx, label in enumerate(labels_list):
                if label in WANTED_LABELS and label not in found_labels:
                    selected_indices.append(idx)
                    found_labels.add(label)
                    if len(selected_indices) >= NUM_PLOTS:
                        break
            
            # If needed, fill remaining slots with any other available samples
            if len(selected_indices) < NUM_PLOTS:
                for idx in range(len(labels_list)):
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        if len(selected_indices) >= NUM_PLOTS:
                            break
                            
            # Check if we found any samples at all
            if not selected_indices:
                print("Could not find any suitable samples to visualize.")
                return

            # --- 2. Subset the Data ---
            
            # Ensure selected_indices is a torch tensor for clean indexing
            selected_indices_tensor = torch.tensor(selected_indices, device=self.device)
            
            unsync_signals_subset = unsync_signals_all[selected_indices_tensor]
            labels_subset = labels_all[selected_indices_tensor]
            snrs_subset = snrs_all[selected_indices] # snrs_all is likely a NumPy array or list
            
            print(f"Generating animation for {unsync_signals_subset.shape[0]} selected samples...")

            # --- 3. Run Dedicated Synchronization Method ---
            # ONLY the selected subset is passed here.
            with torch.no_grad():
                final_signals, progression = self.iterative_synchronization_for_gif(
                    unsync_signals_subset, labels_subset, 
                    num_steps=self.hparams.num_train_timesteps
                )
            
            if not progression:
                print("Error: No progression data generated")
                return
            
            print(f"Generated {len(progression)} progression steps")
            
            # --- 4. Create Animation ---
            # Pass the selected subsets to the visualization function
            filename = self.create_sync_animation_with_classification(
                unsync_signals_subset.cpu(),
                progression,
                labels_subset.cpu(),
                snrs_subset,
                num_samples=unsync_signals_subset.shape[0] # Set num_samples to the exact size of the subset
            )
            
            if filename:
                print(f"Animation created successfully: {filename}")
            
        except Exception as e:
            print(f"Error in visualize_sync_progression: {e}")
            import traceback
            traceback.print_exc()


    def create_snr_vs_accuracy_plot(self):
        """Create SNR vs Classification Accuracy plot"""
        if not self.validation_step_outputs:
            print("No validation outputs available for SNR analysis")
            return

        try:
            # Aggregate validation outputs
            all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
            all_snrs = torch.cat([x['snrs'] for x in self.validation_step_outputs])
            
            # Define SNR bins
            snr_bins = [(-20, -18), (-18, -16), (-16, -14), (-14, -12), (-12, -10),
                        (-10, -8), (-8, -6), (-6, -4), (-4, -2), (-2, 0),
                        (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12),
                        (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
                        (22, 24), (24, 26), (26, 28), (28, 30)]
            snr_centers = [(low + high) / 2 for low, high in snr_bins]

            # Calculate accuracies per SNR bin
            overall_accuracies = []
            class_accuracies = {label_name: [] for label_name in self.label_names}

            for snr_min, snr_max in snr_bins:
                snr_mask = (all_snrs >= snr_min) & (all_snrs < snr_max)

                if snr_mask.sum() > 0:
                    overall_acc = (all_preds[snr_mask] == all_labels[snr_mask]).float().mean().item()
                    overall_accuracies.append(overall_acc)

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

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.plot(snr_centers, overall_accuracies, 'ko-', linewidth=3, markersize=8,
                    label='Overall Accuracy', zorder=3)

            colors = ['red', 'blue', 'green', 'orange', 'purple']
            markers = ['s', '^', 'D', 'v', '<']

            for idx, (label_name, accuracies) in enumerate(class_accuracies.items()):
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                ax.plot(snr_centers, accuracies, color=color, marker=marker,
                        linewidth=2, markersize=6, label=str(label_name), alpha=0.8)

            num_classes = len(self.label_names)
            ax.axhline(y=1/num_classes, color='gray', linestyle='--', alpha=0.5,
                    label=f'Random Guess ({1/num_classes:.3f})')

            ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Classification Accuracy', fontsize=14, fontweight='bold')
            ax.set_title(f'Classification Accuracy vs SNR - Epoch {self.current_epoch}\n'
                        f'Synchronized Signals (Joint Training)', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='lower right', frameon=True, fancybox=True, shadow=True)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(-22, 32)

            # Sample count annotations
            for i, (snr_min, snr_max) in enumerate(snr_bins):
                snr_mask = (all_snrs >= snr_min) & (all_snrs < snr_max)
                count = snr_mask.sum().item()
                if count > 0:
                    ax.annotate(f'n={count}', (snr_centers[i], 0.02),
                            ha='center', fontsize=8, alpha=0.7)

            # Statistics text box
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
                    import wandb
                    self.logger.experiment.log({
                        "snr_vs_accuracy": wandb.Image(fig),
                        "epoch": self.current_epoch
                    })
                    print("✓ SNR vs Accuracy plot logged to WandB")

            # Save locally
            plot_filename = f'snr_vs_accuracy_epoch_{self.current_epoch}.png'
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✓ SNR vs Accuracy plot saved as {plot_filename}")

            plt.close(fig)

        except Exception as e:
            print(f"✗ Error creating SNR vs Accuracy plot: {e}")
            import traceback
            traceback.print_exc()


    def create_sync_animation_with_classification(
        self,
        unsync_signals: torch.Tensor,
        progression: list,
        labels: torch.Tensor,
        snrs: torch.Tensor,
        num_samples: int = 4
    ):
        """
        Create animation showing synchronization progression.
        Visuals:
        1. Gold 'X': Ideal Constellation (Static)
        2. Red Dots: Original Unsynchronized Signal (Static Reference at t=0)
        3. Blue Dots: Synchronized Signal (Dynamic/Evolving)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            import numpy as np
            
            num_samples = min(num_samples, unsync_signals.shape[0])
            num_steps = len(progression)
            
            if num_steps == 0:
                print("✗ No progression data available")
                return None

            # Constants
            SPS = getattr(self.hparams, 'sps', 1)
            signal_length = unsync_signals.shape[-1]
            
            # We sample at the nominal grid (0, 10, 20...)
            # Since apply_predicted_sync shifts the waveform to align with this grid,
            # we don't need to calculate fractional indices for plotting.
            base_symbol_indices = torch.arange(0, signal_length, SPS)

            # Setup figure
            fig = plt.figure(figsize=(6*num_samples, 14))
            gs = fig.add_gridspec(3, num_samples, height_ratios=[1, 0.8, 0.8], hspace=0.4, wspace=0.3)

            scatters = []
            param_lines = []

            # Initialize plots for each sample
            for sample_idx in range(num_samples):
                label = labels[sample_idx].item()
                snr = snrs[sample_idx].item()
                class_name = self.label_names[label] if isinstance(self.label_names, list) else f"Class_{label}"
                
                # Get the initial Unsynchronized signal for this sample
                unsync_sig = unsync_signals[sample_idx]

                # --- Row 1: Constellation Diagram ---
                ax_const = fig.add_subplot(gs[0, sample_idx])
                ax_const.set_xlim(-2.5, 2.5)
                ax_const.set_ylim(-2.5, 2.5)
                ax_const.set_xlabel('In-Phase (I)')
                ax_const.set_ylabel('Quadrature (Q)')
                ax_const.set_title(f'Sample {sample_idx}: {class_name}, SNR {snr:.1f}dB')
                ax_const.grid(True, alpha=0.3)
                ax_const.set_aspect('equal')

                # 1. Ideal constellation (Gold X) - Static
                theoretical_const = self.get_psk_constellation(labels[sample_idx])[0]
                ax_const.scatter(theoretical_const[:,0].numpy(), theoretical_const[:,1].numpy(),
                                c='gold', marker='x', s=80, linewidths=2, label='Ideal', zorder=5)

                # 2. Original Unsynchronized Signal (Red Dots) - Static Reference
                # Plotting the raw signal at base indices to show where it started
                ax_const.scatter(unsync_sig[0, base_symbol_indices], unsync_sig[1, base_symbol_indices],
                                c='red', alpha=0.3, s=20, label='Original', marker='o', zorder=2)

                # 3. Synchronized Symbols (Blue Dots) - Dynamic
                # Initialized empty here; 'animate' function will update data
                scatter_synced = ax_const.scatter([], [], c='blue', alpha=0.9, s=35, label='Synced', marker='o', zorder=4)
                
                ax_const.legend(loc='upper right', fontsize=8)
                
                # Store only the dynamic scatter plot
                scatters.append(scatter_synced)

                # --- Row 2: Sync Parameters ---
                ax_params = fig.add_subplot(gs[1, sample_idx])
                ax_params.set_xlim(0, num_steps-1)
                ax_params.set_xlabel('Step')
                ax_params.set_title('Sync Parameters')
                ax_params.grid(True, alpha=0.3)

                timing_line, = ax_params.plot([], [], 'o-', label='Timing', color='blue', markersize=4)
                freq_line, = ax_params.plot([], [], 's-', label='Freq', color='orange', markersize=4)
                phase_line, = ax_params.plot([], [], '^-', label='Phase', color='purple', markersize=4)
                ax_params.legend(fontsize=8)

                # --- Row 3: Classification Confidence ---
                ax_class = fig.add_subplot(gs[2, sample_idx])
                ax_class.set_xlim(0, num_steps-1)
                ax_class.set_ylim(0, 1.05)
                ax_class.set_xlabel('Step')
                ax_class.set_ylabel('Probability')
                ax_class.set_title('Classifier Confidence')
                ax_class.grid(True, alpha=0.3)

                class_lines = []
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                for c_idx, c_name in enumerate(self.label_names):
                    col = colors[c_idx % len(colors)]
                    lw = 2.5 if c_idx == label else 1.0
                    alpha_val = 1.0 if c_idx == label else 0.4
                    line, = ax_class.plot([], [], label=str(c_name), color=col, linewidth=lw, alpha=alpha_val)
                    class_lines.append(line)
                ax_class.legend(loc='upper left', fontsize=8)

                param_lines.append((timing_line, freq_line, phase_line, class_lines))

            # Animation update function
            def animate(frame):
                updates = []
                
                for s_idx in range(num_samples):
                    scatter_sync = scatters[s_idx]
                    
                    # Get the synchronized signal for this step
                    current_sig = progression[frame]['signal'][s_idx].cpu().numpy()
                    
                    # Update Synced Symbols (Blue Dots)
                    # Because ApplyPredictedSync shifts the waveform, the symbol centers 
                    # should now align with base_symbol_indices.
                    sync_offsets = np.column_stack((
                        current_sig[0, base_symbol_indices], 
                        current_sig[1, base_symbol_indices]
                    ))
                    scatter_sync.set_offsets(sync_offsets)
                    updates.append(scatter_sync)
                    
                    # Update parameter lines
                    timing_l, freq_l, phase_l, class_l = param_lines[s_idx]
                    steps = range(frame + 1)
                    
                    t_hist = [progression[i]['timing_offset'][s_idx].item() for i in steps]
                    f_hist = [progression[i]['freq_offset'][s_idx].item() for i in steps]
                    p_hist = [progression[i]['phase_offset'][s_idx].item() for i in steps]
                    
                    timing_l.set_data(steps, t_hist)
                    freq_l.set_data(steps, f_hist)
                    phase_l.set_data(steps, p_hist)
                    updates.extend([timing_l, freq_l, phase_l])
                    
                    # Auto-scale parameters
                    if frame > 0:
                        all_params = t_hist + f_hist + p_hist
                        p_min, p_max = min(all_params), max(all_params)
                        margin = (p_max - p_min) * 0.1 if p_max != p_min else 0.1
                        timing_l.axes.set_ylim(p_min - margin, p_max + margin)
                    
                    # Update classification
                    for cls_i, line in enumerate(class_l):
                        prob_hist = [progression[i]['class_probs'][s_idx, cls_i].item() for i in steps]
                        line.set_data(steps, prob_hist)
                        updates.append(line)
                
                fig.suptitle(f'Synchronization Progress - Step {frame+1}/{num_steps}', fontsize=14, fontweight='bold')
                return updates

            # Generate animation
            ani = FuncAnimation(fig, animate, frames=num_steps, interval=1000, blit=False)
            
            filename = f'sync_progression_epoch_{self.current_epoch}.gif'
            ani.save(filename, writer='pillow', fps=1)
            print(f"✓ Animation saved: {filename}")
            
            # Log to wandb
            if hasattr(self, 'logger') and self.logger is not None:
                logger_name = self.logger.__class__.__name__
                if 'WandbLogger' in logger_name:
                    try:
                        import wandb
                        self.logger.experiment.log({
                            "synchronization_animation": wandb.Video(filename, fps=1, format="gif"),
                            "epoch": self.current_epoch
                        })
                        print("✓ Animation logged to WandB")
                    except Exception as e:
                        print(f"✗ Failed to log to WandB: {e}")

            plt.close(fig)
            return filename

        except Exception as e:
            print(f"✗ Error creating animation: {e}")
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
        label_names: List[str] = ["qpsk", "bpsk"],
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
        # self.criterion = nn.CrossEntropyLoss()

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
        unsync_signals, (labels, snrs) = batch


        # Classify the raw corrupted signals directly
        logits = self.classifier(unsync_signals)
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
        unsync_signals, (labels, snrs) = batch

        # Classify the raw corrupted signals directly
        logits = self.classifier(unsync_signals)
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
            "corrupted_signals": unsync_signals[:4].detach().cpu() if batch_idx == 0 else None,
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
    def __init__(self, in_ch=2, num_classes=3, conv_hidden=(128, 256, 512),
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
        self.pos_enc = PositionalEncoding(trans_dim, max_len=513)  # +1 for CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim, nhead=n_heads,
            dim_feedforward=trans_dim * 4,
            dropout=0.2, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # LayerNorm before classifier
        self.norm = nn.LayerNorm(trans_dim)

        # Classifier Head with Residual MLP
        self.classifier = nn.Sequential(
            nn.Linear(trans_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, num_classes),
        )
    def forward(self, x):
        # x [32,2,4096]
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

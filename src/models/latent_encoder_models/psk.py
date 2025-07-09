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

class DDPMScheduler(nn.Module):
    """Standard DDPM scheduler for AWGN noise"""

    def __init__(self, n_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, schedule: str = 'cosine'):
        super().__init__()
        self.n_steps = n_steps

        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule == 'cosine':
            # Cosine schedule (typically better)
            s = 0.008
            steps = n_steps + 1
            x = torch.linspace(0, n_steps, steps)
            alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Store as buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.n_steps, (batch_size,), device=device)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add AWGN noise according to DDPM schedule"""
        noise = torch.randn_like(x)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        noisy_signal = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        return noisy_signal, noise

class SyncErrorGenerator(nn.Module):
    """Generate random synchronization errors - vectorized per-sample version"""

    def __init__(self, signal_length: int = 1024, sample_rate: float = 1e6):
        super().__init__()
        self.signal_length = signal_length
        self.sample_rate = sample_rate

        # Corruption parameters
        self.max_timing_offset = 4  # ±0.5 symbols = ±4 samples
        self.freq_offset_range = [10, 50]  # ±10-50 Hz

    def apply_sync_errors(self, signals: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply random synchronization errors to signals - vectorized per-sample

        Args:
            signals: [batch_size, 2, signal_length] I/Q signals

        Returns:
            corrupted_signals: [batch_size, 2, signal_length]
            error_info: Dict containing applied errors for logging
        """
        batch_size = signals.shape[0]
        device = signals.device
        corrupted_signals = signals.clone()

        # 1. Vectorized timing offsets
        timing_offsets = torch.randint(
            -self.max_timing_offset, self.max_timing_offset + 1,
            (batch_size,), device=device
        )

        # Apply timing shifts (unavoidable loop due to different shifts per sample)
        for i in range(batch_size):
            if timing_offsets[i] != 0:
                corrupted_signals[i] = torch.roll(corrupted_signals[i], shifts=timing_offsets[i].item(), dims=-1)

        # 2. Vectorized frequency offsets
        freq_offsets = torch.empty(batch_size, device=device).uniform_(
            self.freq_offset_range[0], self.freq_offset_range[1]
        )
        # Randomly flip sign for each sample
        freq_signs = torch.randint(0, 2, (batch_size,), device=device) * 2 - 1  # -1 or 1
        freq_offsets = freq_offsets * freq_signs.float()

        # Create phase ramps for all samples at once
        n = torch.arange(self.signal_length, device=device, dtype=torch.float32)
        phase_ramps = 2 * torch.pi * freq_offsets.unsqueeze(1) * n.unsqueeze(0) / self.sample_rate

        # Apply frequency offsets
        signal_complex = torch.complex(corrupted_signals[:, 0], corrupted_signals[:, 1])
        freq_rotations = torch.exp(1j * phase_ramps)
        signal_complex = signal_complex * freq_rotations

        corrupted_signals[:, 0] = signal_complex.real
        corrupted_signals[:, 1] = signal_complex.imag

        # 3. Vectorized phase offsets
        phase_offsets = torch.empty(batch_size, device=device).uniform_(0, 2 * torch.pi)

        # Apply phase offsets
        signal_complex = torch.complex(corrupted_signals[:, 0], corrupted_signals[:, 1])
        phase_rotations = torch.exp(1j * phase_offsets.unsqueeze(1))
        signal_complex = signal_complex * phase_rotations

        corrupted_signals[:, 0] = signal_complex.real
        corrupted_signals[:, 1] = signal_complex.imag

        error_info = {
            'timing_offsets': timing_offsets.cpu().numpy().tolist(),
            'phase_offsets': phase_offsets.cpu().numpy().tolist(),
            'freq_offsets': freq_offsets.cpu().numpy().tolist()
        }

        return corrupted_signals, error_info

class SimplifiedReconstructionLoss(nn.Module):
    """Complex MSE + Phase Angle Loss + Complex Correlation Loss + Amplitude Penalty for phase preservation"""
    def __init__(self,
                 complex_mse_weight=1.0,
                 phase_weight=2.0,
                 corr_weight=1.0,
                 amp_weight=0.3,
                 align_global_phase=False):
        super().__init__()
        self.complex_mse_weight = complex_mse_weight
        self.phase_weight = phase_weight
        self.corr_weight = corr_weight
        self.amp_weight = amp_weight
        self.align_global_phase = align_global_phase

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2, T] - predicted I/Q
            target: [B, 2, T] - ground truth I/Q
        """
        if self.align_global_phase:
            pred = self.remove_global_phase(pred, target)

        mse = self.complex_mse(pred, target)
        phase = self.sincos_phase_loss(pred, target)
        corr = self.complex_corr_loss(pred, target)
        amp = self.amplitude_penalty(pred, target)

        total = (self.complex_mse_weight * mse +
                 self.phase_weight * phase +
                 self.corr_weight * corr +
                 self.amp_weight * amp)

        return total, {
            "total": total.item(),
            "mse": mse.item(),
            "phase": phase.item(),
            "corr": corr.item(),
            "amp": amp.item()
        }

    def complex_mse(self, pred, target):
        pred_c = torch.complex(pred[:, 0], pred[:, 1])
        target_c = torch.complex(target[:, 0], target[:, 1])
        return torch.mean(torch.abs(pred_c - target_c) ** 2)

    def sincos_phase_loss(self, pred, target):
        pred_c = torch.complex(pred[:, 0], pred[:, 1])
        target_c = torch.complex(target[:, 0], target[:, 1])

        phase_pred = torch.angle(pred_c)
        phase_target = torch.angle(target_c)
        phase_diff = phase_pred - phase_target

        return torch.mean(torch.sin(phase_diff) ** 2)

    def complex_corr_loss(self, pred, target, eps=1e-8):
        pred_c = torch.complex(pred[:, 0], pred[:, 1])
        target_c = torch.complex(target[:, 0], target[:, 1])

        B = pred_c.shape[0]
        pred_flat = pred_c.view(B, -1)
        target_flat = target_c.view(B, -1)

        pred_flat = pred_flat / (torch.norm(pred_flat, dim=1, keepdim=True) + eps)
        target_flat = target_flat / (torch.norm(target_flat, dim=1, keepdim=True) + eps)

        corr = torch.real(torch.sum(torch.conj(pred_flat) * target_flat, dim=1))
        return torch.mean(1.0 - corr)

    def amplitude_penalty(self, pred, target):
        pred_c = torch.complex(pred[:, 0], pred[:, 1])
        target_c = torch.complex(target[:, 0], target[:, 1])
        return F.mse_loss(torch.abs(pred_c), torch.abs(target_c))

    def remove_global_phase(self, pred, target):
        pred_c = torch.complex(pred[:, 0], pred[:, 1])
        target_c = torch.complex(target[:, 0], target[:, 1])

        phase_offset = torch.angle(torch.mean(pred_c * torch.conj(target_c), dim=1, keepdim=True))
        phase_corr = torch.exp(-1j * phase_offset)

        aligned = pred_c * phase_corr
        return torch.stack([aligned.real, aligned.imag], dim=1)

class ResidualBlock(nn.Module):
    """Basic residual block for 1D convolutions"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = F.relu(out)

        return out

class ResidualStack(nn.Module):
    """Stack of residual blocks with downsampling"""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 2):
        super().__init__()

        layers = []
        # First block handles dimension change
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))

        # Remaining blocks maintain dimensions
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class PSKResNetClassifier(nn.Module):
    """ResNet-based classifier for PSK signals"""

    def __init__(self, input_channels: int = 2, num_classes: int = 3, input_length: int = 1024):
        super().__init__()

        self.input_length = input_length

        # Initial convolution to get to 32 channels
        self.initial_conv = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.initial_bn = nn.BatchNorm1d(32)

        # Residual stacks following the paper architecture
        # Input: 2 × 1024 → 32 × 512 (after initial conv)
        self.stack1 = ResidualStack(32, 32, num_blocks=2, stride=1)   # 32 × 512
        self.stack2 = ResidualStack(32, 32, num_blocks=2, stride=2)   # 32 × 256
        self.stack3 = ResidualStack(32, 32, num_blocks=2, stride=2)   # 32 × 128
        self.stack4 = ResidualStack(32, 32, num_blocks=2, stride=2)   # 32 × 64
        self.stack5 = ResidualStack(32, 32, num_blocks=2, stride=2)   # 32 × 32
        self.stack6 = ResidualStack(32, 32, num_blocks=2, stride=2)   # 32 × 16

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Activation function (SeLU as specified in paper)
        self.selu = nn.SELU()

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))

        # Residual stacks
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        x = self.stack5(x)
        x = self.stack6(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.selu(self.fc1(x))
        x = self.selu(self.fc2(x))
        x = self.fc3(x)  # No activation here, will apply softmax in loss

        return x

class PSKDenoisingClassifier(L.LightningModule):
    """Simplified PSK Denoiser and Classifier with baseline losses"""
    def __init__(
        self,
        unet,
        signal_length: int = 1024,
        learning_rate: float = 1e-3,
        num_diffusion_steps: int = 1000,
        beta_schedule: str = 'cosine',
        sample_rate: float = 1e6,
        # Simplified loss weights
        complex_mse_weight=1.0,
        phase_weight=2.5,
        corr_weight=1.5,
        amp_weight=0.3,
        align_global_phase=True,
        classification_weight: float = 1.0,
        warmup_epochs: int = 3,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['unet'])
        self.automatic_optimization = False

        self.label_names = ['QPSK', '8PSK', '16PSK']
        self.num_classes = 3
        self.warmup_epochs = warmup_epochs

        # Core components
        self.unet = unet
        self.ddpm_scheduler = DDPMScheduler(
            n_steps=num_diffusion_steps,
            schedule=beta_schedule
        )
        self.sync_error_generator = SyncErrorGenerator(
            signal_length=signal_length,
            sample_rate=sample_rate
        )

        # Simplified reconstruction loss: Complex MSE + Cosine Similarity
        self.reconstruction_loss = SimplifiedReconstructionLoss(
            complex_mse_weight=complex_mse_weight,
            phase_weight=phase_weight,
            corr_weight=corr_weight,
            amp_weight=amp_weight,
            align_global_phase=align_global_phase,
        )

        # ResNet classifier
        self.classifier = PSKResNetClassifier(
            input_channels=2,
            num_classes=self.num_classes,
            input_length=signal_length
        )

        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()

        # Classification weight schedule
        self.classification_weight = classification_weight

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict clean signal from corrupted signal"""
        return self.unet(x, t)

    def denoise_signal(self, corrupted_signal: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Denoise a signal using progressive denoising steps
        """
        self.unet.eval()

        # Create timestep schedule for inference
        timesteps = torch.linspace(
            self.ddpm_scheduler.n_steps - 1, 0, num_steps,
            dtype=torch.long, device=corrupted_signal.device
        )

        current_signal = corrupted_signal

        with torch.no_grad():
            for t in timesteps:
                t_batch = t.repeat(corrupted_signal.shape[0])

                # Predict clean signal
                predicted_clean = self.unet(current_signal, t_batch)

                # Simple update rule
                alpha = 0.1
                current_signal = (1 - alpha) * current_signal + alpha * predicted_clean

        return current_signal

    def get_current_classification_weight(self) -> float:
        """Get current classification weight based on training progress"""
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        else:
            # Gradually increase classification weight
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.trainer.max_epochs - self.warmup_epochs)
            return self.classification_weight * min(1.0, progress * 2)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # Handle the new 4-value batch format
        synced_signals, original_unsynced_signals, labels, snrs = batch

        batch_size = synced_signals.shape[0]
        device = synced_signals.device

        # ===================
        # DENOISING TRAINING
        # ===================

        # Sample timesteps for DDPM (keep random for denoising training)
        timesteps = self.ddpm_scheduler.sample_timesteps(batch_size, device)

        # Step 1: Add random synchronization errors to clean synced signals
        corrupted_signals, error_info = self.sync_error_generator.apply_sync_errors(synced_signals)

        # Step 2: Add AWGN noise according to DDPM schedule
        noisy_signals, noise = self.ddpm_scheduler.add_noise(corrupted_signals, timesteps)

        # Step 3: Predict clean synchronized signal from noisy corrupted signal
        predicted_clean = self.unet(noisy_signals, timesteps)

        # Step 4: Compute simplified reconstruction loss
        reconstruction_loss_total, loss_components = self.reconstruction_loss(predicted_clean, synced_signals)

        # ===================
        # CLASSIFICATION TRAINING (on original unsynced data with SNR-aware denoising)
        # ===================

        classification_loss = torch.tensor(0.0, device=device)
        classification_accuracy = torch.tensor(0.0, device=device)

        current_class_weight = self.get_current_classification_weight()

        if current_class_weight > 0:
            # Apply SNR-aware denoising to original unsynced signals (same as validation)
            with torch.no_grad():
                denoised_unsynced = self.adaptive_denoise_signal(original_unsynced_signals, snrs, num_steps=5)

            # Classify the denoised signals
            class_logits = self.classifier(denoised_unsynced)
            classification_loss = self.classification_loss(class_logits, labels)

            # Compute accuracy
            _, predicted_classes = torch.max(class_logits, 1)
            classification_accuracy = (predicted_classes == labels).float().mean()

        # ===================
        # COMBINED LOSS
        # ===================

        total_loss = reconstruction_loss_total + current_class_weight * classification_loss

        # Backpropagation
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # ===================
        # LOGGING
        # ===================

        # Main metrics
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_reconstruction_loss', reconstruction_loss_total)
        self.log('train_classification_loss', classification_loss)
        self.log('train_classification_accuracy', classification_accuracy, prog_bar=True)
        self.log('train_classification_weight', current_class_weight)

        # Individual reconstruction loss components
        for name, loss_val in loss_components.items():
            self.log(f'train_{name}_loss', loss_val)

        # Corruption statistics
        avg_timing_offset = np.mean([abs(x) for x in error_info['timing_offsets']])
        avg_phase_offset = np.mean(error_info['phase_offsets'])
        avg_freq_offset = np.mean([abs(x) for x in error_info['freq_offsets']])

        self.log('train_avg_timing_offset', avg_timing_offset)
        self.log('train_avg_phase_offset', avg_phase_offset)
        self.log('train_avg_freq_offset', avg_freq_offset)

        # SNR-aware training statistics
        if current_class_weight > 0:
            # Log SNR statistics for training
            self.log('train_avg_snr', snrs.float().mean())
            self.log('train_min_snr', snrs.float().min())
            self.log('train_max_snr', snrs.float().max())

            # Log assigned timesteps for classification training
            assigned_timesteps = self.snr_to_timestep(snrs)
            self.log('train_avg_assigned_timestep', assigned_timesteps.float().mean())
            self.log('train_min_assigned_timestep', assigned_timesteps.float().min())
            self.log('train_max_assigned_timestep', assigned_timesteps.float().max())

            # SNR-based accuracy analysis during training
            snr_ranges = [(-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30)]
            for snr_min, snr_max in snr_ranges:
                snr_mask = (snrs >= snr_min) & (snrs < snr_max)
                if snr_mask.sum() > 0:
                    snr_acc = (predicted_classes[snr_mask] == labels[snr_mask]).float().mean()
                    self.log(f'train_accuracy_snr_{snr_min}to{snr_max}', snr_acc)

        return total_loss

    def snr_to_timestep(self, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Map SNR values to appropriate DDPM timesteps for denoising

        Args:
            snr_db: [batch_size] SNR values in dB, range [-20, 30]

        Returns:
            timesteps: [batch_size] DDMP timesteps, range [50, 950]
        """
        # Define SNR to timestep mapping
        # Higher SNR (cleaner signal) → Lower timestep (less denoising needed)
        # Lower SNR (noisier signal) → Higher timestep (more denoising needed)

        # Clamp SNR to expected range
        snr_clamped = torch.clamp(snr_db, -20, 30)

        # Linear mapping: SNR [-20, 30] → timestep [950, 50]
        # timestep = 950 - (snr + 20) * (950 - 50) / (30 - (-20))
        timesteps = 950 - (snr_clamped + 20) * (900 / 50)

        # Round to integers and clamp to valid range
        timesteps = torch.clamp(timesteps.round().long(), 50, 950)

        return timesteps

    def adaptive_denoise_signal(self, corrupted_signal: torch.Tensor, snr_db: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Denoise signal with SNR-aware timestep scheduling

        Args:
            corrupted_signal: [batch_size, 2, signal_length] I/Q signals to denoise
            snr_db: [batch_size] SNR of each signal in dB
            num_steps: Number of denoising steps

        Returns:
            Denoised signals
        """
        self.unet.eval()

        batch_size = corrupted_signal.shape[0]
        device = corrupted_signal.device

        # Get starting timesteps based on SNR
        starting_timesteps = self.snr_to_timestep(snr_db)

        current_signal = corrupted_signal

        with torch.no_grad():
            for step in range(num_steps):
                # Calculate current timestep for each signal based on its starting point
                progress = step / (num_steps - 1) if num_steps > 1 else 1.0

                # Each signal gets its own timestep based on its SNR
                current_timesteps = torch.zeros(batch_size, device=device, dtype=torch.long)
                for i in range(batch_size):
                    start_t = starting_timesteps[i].item()
                    # Linear decay from starting timestep to 0
                    current_timesteps[i] = int(start_t * (1 - progress))

                # Predict clean signal
                predicted_clean = self.unet(current_signal, current_timesteps)

                # Adaptive step size - be more aggressive early on
                alpha = 0.05 + 0.15 * progress  # 0.05 → 0.20
                current_signal = (1 - alpha) * current_signal + alpha * predicted_clean

        return current_signal

    def validation_step(self, batch, batch_idx):
        # Handle the new 4-value batch format
        synced_signals, original_unsynced_signals, labels, snrs = batch

        batch_size = synced_signals.shape[0]
        device = synced_signals.device

        # ===================
        # DENOISING VALIDATION
        # ===================

        # Test 1: Clean synced signals (sanity check)
        # For clean signals, use low timesteps (light denoising)
        low_timesteps = torch.full((batch_size,), 50, device=device)
        slightly_noisy, _ = self.ddpm_scheduler.add_noise(synced_signals, low_timesteps)

        predicted_clean = self.unet(slightly_noisy, low_timesteps)
        clean_preservation_loss, clean_loss_components = self.reconstruction_loss(predicted_clean, synced_signals)

        # Test 2: Synced signals with added corruptions
        corrupted_signals, error_info = self.sync_error_generator.apply_sync_errors(synced_signals)

        # Use medium timesteps for artificially corrupted signals
        medium_timesteps = torch.full((batch_size,), 500, device=device)
        noisy_corrupted, _ = self.ddpm_scheduler.add_noise(corrupted_signals, medium_timesteps)

        predicted_restored = self.unet(noisy_corrupted, medium_timesteps)
        restoration_loss, restoration_loss_components = self.reconstruction_loss(predicted_restored, synced_signals)

        # ===================
        # SNR-AWARE CLASSIFICATION VALIDATION (on original unsynced data)
        # ===================

        # Apply SNR-aware denoising to original unsynced signals (same as training)
        denoised_unsynced = self.adaptive_denoise_signal(original_unsynced_signals, snrs, num_steps=10)

        # Classify the adaptively denoised signals
        class_logits = self.classifier(denoised_unsynced)
        classification_loss = self.classification_loss(class_logits, labels)

        # Compute accuracy
        _, predicted_classes = torch.max(class_logits, 1)
        classification_accuracy = (predicted_classes == labels).float().mean()

        # Per-class accuracy
        for i, class_name in enumerate(self.label_names):
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                class_acc = (predicted_classes[class_mask] == labels[class_mask]).float().mean()
                self.log(f'val_accuracy_{class_name}', class_acc)

        # SNR-based accuracy analysis
        snr_ranges = [(-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30)]
        for snr_min, snr_max in snr_ranges:
            snr_mask = (snrs >= snr_min) & (snrs < snr_max)
            if snr_mask.sum() > 0:
                snr_acc = (predicted_classes[snr_mask] == labels[snr_mask]).float().mean()
                self.log(f'val_accuracy_snr_{snr_min}to{snr_max}', snr_acc)

        # Overall validation loss
        val_loss = (clean_preservation_loss + restoration_loss) / 2

        # ===================
        # LOGGING
        # ===================

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_clean_preservation_loss', clean_preservation_loss)
        self.log('val_restoration_loss', restoration_loss)
        self.log('val_classification_loss', classification_loss)
        self.log('val_classification_accuracy', classification_accuracy, prog_bar=True)

        # Log SNR statistics
        self.log('val_avg_snr', snrs.float().mean())
        self.log('val_min_snr', snrs.float().min())
        self.log('val_max_snr', snrs.float().max())

        # Log timestep statistics
        assigned_timesteps = self.snr_to_timestep(snrs)
        self.log('val_avg_assigned_timestep', assigned_timesteps.float().mean())
        self.log('val_min_assigned_timestep', assigned_timesteps.float().min())
        self.log('val_max_assigned_timestep', assigned_timesteps.float().max())

        # Store data for visualization (first batch only)
        if batch_idx == 0:
            self.val_data = {
                'original_synced': synced_signals[:4].detach().cpu(),
                'original_unsynced': original_unsynced_signals[:4].detach().cpu(),
                'denoised_unsynced': denoised_unsynced[:4].detach().cpu(),
                'corrupted_signals': corrupted_signals[:4].detach().cpu(),
                'predicted_restored': predicted_restored[:4].detach().cpu(),
                'labels': labels[:4].detach().cpu(),
                'snrs': snrs[:4].detach().cpu(),
                'assigned_timesteps': assigned_timesteps[:4].detach().cpu(),
                'predicted_classes': predicted_classes[:4].detach().cpu(),
                'class_logits': class_logits[:4].detach().cpu(),
                'error_info': {k: v[:4] for k, v in error_info.items()}
            }
        return val_loss

    def on_validation_epoch_end(self):
        """Create comprehensive visualization"""
        self._create_combined_visualization()

    def _create_combined_visualization(self):
        """Create comprehensive visualization showing all denoising and synchronization stages"""
        try:
            if not hasattr(self, 'val_data') or self.val_data is None:
                return

            # Create comprehensive visualization
            fig = plt.figure(figsize=(30, 20))  # Increased size for more columns

            # Show 4 samples
            for sample_idx in range(4):
                true_label_idx = self.val_data['labels'][sample_idx].item()
                pred_label_idx = self.val_data['predicted_classes'][sample_idx].item()

                true_label_name = self.label_names[true_label_idx]
                pred_label_name = self.label_names[pred_label_idx]

                # Get class confidence
                class_probs = F.softmax(self.val_data['class_logits'][sample_idx], dim=0)
                confidence = class_probs[pred_label_idx].item()

                # Get ideal constellation
                if true_label_idx == 0:  # QPSK
                    ideal_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
                elif true_label_idx == 1:  # 8PSK
                    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                    ideal_points = np.exp(1j * angles)
                else:  # 16PSK
                    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
                    ideal_points = np.exp(1j * angles)

                # Get error information for this sample
                timing_err = self.val_data['error_info']['timing_offsets'][sample_idx]
                phase_err = self.val_data['error_info']['phase_offsets'][sample_idx]
                freq_err = self.val_data['error_info']['freq_offsets'][sample_idx]

                # Row for this sample
                row = sample_idx

                # Column 1: Clean Synced Signal
                ax1 = plt.subplot(4, 7, row * 7 + 1)
                synced_signal = self.val_data['original_synced'][sample_idx]
                synced_complex = torch.complex(synced_signal[0], synced_signal[1])

                ax1.scatter(synced_complex.real, synced_complex.imag, alpha=0.6, s=20, c='blue')
                ax1.scatter(ideal_points.real, ideal_points.imag,
                        c='black', s=100, marker='x', linewidth=3)
                ax1.set_title(f'{true_label_name}\nClean Synced' if row == 0 else 'Clean Synced')
                ax1.grid(True, alpha=0.3)
                ax1.set_aspect('equal')
                ax1.set_xlim(-1.5, 1.5)
                ax1.set_ylim(-1.5, 1.5)

                # Column 2: Sync Errors Added (no AWGN)
                ax2 = plt.subplot(4, 7, row * 7 + 2)
                corrupted_signal = self.val_data['corrupted_signals'][sample_idx]
                corrupted_complex = torch.complex(corrupted_signal[0], corrupted_signal[1])

                ax2.scatter(corrupted_complex.real, corrupted_complex.imag, alpha=0.6, s=20, c='orange')
                ax2.scatter(ideal_points.real, ideal_points.imag,
                        c='black', s=100, marker='x', linewidth=3)
                ax2.set_title('+ Sync Errors' if row == 0 else '+ Sync Errors')

                # Add error info text
                ax2.text(0.02, 0.98, f'T:{timing_err}\nP:{phase_err:.1f}\nF:{freq_err:.1f}Hz',
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.set_aspect('equal')
                ax2.set_xlim(-1.5, 1.5)
                ax2.set_ylim(-1.5, 1.5)

                # Column 3: Sync Errors + AWGN
                ax3 = plt.subplot(4, 7, row * 7 + 3)
                # FIXED: Move corrupted signal to device and handle device consistency
                device = next(self.parameters()).device  # Get model device

                # Move corrupted signal to device and add batch dimension
                corrupted_single = corrupted_signal.to(device).unsqueeze(0)
                high_timesteps = torch.full((1,), 500, device=device)

                # Recreate noisy corrupted signal
                with torch.no_grad():
                    noisy_corrupted_single, _ = self.ddpm_scheduler.add_noise(corrupted_single, high_timesteps)
                noisy_corrupted = noisy_corrupted_single.squeeze(0).cpu()  # Move back to CPU for plotting

                noisy_complex = torch.complex(noisy_corrupted[0], noisy_corrupted[1])

                ax3.scatter(noisy_complex.real, noisy_complex.imag, alpha=0.6, s=20, c='red')
                ax3.scatter(ideal_points.real, ideal_points.imag,
                        c='black', s=100, marker='x', linewidth=3)
                ax3.set_title('+ AWGN' if row == 0 else '+ AWGN')
                ax3.grid(True, alpha=0.3)
                ax3.set_aspect('equal')
                ax3.set_xlim(-1.5, 1.5)
                ax3.set_ylim(-1.5, 1.5)

                # Column 4: Restored Signal (from sync errors + AWGN)
                ax4 = plt.subplot(4, 7, row * 7 + 4)
                restored_signal = self.val_data['predicted_restored'][sample_idx]
                restored_complex = torch.complex(restored_signal[0], restored_signal[1])

                ax4.scatter(restored_complex.real, restored_complex.imag, alpha=0.6, s=20, c='green')
                ax4.scatter(ideal_points.real, ideal_points.imag,
                        c='black', s=100, marker='x', linewidth=3)
                ax4.set_title('Restored' if row == 0 else 'Restored')

                # Compute restoration quality
                orig_np = synced_signal.numpy()
                restored_np = restored_signal.numpy()
                mse = np.mean((orig_np - restored_np) ** 2)
                ax4.text(0.02, 0.98, f'MSE: {mse:.4f}',
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)

                ax4.grid(True, alpha=0.3)
                ax4.set_aspect('equal')
                ax4.set_xlim(-1.5, 1.5)
                ax4.set_ylim(-1.5, 1.5)

                # Column 5: Original Unsynced Signal
                ax5 = plt.subplot(4, 7, row * 7 + 5)
                unsynced_signal = self.val_data['original_unsynced'][sample_idx]
                unsynced_complex = torch.complex(unsynced_signal[0], unsynced_signal[1])

                ax5.scatter(unsynced_complex.real, unsynced_complex.imag, alpha=0.6, s=20, c='purple')
                ax5.scatter(ideal_points.real, ideal_points.imag,
                        c='black', s=100, marker='x', linewidth=3)
                ax5.set_title('Original\nUnsynced' if row == 0 else 'Original\nUnsynced')
                ax5.grid(True, alpha=0.3)
                ax5.set_aspect('equal')
                ax5.set_xlim(-1.5, 1.5)
                ax5.set_ylim(-1.5, 1.5)

                # Column 6: Denoised Original Unsynced
                ax6 = plt.subplot(4, 7, row * 7 + 6)
                denoised_signal = self.val_data['denoised_unsynced'][sample_idx]
                denoised_complex = torch.complex(denoised_signal[0], denoised_signal[1])

                ax6.scatter(denoised_complex.real, denoised_complex.imag, alpha=0.6, s=20, c='cyan')
                ax6.scatter(ideal_points.real, ideal_points.imag,
                        c='black', s=100, marker='x', linewidth=3)
                ax6.set_title('Denoised\nUnsynced' if row == 0 else 'Denoised\nUnsynced')

                # Compute denoising quality vs original synced
                denoised_np = denoised_signal.numpy()
                denoise_mse = np.mean((orig_np - denoised_np) ** 2)
                ax6.text(0.02, 0.98, f'MSE: {denoise_mse:.4f}',
                        transform=ax6.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)

                ax6.grid(True, alpha=0.3)
                ax6.set_aspect('equal')
                ax6.set_xlim(-1.5, 1.5)
                ax6.set_ylim(-1.5, 1.5)

                # Column 7: Classification Result
                ax7 = plt.subplot(4, 7, row * 7 + 7)
                ax7.bar(range(3), class_probs.cpu().numpy(), alpha=0.7)
                ax7.set_xticks(range(3))
                ax7.set_xticklabels(self.label_names, rotation=45)
                ax7.set_ylabel('Probability')
                ax7.set_title('Classification' if row == 0 else '')

                # Highlight prediction
                correct = (true_label_idx == pred_label_idx)
                color = 'green' if correct else 'red'
                ax7.axvline(pred_label_idx, color=color, linewidth=3, alpha=0.7)

                # Add text with prediction
                status = "✓" if correct else "✗"
                ax7.text(0.02, 0.98, f'{status} {pred_label_name}\nConf: {confidence:.2f}',
                        transform=ax7.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)

            # Add overall title with pipeline description
            pipeline_description = "Clean Synced → + Sync Errors → + AWGN → Restored | Original Unsynced → Denoised → Classified"
            plt.suptitle(f'PSK Pipeline Visualization - Epoch {self.current_epoch}\n{pipeline_description}',
                        fontsize=16, y=0.95)

            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Make room for the title

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({'psk_pipeline_visualization': wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()
    def configure_optimizers(self):
        """Configure optimizers with different learning rates for denoiser and classifier"""

        # Separate parameters for different components
        denoiser_params = list(self.unet.parameters())
        reconstruction_loss_params = list(self.reconstruction_loss.parameters())
        classifier_params = list(self.classifier.parameters())

        # Combine denoiser and reconstruction loss parameters
        denoiser_all_params = denoiser_params + reconstruction_loss_params

        # Create optimizer with parameter groups
        optimizer = AdamW([
            {
                'params': denoiser_all_params,
                'lr': self.hparams.learning_rate,
                'name': 'denoiser'
            },
            {
                'params': classifier_params,
                'lr': self.hparams.learning_rate * 0.1,  # Lower LR for classifier
                'name': 'classifier'
            }
        ], weight_decay=1e-4)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,  # 10% warm-up
            anneal_strategy='cos',
            div_factor=25,  # Initial LR = max_lr / div_factor
            final_div_factor=1e4  # Final LR = initial_LR / final_div_factor
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": "OneCycleLR"
            },
        }

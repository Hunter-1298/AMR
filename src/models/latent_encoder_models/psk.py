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
        # Generate random frequency offsets for each sample
        freq_offsets = torch.empty(batch_size, device=device).uniform_(
            self.freq_offset_range[0], self.freq_offset_range[1]
        )
        # Randomly flip sign for each sample
        freq_signs = torch.randint(0, 2, (batch_size,), device=device) * 2 - 1  # -1 or 1
        freq_offsets = freq_offsets * freq_signs.float()

        # Create phase ramps for all samples at once
        n = torch.arange(self.signal_length, device=device, dtype=torch.float32)
        phase_ramps = 2 * torch.pi * freq_offsets.unsqueeze(1) * n.unsqueeze(0) / self.sample_rate
        # phase_ramps shape: [batch_size, signal_length]

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

class ComplexSignalLoss(nn.Module):
    """Multi-component loss for PSK denoising and synchronization"""

    def __init__(self,
                 mse_weight: float = 1.0,
                 power_weight: float = 2.0,
                 constellation_weight: float = 1.0,
                 envelope_weight: float = 1.0,
                 psd_weight: float = 0.5,
                 phase_coherence_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.power_weight = power_weight
        self.constellation_weight = constellation_weight
        self.envelope_weight = envelope_weight
        self.psd_weight = psd_weight
        self.phase_coherence_weight = phase_coherence_weight

        # Pre-compute Hann window for PSD computation
        self.register_buffer('hann_window', torch.hann_window(1024))

    def forward(self, pred_signal: torch.Tensor, target_signal: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-component loss for PSK signals

        Args:
            pred_signal: [batch_size, 2, signal_length] predicted I/Q
            target_signal: [batch_size, 2, signal_length] target I/Q
        """
        # Convert to complex
        pred_complex = torch.complex(pred_signal[:, 0], pred_signal[:, 1])
        target_complex = torch.complex(target_signal[:, 0], target_signal[:, 1])

        losses = {}

        # 1. Complex MSE (Basic reconstruction)
        mse_loss = torch.mean(torch.abs(pred_complex - target_complex) ** 2)
        losses['mse'] = mse_loss

        # 2. Power preservation (Anti-collapse)
        power_loss = self.power_preservation_loss(pred_complex, target_complex)
        losses['power'] = power_loss

        # 3. Soft constellation loss (Synchronization quality)
        constellation_loss = self.soft_constellation_loss(pred_complex, target_complex)
        losses['constellation'] = constellation_loss

        # 4. Envelope consistency (PSK characteristic)
        envelope_loss = self.envelope_consistency_loss(pred_complex, target_complex)
        losses['envelope'] = envelope_loss

        # 5. Power spectral density matching (Frequency domain)
        psd_loss = self.psd_matching_loss(pred_complex, target_complex)
        losses['psd'] = psd_loss

        # 6. Phase coherence (Phase synchronization)
        phase_coherence_loss = self.phase_coherence_loss(pred_complex, target_complex)
        losses['phase_coherence'] = phase_coherence_loss

        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.power_weight * power_loss +
            self.constellation_weight * constellation_loss +
            self.envelope_weight * envelope_loss +
            self.psd_weight * psd_loss +
            self.phase_coherence_weight * phase_coherence_loss
        )

        return total_loss, losses

    def power_preservation_loss(self, pred_complex: torch.Tensor, target_complex: torch.Tensor) -> torch.Tensor:
        """Preserve signal power to prevent collapse"""
        pred_power = torch.mean(torch.abs(pred_complex) ** 2, dim=1)
        target_power = torch.mean(torch.abs(target_complex) ** 2, dim=1)
        return F.mse_loss(pred_power, target_power)

    def soft_constellation_loss(self, pred_complex: torch.Tensor, target_complex: torch.Tensor) -> torch.Tensor:
        """Soft constellation constraints - modulation agnostic"""
        # Encourage magnitude consistency (PSK property)
        pred_magnitude = torch.abs(pred_complex)
        target_magnitude = torch.abs(target_complex)

        # Mean magnitude should be similar
        pred_mean_mag = torch.mean(pred_magnitude, dim=1)
        target_mean_mag = torch.mean(target_magnitude, dim=1)
        mean_mag_loss = F.mse_loss(pred_mean_mag, target_mean_mag)

        # Encourage clustering in phase space (without specifying number of clusters)
        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)

        # Phase histogram similarity using circular statistics
        pred_phase_cos = torch.cos(pred_phase)
        pred_phase_sin = torch.sin(pred_phase)
        target_phase_cos = torch.cos(target_phase)
        target_phase_sin = torch.sin(target_phase)

        phase_cos_loss = F.mse_loss(torch.mean(pred_phase_cos, dim=1), torch.mean(target_phase_cos, dim=1))
        phase_sin_loss = F.mse_loss(torch.mean(pred_phase_sin, dim=1), torch.mean(target_phase_sin, dim=1))

        return mean_mag_loss + phase_cos_loss + phase_sin_loss

    def envelope_consistency_loss(self, pred_complex: torch.Tensor, target_complex: torch.Tensor) -> torch.Tensor:
        """PSK signals should have constant envelope"""
        pred_magnitude = torch.abs(pred_complex)
        target_magnitude = torch.abs(target_complex)

        # Variance of magnitude should be minimal for PSK
        pred_mag_var = torch.var(pred_magnitude, dim=1)
        target_mag_var = torch.var(target_magnitude, dim=1)

        return F.mse_loss(pred_mag_var, target_mag_var)

    def psd_matching_loss(self, pred_complex: torch.Tensor, target_complex: torch.Tensor) -> torch.Tensor:
        """Power spectral density matching for frequency domain characteristics"""
        batch_size = pred_complex.shape[0]

        # Apply window to reduce spectral leakage
        window = self.hann_window.unsqueeze(0).expand(batch_size, -1)

        pred_windowed = pred_complex * window
        target_windowed = target_complex * window

        # Compute FFT
        pred_fft = torch.fft.fft(pred_windowed, dim=1)
        target_fft = torch.fft.fft(target_windowed, dim=1)

        # Compute power spectral density
        pred_psd = torch.abs(pred_fft) ** 2
        target_psd = torch.abs(target_fft) ** 2

        # Normalize PSDs
        pred_psd_norm = pred_psd / (torch.sum(pred_psd, dim=1, keepdim=True) + 1e-8)
        target_psd_norm = target_psd / (torch.sum(target_psd, dim=1, keepdim=True) + 1e-8)

        # MSE between normalized PSDs
        psd_mse = F.mse_loss(pred_psd_norm, target_psd_norm)

        # Spectral centroid preservation (frequency offset detection)
        freq_bins = torch.arange(pred_psd.shape[1], device=pred_psd.device).float()

        pred_centroid = torch.sum(pred_psd_norm * freq_bins.unsqueeze(0), dim=1)
        target_centroid = torch.sum(target_psd_norm * freq_bins.unsqueeze(0), dim=1)

        centroid_loss = F.mse_loss(pred_centroid, target_centroid)

        return psd_mse + 0.5 * centroid_loss

    def phase_coherence_loss(self, pred_complex: torch.Tensor, target_complex: torch.Tensor) -> torch.Tensor:
        """Phase coherence for synchronization quality"""
        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)

        # Phase derivative (instantaneous frequency)
        pred_phase_diff = torch.diff(pred_phase, dim=1)
        target_phase_diff = torch.diff(target_phase, dim=1)

        # Wrap phase differences to [-π, π]
        pred_phase_diff = torch.atan2(torch.sin(pred_phase_diff), torch.cos(pred_phase_diff))
        target_phase_diff = torch.atan2(torch.sin(target_phase_diff), torch.cos(target_phase_diff))

        # Phase derivative should be similar
        phase_diff_loss = F.mse_loss(pred_phase_diff, target_phase_diff)

        # Phase unwrapping smoothness
        pred_unwrapped = torch.cumsum(pred_phase_diff, dim=1)
        target_unwrapped = torch.cumsum(target_phase_diff, dim=1)

        unwrapped_loss = F.mse_loss(pred_unwrapped, target_unwrapped)

        return phase_diff_loss + 0.1 * unwrapped_loss

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

class PSKDenoiser(L.LightningModule):
    """Combined PSK Denoiser and Classifier using end-to-end training"""

    def __init__(
        self,
        unet,
        signal_length: int = 1024,
        learning_rate: float = 1e-3,
        num_diffusion_steps: int = 1000,
        beta_schedule: str = 'cosine',
        sample_rate: float = 1e6,
        # Loss weights
        mse_weight: float = 2.0,
        power_weight: float = 1.0,
        constellation_weight: float = 1.0,
        envelope_weight: float = 1.0,
        psd_weight: float = 0.001,  # Reduced based on previous discussion
        phase_coherence_weight: float = 0.01,
        classification_weight: float = 0.1,  # Start small
        warmup_epochs: int = 10,  # Epochs before adding classification loss
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

        # Multi-component loss function
        self.complex_loss = ComplexSignalLoss(
            mse_weight=mse_weight,
            power_weight=power_weight,
            constellation_weight=constellation_weight,
            envelope_weight=envelope_weight,
            psd_weight=psd_weight,
            phase_coherence_weight=phase_coherence_weight
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

        Args:
            corrupted_signal: [batch_size, 2, signal_length] corrupted I/Q signal
            num_steps: Number of denoising steps

        Returns:
            Denoised signal
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
            return self.classification_weight * min(1.0, progress * 2)  # Reach full weight at 50% through remaining epochs

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

        # Sample timesteps for DDPM (FIXED TYPO)
        timesteps = self.ddpm_scheduler.sample_timesteps(batch_size, device)

        # Step 1: Add random synchronization errors to clean synced signals
        corrupted_signals, error_info = self.sync_error_generator.apply_sync_errors(synced_signals)

        # Step 2: Add AWGN noise according to DDPM schedule
        noisy_signals, noise = self.ddpm_scheduler.add_noise(corrupted_signals, timesteps)

        # Step 3: Predict clean synchronized signal from noisy corrupted signal
        predicted_clean = self.unet(noisy_signals, timesteps)

        # Step 4: Compute reconstruction loss
        reconstruction_loss, loss_components = self.complex_loss(predicted_clean, synced_signals)

        # ===================
        # CLASSIFICATION TRAINING (on original unsynced data)
        # ===================

        classification_loss = torch.tensor(0.0, device=device)
        classification_accuracy = torch.tensor(0.0, device=device)

        current_class_weight = self.get_current_classification_weight()

        if current_class_weight > 0:
            # Apply denoiser to original unsynced signals
            with torch.no_grad():
                denoised_unsynced = self.denoise_signal(original_unsynced_signals, num_steps=5)

            # Classify the denoised signals
            class_logits = self.classifier(denoised_unsynced)
            classification_loss = self.classification_loss(class_logits, labels)

            # Compute accuracy
            _, predicted_classes = torch.max(class_logits, 1)
            classification_accuracy = (predicted_classes == labels).float().mean()

        # ===================
        # COMBINED LOSS
        # ===================

        total_loss = reconstruction_loss + current_class_weight * classification_loss

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
        self.log('train_reconstruction_loss', reconstruction_loss)
        self.log('train_classification_loss', classification_loss)
        self.log('train_classification_accuracy', classification_accuracy, prog_bar=True)
        self.log('train_classification_weight', current_class_weight)

        # Power monitoring
        pred_power = torch.mean(torch.abs(torch.complex(predicted_clean[:, 0], predicted_clean[:, 1])) ** 2)
        target_power = torch.mean(torch.abs(torch.complex(synced_signals[:, 0], synced_signals[:, 1])) ** 2)
        self.log('train_pred_power', pred_power)
        self.log('train_target_power', target_power)
        self.log('train_power_ratio', pred_power / (target_power + 1e-8))

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

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Handle the new 4-value batch format
        synced_signals, original_unsynced_signals, labels, snrs = batch

        batch_size = synced_signals.shape[0]
        device = synced_signals.device

        # ===================
        # DENOISING VALIDATION
        # ===================

        # Test 1: Clean synced signals (sanity check)
        low_timesteps = torch.full((batch_size,), 50, device=device)
        slightly_noisy, _ = self.ddpm_scheduler.add_noise(synced_signals, low_timesteps)

        predicted_clean = self.unet(slightly_noisy, low_timesteps)
        clean_preservation_loss, _ = self.complex_loss(predicted_clean, synced_signals)

        # Test 2: Synced signals with added corruptions
        corrupted_signals, error_info = self.sync_error_generator.apply_sync_errors(synced_signals)
        high_timesteps = torch.full((batch_size,), 500, device=device)
        noisy_corrupted, _ = self.ddpm_scheduler.add_noise(corrupted_signals, high_timesteps)

        predicted_restored = self.unet(noisy_corrupted, high_timesteps)
        restoration_loss, _ = self.complex_loss(predicted_restored, synced_signals)

        # ===================
        # CLASSIFICATION VALIDATION (on original unsynced data)
        # ===================

        # Apply denoiser to original unsynced signals
        denoised_unsynced = self.denoise_signal(original_unsynced_signals, num_steps=10)

        # Classify the denoised signals
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

        # Store data for visualization (first batch only)
        if batch_idx == 0:
            self.val_data = {
                'original_synced': synced_signals[:4].detach().cpu(),
                'original_unsynced': original_unsynced_signals[:4].detach().cpu(),
                'denoised_unsynced': denoised_unsynced[:4].detach().cpu(),
                'corrupted_signals': corrupted_signals[:4].detach().cpu(),
                'predicted_restored': predicted_restored[:4].detach().cpu(),
                'labels': labels[:4].detach().cpu(),
                'predicted_classes': predicted_classes[:4].detach().cpu(),
                'class_logits': class_logits[:4].detach().cpu(),
                'error_info': {k: v[:4] for k, v in error_info.items()}
            }

        return val_loss

    def on_validation_epoch_end(self):
        """Create comprehensive visualization"""
        self._create_combined_visualization()

    def _create_combined_visualization(self):
        """Create visualization showing denoising and classification results"""
        try:
            if not hasattr(self, 'val_data') or self.val_data is None:
                return

            # Create comprehensive visualization
            fig = plt.figure(figsize=(25, 15))

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

                # Row for this sample
                row = sample_idx

                # Column 1: Original unsynced
                ax1 = plt.subplot(4, 6, row * 6 + 1)
                unsynced_signal = self.val_data['original_unsynced'][sample_idx]
                unsynced_complex = torch.complex(unsynced_signal[0], unsynced_signal[1])

                ax1.scatter(unsynced_complex.real, unsynced_complex.imag, alpha=0.6, s=20, c='red')
                ax1.scatter(ideal_points.real, ideal_points.imag,
                          c='black', s=100, marker='x', linewidth=3)
                ax1.set_title(f'{true_label_name}\nOriginal Unsynced' if row == 0 else 'Original Unsynced')
                ax1.grid(True, alpha=0.3)
                ax1.set_aspect('equal')
                ax1.set_xlim(-1.5, 1.5)
                ax1.set_ylim(-1.5, 1.5)

                # Column 2: Denoised from unsynced
                ax2 = plt.subplot(4, 6, row * 6 + 2)
                denoised_signal = self.val_data['denoised_unsynced'][sample_idx]
                denoised_complex = torch.complex(denoised_signal[0], denoised_signal[1])

                ax2.scatter(denoised_complex.real, denoised_complex.imag, alpha=0.6, s=20, c='green')
                ax2.scatter(ideal_points.real, ideal_points.imag,
                          c='black', s=100, marker='x', linewidth=3)
                ax2.set_title('Denoised' if row == 0 else '')
                ax2.grid(True, alpha=0.3)
                ax2.set_aspect('equal')
                ax2.set_xlim(-1.5, 1.5)
                ax2.set_ylim(-1.5, 1.5)

                # Column 3: Original synced (reference)
                ax3 = plt.subplot(4, 6, row * 6 + 3)
                synced_signal = self.val_data['original_synced'][sample_idx]
                synced_complex = torch.complex(synced_signal[0], synced_signal[1])

                ax3.scatter(synced_complex.real, synced_complex.imag, alpha=0.6, s=20, c='blue')
                ax3.scatter(ideal_points.real, ideal_points.imag,
                          c='black', s=100, marker='x', linewidth=3)
                ax3.set_title('Manual Sync\n(Reference)' if row == 0 else 'Manual Sync')
                ax3.grid(True, alpha=0.3)
                ax3.set_aspect('equal')
                ax3.set_xlim(-1.5, 1.5)
                ax3.set_ylim(-1.5, 1.5)

                # Column 4: Corrupted + restoration test
                ax4 = plt.subplot(4, 6, row * 6 + 4)
                corrupted_signal = self.val_data['corrupted_signals'][sample_idx]
                corrupted_complex = torch.complex(corrupted_signal[0], corrupted_signal[1])

                ax4.scatter(corrupted_complex.real, corrupted_complex.imag, alpha=0.6, s=20, c='orange')
                ax4.scatter(ideal_points.real, ideal_points.imag,
                          c='black', s=100, marker='x', linewidth=3)
                ax4.set_title('Corrupted' if row == 0 else '')
                ax4.grid(True, alpha=0.3)
                ax4.set_aspect('equal')
                ax4.set_xlim(-1.5, 1.5)
                ax4.set_ylim(-1.5, 1.5)

                # Column 5: Restored
                ax5 = plt.subplot(4, 6, row * 6 + 5)
                restored_signal = self.val_data['predicted_restored'][sample_idx]
                restored_complex = torch.complex(restored_signal[0], restored_signal[1])

                ax5.scatter(restored_complex.real, restored_complex.imag, alpha=0.6, s=20, c='purple')
                ax5.scatter(ideal_points.real, ideal_points.imag,
                          c='black', s=100, marker='x', linewidth=3)
                ax5.set_title('Restored' if row == 0 else '')
                ax5.grid(True, alpha=0.3)
                ax5.set_aspect('equal')
                ax5.set_xlim(-1.5, 1.5)
                ax5.set_ylim(-1.5, 1.5)

                # Column 6: Classification result
                ax6 = plt.subplot(4, 6, row * 6 + 6)
                ax6.bar(range(3), class_probs.cpu().numpy(), alpha=0.7)
                ax6.set_xticks(range(3))
                ax6.set_xticklabels(self.label_names)
                ax6.set_ylabel('Probability')
                ax6.set_title('Classification' if row == 0 else '')

                # Highlight prediction
                correct = (true_label_idx == pred_label_idx)
                color = 'green' if correct else 'red'
                ax6.axvline(pred_label_idx, color=color, linewidth=3, alpha=0.7)

                # Add text with prediction
                ax6.text(0.02, 0.98, f'Pred: {pred_label_name}\nConf: {confidence:.2f}',
                        transform=ax6.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)

            plt.suptitle(f'PSK Denoising & Classification - Epoch {self.current_epoch}', fontsize=16)
            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({'psk_denoising_classification': wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()

    def configure_optimizers(self):
        # Use different learning rates for denoiser and classifier
        denoiser_params = list(self.unet.parameters()) + list(self.complex_loss.parameters())
        classifier_params = list(self.classifier.parameters())

        optimizer = AdamW([
            {'params': denoiser_params, 'lr': self.hparams.learning_rate},
            {'params': classifier_params, 'lr': self.hparams.learning_rate * 0.1}  # Lower LR for classifier
        ], weight_decay=1e-4)

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

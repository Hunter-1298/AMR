import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
import numpy as np
import wandb
from sklearn.manifold import TSNE


def check_and_handle_nan(tensor, name="tensor", replace_with_zero=True):
    """Check for NaN values and optionally replace them"""
    if torch.isnan(tensor).any():
        print(f"Warning: NaN detected in {name}")
        if replace_with_zero:
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        return tensor, True
    return tensor, False


def compute_instantaneous_frequency_shared(complex_signal):
    """Shared method to compute instantaneous frequency consistently with NaN protection"""
    try:
        # Handle both single sample and batch inputs
        if complex_signal.dim() == 1:
            # Single sample case
            if isinstance(complex_signal, torch.Tensor):
                complex_numpy = complex_signal.detach().cpu().numpy()
            else:
                complex_numpy = complex_signal

            # Check for NaN/inf in input
            if not np.isfinite(complex_numpy).all():
                print("Warning: Non-finite values in complex signal for IF computation")
                return (
                    torch.zeros_like(complex_signal.real)
                    if isinstance(complex_signal, torch.Tensor)
                    else np.zeros_like(np.real(complex_numpy))
                )

            # Add small epsilon to avoid zero magnitude
            magnitude = np.abs(complex_numpy)
            if np.any(magnitude < 1e-12):
                complex_numpy = complex_numpy + 1e-12 * (1 + 1j)

            # Compute phase and unwrap
            phase = np.angle(complex_numpy)

            # Check for valid phase values
            if not np.isfinite(phase).all():
                print("Warning: Non-finite phase values")
                return (
                    torch.zeros_like(complex_signal.real)
                    if isinstance(complex_signal, torch.Tensor)
                    else np.zeros_like(np.real(complex_numpy))
                )

            phase_unwrapped = np.unwrap(phase)

            # Compute derivative using gradient with edge handling
            if len(phase_unwrapped) > 1:
                inst_freq = np.gradient(phase_unwrapped)
            else:
                inst_freq = np.array([0.0])

            # Clip extreme values
            inst_freq = np.clip(inst_freq, -np.pi, np.pi)

            # Convert back to tensor if needed
            if isinstance(complex_signal, torch.Tensor):
                return torch.tensor(
                    inst_freq,
                    device=complex_signal.device,
                    dtype=complex_signal.real.dtype,
                )
            else:
                return inst_freq

        else:
            # Batch case
            batch_size = complex_signal.shape[0]
            inst_freqs = []

            for i in range(batch_size):
                # Get single sample
                signal_sample = complex_signal[i].detach().cpu().numpy()

                # Check for NaN/inf
                if not np.isfinite(signal_sample).all():
                    inst_freq_tensor = torch.zeros(
                        signal_sample.shape[0],
                        device=complex_signal.device,
                        dtype=complex_signal.real.dtype,
                    )
                    inst_freqs.append(inst_freq_tensor)
                    continue

                # Add small epsilon to avoid zero magnitude
                magnitude = np.abs(signal_sample)
                if np.any(magnitude < 1e-12):
                    signal_sample = signal_sample + 1e-12 * (1 + 1j)

                # Compute phase and unwrap
                phase = np.angle(signal_sample)
                if not np.isfinite(phase).all():
                    inst_freq_tensor = torch.zeros(
                        signal_sample.shape[0],
                        device=complex_signal.device,
                        dtype=complex_signal.real.dtype,
                    )
                    inst_freqs.append(inst_freq_tensor)
                    continue

                phase_unwrapped = np.unwrap(phase)

                # Compute derivative
                if len(phase_unwrapped) > 1:
                    inst_freq = np.gradient(phase_unwrapped)
                else:
                    inst_freq = np.array([0.0])

                # Clip extreme values
                inst_freq = np.clip(inst_freq, -np.pi, np.pi)

                # Convert back to tensor
                inst_freq_tensor = torch.tensor(
                    inst_freq,
                    device=complex_signal.device,
                    dtype=complex_signal.real.dtype,
                )
                inst_freqs.append(inst_freq_tensor)

            return torch.stack(inst_freqs, dim=0)

    except Exception as e:
        print(f"Error in shared instantaneous frequency computation: {e}")
        # Fallback: return zeros
        if isinstance(complex_signal, torch.Tensor):
            if complex_signal.dim() == 1:
                return torch.zeros_like(complex_signal.real)
            else:
                return torch.zeros_like(complex_signal.real)
        else:
            return np.zeros_like(np.real(complex_signal))


class ResidualBlock1D(nn.Module):
    """1D Residual block for better gradient flow"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return F.relu(out + identity)


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution with numerical stability"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.conv_imag = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )

        # Initialize with smaller weights for stability
        self._init_weights()

    def _init_weights(self):
        # Use Xavier initialization with smaller gain
        for conv in [self.conv_real, self.conv_imag]:
            nn.init.xavier_uniform_(conv.weight, gain=0.5)  # Reduced gain
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        batch_size, channels, length = x.shape
        in_channels = channels // 2

        # Split I and Q channels
        i_channels = x[:, :in_channels]
        q_channels = x[:, in_channels:]

        # Add small epsilon for numerical stability
        eps = 1e-8
        i_channels = i_channels + eps * torch.randn_like(i_channels)
        q_channels = q_channels + eps * torch.randn_like(q_channels)

        # Complex multiplication with clamping
        real_part = self.conv_real(i_channels) - self.conv_imag(q_channels)
        imag_part = self.conv_real(q_channels) + self.conv_imag(i_channels)

        # Clamp to prevent extreme values
        real_part = torch.clamp(real_part, -10.0, 10.0)
        imag_part = torch.clamp(imag_part, -10.0, 10.0)

        return torch.cat([real_part, imag_part], dim=1)


class EnhancedReconstructionLoss(nn.Module):
    def __init__(self, signal_length=1024, normalize_fft=True):  # Updated default
        super().__init__()
        self.signal_length = signal_length
        self.normalize_fft = normalize_fft

    def complex_mse_loss(self, pred_signal, target_signal):
        """Complex MSE Loss: L_ComplexMSE = (1/N) * Σ|s_t - ŝ_t|²"""
        pred_i, pred_q = pred_signal[:, 0], pred_signal[:, 1]
        target_i, target_q = target_signal[:, 0], target_signal[:, 1]

        pred_complex = torch.complex(pred_i.float(), pred_q.float())
        target_complex = torch.complex(target_i.float(), target_q.float())

        complex_diff = pred_complex - target_complex
        complex_mse = torch.mean(torch.abs(complex_diff) ** 2)
        return complex_mse

    def phase_loss(self, pred_signal, target_signal):
        """
        Time-Domain Phase Loss with NaN protection
        """
        pred_i, pred_q = pred_signal[:, 0], pred_signal[:, 1]
        target_i, target_q = target_signal[:, 0], target_signal[:, 1]

        pred_complex = torch.complex(pred_i.float(), pred_q.float())
        target_complex = torch.complex(target_i.float(), target_q.float())

        # Add small epsilon to avoid zero magnitude
        pred_magnitude = torch.abs(pred_complex) + 1e-12
        target_magnitude = torch.abs(target_complex) + 1e-12

        pred_complex = pred_complex + 1e-12 * torch.exp(
            1j * torch.randn_like(pred_complex.real)
        )
        target_complex = target_complex + 1e-12 * torch.exp(
            1j * torch.randn_like(target_complex.real)
        )

        # Compute phases
        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)

        # Check for NaN values
        pred_phase, _ = check_and_handle_nan(pred_phase, "pred_phase")
        target_phase, _ = check_and_handle_nan(target_phase, "target_phase")

        # Handle phase wrapping
        phase_diff = pred_phase - target_phase
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

        # Check final result
        phase_diff, _ = check_and_handle_nan(phase_diff, "phase_diff")

        # MSE of phase differences
        phase_mse = torch.mean(phase_diff**2)

        # Final NaN check
        if torch.isnan(phase_mse):
            print("Warning: NaN in phase loss, returning zero")
            return torch.tensor(0.0, device=pred_signal.device, dtype=pred_signal.dtype)

        return phase_mse

    def spectral_fft_loss(self, pred_signal, target_signal):
        """Spectral FFT Loss with NaN protection"""
        pred_i, pred_q = pred_signal[:, 0], pred_signal[:, 1]
        target_i, target_q = target_signal[:, 0], target_signal[:, 1]

        pred_complex = torch.complex(pred_i.float(), pred_q.float())
        target_complex = torch.complex(target_i.float(), target_q.float())

        # Check input for NaN
        pred_complex, _ = check_and_handle_nan(pred_complex, "pred_complex_fft")
        target_complex, _ = check_and_handle_nan(target_complex, "target_complex_fft")

        # Compute FFT magnitudes
        pred_fft = torch.fft.fft(pred_complex, dim=-1)
        target_fft = torch.fft.fft(target_complex, dim=-1)

        pred_fft_mag = torch.abs(pred_fft)
        target_fft_mag = torch.abs(target_fft)

        # Check FFT results
        pred_fft_mag, _ = check_and_handle_nan(pred_fft_mag, "pred_fft_mag")
        target_fft_mag, _ = check_and_handle_nan(target_fft_mag, "target_fft_mag")

        # Optional: Normalize FFT magnitudes with safe division
        if self.normalize_fft:
            pred_mean = torch.mean(pred_fft_mag, dim=-1, keepdim=True)
            target_mean = torch.mean(target_fft_mag, dim=-1, keepdim=True)

            pred_fft_mag = pred_fft_mag / (pred_mean + 1e-8)
            target_fft_mag = target_fft_mag / (target_mean + 1e-8)

        fft_loss = torch.mean((pred_fft_mag - target_fft_mag) ** 2)

        # Final NaN check
        if torch.isnan(fft_loss):
            print("Warning: NaN in FFT loss, returning zero")
            return torch.tensor(0.0, device=pred_signal.device, dtype=pred_signal.dtype)

        return fft_loss

    def instantaneous_frequency_loss(self, pred_signal, target_signal):
        """Instantaneous Frequency Loss with NaN protection"""
        pred_i, pred_q = pred_signal[:, 0], pred_signal[:, 1]
        target_i, target_q = target_signal[:, 0], target_signal[:, 1]

        pred_complex = torch.complex(pred_i.float(), pred_q.float())
        target_complex = torch.complex(target_i.float(), target_q.float())

        # Check input for NaN
        pred_complex, _ = check_and_handle_nan(pred_complex, "pred_complex_if")
        target_complex, _ = check_and_handle_nan(target_complex, "target_complex_if")

        # Compute instantaneous frequencies with protection
        try:
            pred_inst_freq = compute_instantaneous_frequency_shared(pred_complex)
            target_inst_freq = compute_instantaneous_frequency_shared(target_complex)

            # Check results
            pred_inst_freq, _ = check_and_handle_nan(pred_inst_freq, "pred_inst_freq")
            target_inst_freq, _ = check_and_handle_nan(
                target_inst_freq, "target_inst_freq"
            )

            if_loss = torch.mean((pred_inst_freq - target_inst_freq) ** 2)

            # Final check
            if torch.isnan(if_loss):
                print("Warning: NaN in IF loss, returning zero")
                return torch.tensor(
                    0.0, device=pred_signal.device, dtype=pred_signal.dtype
                )

            return if_loss

        except Exception as e:
            print(f"Error in IF loss computation: {e}")
            return torch.tensor(0.0, device=pred_signal.device, dtype=pred_signal.dtype)

    def forward(self, decoder_output, target_signal, labels=None):
        pred_signal = (
            decoder_output["signal"]
            if isinstance(decoder_output, dict)
            else decoder_output
        )

        # Four focused losses including explicit phase
        complex_mse = self.complex_mse_loss(pred_signal, target_signal)
        phase_loss = self.phase_loss(pred_signal, target_signal)  # NEW!
        fft_loss = self.spectral_fft_loss(pred_signal, target_signal)
        if_loss = self.instantaneous_frequency_loss(pred_signal, target_signal)

        # Weighted combination
        total_loss = (
            10.0 * complex_mse  # Primary: Complex MSE
            + 1.0 * phase_loss  # Explicit phase preservation
            + 0.1 * fft_loss  # Spectral preservation
            + 0.1 * if_loss  # Modulation structure
        )

        return total_loss, {
            "complex_mse": complex_mse,
            "phase_loss": phase_loss,
            "fft_loss": fft_loss,
            "instantaneous_frequency": if_loss,
        }


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales with gradient stability"""

    def __init__(self, in_channels=1):
        super().__init__()

        # Reduced kernel sizes to prevent extreme gradients
        kernel_sizes = [3, 7, 15, 31]  # Removed the largest kernel (63)

        self.conv_blocks = nn.ModuleList()

        for i, kernel_size in enumerate(kernel_sizes):
            block = self._make_stable_conv_block(
                in_channels, 16, kernel_size, f"scale_{i}"
            )
            self.conv_blocks.append(block)

        # Learnable attention weights for each scale
        self.scale_weights = nn.Parameter(torch.ones(len(kernel_sizes)) * 0.25)

        # Complex processing with reduced channels
        # Input will be 16*4 = 64 channels for I and Q each, so 128 total
        self.complex_conv = self._make_stable_complex_conv(64, 32)

        # Final stabilization
        self.output_norm = nn.LayerNorm(64)  # 32*2 channels from complex conv

    def _make_stable_conv_block(self, in_ch, out_ch, kernel_size, name):
        padding = kernel_size // 2

        # Gradient scaling factor based on kernel size
        # Larger kernels get smaller initialization to prevent gradient explosion
        scale_factor = 1.0 / np.sqrt(kernel_size)

        block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01),  # Conservative BN
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
        )

        # Custom initialization for stability
        with torch.no_grad():
            nn.init.xavier_uniform_(block[0].weight, gain=scale_factor)

        return block

    def _make_stable_complex_conv(self, in_channels, out_channels):
        """Stable complex convolution"""
        return nn.Sequential(
            nn.Conv1d(in_channels * 2, out_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * 2, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # x shape: [batch, 2, length] where dim=1 is [I, Q]
        batch_size, _, length = x.shape

        # Process I and Q channels separately
        i_channel = x[:, 0:1]  # [batch, 1, length]
        q_channel = x[:, 1:2]  # [batch, 1, length]

        # Extract multi-scale features with weighted combination
        i_features = []
        q_features = []

        for idx, block in enumerate(self.conv_blocks):
            # Apply block
            i_feat = block(i_channel)
            q_feat = block(q_channel)

            # Apply learnable scale weights
            scale_weight = torch.sigmoid(self.scale_weights[idx])  # Ensure positive
            i_feat = i_feat * scale_weight
            q_feat = q_feat * scale_weight

            # Gradient checkpointing for memory efficiency (optional)
            # i_feat = torch.utils.checkpoint.checkpoint(lambda x: x, i_feat)
            # q_feat = torch.utils.checkpoint.checkpoint(lambda x: x, q_feat)

            i_features.append(i_feat)
            q_features.append(q_feat)

        # Concatenate features
        i_concat = torch.cat(i_features, dim=1)  # [batch, 64, length]
        q_concat = torch.cat(q_features, dim=1)  # [batch, 64, length]

        # Combine I and Q for complex processing
        combined = torch.cat([i_concat, q_concat], dim=1)  # [batch, 128, length]

        # Clamp before complex processing
        combined = torch.clamp(combined, -10, 10)

        # Apply stable complex convolution
        complex_features = self.complex_conv(combined)  # [batch, 64, length]

        # Apply layer normalization for stability
        # Reshape for LayerNorm: [batch, length, channels]
        complex_features = complex_features.permute(0, 2, 1)
        complex_features = self.output_norm(complex_features)
        complex_features = complex_features.permute(0, 2, 1)  # Back to [batch, channels, length]

        # Final clamping
        complex_features = torch.clamp(complex_features, -5, 5)

        return complex_features

class FrequencyDomainProcessor(nn.Module):
    """Process frequency domain features with numerical stability"""

    def __init__(self, signal_length=1024):
        super().__init__()
        self.signal_length = signal_length

        self.mag_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32, eps=1e-5),  # Increase BatchNorm epsilon
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32, eps=1e-5),
            nn.ReLU(),
        )

        self.phase_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32, eps=1e-5),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size = x.shape[0]

        # Convert to float32 for stability
        i_channel = x[:, 0].float()
        q_channel = x[:, 1].float()

        # Add small noise for numerical stability
        eps = 1e-8
        i_channel = i_channel + eps * torch.randn_like(i_channel)
        q_channel = q_channel + eps * torch.randn_like(q_channel)

        # Convert to complex tensor
        complex_signal = torch.complex(i_channel, q_channel)

        # Add minimum magnitude to avoid zero division
        magnitude = torch.abs(complex_signal)
        min_mag = 1e-12
        complex_signal = complex_signal + min_mag * torch.exp(
            1j * torch.angle(complex_signal + min_mag)
        )

        # FFT with window to reduce spectral leakage
        # Apply Hann window
        window = torch.hann_window(complex_signal.shape[-1], device=x.device)
        windowed_signal = complex_signal * window

        # FFT
        fft_signal = torch.fft.fft(windowed_signal, dim=-1)

        # Extract magnitude and phase with stability
        magnitude = torch.abs(fft_signal) + 1e-12
        phase = torch.angle(fft_signal)

        # Clamp extreme values
        magnitude = torch.clamp(magnitude, 1e-12, 100.0)
        phase = torch.clamp(phase, -np.pi, np.pi)

        # Log-scale magnitude for better numerical properties
        log_magnitude = torch.log(magnitude + 1e-12).unsqueeze(1)
        phase = phase.unsqueeze(1)

        # Convert back to original dtype
        log_magnitude = log_magnitude.to(x.dtype)
        phase = phase.to(x.dtype)

        # Process features
        mag_features = self.mag_processor(log_magnitude)
        phase_features = self.phase_processor(phase)

        return torch.cat([mag_features, phase_features], dim=1)


class RFEncoder(nn.Module):
    """Enhanced encoder with numerical stability"""

    def __init__(self, signal_length=1024, latent_dim=256):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim

        # Multi-scale time domain features
        self.time_features = MultiScaleFeatureExtractor(in_channels=1)
        # Frequency domain features
        self.freq_features = FrequencyDomainProcessor(signal_length)

        combined_channels = 128

        # Feature-level transformer with stability improvements
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=combined_channels,
            nhead=8,
            dim_feedforward=combined_channels * 2,  # Reduced from 4x
            dropout=0.2,  # Increased dropout
            activation="gelu",  # GELU is more stable than ReLU
            batch_first=False,
            norm_first=True,
        )
        self.feature_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Encoder layers with batch normalization and stability
        self.conv1 = nn.Sequential(
            nn.Conv1d(combined_channels, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256, eps=1e-5),
            nn.GELU(),  # More stable than ReLU
            nn.Dropout(0.2),
        )

        # Reduced transformer complexity
        encoder_layer_256 = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,  # Reduced
            dropout=0.3,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.transformer_256 = nn.TransformerEncoder(encoder_layer_256, num_layers=1)

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        encoder_layer_128 = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,  # Reduced
            dropout=0.3,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.transformer_128 = nn.TransformerEncoder(encoder_layer_128, num_layers=1)

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Stable final mapping with layer normalization
        self.to_latent = nn.Sequential(
            nn.Linear(4096, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim, eps=1e-5),  # Final normalization
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # Add input noise for regularization
        if self.training:
            x = x + 1e-6 * torch.randn_like(x)

        # Extract features with stability checks
        time_features = self.time_features(x)
        freq_features = self.freq_features(x)

        # Check for NaN after feature extraction
        time_features = torch.where(
            torch.isnan(time_features), torch.zeros_like(time_features), time_features
        )
        freq_features = torch.where(
            torch.isnan(freq_features), torch.zeros_like(freq_features), freq_features
        )

        # Combine features
        combined_features = torch.cat([time_features, freq_features], dim=1)

        # Clamp combined features
        combined_features = torch.clamp(combined_features, -10.0, 10.0)

        # Apply feature transformer with stability
        x = combined_features.permute(2, 0, 1)
        x = self.feature_transformer(x)
        x = x.permute(1, 2, 0)

        # Progressive encoding with stability checks
        x = self.conv1(x)
        x = torch.clamp(x, -10.0, 10.0)  # Clamp after each major operation

        x = x.permute(2, 0, 1)
        x = self.transformer_256(x)
        x = x.permute(1, 2, 0)
        x = torch.clamp(x, -10.0, 10.0)

        x = self.conv2(x)
        x = torch.clamp(x, -10.0, 10.0)

        x = x.permute(2, 0, 1)
        x = self.transformer_128(x)
        x = x.permute(1, 2, 0)
        x = torch.clamp(x, -10.0, 10.0)

        x = self.conv3(x)
        x = torch.clamp(x, -10.0, 10.0)

        x = self.conv4(x)
        x = torch.clamp(x, -10.0, 10.0)

        # Flatten and get latent representation
        encoded_flat = x.view(x.shape[0], -1)
        encoded_flat = torch.clamp(encoded_flat, -10.0, 10.0)

        latent = self.to_latent(encoded_flat)
        latent = torch.clamp(latent, -5.0, 5.0)  # Final clamping

        return latent


class RFDecoderEnhanced(nn.Module):
    """Decoder with numerical stability"""

    def __init__(self, latent_dim=256, signal_length=1024):
        super().__init__()
        self.signal_length = signal_length
        self.init_size = signal_length // 16

        # Initial mapping with layer normalization
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256 * self.init_size),
            nn.LayerNorm(256 * self.init_size, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Main decoder layers with stability
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose1d(
                256, 512, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(512, eps=1e-5),
            nn.GELU(),
            ResidualBlock1D(512, 512),
            nn.ConvTranspose1d(
                512, 256, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(256, eps=1e-5),
            nn.GELU(),
            ResidualBlock1D(256, 256),
            nn.ConvTranspose1d(
                256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(128, eps=1e-5),
            nn.GELU(),
            ResidualBlock1D(128, 128),
            nn.ConvTranspose1d(
                128, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(128, eps=1e-5),
            nn.GELU(),
        )

        # Transformer with reduced complexity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,  # Reduced from 512
            dropout=0.2,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )  # Reduced layers

        # Final refinement layers
        self.final_layers = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(32, 2, kernel_size=7, padding=3),
            nn.Tanh(),  # Bound output to [-1, 1]
        )

        # Learnable but bounded scale
        self.output_scale = nn.Parameter(torch.tensor(0.5))  # Start smaller
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, latent):
        # Clamp input latent
        latent = torch.clamp(latent, -5.0, 5.0)

        x = self.from_latent(latent)
        x = torch.clamp(x, -10.0, 10.0)

        x = x.view(x.shape[0], 256, self.init_size)

        # Apply decoder layers with stability checks
        x = self.decoder_layers(x)
        x = torch.clamp(x, -10.0, 10.0)

        # Apply transformer
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)
        x = torch.clamp(x, -10.0, 10.0)

        # Final processing
        x = self.final_layers(x)

        # Clamp scale parameter
        scale = torch.clamp(self.output_scale, 0.1, 2.0)
        reconstructed = x * scale

        # Ensure correct length
        if reconstructed.shape[-1] != self.signal_length:
            reconstructed = F.interpolate(
                reconstructed,
                size=self.signal_length,
                mode="linear",
                align_corners=False,
            )

        # Final output clamping
        reconstructed = torch.clamp(reconstructed, -5.0, 5.0)

        return reconstructed


class AdaptiveArcFaceLoss(nn.Module):
    """ArcFace loss with SNR-adaptive margins"""

    def __init__(self, embedding_dim, num_classes, base_margin=0.5, scale=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.base_margin = base_margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def get_adaptive_margin(self, snrs):
        """Compute adaptive margin based on SNR"""
        # Higher margin for high SNR (easier to separate)
        # Lower margin for low SNR (harder to separate)
        normalized_snr = torch.clamp(
            (snrs + 20) / 38, 0, 1
        )  # Normalize -20 to 18 dB to [0,1]
        adaptive_margin = self.base_margin * (
            0.2 + 0.8 * normalized_snr
        )  # Range: 0.2 to 1.0
        return adaptive_margin

    def forward(self, embeddings, labels, snrs):
        embeddings = embeddings.float()
        device = embeddings.device

        # Check for NaN in embeddings
        embeddings, had_nan = check_and_handle_nan(embeddings, "arcface_embeddings")
        if had_nan:
            print("Warning: NaN values found in ArcFace embeddings")

        # Normalize embeddings and weights with safe normalization
        embeddings_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings = embeddings / (embeddings_norm + 1e-8)

        weight = F.normalize(self.weight.float(), p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)

        # Clamp cosine values to prevent NaN in acos
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)

        # Get target cosine values
        target_cosine = cosine[torch.arange(len(labels), device=device), labels]

        # Get adaptive margins
        adaptive_margins = self.get_adaptive_margin(snrs.squeeze())

        # Add adaptive margin to target with safe acos/cos operations
        target_theta = torch.acos(torch.clamp(target_cosine, -1 + 1e-7, 1 - 1e-7))
        target_theta_margin = target_theta + adaptive_margins
        target_cosine_margin = torch.cos(target_theta_margin)

        # Replace target values with margin-adjusted values
        logits = cosine.clone()
        target_cosine_margin = target_cosine_margin.to(logits.dtype)
        logits[torch.arange(len(labels), device=device), labels] = target_cosine_margin

        # Scale logits
        logits *= self.scale

        # Check for NaN in final logits
        logits, had_nan = check_and_handle_nan(logits, "arcface_logits")
        if had_nan:
            print("Warning: NaN in ArcFace logits")

        return F.cross_entropy(logits, labels)


class RFEncoderDecoder(L.LightningModule):
    """Enhanced Phase and frequency aware encoder-decoder"""

    def __init__(
        self,
        label_names,
        signal_length=1024,
        latent_dim=256,
        learning_rate=1e-3,
        arcface_margin=0.4,
        arcface_scale=32,
        reconstruction_weight=1.0,
        arcface_weight=0.1,
        curriculum_learning=False,
        initial_snr_threshold=10,
        final_snr_threshold=-20,
        curriculum_epochs=20,
        use_enhanced_decoder=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.label_names = label_names
        self.num_classes = len(label_names)
        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.arcface_weight = arcface_weight
        self.use_enhanced_decoder = use_enhanced_decoder

        # Curriculum learning parameters
        self.curriculum_learning = curriculum_learning
        self.initial_snr_threshold = initial_snr_threshold
        self.final_snr_threshold = final_snr_threshold
        self.curriculum_epochs = curriculum_epochs

        # Networks - choose encoder type
        self.encoder = RFEncoder(signal_length, latent_dim)

        # Choose decoder type
        self.decoder = RFDecoderEnhanced(latent_dim, signal_length)  # Single decoder

        # Loss functions
        self.arcface_loss = AdaptiveArcFaceLoss(
            latent_dim, self.num_classes, arcface_margin, arcface_scale
        )

        self.reconstruction_loss = EnhancedReconstructionLoss(signal_length)

    def get_current_snr_threshold(self):
        """Calculate current SNR threshold based on training progress"""
        if not self.curriculum_learning:
            return self.final_snr_threshold

        # Linear decay from initial to final threshold
        progress = min(self.current_epoch / self.curriculum_epochs, 1.0)
        current_threshold = (
            self.initial_snr_threshold * (1 - progress)
            + self.final_snr_threshold * progress
        )

        return current_threshold

    def get_arcface_embeddings(self, x):
        """Get the normalized embeddings used by ArcFace (before classification)"""
        with torch.no_grad():
            encoder_output = self.encode(x)

            if isinstance(encoder_output, dict):
                # For SNR-aware encoder, use signal features or full embedding
                if "signal_features" in encoder_output:
                    embeddings = encoder_output["signal_features"]
                else:
                    embeddings = encoder_output["full_embedding"]
            else:
                # For regular encoder
                embeddings = encoder_output

            # Normalize embeddings the same way ArcFace does
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

            return normalized_embeddings

    def get_raw_embeddings(self, x):
        """Get the raw (unnormalized) embeddings"""
        with torch.no_grad():
            encoder_output = self.encode(x)

            if isinstance(encoder_output, dict):
                if "signal_features" in encoder_output:
                    return encoder_output["signal_features"]
                else:
                    return encoder_output["full_embedding"]
            else:
                return encoder_output

    def create_curriculum_mask(self, snrs, current_threshold):
        """Create mask for samples that should participate in training"""
        # Only train on samples with SNR >= current_threshold
        mask = snrs >= current_threshold
        return mask

    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)

    def decode(self, latent):
        """Reconstruct from latent"""
        if isinstance(latent, dict):
            # If using SNR-aware encoder, use full embedding for reconstruction
            return self.decoder(latent["full_embedding"])
        else:
            # Original encoder returns tensor directly
            return self.decoder(latent)

    def forward(self, x):
        """Full forward pass"""
        encoder_output = self.encode(x)
        decoder_output = self.decode(encoder_output)
        return encoder_output, decoder_output

    def training_step(self, batch, batch_idx):
        try:
            import ipdb

            ipdb.set_trace()
            x, labels, snrs = batch
            batch_size = x.shape[0]

            # Check input for NaN
            x, had_nan = check_and_handle_nan(x, "input_signals")
            if had_nan:
                print(f"Warning: NaN in input signals at batch {batch_idx}")

            # Get current SNR threshold for curriculum learning
            current_threshold = self.get_current_snr_threshold()

            # Create curriculum mask
            # curriculum_mask = self.create_curriculum_mask(
            #     snrs.squeeze(), current_threshold
            # )

            # Check if any samples in batch meet the curriculum criteria
            # if not curriculum_mask.any():
            #     self.log("curriculum_snr_threshold", current_threshold, prog_bar=True)
            #     self.log("curriculum_batch_skipped", 1.0)
            #     self.log("curriculum_samples_used", 0.0)
            #     return None

            # Filter batch to only include curriculum samples
            # x_curriculum = x[curriculum_mask]
            # labels_curriculum = labels[curriculum_mask]
            # snrs_curriculum = snrs[curriculum_mask]

            # Forward pass
            encoder_output, decoder_output = self.forward(x)

            # Check encoder/decoder outputs
            encoder_output, _ = check_and_handle_nan(encoder_output, "encoder_output")
            decoder_output, _ = check_and_handle_nan(decoder_output, "decoder_output")

            # Reconstruction loss
            recon_loss, loss_components = self.reconstruction_loss(
                decoder_output, x, labels
            )

            # Check reconstruction loss
            if torch.isnan(recon_loss):
                print(f"Warning: NaN in reconstruction loss at batch {batch_idx}")
                recon_loss = torch.tensor(0.0, device=x.device, requires_grad=True)

            # ArcFace loss
            try:
                arcface_loss = self.arcface_loss(
                    encoder_output, labels.squeeze(), snrs
                )

                if torch.isnan(arcface_loss):
                    print(f"Warning: NaN in ArcFace loss at batch {batch_idx}")
                    arcface_loss = torch.tensor(
                        0.0, device=x.device, requires_grad=True
                    )

            except Exception as e:
                print(f"Error in ArcFace loss computation: {e}")
                arcface_loss = torch.tensor(0.0, device=x.device, requires_grad=True)

            # Combined loss
            total_loss = (
                self.reconstruction_weight * recon_loss
                + self.arcface_weight * arcface_loss
            )

            # Final NaN check
            if torch.isnan(total_loss):
                print(
                    f"Warning: NaN in total loss at batch {batch_idx}, skipping batch"
                )
                return None

            # Clean logging
            self.log("train_loss", total_loss, prog_bar=True)
            self.log("train_recon_loss", recon_loss)
            self.log("train_arcface_loss", arcface_loss)

            return total_loss

        except Exception as e:
            print(f"Error in training step: {e}")
            return None

    def validation_step(self, batch, batch_idx):
        try:
            x, labels, snrs = batch

            # Check input
            x, had_nan = check_and_handle_nan(x, "val_input_signals")
            if had_nan:
                print(f"Warning: NaN in validation input at batch {batch_idx}")

            # Forward pass
            encoder_output, decoder_output = self.forward(x)

            # Check outputs
            encoder_output, _ = check_and_handle_nan(
                encoder_output, "val_encoder_output"
            )
            decoder_output, _ = check_and_handle_nan(
                decoder_output, "val_decoder_output"
            )

            # Losses
            try:
                arcface_loss = self.arcface_loss(encoder_output, labels.squeeze(), snrs)
                if torch.isnan(arcface_loss):
                    arcface_loss = torch.tensor(0.0, device=x.device)
            except:
                arcface_loss = torch.tensor(0.0, device=x.device)

            recon_loss, loss_components = self.reconstruction_loss(
                decoder_output, x, labels
            )

            if torch.isnan(recon_loss):
                recon_loss = torch.tensor(0.0, device=x.device)

            # Combined loss
            total_loss = (
                self.reconstruction_weight * recon_loss
                + self.arcface_weight * arcface_loss
            )

            if torch.isnan(total_loss):
                print(f"Warning: NaN in validation loss at batch {batch_idx}")
                return torch.tensor(0.0, device=x.device)

            # Logging
            self.log("val_loss", total_loss, prog_bar=True)
            self.log("val_recon_loss", recon_loss)
            self.log("val_arcface_loss", arcface_loss)

            for key, value in loss_components.items():
                if not torch.isnan(value):
                    self.log(f"val_{key}", value)

            # Save for visualization (first batch only)
            if batch_idx == 0:
                self.val_latents = encoder_output.detach().cpu()
                self.val_labels = labels.detach().cpu()
                self.val_snrs = snrs.detach().cpu()
                self.val_original = x.detach().cpu()
                self.val_reconstructed = decoder_output.detach().cpu()

            return total_loss

        except Exception as e:
            print(f"Error in validation step: {e}")
            return torch.tensor(0.0, device=batch[0].device)

    def on_train_epoch_end(self):
        """Log curriculum progress at end of each epoch"""
        current_threshold = self.get_current_snr_threshold()
        progress = min(self.current_epoch / self.curriculum_epochs, 1.0) * 100

        # Calculate approximate data coverage
        total_snr_range = 30 - (-20)  # 38 dB range
        included_range = 30 - current_threshold
        data_coverage = min(included_range / total_snr_range * 100, 100)

        print(
            f"Epoch {self.current_epoch}: SNR threshold = {current_threshold:.1f} dB "
            f"(~{data_coverage:.0f}% of data, Progress: {progress:.1f}%)"
        )

        if self.current_epoch >= self.curriculum_epochs:
            print("Curriculum learning completed - training on all SNR levels")

    def _plot_reconstruction_quality(self):
        """Enhanced visualization including phase analysis"""
        if not hasattr(self, "val_original") or not hasattr(self, "val_reconstructed"):
            return

        try:
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))  # 3x3 grid now

            # Use first sample for detailed analysis
            sample_idx = 0
            orig_i = self.val_original[sample_idx, 0].float()
            orig_q = self.val_original[sample_idx, 1].float()
            recon_i = self.val_reconstructed[sample_idx, 0].float()
            recon_q = self.val_reconstructed[sample_idx, 1].float()

            # Create complex signals
            orig_complex = torch.complex(orig_i, orig_q)
            recon_complex = torch.complex(recon_i, recon_q)
            time_axis = np.arange(len(orig_i))

            # 1. I/Q Time Domain
            axes[0, 0].plot(
                time_axis, orig_i.numpy(), "b-", label="Original I", alpha=0.8
            )
            axes[0, 0].plot(
                time_axis, recon_i.numpy(), "r--", label="Reconstructed I", alpha=0.8
            )
            axes[0, 0].plot(
                time_axis, orig_q.numpy(), "c-", label="Original Q", alpha=0.8
            )
            axes[0, 0].plot(
                time_axis, recon_q.numpy(), "m--", label="Reconstructed Q", alpha=0.8
            )

            complex_mse = torch.mean(
                torch.abs(orig_complex - recon_complex) ** 2
            ).item()
            axes[0, 0].set_title(f"I/Q Reconstruction\nComplex MSE: {complex_mse:.6f}")
            axes[0, 0].set_xlabel("Time Sample")
            axes[0, 0].set_ylabel("Amplitude")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Phase Comparison (NEW!)
            orig_phase = torch.angle(orig_complex).numpy()
            recon_phase = torch.angle(recon_complex).numpy()

            # Unwrap phases for better visualization
            orig_phase_unwrapped = np.unwrap(orig_phase)
            recon_phase_unwrapped = np.unwrap(recon_phase)

            axes[0, 1].plot(
                time_axis, orig_phase_unwrapped, "b-", label="Original Phase", alpha=0.8
            )
            axes[0, 1].plot(
                time_axis,
                recon_phase_unwrapped,
                "r--",
                label="Reconstructed Phase",
                alpha=0.8,
            )

            # Calculate phase loss
            phase_diff = torch.atan2(
                torch.sin(torch.angle(orig_complex) - torch.angle(recon_complex)),
                torch.cos(torch.angle(orig_complex) - torch.angle(recon_complex)),
            )
            phase_mse = torch.mean(phase_diff**2).item()

            axes[0, 1].set_title(f"Phase Comparison\nPhase MSE: {phase_mse:.6f} rad²")
            axes[0, 1].set_xlabel("Time Sample")
            axes[0, 1].set_ylabel("Phase (radians)")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Phase Error (NEW!)
            phase_error = phase_diff.numpy()
            axes[0, 2].plot(time_axis, phase_error, "r-", alpha=0.8)
            axes[0, 2].axhline(y=0, color="k", linestyle="--", alpha=0.3)
            axes[0, 2].set_title(
                f"Phase Error\nMean: {np.mean(phase_error):.4f}, Std: {np.std(phase_error):.4f}"
            )
            axes[0, 2].set_xlabel("Time Sample")
            axes[0, 2].set_ylabel("Phase Error (radians)")
            axes[0, 2].grid(True, alpha=0.3)

            # 4. Constellation Diagram
            axes[1, 0].scatter(
                orig_complex.real.numpy(),
                orig_complex.imag.numpy(),
                alpha=0.6,
                s=20,
                c="blue",
                label="Original",
            )
            axes[1, 0].scatter(
                recon_complex.real.numpy(),
                recon_complex.imag.numpy(),
                alpha=0.6,
                s=20,
                c="red",
                label="Reconstructed",
            )
            axes[1, 0].set_title("Constellation Diagram")
            axes[1, 0].set_xlabel("I (Real)")
            axes[1, 0].set_ylabel("Q (Imaginary)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axis("equal")

            # 5. FFT Magnitude Spectrum
            orig_fft = torch.fft.fft(orig_complex)
            recon_fft = torch.fft.fft(recon_complex)
            freqs = np.arange(len(orig_fft))

            fft_loss = torch.mean(
                (torch.abs(orig_fft) - torch.abs(recon_fft)) ** 2
            ).item()

            axes[1, 1].plot(
                freqs, torch.abs(orig_fft).numpy(), "b-", label="Original", alpha=0.8
            )
            axes[1, 1].plot(
                freqs,
                torch.abs(recon_fft).numpy(),
                "r--",
                label="Reconstructed",
                alpha=0.8,
            )
            axes[1, 1].set_title(f"FFT Magnitude Spectrum\nFFT Loss: {fft_loss:.6f}")
            axes[1, 1].set_xlabel("Frequency Bin")
            axes[1, 1].set_ylabel("FFT Magnitude")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # 6. FFT Phase Spectrum (NEW!)
            orig_fft_phase = torch.angle(orig_fft).numpy()
            recon_fft_phase = torch.angle(recon_fft).numpy()

            axes[1, 2].plot(freqs, orig_fft_phase, "b-", label="Original", alpha=0.8)
            axes[1, 2].plot(
                freqs, recon_fft_phase, "r--", label="Reconstructed", alpha=0.8
            )
            axes[1, 2].set_title("FFT Phase Spectrum")
            axes[1, 2].set_xlabel("Frequency Bin")
            axes[1, 2].set_ylabel("Phase (radians)")
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

            # 7. Instantaneous Frequency
            orig_inst_freq = self._compute_instantaneous_frequency(orig_complex)
            recon_inst_freq = self._compute_instantaneous_frequency(recon_complex)

            if_loss = torch.mean((orig_inst_freq - recon_inst_freq) ** 2).item()

            axes[2, 0].plot(
                time_axis, orig_inst_freq.numpy(), "b-", label="Original", alpha=0.8
            )
            axes[2, 0].plot(
                time_axis,
                recon_inst_freq.numpy(),
                "r--",
                label="Reconstructed",
                alpha=0.8,
            )
            axes[2, 0].set_title(f"Instantaneous Frequency\nIF Loss: {if_loss:.6f}")
            axes[2, 0].set_xlabel("Time Sample")
            axes[2, 0].set_ylabel("Frequency (rad/sample)")
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

            # 8. Power Spectral Density
            orig_psd = torch.abs(orig_fft) ** 2
            recon_psd = torch.abs(recon_fft) ** 2

            axes[2, 1].plot(
                freqs,
                10 * torch.log10(orig_psd + 1e-10).numpy(),
                "b-",
                label="Original",
                alpha=0.8,
            )
            axes[2, 1].plot(
                freqs,
                10 * torch.log10(recon_psd + 1e-10).numpy(),
                "r--",
                label="Reconstructed",
                alpha=0.8,
            )
            axes[2, 1].set_title("Power Spectral Density")
            axes[2, 1].set_xlabel("Frequency Bin")
            axes[2, 1].set_ylabel("Power (dB)")
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)

            # 9. Loss Component Breakdown
            with torch.no_grad():
                sample_orig = self.val_original[sample_idx : sample_idx + 1]
                sample_recon = self.val_reconstructed[sample_idx : sample_idx + 1]

                loss_fn = EnhancedReconstructionLoss(self.hparams.signal_length)
                _, loss_breakdown = loss_fn(sample_recon, sample_orig)

                loss_names = [
                    "Complex MSE",
                    "Phase Loss",
                    "FFT Loss",
                    "Inst. Freq",
                ]  # Updated
                loss_values = [
                    loss_breakdown["complex_mse"].item(),
                    loss_breakdown["phase_loss"].item(),  # NEW!
                    loss_breakdown["fft_loss"].item(),
                    loss_breakdown["instantaneous_frequency"].item(),
                ]

                bars = axes[2, 2].bar(
                    loss_names,
                    loss_values,
                    color=["blue", "red", "orange", "green"],
                    alpha=0.7,
                )
                axes[2, 2].set_title("Loss Component Breakdown")
                axes[2, 2].set_ylabel("Loss Value")
                axes[2, 2].tick_params(axis="x", rotation=45)

                # Add value labels on bars
                for bar, value in zip(bars, loss_values):
                    height = bar.get_height()
                    axes[2, 2].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.4f}",
                        ha="center",
                        va="bottom",
                    )

            plt.tight_layout()

            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.log({"reconstruction_quality": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in reconstruction quality visualization: {e}")
            plt.close("all")

    def _plot_arcface_embeddings(self):
        """Core ArcFace embedding analysis: class separation and angular distances"""
        if not hasattr(self, "val_latents") or not hasattr(self, "val_labels"):
            return

        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Normalize embeddings (should already be normalized by ArcFace)
            embeddings_norm = F.normalize(self.val_latents, p=2, dim=1).numpy()

            # Check for NaN values and remove them
            nan_mask = np.isnan(embeddings_norm).any(axis=1)
            if nan_mask.any():
                print(
                    f"Warning: Found {nan_mask.sum()} samples with NaN values in embeddings. Removing them for visualization."
                )
                embeddings_norm = embeddings_norm[~nan_mask]
                val_labels_clean = self.val_labels[~nan_mask]
            else:
                val_labels_clean = self.val_labels

            # Check if we have enough samples left
            if len(embeddings_norm) < 2:
                print(
                    "Warning: Not enough valid samples for t-SNE visualization after NaN removal."
                )
                return

            unique_labels = torch.unique(val_labels_clean).numpy()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

            # 1. t-SNE visualization
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(embeddings_norm) // 4),
            )
            embeddings_2d = tsne.fit_transform(embeddings_norm)

            for i, label in enumerate(unique_labels):
                mask = val_labels_clean.squeeze().numpy() == label
                if mask.sum() > 0:
                    axes[0].scatter(
                        embeddings_2d[mask, 0],
                        embeddings_2d[mask, 1],
                        c=[colors[i]],
                        label=self.label_names[int(label)],
                        alpha=0.7,
                    )

            axes[0].set_title("t-SNE: Class Separation")
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[0].grid(True, alpha=0.3)

            # 2. Angular distances on unit circle (first 2 dims)
            embeddings_2d_circle = embeddings_norm[:, :2]
            # Add small epsilon to avoid division by zero
            norms = np.linalg.norm(embeddings_2d_circle, axis=1, keepdims=True)
            embeddings_2d_circle = embeddings_2d_circle / (norms + 1e-8)

            for i, label in enumerate(unique_labels):
                mask = val_labels_clean.squeeze().numpy() == label
                if mask.sum() > 0:
                    axes[1].scatter(
                        embeddings_2d_circle[mask, 0],
                        embeddings_2d_circle[mask, 1],
                        c=[colors[i]],
                        label=self.label_names[int(label)],
                        alpha=0.7,
                    )

            # Draw unit circle
            circle = plt.Circle(
                (0, 0), 1, fill=False, color="gray", linestyle="--", alpha=0.8
            )
            axes[1].add_artist(circle)
            axes[1].set_xlim(-1.2, 1.2)
            axes[1].set_ylim(-1.2, 1.2)
            axes[1].set_aspect("equal")
            axes[1].set_title("Unit Circle Projection")
            axes[1].grid(True, alpha=0.3)

            # 3. Inter-class angular distance matrix
            n_classes = len(unique_labels)
            distance_matrix = np.zeros((n_classes, n_classes))
            class_names = []

            # Calculate class centroids
            for i, label in enumerate(unique_labels):
                mask = val_labels_clean.squeeze().numpy() == label
                if mask.sum() > 0:
                    centroid_i = embeddings_norm[mask].mean(axis=0)
                    # Add small epsilon to avoid division by zero
                    centroid_i = centroid_i / (np.linalg.norm(centroid_i) + 1e-8)
                    class_names.append(self.label_names[int(label)])

                    for j, label_j in enumerate(unique_labels):
                        mask_j = val_labels_clean.squeeze().numpy() == label_j
                        if mask_j.sum() > 0:
                            centroid_j = embeddings_norm[mask_j].mean(axis=0)
                            centroid_j = centroid_j / (
                                np.linalg.norm(centroid_j) + 1e-8
                            )

                            # Angular distance in degrees
                            cos_sim = np.clip(np.dot(centroid_i, centroid_j), -1.0, 1.0)
                            angular_dist = np.arccos(cos_sim) * 180 / np.pi
                            distance_matrix[i, j] = angular_dist

            im = axes[2].imshow(distance_matrix, cmap="viridis")
            axes[2].set_xticks(range(len(class_names)))
            axes[2].set_yticks(range(len(class_names)))
            axes[2].set_xticklabels(class_names, rotation=45, ha="right")
            axes[2].set_yticklabels(class_names)
            axes[2].set_title("Angular Distances (degrees)")

            # Add text annotations
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    axes[2].text(
                        j,
                        i,
                        f"{distance_matrix[i, j]:.1f}°",
                        ha="center",
                        va="center",
                        color="white"
                        if distance_matrix[i, j] > distance_matrix.max() / 2
                        else "black",
                        fontsize=8,
                    )

            plt.colorbar(im, ax=axes[2])
            plt.tight_layout()

            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.log({"arcface_analysis": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in ArcFace visualization: {e}")
            plt.close("all")

    def on_validation_epoch_end(self):
        """Simplified validation visualization - only core metrics"""
        if not hasattr(self, "val_latents"):
            return

        # Only plot the essential visualizations
        self._plot_reconstruction_quality()  # I/Q, constellation, frequency/phase
        self._plot_arcface_embeddings()  # Class separation and angular distances

    def configure_optimizers(self):
        """Using PyTorch's built-in warmup scheduler"""

        # Target learning rates
        max_lr = 1e-4
        min_lr = 1e-6

        optimizer = AdamW(
            self.parameters(),
            lr=max_lr,  # This will be the peak LR
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,  # Start at 1% of max_lr
            end_factor=1.0,  # End at 100% of max_lr
            total_iters=10,  # 10 epochs of warmup
        )

        # Main scheduler (starts after warmup)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=40,  # 90 epochs of cosine annealing
            eta_min=min_lr,  # Minimum LR
        )

        # Sequential scheduler: warmup -> cosine annealing
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[10],  # Switch at epoch 10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "sequential_warmup_cosine",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        """Comprehensive gradient handling for Inf/NaN prevention"""

        # Track gradient statistics
        total_norm = 0
        param_count = 0
        problematic_params = []

        # First pass: analyze all gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

                # Check for problematic gradients
                if torch.isinf(param.grad).any():
                    problematic_params.append(f"{name}:Inf")
                    param.grad.data.zero_()  # Zero out infinite gradients

                elif torch.isnan(param.grad).any():
                    problematic_params.append(f"{name}:NaN")
                    param.grad.data.zero_()  # Zero out NaN gradients

                elif param_norm > 100:  # Very large gradients
                    problematic_params.append(f"{name}:Large({param_norm:.2f})")
                    param.grad.data = param.grad.data / max(
                        param_norm / 10, 1
                    )  # Scale down

        total_norm = total_norm ** (1.0 / 2)

        if problematic_params:
            print(
                f"Epoch {self.current_epoch}, Step: Fixed gradients in: {problematic_params[:10]}..."
            )  # Limit output

        # Log gradient norm for monitoring
        if param_count > 0:
            self.log("grad_norm", total_norm, prog_bar=True)

            # If total norm is still too large, apply aggressive clipping
            if total_norm > 10.0:
                print(
                    f"Large gradient norm: {total_norm:.2f}, applying aggressive clipping"
                )
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            elif total_norm > 5.0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

    def on_after_backward(self):
        """Check gradients immediately after backward pass"""
        # Check for exploding gradients right after backward
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if grad_norm > 1000 or torch.isinf(grad_norm) or torch.isnan(grad_norm):
                    print(f"Extreme gradient in {name}: {grad_norm}")
                    param.grad.data.clamp_(-10, 10)  # Hard clamp

    def _compute_instantaneous_frequency(self, complex_signal):
        """Use shared instantaneous frequency method"""
        return compute_instantaneous_frequency_shared(complex_signal)

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


def compute_instantaneous_frequency_shared(complex_signal):
    """Shared method to compute instantaneous frequency consistently"""
    try:
        # Handle both single sample and batch inputs
        if complex_signal.dim() == 1:
            # Single sample case
            if isinstance(complex_signal, torch.Tensor):
                complex_numpy = complex_signal.detach().cpu().numpy()
            else:
                complex_numpy = complex_signal

            # Compute phase and unwrap
            phase = np.angle(complex_numpy)
            phase_unwrapped = np.unwrap(phase)

            # Compute derivative using gradient
            inst_freq = np.gradient(phase_unwrapped)

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

                # Compute phase and unwrap
                phase = np.angle(signal_sample)
                phase_unwrapped = np.unwrap(phase)

                # Compute derivative using gradient
                inst_freq = np.gradient(phase_unwrapped)

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


class EnhancedPhaseFrequencyAwareDecoder(nn.Module):
    """Enhanced decoder with significantly more capacity for complex RF signal reconstruction"""

    def __init__(self, latent_dim=256, signal_length=128):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.init_size = signal_length // 16  # Start smaller for more upsampling layers

        # Increased component dimensions for better separation
        self.magnitude_dim = latent_dim // 4  # 64 dims for magnitude
        self.phase_dim = latent_dim // 4  # 64 dims for phase
        self.frequency_dim = latent_dim // 4  # 64 dims for frequency
        self.modulation_dim = latent_dim // 4  # 64 dims for modulation features

        # Much larger separate decoders for different signal components
        self.magnitude_decoder = self._build_enhanced_component_decoder(
            self.magnitude_dim, "magnitude"
        )
        self.phase_decoder = self._build_enhanced_component_decoder(
            self.phase_dim, "phase"
        )
        self.frequency_decoder = self._build_enhanced_component_decoder(
            self.frequency_dim, "frequency"
        )
        self.modulation_decoder = self._build_enhanced_component_decoder(
            self.modulation_dim, "modulation"
        )

        # Enhanced fusion network with much more capacity
        self.fusion_network = nn.Sequential(
            # Initial feature processing
            nn.Conv1d(
                8, 128, kernel_size=7, padding=3
            ),  # 4 components * 2 channels each
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Residual blocks for better gradient flow
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 128),
            # Progressive refinement
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock1D(256, 256),
            ResidualBlock1D(256, 256),
            # Attention mechanism for feature selection
            SelfAttention1D(256),
            # Final refinement layers
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),  # Final I/Q channels
            # nn.Tanh(),
        )

        # Enhanced constellation-aware refinement
        self.constellation_refiner = EnhancedConstellationRefiner(signal_length)

        # Additional phase consistency network
        self.phase_consistency_network = PhaseConsistencyNetwork(signal_length)

    def _build_enhanced_component_decoder(self, input_dim, component_type):
        """Build enhanced decoder for specific signal component with more capacity"""
        activation = nn.Tanh() if component_type == "phase" else nn.ReLU()

        # Much larger initial mapping
        initial_channels = 128 if component_type in ["magnitude", "phase"] else 96

        return nn.Sequential(
            # Enhanced initial mapping
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, initial_channels * self.init_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Reshape for convolutions
            nn.Unflatten(1, (initial_channels, self.init_size)),
            # First upsampling block (init_size -> 2*init_size)
            nn.ConvTranspose1d(
                initial_channels,
                256,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock1D(256, 256),
            # Second upsampling block (2*init_size -> 4*init_size)
            nn.ConvTranspose1d(
                256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock1D(128, 128),
            # Third upsampling block (4*init_size -> 8*init_size)
            nn.ConvTranspose1d(
                128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock1D(64, 64),
            # Fourth upsampling block (8*init_size -> 16*init_size = signal_length)
            nn.ConvTranspose1d(
                64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Component-specific refinement
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1),  # Output I/Q for this component
            # activation,
        )

    def forward(self, latent):
        batch_size = latent.shape[0]

        # Split latent into components
        mag_latent = latent[:, : self.magnitude_dim]
        phase_latent = latent[
            :, self.magnitude_dim : self.magnitude_dim + self.phase_dim
        ]
        freq_latent = latent[
            :,
            self.magnitude_dim + self.phase_dim : self.magnitude_dim
            + self.phase_dim
            + self.frequency_dim,
        ]
        mod_latent = latent[:, -self.modulation_dim :]

        # Decode each component with enhanced decoders
        magnitude_component = self.magnitude_decoder(mag_latent)
        phase_component = self.phase_decoder(phase_latent)
        frequency_component = self.frequency_decoder(freq_latent)
        modulation_component = self.modulation_decoder(mod_latent)

        # Ensure all components have the same length
        target_length = self.signal_length
        magnitude_component = F.interpolate(
            magnitude_component, size=target_length, mode="linear", align_corners=False
        )
        phase_component = F.interpolate(
            phase_component, size=target_length, mode="linear", align_corners=False
        )
        frequency_component = F.interpolate(
            frequency_component, size=target_length, mode="linear", align_corners=False
        )
        modulation_component = F.interpolate(
            modulation_component, size=target_length, mode="linear", align_corners=False
        )

        # Concatenate all components
        combined_features = torch.cat(
            [
                magnitude_component,
                phase_component,
                frequency_component,
                modulation_component,
            ],
            dim=1,
        )  # [batch, 8, signal_length]

        # Enhanced fusion with attention
        fused_signal = self.fusion_network(combined_features)

        # Apply enhanced constellation refinement
        refined_signal = self.constellation_refiner(fused_signal)

        # Apply phase consistency
        final_signal = self.phase_consistency_network(refined_signal)

        return {
            "signal": final_signal,
            "magnitude": magnitude_component,
            "phase": phase_component,
            "frequency": frequency_component,
            "modulation": modulation_component,
            "fused": fused_signal,  # Before final refinement
            "refined": refined_signal,  # After constellation refinement
        }


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


class SelfAttention1D(nn.Module):
    """Self-attention mechanism for 1D signals"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, length = x.shape

        # Generate query, key, value
        q = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C//8]
        k = self.key(x).view(batch_size, -1, length)  # [B, C//8, L]
        v = self.value(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C]

        # Attention weights
        attention = torch.bmm(q, k)  # [B, L, L]
        attention = F.softmax(attention, dim=-1)

        # Apply attention
        out = torch.bmm(attention, v)  # [B, L, C]
        out = out.permute(0, 2, 1).view(batch_size, channels, length)

        return self.gamma * out + x


class EnhancedConstellationRefiner(nn.Module):
    """Enhanced constellation refiner with more sophisticated processing"""

    def __init__(self, signal_length):
        super().__init__()
        self.signal_length = signal_length

        # Learnable constellation templates (increased capacity)
        self.register_buffer(
            "constellation_templates", torch.randn(11, 64, 2)
        )  # More constellation points

        # Enhanced constellation selector with more capacity
        self.constellation_selector = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 11),  # 11 modulation types
            nn.Softmax(dim=1),
        )

        # Multi-scale refinement network
        self.refiner = nn.Sequential(
            # Multi-scale processing
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock1D(128, 128),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),
            # nn.Tanh(),
        )

    def forward(self, signal):
        # Select appropriate constellation
        constellation_weights = self.constellation_selector(signal)  # [batch, 11]

        # Apply enhanced refinement
        refined = self.refiner(signal)

        # Stronger residual connection with learnable weight
        output = signal + 0.2 * refined  # Increased residual strength

        return output


class PhaseConsistencyNetwork(nn.Module):
    """Network to ensure phase consistency across the signal"""

    def __init__(self, signal_length):
        super().__init__()
        self.signal_length = signal_length

        # Phase unwrapping and consistency network
        self.phase_processor = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock1D(128, 128),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),
            # nn.Tanh(),
        )

        # Learnable phase correction weight
        self.phase_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, signal):
        # Process for phase consistency
        phase_correction = self.phase_processor(signal)

        # Apply phase correction with learnable weight
        corrected_signal = signal + self.phase_weight * phase_correction

        return corrected_signal


class PhaseFrequencyAwareDecoder(nn.Module):
    """Decoder that explicitly reconstructs phase and frequency components"""

    def __init__(self, latent_dim=256, signal_length=128):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.init_size = signal_length // 8

        # Split latent space into different components
        self.magnitude_dim = latent_dim // 4  # 64 dims for magnitude
        self.phase_dim = latent_dim // 4  # 64 dims for phase
        self.frequency_dim = latent_dim // 4  # 64 dims for frequency
        self.modulation_dim = latent_dim // 4  # 64 dims for modulation features

        # Separate decoders for different signal components
        self.magnitude_decoder = self._build_component_decoder(
            self.magnitude_dim, "magnitude"
        )
        self.phase_decoder = self._build_component_decoder(self.phase_dim, "phase")
        self.frequency_decoder = self._build_component_decoder(
            self.frequency_dim, "frequency"
        )
        self.modulation_decoder = self._build_component_decoder(
            self.modulation_dim, "modulation"
        )

        # Fusion network to combine components
        self.fusion_network = nn.Sequential(
            nn.Conv1d(
                8, 64, kernel_size=7, padding=3
            ),  # 4 components * 2 channels each
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1),  # Final I/Q channels
            # nn.Tanh(),
        )

        # Constellation-aware refinement
        self.constellation_refiner = ConstellationRefiner(signal_length)

    def _build_component_decoder(self, input_dim, component_type):
        """Build decoder for specific signal component"""
        activation = nn.Tanh() if component_type == "phase" else nn.ReLU()

        return nn.Sequential(
            nn.Linear(input_dim, 32 * self.init_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Reshape and upsample
            nn.Unflatten(1, (32, self.init_size)),
            nn.ConvTranspose1d(
                32, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(
                64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, 16, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1),  # Output I/Q for this component
            # activation,
        )

    def forward(self, latent):
        batch_size = latent.shape[0]

        # Split latent into components
        mag_latent = latent[:, : self.magnitude_dim]
        phase_latent = latent[
            :, self.magnitude_dim : self.magnitude_dim + self.phase_dim
        ]
        freq_latent = latent[
            :,
            self.magnitude_dim + self.phase_dim : self.magnitude_dim
            + self.phase_dim
            + self.frequency_dim,
        ]
        mod_latent = latent[:, -self.modulation_dim :]

        # Decode each component
        magnitude_component = self.magnitude_decoder(mag_latent)
        phase_component = self.phase_decoder(phase_latent)
        frequency_component = self.frequency_decoder(freq_latent)
        modulation_component = self.modulation_decoder(mod_latent)

        # Ensure all components have the same length
        target_length = self.signal_length
        magnitude_component = F.interpolate(
            magnitude_component, size=target_length, mode="linear", align_corners=False
        )
        phase_component = F.interpolate(
            phase_component, size=target_length, mode="linear", align_corners=False
        )
        frequency_component = F.interpolate(
            frequency_component, size=target_length, mode="linear", align_corners=False
        )
        modulation_component = F.interpolate(
            modulation_component, size=target_length, mode="linear", align_corners=False
        )

        # Concatenate all components
        combined_features = torch.cat(
            [
                magnitude_component,
                phase_component,
                frequency_component,
                modulation_component,
            ],
            dim=1,
        )  # [batch, 8, signal_length]

        # Fuse components
        fused_signal = self.fusion_network(combined_features)

        # Apply constellation refinement
        refined_signal = self.constellation_refiner(fused_signal)

        return {
            "signal": refined_signal,
            "magnitude": magnitude_component,
            "phase": phase_component,
            "frequency": frequency_component,
            "modulation": modulation_component,
        }


class ConstellationRefiner(nn.Module):
    """Refines signals based on constellation diagram constraints"""

    def __init__(self, signal_length):
        super().__init__()
        self.signal_length = signal_length

        # Learnable constellation points for different modulation schemes
        # These will be updated during training to match actual constellations
        self.register_buffer(
            "constellation_templates", torch.randn(11, 16, 2)
        )  # 11 mod types, 16 points each, I/Q

        # Attention mechanism to select appropriate constellation
        self.constellation_selector = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 11),  # 11 modulation types
            nn.Softmax(dim=1),
        )

        # Refinement network
        self.refiner = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),
            # nn.Tanh(),
        )

    def forward(self, signal):
        # Select appropriate constellation
        constellation_weights = self.constellation_selector(signal)  # [batch, 11]

        # Apply refinement
        refined = self.refiner(signal)

        # Add residual connection
        output = signal + 0.1 * refined  # Small residual correction

        return output


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution for I/Q processing"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Each conv layer takes in_channels and outputs out_channels
        # in_channels should be the number of feature channels, not 2x
        self.conv_real = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.conv_imag = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )

    def forward(self, x):
        # x shape: [batch, 2*in_channels, length] where first half is I, second half is Q
        batch_size, channels, length = x.shape
        in_channels = channels // 2

        # Split I and Q channels
        i_channels = x[:, :in_channels]  # First half are I channels
        q_channels = x[:, in_channels:]  # Second half are Q channels

        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_part = self.conv_real(i_channels) - self.conv_imag(q_channels)
        imag_part = self.conv_real(q_channels) + self.conv_imag(i_channels)

        return torch.cat([real_part, imag_part], dim=1)


class EnhancedReconstructionLoss(nn.Module):
    def __init__(self, signal_length=128, normalize_fft=True):
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
        Time-Domain Phase Loss: L_Phase = MSE(phase(pred), phase(target))
        Handles phase wrapping properly
        """
        pred_i, pred_q = pred_signal[:, 0], pred_signal[:, 1]
        target_i, target_q = target_signal[:, 0], target_signal[:, 1]

        pred_complex = torch.complex(pred_i.float(), pred_q.float())
        target_complex = torch.complex(target_i.float(), target_q.float())

        # Compute phases
        pred_phase = torch.angle(pred_complex)      # [-π, π]
        target_phase = torch.angle(target_complex)  # [-π, π]

        # Handle phase wrapping: compute shortest angular distance
        phase_diff = pred_phase - target_phase

        # Wrap to [-π, π]
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

        # MSE of phase differences
        phase_mse = torch.mean(phase_diff ** 2)

        return phase_mse

    def spectral_fft_loss(self, pred_signal, target_signal):
        """Spectral FFT Loss: L_FFT = ||FFT(s)| - |FFT(ŝ)||²₂"""
        pred_i, pred_q = pred_signal[:, 0], pred_signal[:, 1]
        target_i, target_q = target_signal[:, 0], target_signal[:, 1]

        pred_complex = torch.complex(pred_i.float(), pred_q.float())
        target_complex = torch.complex(target_i.float(), target_q.float())

        # Compute FFT magnitudes
        pred_fft = torch.fft.fft(pred_complex, dim=-1)
        target_fft = torch.fft.fft(target_complex, dim=-1)

        pred_fft_mag = torch.abs(pred_fft)
        target_fft_mag = torch.abs(target_fft)

        # Optional: Normalize FFT magnitudes
        if self.normalize_fft:
            pred_fft_mag = pred_fft_mag / (torch.mean(pred_fft_mag, dim=-1, keepdim=True) + 1e-8)
            target_fft_mag = target_fft_mag / (torch.mean(target_fft_mag, dim=-1, keepdim=True) + 1e-8)

        fft_loss = torch.mean((pred_fft_mag - target_fft_mag) ** 2)
        return fft_loss

    def instantaneous_frequency_loss(self, pred_signal, target_signal):
        """Instantaneous Frequency Loss: L_IF = ||f_inst(s) - f_inst(ŝ)||²₂"""
        pred_i, pred_q = pred_signal[:, 0], pred_signal[:, 1]
        target_i, target_q = target_signal[:, 0], target_signal[:, 1]

        pred_complex = torch.complex(pred_i.float(), pred_q.float())
        target_complex = torch.complex(target_i.float(), target_q.float())

        # Compute instantaneous frequencies
        pred_inst_freq = compute_instantaneous_frequency_shared(pred_complex)
        target_inst_freq = compute_instantaneous_frequency_shared(target_complex)

        if_loss = torch.mean((pred_inst_freq - target_inst_freq) ** 2)
        return if_loss

    def forward(self, decoder_output, target_signal, labels=None):
        pred_signal = decoder_output["signal"] if isinstance(decoder_output, dict) else decoder_output

        # Four focused losses including explicit phase
        complex_mse = self.complex_mse_loss(pred_signal, target_signal)
        phase_loss = self.phase_loss(pred_signal, target_signal)          # NEW!
        fft_loss = self.spectral_fft_loss(pred_signal, target_signal)
        if_loss = self.instantaneous_frequency_loss(pred_signal, target_signal)

        # Weighted combination
        total_loss = (
            10.0 * complex_mse +      # Primary: Complex MSE
            1.0 * phase_loss +       # Explicit phase preservation
            0.1 * fft_loss +         # Spectral preservation
            0.1 * if_loss            # Modulation structure
        )

        return total_loss, {
            "complex_mse": complex_mse,
            "phase_loss": phase_loss,
            "fft_loss": fft_loss,
            "instantaneous_frequency": if_loss,
        }


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for RF signals"""

    def __init__(self, in_channels=1):
        super().__init__()
        # Different kernel sizes to capture different temporal patterns
        self.conv_blocks = nn.ModuleList(
            [
                self._make_conv_block(
                    in_channels, 16, kernel_size=3
                ),  # Reduced channel count
                self._make_conv_block(in_channels, 16, kernel_size=7),
                self._make_conv_block(in_channels, 16, kernel_size=15),
                self._make_conv_block(in_channels, 16, kernel_size=31),
            ]
        )

        # Complex processing after initial feature extraction
        # Input will be 16*4 = 64 channels for I and Q each, so 128 total
        # But ComplexConv1d expects the number of feature channels (64), not total (128)
        self.complex_conv = ComplexConv1d(
            64, 32, kernel_size=5, padding=2
        )  # 64 -> 32 per I/Q

    def _make_conv_block(self, in_ch, out_ch, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # x shape: [batch, 2, length] where dim=1 is [I, Q]
        batch_size, _, length = x.shape

        # Process I and Q channels separately first
        i_channel = x[:, 0:1]  # [batch, 1, length]
        q_channel = x[:, 1:2]  # [batch, 1, length]

        # Extract multi-scale features for each channel
        i_features = []
        q_features = []

        for block in self.conv_blocks:
            i_features.append(block(i_channel))
            q_features.append(block(q_channel))

        # Concatenate features: [I_scale1, I_scale2, ..., Q_scale1, Q_scale2, ...]
        i_concat = torch.cat(i_features, dim=1)  # [batch, 16*4=64, length]
        q_concat = torch.cat(q_features, dim=1)  # [batch, 16*4=64, length]

        # Combine I and Q for complex processing
        combined = torch.cat([i_concat, q_concat], dim=1)  # [batch, 64*2=128, length]

        # Apply complex convolution
        complex_features = self.complex_conv(combined)  # [batch, 32*2=64, length]

        return complex_features


class AttentionBlock(nn.Module):
    """Self-attention for important feature selection"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv1d(
            channels, max(1, channels // 8), 1
        )  # Ensure at least 1 channel
        self.key = nn.Conv1d(channels, max(1, channels // 8), 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, length = x.shape

        # Generate query, key, value
        q = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C//8]
        k = self.key(x).view(batch_size, -1, length)  # [B, C//8, L]
        v = self.value(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C]

        # Attention weights
        attention = torch.bmm(q, k)  # [B, L, L]
        attention = F.softmax(attention, dim=-1)

        # Apply attention
        out = torch.bmm(attention, v)  # [B, L, C]
        out = out.permute(0, 2, 1).view(batch_size, channels, length)

        return self.gamma * out + x


class FrequencyDomainProcessor(nn.Module):
    """Process frequency domain features"""

    def __init__(self, signal_length=128):
        super().__init__()
        self.signal_length = signal_length
        # Process magnitude and phase separately
        self.mag_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.phase_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, x):
        # x shape: [batch, 2, length] - I/Q channels
        batch_size = x.shape[0]

        # Convert to float32 for complex operations
        i_channel = x[:, 0].float()
        q_channel = x[:, 1].float()

        # Convert to complex tensor
        complex_signal = torch.complex(i_channel, q_channel)  # [batch, length]

        # FFT
        fft_signal = torch.fft.fft(complex_signal, dim=-1)

        # Extract magnitude and phase
        magnitude = torch.abs(fft_signal).unsqueeze(1)  # [batch, 1, length]
        phase = torch.angle(fft_signal).unsqueeze(1)  # [batch, 1, length]

        # Convert back to original dtype
        magnitude = magnitude.to(x.dtype)
        phase = phase.to(x.dtype)

        # Process magnitude and phase
        mag_features = self.mag_processor(magnitude)  # [batch, 32, length]
        phase_features = self.phase_processor(phase)  # [batch, 32, length]

        return torch.cat([mag_features, phase_features], dim=1)  # [batch, 64, length]


class SNRAwareEncoder(nn.Module):
    """Encoder that separates signal features from SNR effects"""

    def __init__(self, signal_length=128, latent_dim=256):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim

        # Multi-scale time domain features
        self.time_features = MultiScaleFeatureExtractor(in_channels=1)

        # Frequency domain features
        self.freq_features = FrequencyDomainProcessor(signal_length)

        # Combine features (64 from time + 64 from freq = 128 channels)
        combined_channels = 64 + 64

        # Attention mechanism
        self.attention = AttentionBlock(combined_channels)

        # Encoder layers with residual connections
        self.encoder_layers = nn.Sequential(
            nn.Conv1d(combined_channels, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Calculate the size after convolutions
        self.encoded_size = self._get_encoded_size()

        # Split latent space
        self.signal_features_dim = latent_dim // 2  # 128 dims for modulation features
        self.noise_features_dim = latent_dim // 2  # 128 dims for noise/SNR features

        # Final latent representation
        self.to_latent = nn.Sequential(
            nn.Linear(self.encoded_size, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # SNR prediction head
        self.snr_predictor = nn.Sequential(
            nn.Linear(self.noise_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Predict SNR value
        )

    def _get_encoded_size(self):
        """Calculate the size after convolutions"""
        with torch.no_grad():
            x = torch.randn(1, 2, self.signal_length)
            time_feat = self.time_features(x)
            freq_feat = self.freq_features(x)
            combined = torch.cat([time_feat, freq_feat], dim=1)
            attended = self.attention(combined)
            encoded = self.encoder_layers(attended)
            return encoded.numel()

    def forward(self, x):
        # Extract time and frequency domain features
        time_features = self.time_features(x)
        freq_features = self.freq_features(x)

        # Combine features
        combined_features = torch.cat([time_features, freq_features], dim=1)

        # Apply attention
        attended_features = self.attention(combined_features)

        # Encode
        encoded = self.encoder_layers(attended_features)

        # Flatten and get latent representation
        encoded_flat = encoded.view(encoded.shape[0], -1)
        full_embedding = self.to_latent(encoded_flat)

        # Split into signal and noise features
        signal_features = full_embedding[:, : self.signal_features_dim]
        noise_features = full_embedding[:, self.signal_features_dim :]

        # Predict SNR from noise features
        predicted_snr = self.snr_predictor(noise_features)

        return {
            "full_embedding": full_embedding,
            "signal_features": signal_features,
            "noise_features": noise_features,
            "predicted_snr": predicted_snr,
        }


class RFEncoder(nn.Module):
    """Original encoder for backward compatibility"""

    def __init__(self, signal_length=128, latent_dim=256):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim

        # Multi-scale time domain features
        self.time_features = MultiScaleFeatureExtractor(in_channels=1)

        # Frequency domain features
        self.freq_features = FrequencyDomainProcessor(signal_length)

        # Combine features (64 from time + 64 from freq = 128 channels)
        combined_channels = 64 + 64  # Updated based on actual output sizes

        # Attention mechanism
        self.attention = AttentionBlock(combined_channels)

        # Encoder layers with residual connections
        self.encoder_layers = nn.Sequential(
            nn.Conv1d(combined_channels, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Calculate the size after convolutions
        self.encoded_size = self._get_encoded_size()

        # Final latent representation
        self.to_latent = nn.Sequential(
            nn.Linear(self.encoded_size, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def _get_encoded_size(self):
        """Calculate the size after convolutions"""
        with torch.no_grad():
            x = torch.randn(1, 2, self.signal_length)
            time_feat = self.time_features(x)
            freq_feat = self.freq_features(x)
            combined = torch.cat([time_feat, freq_feat], dim=1)
            attended = self.attention(combined)
            encoded = self.encoder_layers(attended)
            return encoded.numel()

    def forward(self, x):
        # Extract time and frequency domain features
        time_features = self.time_features(x)
        freq_features = self.freq_features(x)

        # Combine features
        combined_features = torch.cat([time_features, freq_features], dim=1)

        # Apply attention
        attended_features = self.attention(combined_features)

        # Encode
        encoded = self.encoder_layers(attended_features)

        # Flatten and get latent representation
        encoded_flat = encoded.view(encoded.shape[0], -1)
        latent = self.to_latent(encoded_flat)
        return latent


class RFDecoderEnhanced(nn.Module):
    """Enhanced decoder with more capacity for noisy signals"""

    def __init__(self, latent_dim=256, signal_length=128):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim

        # Calculate initial size for decoder
        self.init_size = signal_length // 8  # After 3 upsampling layers

        # Moderate initial mapping (much smaller than 40M version)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size),  # Reasonable increase from 64
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Balanced decoder layers
        self.decoder_layers = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose1d(
                128, 256, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # Second upsampling block
            nn.ConvTranspose1d(
                256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # Third upsampling block
            nn.ConvTranspose1d(
                128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Additional refinement layer (this is the key improvement)
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Final layers
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Conv1d(32, 2, kernel_size=5, padding=2),
            # nn.Tanh(),
        )

    def forward(self, latent):
        # From latent to feature map
        x = self.from_latent(latent)
        x = x.view(x.shape[0], 128, self.init_size)

        # Decode
        reconstructed = self.decoder_layers(x)

        # Ensure correct output size
        if reconstructed.shape[-1] != self.signal_length:
            reconstructed = F.interpolate(
                reconstructed,
                size=self.signal_length,
                mode="linear",
                align_corners=False,
            )

        return reconstructed


class SNRAwareArcFaceLoss(nn.Module):
    """ArcFace loss that focuses on signal features, not noise"""

    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Weight matrix for signal features only
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, signal_embeddings, labels):
        # Use only signal features for classification, ignore noise features
        embeddings = signal_embeddings.float()
        device = embeddings.device

        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight.float(), p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)

        # Get target cosine values
        target_cosine = cosine[torch.arange(len(labels), device=device), labels]

        # Add margin to target
        target_theta = torch.acos(torch.clamp(target_cosine, -1 + 1e-7, 1 - 1e-7))
        target_theta_margin = target_theta + self.margin
        target_cosine_margin = torch.cos(target_theta_margin)

        # Replace target values with margin-adjusted values
        logits = cosine.clone()
        logits[torch.arange(len(labels), device=device), labels] = target_cosine_margin

        # Scale logits
        logits *= self.scale

        return F.cross_entropy(logits, labels)


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

        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight.float(), p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)

        # Get target cosine values
        target_cosine = cosine[torch.arange(len(labels), device=device), labels]

        # Get adaptive margins
        adaptive_margins = self.get_adaptive_margin(snrs.squeeze())

        # Add adaptive margin to target
        target_theta = torch.acos(torch.clamp(target_cosine, -1 + 1e-7, 1 - 1e-7))
        target_theta_margin = target_theta + adaptive_margins
        target_cosine_margin = torch.cos(target_theta_margin)

        # Replace target values with margin-adjusted values
        logits = cosine.clone()
        logits[torch.arange(len(labels), device=device), labels] = target_cosine_margin

        # Scale logits
        logits *= self.scale

        return F.cross_entropy(logits, labels)


class ArcFaceLoss(nn.Module):
    """Original ArcFace loss for backward compatibility"""

    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Force float32 computation for numerical stability
        original_dtype = embeddings.dtype
        embeddings = embeddings.float()
        device = embeddings.device

        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight.float(), p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)

        # Get target cosine values
        target_cosine = cosine[torch.arange(len(labels), device=device), labels]

        # Add margin to target
        target_theta = torch.acos(torch.clamp(target_cosine, -1 + 1e-7, 1 - 1e-7))
        target_theta_margin = target_theta + self.margin
        target_cosine_margin = torch.cos(target_theta_margin)

        # Replace target values with margin-adjusted values
        logits = cosine.clone()
        logits[torch.arange(len(labels), device=device), labels] = target_cosine_margin

        # Scale logits
        logits *= self.scale

        # Convert back to original dtype if needed
        if original_dtype != torch.float32:
            logits = logits.to(dtype=original_dtype)

        return F.cross_entropy(logits, labels)


class RFEncoderDecoder(L.LightningModule):
    """Enhanced Phase and frequency aware encoder-decoder"""

    def __init__(
        self,
        label_names,
        signal_length=128,
        latent_dim=256,
        learning_rate=1e-3,
        arcface_margin=0.4,
        arcface_scale=32,
        reconstruction_weight=1.0,
        arcface_weight=0.0,
        curriculum_learning=True,
        initial_snr_threshold=16,
        final_snr_threshold=-20,
        curriculum_epochs=75,
        use_snr_aware=False,
        use_enhanced_decoder=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.label_names = label_names
        self.num_classes = len(label_names)
        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.arcface_weight = arcface_weight
        self.use_snr_aware = use_snr_aware
        self.use_enhanced_decoder = use_enhanced_decoder

        # Curriculum learning parameters
        self.curriculum_learning = curriculum_learning
        self.initial_snr_threshold = initial_snr_threshold
        self.final_snr_threshold = final_snr_threshold
        self.curriculum_epochs = curriculum_epochs

        # Networks - choose encoder type
        if use_snr_aware:
            self.encoder = SNRAwareEncoder(signal_length, latent_dim)
        else:
            self.encoder = RFEncoder(signal_length, latent_dim)

        # Choose decoder type
        self.decoder = RFDecoderEnhanced(latent_dim, signal_length)  # Single decoder

        # Loss functions
        if use_snr_aware:
            signal_dim = latent_dim // 2
            self.arcface_loss = SNRAwareArcFaceLoss(
                signal_dim, self.num_classes, arcface_margin, arcface_scale
            )
        else:
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
        x, labels, snrs = batch
        batch_size = x.shape[0]

        # Get current SNR threshold for curriculum learning
        current_threshold = self.get_current_snr_threshold()

        # Create curriculum mask
        curriculum_mask = self.create_curriculum_mask(snrs.squeeze(), current_threshold)

        # Check if any samples in batch meet the curriculum criteria
        if not curriculum_mask.any():
            # Skip this batch if no samples meet criteria
            self.log("curriculum_snr_threshold", current_threshold, prog_bar=True)
            self.log("curriculum_batch_skipped", 1.0)
            self.log("curriculum_samples_used", 0.0)
            return None

        # Filter batch to only include curriculum samples
        x_curriculum = x[curriculum_mask]
        labels_curriculum = labels[curriculum_mask]
        snrs_curriculum = snrs[curriculum_mask]

        # Forward pass
        encoder_output, decoder_output = self.forward(x_curriculum)

        # Simple reconstruction loss (no component losses)
        recon_loss, loss_components = self.reconstruction_loss(
            decoder_output, x_curriculum, labels_curriculum
        )

        # ArcFace loss
        if self.use_snr_aware and isinstance(encoder_output, dict):
            arcface_loss = self.arcface_loss(
                encoder_output["signal_features"], labels_curriculum.squeeze()
            )
        else:
            arcface_loss = self.arcface_loss(
                encoder_output, labels_curriculum.squeeze(), snrs_curriculum
            )

        # Combined loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.arcface_weight * arcface_loss
        )

        # Clean logging
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_arcface_loss", arcface_loss)

        for key, value in loss_components.items():
            self.log(f"train_{key}", value)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, labels, snrs = batch

        # Forward pass
        encoder_output, decoder_output = self.forward(x)

        # Handle different encoder types for ArcFace
        if self.use_snr_aware and isinstance(encoder_output, dict):
            latent_for_arcface = encoder_output["signal_features"]
            full_latent = encoder_output["full_embedding"]

            # SNR prediction loss
            snr_loss = F.mse_loss(
                encoder_output["predicted_snr"].squeeze(), snrs.float()
            )

            # ArcFace loss on signal features only
            arcface_loss = self.arcface_loss(latent_for_arcface, labels.squeeze())

        else:
            latent_for_arcface = encoder_output
            snr_loss = 0
            # ArcFace loss with SNR adaptation
            arcface_loss = self.arcface_loss(latent_for_arcface, labels.squeeze(), snrs)

        # Compute focused reconstruction loss
        recon_loss, loss_components = self.reconstruction_loss(
            decoder_output, x, labels
        )

        # Combined loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.arcface_weight * arcface_loss
        )

        if self.use_snr_aware and isinstance(encoder_output, dict):
            total_loss += 0.1 * snr_loss

        # Logging - clean and focused
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_arcface_loss", arcface_loss)

        if self.use_snr_aware and isinstance(encoder_output, dict):
            self.log("val_snr_loss", snr_loss)

        # Log the three core reconstruction loss components
        for key, value in loss_components.items():
            self.log(f"val_{key}", value)

        # Save for visualization (first batch only)
        if batch_idx == 0:
            # Save embeddings for visualization
            if self.use_snr_aware and isinstance(encoder_output, dict):
                self.val_latents = latent_for_arcface.detach().cpu()  # Use signal features
                self.val_encoder_output = {
                    k: v.detach().cpu() for k, v in encoder_output.items()
                }
            else:
                self.val_latents = latent_for_arcface.detach().cpu()

            # Save basic data for visualization
            self.val_labels = labels.detach().cpu()
            self.val_snrs = snrs.detach().cpu()
            self.val_original = x.detach().cpu()

            # Handle decoder output
            if isinstance(decoder_output, dict):
                self.val_reconstructed = decoder_output["signal"].detach().cpu()
            else:
                self.val_reconstructed = decoder_output.detach().cpu()

        return total_loss

    def on_train_epoch_end(self):
        """Log curriculum progress at end of each epoch"""
        current_threshold = self.get_current_snr_threshold()
        progress = min(self.current_epoch / self.curriculum_epochs, 1.0) * 100

        # Calculate approximate data coverage
        total_snr_range = 18 - (-20)  # 38 dB range
        included_range = 18 - current_threshold
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
            axes[0, 0].plot(time_axis, orig_i.numpy(), "b-", label="Original I", alpha=0.8)
            axes[0, 0].plot(time_axis, recon_i.numpy(), "r--", label="Reconstructed I", alpha=0.8)
            axes[0, 0].plot(time_axis, orig_q.numpy(), "c-", label="Original Q", alpha=0.8)
            axes[0, 0].plot(time_axis, recon_q.numpy(), "m--", label="Reconstructed Q", alpha=0.8)

            complex_mse = torch.mean(torch.abs(orig_complex - recon_complex) ** 2).item()
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

            axes[0, 1].plot(time_axis, orig_phase_unwrapped, "b-", label="Original Phase", alpha=0.8)
            axes[0, 1].plot(time_axis, recon_phase_unwrapped, "r--", label="Reconstructed Phase", alpha=0.8)

            # Calculate phase loss
            phase_diff = torch.atan2(torch.sin(torch.angle(orig_complex) - torch.angle(recon_complex)),
                                    torch.cos(torch.angle(orig_complex) - torch.angle(recon_complex)))
            phase_mse = torch.mean(phase_diff ** 2).item()

            axes[0, 1].set_title(f"Phase Comparison\nPhase MSE: {phase_mse:.6f} rad²")
            axes[0, 1].set_xlabel("Time Sample")
            axes[0, 1].set_ylabel("Phase (radians)")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Phase Error (NEW!)
            phase_error = phase_diff.numpy()
            axes[0, 2].plot(time_axis, phase_error, "r-", alpha=0.8)
            axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[0, 2].set_title(f"Phase Error\nMean: {np.mean(phase_error):.4f}, Std: {np.std(phase_error):.4f}")
            axes[0, 2].set_xlabel("Time Sample")
            axes[0, 2].set_ylabel("Phase Error (radians)")
            axes[0, 2].grid(True, alpha=0.3)

            # 4. Constellation Diagram
            axes[1, 0].scatter(orig_complex.real.numpy(), orig_complex.imag.numpy(),
                            alpha=0.6, s=20, c="blue", label="Original")
            axes[1, 0].scatter(recon_complex.real.numpy(), recon_complex.imag.numpy(),
                            alpha=0.6, s=20, c="red", label="Reconstructed")
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

            fft_loss = torch.mean((torch.abs(orig_fft) - torch.abs(recon_fft)) ** 2).item()

            axes[1, 1].plot(freqs, torch.abs(orig_fft).numpy(), "b-", label="Original", alpha=0.8)
            axes[1, 1].plot(freqs, torch.abs(recon_fft).numpy(), "r--", label="Reconstructed", alpha=0.8)
            axes[1, 1].set_title(f"FFT Magnitude Spectrum\nFFT Loss: {fft_loss:.6f}")
            axes[1, 1].set_xlabel("Frequency Bin")
            axes[1, 1].set_ylabel("FFT Magnitude")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # 6. FFT Phase Spectrum (NEW!)
            orig_fft_phase = torch.angle(orig_fft).numpy()
            recon_fft_phase = torch.angle(recon_fft).numpy()

            axes[1, 2].plot(freqs, orig_fft_phase, "b-", label="Original", alpha=0.8)
            axes[1, 2].plot(freqs, recon_fft_phase, "r--", label="Reconstructed", alpha=0.8)
            axes[1, 2].set_title("FFT Phase Spectrum")
            axes[1, 2].set_xlabel("Frequency Bin")
            axes[1, 2].set_ylabel("Phase (radians)")
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

            # 7. Instantaneous Frequency
            orig_inst_freq = self._compute_instantaneous_frequency(orig_complex)
            recon_inst_freq = self._compute_instantaneous_frequency(recon_complex)

            if_loss = torch.mean((orig_inst_freq - recon_inst_freq) ** 2).item()

            axes[2, 0].plot(time_axis, orig_inst_freq.numpy(), "b-", label="Original", alpha=0.8)
            axes[2, 0].plot(time_axis, recon_inst_freq.numpy(), "r--", label="Reconstructed", alpha=0.8)
            axes[2, 0].set_title(f"Instantaneous Frequency\nIF Loss: {if_loss:.6f}")
            axes[2, 0].set_xlabel("Time Sample")
            axes[2, 0].set_ylabel("Frequency (rad/sample)")
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

            # 8. Power Spectral Density
            orig_psd = torch.abs(orig_fft) ** 2
            recon_psd = torch.abs(recon_fft) ** 2

            axes[2, 1].plot(freqs, 10 * torch.log10(orig_psd + 1e-10).numpy(),
                        "b-", label="Original", alpha=0.8)
            axes[2, 1].plot(freqs, 10 * torch.log10(recon_psd + 1e-10).numpy(),
                        "r--", label="Reconstructed", alpha=0.8)
            axes[2, 1].set_title("Power Spectral Density")
            axes[2, 1].set_xlabel("Frequency Bin")
            axes[2, 1].set_ylabel("Power (dB)")
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)

            # 9. Loss Component Breakdown
            with torch.no_grad():
                sample_orig = self.val_original[sample_idx:sample_idx+1]
                sample_recon = self.val_reconstructed[sample_idx:sample_idx+1]

                loss_fn = EnhancedReconstructionLoss(self.hparams.signal_length)
                _, loss_breakdown = loss_fn(sample_recon, sample_orig)

                loss_names = ['Complex MSE', 'Phase Loss', 'FFT Loss', 'Inst. Freq']  # Updated
                loss_values = [
                    loss_breakdown['complex_mse'].item(),
                    loss_breakdown['phase_loss'].item(),      # NEW!
                    loss_breakdown['fft_loss'].item(),
                    loss_breakdown['instantaneous_frequency'].item()
                ]

                bars = axes[2, 2].bar(loss_names, loss_values,
                                    color=['blue', 'red', 'orange', 'green'], alpha=0.7)
                axes[2, 2].set_title("Loss Component Breakdown")
                axes[2, 2].set_ylabel("Loss Value")
                axes[2, 2].tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for bar, value in zip(bars, loss_values):
                    height = bar.get_height()
                    axes[2, 2].text(bar.get_x() + bar.get_width()/2., height,
                                f'{value:.4f}', ha='center', va='bottom')

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
            unique_labels = torch.unique(self.val_labels).numpy()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

            # 1. t-SNE visualization
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(embeddings_norm) // 4),
            )
            embeddings_2d = tsne.fit_transform(embeddings_norm)

            for i, label in enumerate(unique_labels):
                mask = self.val_labels.squeeze().numpy() == label
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
            embeddings_2d_circle = embeddings_2d_circle / np.linalg.norm(
                embeddings_2d_circle, axis=1, keepdims=True
            )

            for i, label in enumerate(unique_labels):
                mask = self.val_labels.squeeze().numpy() == label
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
                mask = self.val_labels.squeeze().numpy() == label
                if mask.sum() > 0:
                    centroid_i = embeddings_norm[mask].mean(axis=0)
                    centroid_i = centroid_i / np.linalg.norm(centroid_i)
                    class_names.append(self.label_names[int(label)])

                    for j, label_j in enumerate(unique_labels):
                        mask_j = self.val_labels.squeeze().numpy() == label_j
                        if mask_j.sum() > 0:
                            centroid_j = embeddings_norm[mask_j].mean(axis=0)
                            centroid_j = centroid_j / np.linalg.norm(centroid_j)

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
        """
        Cosine Annealing with Warm Restarts - periodically resets to high LR
        """
        # Main optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,  # This becomes the max LR
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Cosine Annealing with Warm Restarts
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,        # Initial restart period (epochs)
                T_mult=2,      # Multiply restart period by this after each restart
                eta_min=1e-6,  # Minimum learning rate
                last_epoch=-1
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
            "name": "cosine_warm_restarts"
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _compute_instantaneous_frequency(self, complex_signal):
        """Use shared instantaneous frequency method"""
        return compute_instantaneous_frequency_shared(complex_signal)

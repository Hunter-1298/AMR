import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Optional, Dict, Any, Tuple
from .unet_1d import UNet1DModel
from ..latent_encoder_models import ResNet1D, Decoder1D
from ..classifier.rfnet import RFNet
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import io
from PIL import Image
import wandb

class ArcFaceCenters(nn.Module):
    """
    Class to extract, manage, and compute with ArcFace class centers
    """

    def __init__(self, encoder_decoder, num_classes: int, label_names: list):
        super().__init__()
        self.num_classes = num_classes
        self.label_names = label_names
        self.encoder_decoder = encoder_decoder

        # Extract and register ArcFace centers
        self._extract_centers()

    def __getitem__(self, index):
        """Allow indexing like arcface_centers[class_labels]"""
        return self.centers[index]

    def __len__(self):
        """Return number of classes"""
        return self.num_classes

    def _extract_centers(self):
        """Extract ArcFace class centers from pre-trained encoder-decoder"""
        try:
            # Try multiple possible locations for ArcFace loss
            arcface_loss = None

            # Location 1: Direct attribute
            if hasattr(self.encoder_decoder, 'arcface_loss'):
                arcface_loss = self.encoder_decoder.arcface_loss
                print("Found ArcFace loss at encoder_decoder.arcface_loss")

            # Location 2: Nested in model attribute
            elif hasattr(self.encoder_decoder, 'model') and hasattr(self.encoder_decoder.model, 'arcface_loss'):
                arcface_loss = self.encoder_decoder.model.arcface_loss
                print("Found ArcFace loss at encoder_decoder.model.arcface_loss")

            # Location 3: Look through named modules
            else:
                for name, module in self.encoder_decoder.named_modules():
                    if 'arcface' in name.lower() and hasattr(module, 'weight'):
                        arcface_loss = module
                        print(f"Found ArcFace loss at {name}")
                        break

            if arcface_loss is not None and hasattr(arcface_loss, 'weight'):
                # Extract the weight matrix (these are the class centers)
                centers = arcface_loss.weight.data.clone()  # [num_classes, embedding_dim]

                # Normalize the centers (ArcFace uses normalized weights)
                centers = F.normalize(centers, p=2, dim=1)

                # Register as buffer (not a parameter, but part of model state)
                self.register_buffer("centers", centers)

                print(f"✅ Successfully extracted ArcFace centers: {centers.shape}")
                print(f"   - Number of classes: {centers.shape[0]}")
                print(f"   - Embedding dimension: {centers.shape[1]}")

                # Verify we have the right number of classes
                if centers.shape[0] != self.num_classes:
                    print(f"⚠️  Warning: Expected {self.num_classes} classes, got {centers.shape[0]}")

            else:
                raise AttributeError("Could not find ArcFace loss with weight attribute")

        except Exception as e:
            print(f"❌ Error extracting ArcFace centers: {e}")
            self._create_fallback_centers()

    def _create_fallback_centers(self):
        """Create random normalized centers as fallback"""
        print("Creating fallback random centers...")

        # Try to determine embedding dimension
        embedding_dim = self._get_embedding_dimension()

        # Create random normalized centers
        centers = torch.randn(self.num_classes, embedding_dim)
        centers = F.normalize(centers, p=2, dim=1)

        self.register_buffer("centers", centers)
        print(f"✅ Created fallback centers: {centers.shape}")

    def _get_embedding_dimension(self):
        """Try to determine the embedding dimension from the encoder"""
        try:
            # Try to get from encoder hyperparameters
            if hasattr(self.encoder_decoder, 'encoder'):
                encoder = self.encoder_decoder.encoder

                if hasattr(encoder, 'hparams') and hasattr(encoder.hparams, 'latent_dim'):
                    return encoder.hparams.latent_dim
                elif hasattr(encoder, 'latent_dim'):
                    return encoder.latent_dim

            # Try to get from encoder_decoder hyperparameters
            if hasattr(self.encoder_decoder, 'hparams') and hasattr(self.encoder_decoder.hparams, 'latent_dim'):
                return self.encoder_decoder.hparams.latent_dim
            elif hasattr(self.encoder_decoder, 'latent_dim'):
                return self.encoder_decoder.latent_dim

            # Try to infer from a forward pass
            dummy_input = torch.randn(1, 2, 128)  # Assuming I/Q input
            with torch.no_grad():
                if hasattr(self.encoder_decoder, 'encode'):
                    dummy_output = self.encoder_decoder.encode(dummy_input)
                elif hasattr(self.encoder_decoder, 'encoder'):
                    dummy_output = self.encoder_decoder.encoder(dummy_input)
                else:
                    dummy_output = self.encoder_decoder(dummy_input)

                if isinstance(dummy_output, torch.Tensor):
                    # If 3D tensor [batch, channels, length], use channels
                    if dummy_output.dim() == 3:
                        return dummy_output.shape[1]
                    # If 2D tensor [batch, features], use features
                    elif dummy_output.dim() == 2:
                        return dummy_output.shape[1]

        except Exception as e:
            print(f"Could not determine embedding dimension: {e}")

        # Default fallback
        return 256

    def get_centers(self):
        """Get all ArcFace centers"""
        return self.centers

    def get_center_for_class(self, class_id: int):
        """Get center for specific class"""
        return self.centers[class_id]

    def get_centers_for_batch(self, class_labels: torch.Tensor):
        """Get centers for a batch of class labels"""
        return self.centers[class_labels]

    def compute_alignment_loss(self, embeddings: torch.Tensor, class_labels: torch.Tensor):
        """
        Compute alignment loss between embeddings and their class centers

        Args:
            embeddings: Embeddings to align [batch, embedding_dim] or [batch, channels, length]
            class_labels: Class labels [batch]

        Returns:
            alignment_loss: Cosine similarity based loss
        """
        # Handle different embedding shapes
        if embeddings.dim() == 3:
            # Pool spatial dimensions if needed [batch, channels, length] -> [batch, channels]
            embeddings = F.adaptive_avg_pool1d(embeddings, 1).squeeze(-1)

        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # Get target centers for this batch
        target_centers = self.get_centers_for_batch(class_labels)  # [batch, embedding_dim]
        target_centers_norm = F.normalize(target_centers, p=2, dim=1)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(embeddings_norm, target_centers_norm, dim=1)

        # Convert to loss (maximize similarity = minimize negative similarity)
        alignment_loss = -cosine_sim.mean()

        return alignment_loss

    def compute_inter_class_distances(self):
        """Compute pairwise distances between class centers"""
        centers_norm = F.normalize(self.centers, p=2, dim=1)

        # Compute pairwise cosine similarities
        similarities = torch.mm(centers_norm, centers_norm.t())

        # Convert to angular distances (in degrees)
        cosine_similarities = torch.clamp(similarities, -1, 1)
        angular_distances = torch.acos(cosine_similarities) * 180 / torch.pi

        return angular_distances

    def visualize_centers(self, logger=None):
        """Visualize ArcFace centers using t-SNE"""
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import numpy as np

            # Get centers as numpy
            centers_np = self.centers.detach().cpu().numpy()

            # Apply t-SNE for 2D visualization
            perplexity = min(30, self.num_classes - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            centers_2d = tsne.fit_transform(centers_np)

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: t-SNE visualization
            colors = plt.cm.tab20(np.linspace(0, 1, self.num_classes))

            for i in range(self.num_classes):
                ax1.scatter(centers_2d[i, 0], centers_2d[i, 1],
                           c=[colors[i]], s=100, label=self.label_names[i])

            ax1.set_title('ArcFace Class Centers (t-SNE)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Distance matrix
            distances = self.compute_inter_class_distances().cpu().numpy()

            im = ax2.imshow(distances, cmap='viridis')
            ax2.set_title('Inter-Class Angular Distances (degrees)')
            ax2.set_xticks(range(self.num_classes))
            ax2.set_yticks(range(self.num_classes))
            ax2.set_xticklabels(self.label_names, rotation=45, ha='right')
            ax2.set_yticklabels(self.label_names)

            # Add colorbar
            plt.colorbar(im, ax=ax2)

            # Add text annotations for distances
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:  # Skip diagonal
                        text_color = 'white' if distances[i, j] > distances.max()/2 else 'black'
                        ax2.text(j, i, f'{distances[i, j]:.1f}°',
                                ha='center', va='center', color=text_color, fontsize=8)

            plt.tight_layout()

            # Log to wandb if logger provided
            if logger:
                logger.experiment.log({"arcface_centers_analysis": wandb.Image(fig)})

            plt.show()
            plt.close()

        except Exception as e:
            print(f"Error visualizing ArcFace centers: {e}")

    def get_embedding_dim(self):
        """Get the embedding dimension"""
        return self.centers.shape[1]

    def to(self, device):
        """Override to method to ensure centers are moved to correct device"""
        super().to(device)
        return self

    def __repr__(self):
        return f"ArcFaceCenters(num_classes={self.num_classes}, embedding_dim={self.get_embedding_dim()})"

class AWGNScheduler(nn.Module):
    """AWGN noise scheduler that adds noise to existing noisy signals"""

    def __init__(self, n_steps: int = 1000, snr_range: Tuple[float, float] = (-20.0, 20.0)):
        super().__init__()
        self.n_steps = n_steps
        self.snr_min, self.snr_max = snr_range

    def add_awgn_noise(self, signal: torch.Tensor, current_snr_db: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add AWGN noise to degrade signal from current SNR to target SNR

        Args:
            signal: RF signal [batch, 2, length] - already has some noise
            current_snr_db: Current SNR of the signal [batch]
            t: Timestep [batch] - determines target SNR

        Returns:
            noisy_signal: Signal degraded to target SNR
            noise: Additional noise that was added
        """
        # Convert timestep to target SNR
        target_snr_db = self.timestep_to_snr(t)

        # Calculate signal power (includes existing signal + existing noise)
        total_power = torch.mean(signal ** 2, dim=(1, 2), keepdim=True)  # [batch, 1, 1]

        # Convert SNRs to linear scale
        current_snr_linear = 10 ** (current_snr_db / 10)  # [batch]
        target_snr_linear = 10 ** (target_snr_db / 10)   # [batch]

        # Calculate clean signal power from current noisy signal
        # current_snr = signal_power / current_noise_power
        # total_power = signal_power + current_noise_power
        # Solving: signal_power = total_power * current_snr / (1 + current_snr)
        signal_power = total_power * current_snr_linear.view(-1, 1, 1) / (1 + current_snr_linear.view(-1, 1, 1))

        # Calculate current noise power
        current_noise_power = total_power - signal_power

        # Calculate required total noise power for target SNR
        target_total_noise_power = signal_power / target_snr_linear.view(-1, 1, 1)

        # Additional noise power needed
        additional_noise_power = target_total_noise_power - current_noise_power

        # Only add noise if target SNR is lower than current SNR
        additional_noise_power = torch.clamp(additional_noise_power, min=0)

        # Generate additional AWGN noise
        eps = 1e-12
        additional_noise = torch.randn_like(signal) * torch.sqrt(additional_noise_power + eps)

        # Add additional noise to signal
        noisy_signal = signal + additional_noise

        return noisy_signal, additional_noise

    def snr_to_timestep(self, snr_db: torch.Tensor) -> torch.Tensor:
        """Convert SNR (dB) to timestep"""
        normalized_snr = (snr_db - self.snr_min) / (self.snr_max - self.snr_min)
        normalized_snr = torch.clamp(normalized_snr, 0, 1)
        timestep = (1 - normalized_snr) * (self.n_steps - 1)
        return timestep.long()

    def timestep_to_snr(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert timestep to SNR (dB)"""
        normalized_t = timestep.float() / (self.n_steps - 1)
        snr_db = self.snr_max - normalized_t * (self.snr_max - self.snr_min)
        return snr_db


# class ConstellationPrototypeLoss(nn.Module):
#     """
#     Constellation prototype loss that matches modulation-specific characteristics
#     instead of exact signal-to-signal constellation matching
#     """

#     def __init__(self, label_names, temperature=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.label_names = label_names

#     def forward(self, x_denoised: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
#         """
#         Compute constellation prototype loss for each signal based on its modulation class

#         Args:
#             x_denoised: Denoised signal [batch, 2, length] (I/Q)
#             class_labels: Class labels [batch]

#         Returns:
#             loss: Average prototype loss across batch
#         """
#         batch_size = x_denoised.shape[0]
#         losses = []

#         for batch_idx in range(batch_size):
#             signal = x_denoised[batch_idx:batch_idx+1]  # [1, 2, length]
#             class_id = class_labels[batch_idx].item()
#             modulation = self.label_names[class_id]

#             # Route to appropriate prototype loss based on modulation type
#             if 'PSK' in modulation.upper():
#                 loss = self.psk_prototype_loss(signal)
#             elif 'QAM' in modulation.upper():
#                 loss = self.qam_prototype_loss(signal)
#             elif 'FSK' in modulation.upper():
#                 loss = self.fsk_prototype_loss(signal)
#             elif 'AM' in modulation.upper():
#                 loss = self.am_prototype_loss(signal)
#             else:
#                 # Generic modulation loss for unknown types
#                 loss = self.generic_modulation_loss(signal)

#             losses.append(loss)

#         return torch.stack(losses).mean()

#     def extract_constellation(self, signal: torch.Tensor) -> torch.Tensor:
#         """Extract complex constellation from I/Q signal"""
#         i_channel = signal[:, 0]  # [batch, length]
#         q_channel = signal[:, 1]  # [batch, length]
#         return torch.complex(i_channel, q_channel)

#     def psk_prototype_loss(self, signal: torch.Tensor) -> torch.Tensor:
#         """
#         PSK prototype: uniform amplitude, clustered phases
#         Information encoded in phase only
#         """
#         const = self.extract_constellation(signal)  # [1, length]
#         const = const.squeeze(0)  # [length]

#         # 1. Amplitude should be relatively uniform (constant envelope)
#         amplitude = torch.abs(const)
#         amplitude_mean = torch.mean(amplitude)
#         amplitude_variance = torch.var(amplitude) / (amplitude_mean ** 2 + 1e-8)  # Normalized variance

#         # 2. Phase should form clusters (low entropy when quantized)
#         phase = torch.angle(const)  # [-π, π]

#         # Create phase histogram
#         n_bins = 32
#         phase_hist = self._compute_phase_histogram(phase, n_bins)

#         # Encourage clustering by minimizing entropy
#         phase_entropy = -torch.sum(phase_hist * torch.log(phase_hist + 1e-8))
#         max_entropy = torch.log(torch.tensor(float(n_bins)))  # Normalize
#         normalized_entropy = phase_entropy / max_entropy

#         # 3. Phase consistency (phases should be stable, not random)
#         phase_unwrapped = torch.unwrap(phase)
#         phase_derivative = torch.diff(phase_unwrapped)
#         phase_stability = torch.var(phase_derivative)

#         # Combine losses (lower is better)
#         psk_loss = (
#             1.0 * amplitude_variance +      # Penalize amplitude variation
#             0.5 * normalized_entropy +      # Penalize phase randomness
#             0.3 * phase_stability           # Penalize phase instability
#         )

#         return psk_loss

#     def qam_prototype_loss(self, signal: torch.Tensor) -> torch.Tensor:
#         """
#         QAM prototype: grid structure in I/Q plane
#         Information encoded in both amplitude and phase
#         """
#         const = self.extract_constellation(signal).squeeze(0)  # [length]

#         i_channel = torch.real(const)
#         q_channel = torch.imag(const)

#         # 1. Both I and Q should have discrete levels (grid structure)
#         i_discreteness = self._measure_discreteness(i_channel)
#         q_discreteness = self._measure_discreteness(q_channel)

#         # 2. Constellation should form clusters around grid points
#         grid_structure_loss = self._measure_grid_structure(i_channel, q_channel)

#         # 3. Power should be relatively uniform across symbols
#         power = torch.abs(const) ** 2
#         power_variance = torch.var(power) / (torch.mean(power) ** 2 + 1e-8)

#         qam_loss = (
#             0.4 * (2.0 - i_discreteness - q_discreteness) +  # Encourage discreteness
#             0.4 * grid_structure_loss +                      # Encourage grid structure
#             0.2 * power_variance                             # Moderate power variation
#         )

#         return qam_loss

#     def fsk_prototype_loss(self, signal: torch.Tensor) -> torch.Tensor:
#         """
#         FSK prototype: frequency domain characteristics
#         Information encoded as frequency shifts
#         """
#         const = self.extract_constellation(signal).squeeze(0)  # [length]

#         # 1. Instantaneous frequency should have discrete levels
#         phase = torch.angle(const)
#         phase_unwrapped = torch.unwrap(phase)
#         inst_freq = torch.gradient(phase_unwrapped)[0]  # Approximate derivative

#         freq_discreteness = self._measure_discreteness(inst_freq)

#         # 2. Frequency transitions should be clean (not gradual)
#         freq_derivative = torch.gradient(inst_freq)[0]
#         transition_sharpness = torch.mean(torch.abs(freq_derivative))

#         # 3. Power should be relatively constant (like PSK)
#         amplitude = torch.abs(const)
#         amplitude_variance = torch.var(amplitude) / (torch.mean(amplitude) ** 2 + 1e-8)

#         fsk_loss = (
#             0.5 * (1.0 - freq_discreteness) +     # Encourage discrete frequencies
#             0.3 * (-transition_sharpness) +       # Encourage sharp transitions
#             0.2 * amplitude_variance              # Discourage amplitude variation
#         )

#         return fsk_loss

#     def am_prototype_loss(self, signal: torch.Tensor) -> torch.Tensor:
#         """
#         AM prototype: amplitude modulation characteristics
#         Information encoded in amplitude variations
#         """
#         const = self.extract_constellation(signal).squeeze(0)  # [length]

#         # 1. Amplitude should vary significantly (that's where the information is)
#         amplitude = torch.abs(const)
#         amplitude_range = torch.max(amplitude) - torch.min(amplitude)
#         amplitude_mean = torch.mean(amplitude)
#         amplitude_variation = amplitude_range / (amplitude_mean + 1e-8)

#         # 2. Phase should be relatively stable
#         phase = torch.angle(const)
#         phase_unwrapped = torch.unwrap(phase)
#         phase_stability = torch.var(torch.gradient(phase_unwrapped)[0])

#         # 3. Amplitude changes should be smooth (not abrupt)
#         amplitude_smoothness = torch.var(torch.gradient(amplitude)[0])

#         am_loss = (
#             0.4 * (-amplitude_variation) +        # Encourage amplitude variation
#             0.4 * phase_stability +               # Discourage phase variation
#             0.2 * amplitude_smoothness            # Encourage smooth amplitude changes
#         )

#         return am_loss

#     def generic_modulation_loss(self, signal: torch.Tensor) -> torch.Tensor:
#         """Generic loss for unknown modulation types"""
#         const = self.extract_constellation(signal).squeeze(0)

#         # Basic modulation characteristics
#         amplitude = torch.abs(const)
#         phase = torch.angle(const)

#         # Encourage some structure (not completely random)
#         amplitude_structure = -torch.var(amplitude)  # Some amplitude variation
#         phase_structure = -torch.var(torch.gradient(torch.unwrap(phase))[0])  # Some phase structure

#         return 0.5 * amplitude_structure + 0.5 * phase_structure

#     def _compute_phase_histogram(self, phase: torch.Tensor, n_bins: int = 32) -> torch.Tensor:
#         """Compute soft histogram of phase values"""
#         # Create bin edges from -π to π
#         bin_edges = torch.linspace(-torch.pi, torch.pi, n_bins + 1, device=phase.device)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#         bin_width = bin_edges[1] - bin_edges[0]

#         # Soft binning using Gaussian kernels
#         phase_expanded = phase.unsqueeze(-1)  # [length, 1]
#         centers_expanded = bin_centers.unsqueeze(0)  # [1, n_bins]

#         # Gaussian kernel for soft assignment
#         sigma = bin_width / 3
#         weights = torch.exp(-0.5 * ((phase_expanded - centers_expanded) / sigma) ** 2)

#         # Normalize to get probability distribution
#         histogram = torch.sum(weights, dim=0)  # [n_bins]
#         histogram = histogram / (torch.sum(histogram) + 1e-8)

#         return histogram

#     def _measure_discreteness(self, values: torch.Tensor) -> torch.Tensor:
#         """
#         Measure how discrete/quantized a signal is
#         Returns value between 0 (continuous) and 1 (perfectly discrete)
#         """
#         # Compute histogram
#         n_bins = 16
#         hist = torch.histc(values, bins=n_bins, min=values.min(), max=values.max())
#         hist = hist / (torch.sum(hist) + 1e-8)

#         # High peaks indicate discreteness
#         # Use negative entropy as discreteness measure
#         entropy = -torch.sum(hist * torch.log(hist + 1e-8))
#         max_entropy = torch.log(torch.tensor(float(n_bins)))

#         discreteness = 1.0 - (entropy / max_entropy)
#         return torch.clamp(discreteness, 0, 1)

#     def _measure_grid_structure(self, i_values: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
#         """Measure how well I/Q values form a grid structure"""
#         # For QAM, I and Q values should be uncorrelated and form rectangular grid

#         # 1. Measure correlation (should be low for rectangular grid)
#         i_centered = i_values - torch.mean(i_values)
#         q_centered = q_values - torch.mean(q_values)

#         correlation = torch.abs(torch.mean(i_centered * q_centered))
#         correlation_normalized = correlation / (torch.std(i_values) * torch.std(q_values) + 1e-8)

#         # 2. Both I and Q should span reasonable range (not collapsed to single value)
#         i_range = torch.max(i_values) - torch.min(i_values)
#         q_range = torch.max(q_values) - torch.min(q_values)
#         range_penalty = torch.exp(-(i_range + q_range))  # Penalty if ranges are too small

#         grid_loss = correlation_normalized + range_penalty
#         return grid_loss


class LatentDiffusion(L.LightningModule):
    def __init__(
        self,
        unet,
        rfnet,
        encoder,  # This should be your trained RFEncoderDecoder model
        label_names,
        n_steps: int = 1000,
        latent_scaling: float = 0.18215,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        num_classes: int = 11,
        # AWGN scheduling parameters
        snr_range: Tuple[float, float] = (-20.0, 20.0),  # SNR range in dB
        # Loss weights
        latent_denoise_weight: float = 1.0,
        reconstruction_loss_weight: float = 0.5,
        constellation_loss_weight: float = 0.3,
        arcface_alignment_weight: float = 0.2,
        classification_loss_weight: float = 0.1,
        # Curriculum learning
        curriculum_epochs: int = 150,
        high_snr_start: float = 10.0,
        low_snr_end: float = -20.0,
        # Training strategy
        predict_noise: bool = True,  # Whether to predict noise or clean latent
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['unet', 'encoder_decoder'])
        self.automatic_optimization = False

        # Core parameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.latent_scaling = latent_scaling
        self.num_classes = num_classes
        self.label_names = label_names
        self.predict_noise = predict_noise

        # Loss weights
        self.latent_denoise_weight = latent_denoise_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.constellation_loss_weight = constellation_loss_weight
        self.arcface_alignment_weight = arcface_alignment_weight
        self.classification_loss_weight = classification_loss_weight

        # Curriculum learning
        self.curriculum_epochs = curriculum_epochs
        self.high_snr_start = high_snr_start
        self.low_snr_end = low_snr_end

        # Initialize models
        self.unet = unet
        self.encoder_decoder = encoder

        # Extract encoder and decoder components
        self.encoder = self.encoder_decoder.encoder
        self.decoder = self.encoder_decoder.decoder

        # AWGN noise scheduler
        self.awgn_scheduler = AWGNScheduler(n_steps=n_steps, snr_range=snr_range)

        # Initialize RFNet classifier
        self.rfnet = rfnet
        # Initialize ArcFace centers
        self.arcface_centers = ArcFaceCenters(
            encoder_decoder=encoder,
            num_classes=num_classes,
            label_names=label_names
        )

        self.arcface_centers.visualize_centers(logger=None)

        # Constellation loss
        # self.constellation_loss = ConstellationLoss()

        # Enhanced reconstruction loss (from your encoder-decoder)
        self.enhanced_reconstruction_loss = self.encoder_decoder.reconstruction_loss

        # Extract ArcFace centers from pre-trained encoder-decoder
        self._extract_arcface_centers()

    def _extract_arcface_centers(self):
        """Extract ArcFace class centers from pre-trained encoder-decoder"""
        if hasattr(self.encoder_decoder, 'arcface_loss') and hasattr(self.encoder_decoder.arcface_loss, 'weight'):
            # Get normalized ArcFace centers
            arcface_centers = F.normalize(self.encoder_decoder.arcface_loss.weight, p=2, dim=1)
            self.register_buffer("arcface_centers", arcface_centers)
        else:
            # Fallback: create learnable class prototypes
            print("Warning: ArcFace centers not found, creating random prototypes")
            centers = torch.randn(self.num_classes, self.encoder.hparams.latent_dim)
            centers = F.normalize(centers, p=2, dim=1)
            self.register_buffer("arcface_centers", centers)

    def get_curriculum_snr_threshold(self) -> float:
        """Get current SNR threshold for curriculum learning"""
        if self.current_epoch >= self.curriculum_epochs:
            return self.low_snr_end

        progress = self.current_epoch / self.curriculum_epochs
        threshold = self.high_snr_start + (self.low_snr_end - self.high_snr_start) * progress
        return threshold

    def compute_arcface_alignment_loss(self, z_denoised, class_labels):
        """Compute ArcFace alignment loss"""
        return self.arcface_centers.compute_alignment_loss(z_denoised, class_labels)

    def get_class_embeddings(self, class_labels):
        """Get class embeddings for conditioning"""
        return self.arcface_centers.get_centers_for_batch(class_labels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space and scale"""
        z = self.encoder(x) * self.latent_scaling
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        z_unscaled = z / self.latent_scaling
        return self.decoder(z_unscaled)

    def forward(
        self,
        z_noisy: torch.Tensor,
        t: torch.Tensor,
        class_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through UNet for latent space denoising

        Args:
            z_noisy: Noisy latent representation [batch, channels, length]
            t: Timestep [batch]
            class_embedding: Class conditioning [batch, embedding_dim]
        """
        return self.unet(z_noisy, t, class_embedding)

    def compute_reconstruction_losses(
        self,
        x_denoised: torch.Tensor,
        x_clean: torch.Tensor,
        x_noisy: torch.Tensor,
        is_paired: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction losses using the decoder

        Args:
            x_denoised: Denoised signal reconstruction
            x_clean: Original clean signal
            x_noisy: Noisy input signal
            is_paired: Whether this is paired data (synthetic) or unpaired (real low-SNR)
        """
        losses = {}

        if is_paired:
            # For paired data, use full reconstruction loss against clean signal
            recon_loss, loss_components = self.enhanced_reconstruction_loss(
                x_denoised, x_clean
            )
            losses.update(loss_components)
            losses['total_reconstruction'] = recon_loss
        # else:
        #     # For unpaired data, use constellation loss
        #     constellation_loss = self.constellation_loss(x_denoised, x_clean)
        #     losses['constellation_loss'] = constellation_loss
        #     losses['total_reconstruction'] = constellation_loss

        return losses

    def training_step(self, batch, batch_idx):
        # Get optimizer
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # Unpack batch
        x_clean, class_labels, original_snr = batch

        # Apply curriculum learning - filter based on original SNR
        snr_threshold = self.get_curriculum_snr_threshold()
        mask = original_snr >= snr_threshold

        if mask.sum() == 0:
            return None

        # Filter batch
        x_clean = x_clean[mask]
        class_labels = class_labels[mask]
        original_snr = original_snr[mask]

        batch_size = x_clean.shape[0]

        # Sample random timesteps for AWGN noise addition
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device).long()

        # Add AWGN noise to clean signal based on timestep
        x_noisy, signal_noise = self.awgn_scheduler.add_awgn_noise(x_clean, t)

        # Encode both clean and noisy signals
        z_clean = self.encode(x_clean)
        z_noisy = self.encode(x_noisy)

        # Compute target for latent space denoising
        if self.predict_noise:
            # Predict the noise that was added in latent space
            latent_noise = z_noisy - z_clean
            target = latent_noise
        else:
            # Predict the clean latent directly
            target = z_clean

        # Get class conditioning from ArcFace centers
        class_embedding = self.arcface_centers[class_labels]

        # Predict using UNet in latent space
        predicted = self.forward(z_noisy, t, class_embedding)

        # Primary latent denoising loss
        if self.predict_noise:
            latent_denoise_loss = F.mse_loss(predicted, target)
            # Reconstruct clean latent
            z_denoised = z_noisy - predicted
        else:
            latent_denoise_loss = F.mse_loss(predicted, target)
            z_denoised = predicted

        # Decode for reconstruction losses
        x_denoised = self.decode(z_denoised)

        # Determine if we have paired data (synthetic noise) or unpaired (real low-SNR)
        current_snr = self.awgn_scheduler.get_snr_at_timestep(t)
        is_paired = True  # We're always adding synthetic AWGN noise

        # Compute reconstruction losses
        reconstruction_losses = self.compute_reconstruction_losses(
            x_denoised, x_clean, x_noisy, is_paired=is_paired
        )

        # ArcFace alignment loss
        arcface_loss = self.compute_arcface_alignment_loss(z_denoised, class_labels)

        # Classification loss using RFNet on denoised signal
        class_logits, class_features = self.rfnet(x_denoised, return_features=True)
        classification_loss = F.cross_entropy(class_logits, class_labels)

        # Combined loss
        total_loss = (
            self.latent_denoise_weight * latent_denoise_loss +
            self.reconstruction_loss_weight * reconstruction_losses['total_reconstruction'] +
            self.arcface_alignment_weight * arcface_loss +
            self.classification_loss_weight * classification_loss
        )

        # Backprop
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()
        scheduler.step()

        # Logging
        self.log("train/latent_denoise_loss", latent_denoise_loss, prog_bar=True)
        self.log("train/reconstruction_loss", reconstruction_losses['total_reconstruction'])
        self.log("train/arcface_alignment_loss", arcface_loss)
        self.log("train/classification_loss", classification_loss)
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/snr_threshold", snr_threshold)
        self.log("train/avg_target_snr", current_snr.mean())

        # Log specific reconstruction components
        for key, value in reconstruction_losses.items():
            if key != 'total_reconstruction':
                self.log(f"train/{key}", value)

        # Classification accuracy
        with torch.no_grad():
            pred_classes = torch.argmax(class_logits, dim=1)
            acc = (pred_classes == class_labels).float().mean()
            self.log("train/classification_acc", acc, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x_clean, class_labels, original_snr = batch

        # Store for confusion matrix
        if not hasattr(self, "val_preds"):
            self.val_preds = []
            self.val_labels = []
            self.val_snrs = []
            self.val_denoised_preds = []

        batch_size = x_clean.shape[0]

        # Test denoising at multiple noise levels
        test_timesteps = [100, 300, 500, 700, 900]  # Different noise levels
        total_losses = []

        for test_t in test_timesteps:
            t = torch.full((batch_size,), test_t, device=self.device).long()

            # Add AWGN noise
            x_noisy, signal_noise = self.awgn_scheduler.add_awgn_noise(x_clean, t)

            # Encode
            z_clean = self.encode(x_clean)
            z_noisy = self.encode(x_noisy)

            # Compute target
            if self.predict_noise:
                latent_noise = z_noisy - z_clean
                target = latent_noise
            else:
                target = z_clean

            # Get class conditioning
            class_embedding = self.arcface_centers[class_labels]

            # Denoise in latent space
            predicted = self.forward(z_noisy, t, class_embedding)
            latent_denoise_loss = F.mse_loss(predicted, target)

            # Reconstruct
            if self.predict_noise:
                z_denoised = z_noisy - predicted
            else:
                z_denoised = predicted

            x_denoised = self.decode(z_denoised)

            # Reconstruction losses
            recon_losses = self.compute_reconstruction_losses(
                x_denoised, x_clean, x_noisy, is_paired=True
            )

            # ArcFace alignment
            arcface_loss = self.compute_arcface_alignment_loss(z_denoised, class_labels)

            # Classification on denoised signal
            class_logits, _ = self.rfnet(x_denoised, return_features=True)
            classification_loss = F.cross_entropy(class_logits, class_labels)

            total_loss = (
                self.latent_denoise_weight * latent_denoise_loss +
                self.reconstruction_loss_weight * recon_losses['total_reconstruction'] +
                self.arcface_alignment_weight * arcface_loss +
                self.classification_loss_weight * classification_loss
            )

            total_losses.append(total_loss)

            # Log per timestep
            target_snr = self.awgn_scheduler.get_snr_at_timestep(t[0])
            self.log(f"val/latent_denoise_loss_t{test_t}", latent_denoise_loss)
            self.log(f"val/total_loss_t{test_t}", total_loss)

        # Average validation loss
        avg_val_loss = torch.stack(total_losses).mean()
        self.log("val_loss", avg_val_loss, prog_bar=True)

        # Classification on original clean signal for comparison
        original_logits, _ = self.rfnet(x_clean, return_features=True)
        original_pred_classes = torch.argmax(original_logits, dim=1)

        # Classification on noisy signal (worst case - highest noise)
        t_worst = torch.full((batch_size,), self.n_steps - 1, device=self.device).long()
        x_very_noisy, _ = self.awgn_scheduler.add_awgn_noise(x_clean, t_worst)
        z_very_noisy = self.encode(x_very_noisy)

        # Denoise the very noisy signal
        class_embedding = self.arcface_centers[class_labels]
        predicted_noise_or_clean = self.forward(z_very_noisy, t_worst, class_embedding)

        if self.predict_noise:
            z_final_denoised = z_very_noisy - predicted_noise_or_clean
        else:
            z_final_denoised = predicted_noise_or_clean

        x_final_denoised = self.decode(z_final_denoised)
        denoised_logits, _ = self.rfnet(x_final_denoised, return_features=True)
        denoised_pred_classes = torch.argmax(denoised_logits, dim=1)

        # Store predictions
        self.val_preds.append(original_pred_classes.detach().cpu())
        self.val_denoised_preds.append(denoised_pred_classes.detach().cpu())
        self.val_labels.append(class_labels.detach().cpu())
        self.val_snrs.append(original_snr.detach().cpu())

        # Log accuracies
        original_acc = (original_pred_classes == class_labels).float().mean()
        denoised_acc = (denoised_pred_classes == class_labels).float().mean()

        self.log("val/original_acc", original_acc)
        self.log("val/denoised_acc", denoised_acc, prog_bar=True)

        return avg_val_loss

    def on_validation_epoch_end(self):
        if hasattr(self, "val_preds") and len(self.val_preds) > 0:
            # Concatenate predictions
            original_preds = torch.cat(self.val_preds).cpu().numpy()
            denoised_preds = torch.cat(self.val_denoised_preds).cpu().numpy()
            all_labels = torch.cat(self.val_labels).cpu().numpy()
            all_snrs = torch.cat(self.val_snrs).cpu().numpy()

            # Create comprehensive analysis plots
            self._plot_validation_analysis(original_preds, denoised_preds, all_labels, all_snrs)

            # Clear stored predictions
            self.val_preds.clear()
            self.val_denoised_preds.clear()
            self.val_labels.clear()
            self.val_snrs.clear()

    def _plot_validation_analysis(self, preds, labels, snrs):
        """Plot confusion matrix and SNR vs accuracy analysis"""
        label_names = [self.label_names[i] for i in range(self.num_classes)]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names, ax=axes[0, 0]
        )
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')

        # Accuracy vs SNR
        snr_bins = np.arange(-20, 25, 5)
        accuracies = []

        for i in range(len(snr_bins) - 1):
            mask = (snrs >= snr_bins[i]) & (snrs < snr_bins[i + 1])
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean()
                accuracies.append(acc)
            else:
                accuracies.append(0)

        axes[0, 1].plot(snr_bins[:-1] + 2.5, accuracies, 'b-o')
        axes[0, 1].set_xlabel('SNR (dB)')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Classification Accuracy vs SNR')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.05)

        # Per-class accuracy
        class_accuracies = []
        for class_id in range(self.num_classes):
            mask = labels == class_id
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean()
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0)

        axes[1, 0].bar(range(self.num_classes), class_accuracies)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Per-Class Accuracy')
        axes[1, 0].set_xticks(range(self.num_classes))
        axes[1, 0].set_xticklabels(label_names, rotation=45)

        # SNR distribution
        axes[1, 1].hist(snrs, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('SNR (dB)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('SNR Distribution')
        axes[1, 1].axvline(self.get_curriculum_snr_threshold(), color='red',
                          linestyle='--', label=f'Current Threshold')
        axes[1, 1].legend()

        plt.tight_layout()

        # Log to wandb
        if self.logger:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            self.logger.experiment.log({"validation_analysis": wandb.Image(img)})

        plt.close()

    def configure_optimizers(self):
        # Single optimizer for all components
        optimizer = torch.optim.AdamW(
            [
                {'params': self.unet.parameters()},
                {'params': self.rfnet.parameters(), 'lr': self.learning_rate * 2},  # Higher LR for classifier
            ],
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.1,
            anneal_strategy='cos'
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

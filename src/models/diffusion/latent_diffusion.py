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
            if hasattr(self.encoder_decoder, "arcface_loss"):
                arcface_loss = self.encoder_decoder.arcface_loss
                print("Found ArcFace loss at encoder_decoder.arcface_loss")

            # Location 2: Nested in model attribute
            elif hasattr(self.encoder_decoder, "model") and hasattr(
                self.encoder_decoder.model, "arcface_loss"
            ):
                arcface_loss = self.encoder_decoder.model.arcface_loss
                print("Found ArcFace loss at encoder_decoder.model.arcface_loss")

            # Location 2: Look through named modules
            else:
                for name, module in self.encoder_decoder.named_modules():
                    if "arcface" in name.lower() and hasattr(module, "weight"):
                        arcface_loss = module
                        print(f"Found ArcFace loss at {name}")
                        break

            if arcface_loss is not None and hasattr(arcface_loss, "weight"):
                # Extract the weight matrix (these are the class centers)
                centers = (
                    arcface_loss.weight.data.clone()
                )  # [num_classes, embedding_dim]

                # Normalize the centers (ArcFace uses normalized weights)
                centers = F.normalize(centers, p=2, dim=1)

                # Register as buffer (not a parameter, but part of model state)
                self.register_buffer("centers", centers)

                print(f"✅ Successfully extracted ArcFace centers: {centers.shape}")
                print(f"   - Number of classes: {centers.shape[0]}")
                print(f"   - Embedding dimension: {centers.shape[1]}")

                # Verify we have the right number of classes
                if centers.shape[0] != self.num_classes:
                    print(
                        f"⚠️  Warning: Expected {self.num_classes} classes, got {centers.shape[0]}"
                    )

            else:
                raise AttributeError(
                    "Could not find ArcFace loss with weight attribute"
                )

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
            if hasattr(self.encoder_decoder, "encoder"):
                encoder = self.encoder_decoder.encoder

                if hasattr(encoder, "hparams") and hasattr(
                    encoder.hparams, "latent_dim"
                ):
                    return encoder.hparams.latent_dim
                elif hasattr(encoder, "latent_dim"):
                    return encoder.latent_dim

            # Try to get from encoder_decoder hyperparameters
            if hasattr(self.encoder_decoder, "hparams") and hasattr(
                self.encoder_decoder.hparams, "latent_dim"
            ):
                return self.encoder_decoder.hparams.latent_dim
            elif hasattr(self.encoder_decoder, "latent_dim"):
                return self.encoder_decoder.latent_dim

            # Try to infer from a forward pass
            dummy_input = torch.randn(1, 2, 128)  # Assuming I/Q input
            with torch.no_grad():
                if hasattr(self.encoder_decoder, "encode"):
                    dummy_output = self.encoder_decoder.encode(dummy_input)
                elif hasattr(self.encoder_decoder, "encoder"):
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

    def compute_alignment_loss(
        self, embeddings: torch.Tensor, class_labels: torch.Tensor
    ):
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
            # Reshape from [batch, channels, length] to [batch, channels*length]
            batch_size = embeddings.shape[0]
            embeddings = embeddings.reshape(batch_size, -1)

        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # Get target centers for this batch
        target_centers = self.get_centers_for_batch(
            class_labels
        )  # [batch, embedding_dim]
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
                ax1.scatter(
                    centers_2d[i, 0],
                    centers_2d[i, 1],
                    c=[colors[i]],
                    s=100,
                    label=self.label_names[i],
                )

            ax1.set_title("ArcFace Class Centers (t-SNE)")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax1.grid(True, alpha=0.3)

            # Plot 2: Distance matrix
            distances = self.compute_inter_class_distances().cpu().numpy()

            im = ax2.imshow(distances, cmap="viridis")
            ax2.set_title("Inter-Class Angular Distances (degrees)")
            ax2.set_xticks(range(self.num_classes))
            ax2.set_yticks(range(self.num_classes))
            ax2.set_xticklabels(self.label_names, rotation=45, ha="right")
            ax2.set_yticklabels(self.label_names)

            # Add colorbar
            plt.colorbar(im, ax=ax2)

            # Add text annotations for distances
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:  # Skip diagonal
                        text_color = (
                            "white"
                            if distances[i, j] > distances.max() / 2
                            else "black"
                        )
                        ax2.text(
                            j,
                            i,
                            f"{distances[i, j]:.1f}°",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8,
                        )

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

    def __init__(
        self, n_steps: int = 1000, snr_range: Tuple[float, float] = (-20.0, 20.0)
    ):
        super().__init__()
        self.n_steps = n_steps
        self.snr_min, self.snr_max = snr_range

    def sample_valid_timesteps_direct(
        self, current_snr_db: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Directly sample target SNRs that are lower than current SNR, then convert to timesteps
        """
        device = current_snr_db.device

        # Sample target SNRs that are lower than current SNR
        # Leave at least 1dB margin to ensure meaningful noise addition
        min_target_snr = self.snr_min
        max_target_snr = current_snr_db - 2

        # Handle edge cases where max < min
        valid_range = max_target_snr > min_target_snr
        max_target_snr = torch.where(valid_range, max_target_snr, current_snr_db - 2.0)
        min_target_snr = torch.where(valid_range, min_target_snr, self.snr_min)

        # Sample uniformly between min and max target SNR
        random_factors = torch.rand(batch_size, device=device)
        target_snrs = min_target_snr + random_factors * (
            max_target_snr - min_target_snr
        )

        # Convert target SNRs to timesteps
        timesteps = self.snr_to_timestep(target_snrs)

        return timesteps

    def add_awgn_noise(
        self, signal: torch.Tensor, current_snr_db: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add AWGN noise to degrade signal from current SNR to target SNR
        """
        # Convert timestep to target SNR
        target_snr_db = self.timestep_to_snr(t)

        # Force target SNR to be lower than current SNR (safety check)
        target_snr_db = torch.min(target_snr_db, current_snr_db - 0.5)

        # Clamp target SNR to valid range using element-wise operations
        target_snr_db = torch.clamp(target_snr_db, min=self.snr_min)
        target_snr_db = torch.min(target_snr_db, current_snr_db - 0.5)

        # Debug: Check for any remaining invalid cases
        invalid_mask = target_snr_db >= current_snr_db
        if invalid_mask.any():
            print(f"Fixing {invalid_mask.sum()} invalid samples")
            target_snr_db = torch.where(
                invalid_mask, current_snr_db - 1.0, target_snr_db
            )

        # Calculate signal power
        total_power = torch.mean(signal**2, dim=(1, 2), keepdim=True)  # [batch, 1, 1]

        # Convert SNRs to linear scale
        current_snr_linear = 10 ** (current_snr_db.view(-1, 1, 1) / 10)
        target_snr_linear = 10 ** (target_snr_db.view(-1, 1, 1) / 10)

        # Calculate clean signal power (assuming current signal = clean + noise)
        signal_power = total_power * current_snr_linear / (1 + current_snr_linear)

        # Calculate current noise power
        current_noise_power = total_power - signal_power

        # Calculate required total noise power for target SNR
        target_total_noise_power = signal_power / target_snr_linear

        # Additional noise power needed (should always be positive now)
        additional_noise_power = target_total_noise_power - current_noise_power
        additional_noise_power = torch.clamp(additional_noise_power, min=1e-10)

        # Generate additional AWGN noise
        additional_noise = torch.randn_like(signal) * torch.sqrt(additional_noise_power)

        # Add additional noise to signal
        noisy_signal = signal + additional_noise

        # Apply realistic hardware clipping at [-1, 1]
        noisy_signal = torch.clamp(noisy_signal, -1.0, 1.0)

        return noisy_signal, additional_noise

    def snr_to_timestep(self, snr_db: torch.Tensor) -> torch.Tensor:
        """Convert SNR (dB) to timestep - higher SNR = lower timestep"""
        normalized_snr = (snr_db - self.snr_min) / (self.snr_max - self.snr_min)
        normalized_snr = torch.clamp(normalized_snr, 0, 1)
        # Higher SNR = lower timestep (less noise to add)
        timestep = (1 - normalized_snr) * (self.n_steps - 1)
        return timestep.long()

    def timestep_to_snr(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert timestep to SNR (dB) - higher timestep = lower SNR"""
        normalized_t = timestep.float() / (self.n_steps - 1)
        # Higher timestep = lower SNR (more degraded)
        snr_db = self.snr_max - normalized_t * (self.snr_max - self.snr_min)
        return snr_db


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
        recon_base: float = 1.0,
        arc_base: float = 0.1,
        class_base: float = 0.1,
        # Curriculum learning
        curriculum_epochs: int = 1,
        high_snr_start: float = 16.0,
        low_snr_end: float = -20.0,
        # Training strategy
        predict_noise: bool = True,  # Whether to predict noise or clean latent
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet", "encoder_decoder"])
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
        self.recon_base = recon_base
        self.arc_base = arc_base
        self.class_base = class_base

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
            encoder_decoder=encoder, num_classes=num_classes, label_names=label_names
        )

        self.arcface_centers.visualize_centers(logger=None)

        # Constellation loss
        # self.constellation_loss = ConstellationLoss()

        # Enhanced reconstruction loss (from your encoder-decoder)
        self.enhanced_reconstruction_loss = self.encoder_decoder.reconstruction_loss

    def get_curriculum_snr_threshold(self) -> float:
        """Get current SNR threshold for curriculum learning"""
        if self.current_epoch >= self.curriculum_epochs:
            return self.low_snr_end

        progress = self.current_epoch / self.curriculum_epochs
        threshold = (
            self.high_snr_start + (self.low_snr_end - self.high_snr_start) * progress
        )
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

        # Reshape from [batch_size, 256] to [batch_size, 32, 8]
        # Distributing 256 features as 32 channels × 8 sequence length
        z = z.reshape(z.shape[0], 32, 8)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        # Reshape from [batch_size, 32, 8] back to [batch_size, 256]
        batch_size = z.shape[0]
        z_flat = z.reshape(batch_size, 256)

        z_unscaled = z_flat / self.latent_scaling
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
        is_paired: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction losses using the decoder

        Args:
            x_denoised: Denoised signal reconstruction
            x_clean: Original clean signal (or reference signal for unpaired)
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
            losses["total_reconstruction"] = recon_loss
        else:
            # For unpaired data, use basic reconstruction metrics
            # Since we don't have ground truth, use basic signal quality metrics

            # MSE between denoised and reference (original noisy)
            mse_loss = F.mse_loss(x_denoised, x_clean)
            losses["mse_loss"] = mse_loss

            # Signal power preservation
            signal_power_original = torch.mean(x_clean**2)
            signal_power_denoised = torch.mean(x_denoised**2)
            power_diff = F.mse_loss(signal_power_denoised, signal_power_original)
            losses["power_preservation"] = power_diff

            # Use constellation loss if available
            if hasattr(self, "constellation_loss"):
                constellation_loss = self.constellation_loss(x_denoised, x_clean)
                losses["constellation_loss"] = constellation_loss
                losses["total_reconstruction"] = (
                    mse_loss + 0.1 * power_diff + 0.3 * constellation_loss
                )
            else:
                # Fallback: weighted combination of available losses
                losses["total_reconstruction"] = mse_loss + 0.1 * power_diff

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

        # Use direct SNR-based sampling for more reliable timestep generation
        t = self.awgn_scheduler.sample_valid_timesteps_direct(original_snr, batch_size)

        # Verify the target SNRs are valid
        target_snr = self.awgn_scheduler.timestep_to_snr(t)

        # Debug logging (less frequent)
        if batch_idx % 100 == 0:
            print(f"Original SNR: {original_snr.min():.1f} to {original_snr.max():.1f}")
            print(f"Target SNR: {target_snr.min():.1f} to {target_snr.max():.1f}")
            print(
                f"SNR reduction: {(original_snr - target_snr).min():.1f} to {(original_snr - target_snr).max():.1f}"
            )

            # Verify all samples have valid degradation
            valid_degradation = (original_snr - target_snr) > 0
            print(
                f"Valid degradation: {valid_degradation.sum()}/{len(valid_degradation)} samples"
            )

        # Add AWGN noise to clean signal based on timestep
        x_noisy, signal_noise = self.awgn_scheduler.add_awgn_noise(
            x_clean, original_snr, t
        )

        # Rest of training step remains the same...
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
        predicted = self.forward(z_noisy, t, class_labels)

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
        current_snr = self.awgn_scheduler.timestep_to_snr(t)
        is_paired = True  # We're always adding synthetic AWGN noise

        # Compute reconstruction losses
        reconstruction_losses = self.compute_reconstruction_losses(
            x_denoised, x_clean, x_noisy, is_paired=is_paired
        )

        # ArcFace alignment loss
        arcface_loss = self.compute_arcface_alignment_loss(z_denoised, class_labels)

        # Classification loss using RFNet on denoised signal
        # class_logits, class_features = self.rfnet(z_denoised, return_features=True)
        class_logits, class_features = self.rfnet(x_denoised, return_features=True)
        classification_loss = F.cross_entropy(
            class_logits, class_labels)#, reduction="none"
        # , reduction="none"

        snr_factor = torch.sigmoid((original_snr - snr_threshold) / 5.0)  # [B]

        # 2. Compute per-sample weights
        r_w = self.recon_base * snr_factor  # [B]
        a_w = self.arc_base + self.recon_base * (1 - snr_factor) * 0.5  # [B]
        c_w = self.class_base + self.recon_base * (1 - snr_factor) * 0.5  # [B]

        # 3. Compute per-sample losses (you might already have these)
        recon_losses = reconstruction_losses["total_reconstruction"]  # [B]
        arc_losses = arcface_loss  # compute per-sample

        total_loss = classification_loss
        # total_loss = (
        #     self.latent_denoise_weight * latent_denoise_loss  # scalar
        #     + (r_w * recon_losses).mean()
        #     + (a_w * arc_losses).mean()
        #     + (classification_loss).mean()
        # )

        # Backprop
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()
        scheduler.step()
        # Logging
        self.log("train/latent_denoise_loss", latent_denoise_loss, prog_bar=True)
        self.log(
            "train/reconstruction_loss", reconstruction_losses["total_reconstruction"]
        )
        self.log("train/arcface_alignment_loss", arcface_loss)
        self.log("train/classification_loss", classification_loss.mean())
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/snr_threshold", snr_threshold)
        self.log("train/avg_target_snr", current_snr.mean())

        # Log specific reconstruction components
        for key, value in reconstruction_losses.items():
            if key != "total_reconstruction":
                self.log(f"train/{key}", value)

        # Classification accuracy
        with torch.no_grad():
            pred_classes = torch.argmax(class_logits, dim=1)
            acc = (pred_classes == class_labels).float().mean()
            self.log("train/classification_acc", acc, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x_noisy, class_labels, original_snr = batch  # x_noisy = naturally noisy signal

        # Init storage
        if not hasattr(self, "val_preds"):
            self.val_preds, self.val_labels, self.val_snrs = [], [], []
            self.val_denoised_preds, self.val_latents = [], []

        if not hasattr(self, "viz_examples"):
            self.viz_examples = {
                "clean": [],  # original noisy input
                "noisy": [],  # synthetic AWGN added for viz
                "denoised": [],
                "labels": [],
                "original_snr": [],
                "timesteps": [],
            }

        batch_size = x_noisy.size(0)

        # Map real SNR to timestep for denoising
        t = self.awgn_scheduler.snr_to_timestep(original_snr).long().to(self.device)

        # Debug logging (only for first batch)
        if batch_idx == 0:
            target_snr_from_timestep = self.awgn_scheduler.timestep_to_snr(t)
            print(
                f"Validation - Original SNR: {original_snr.min():.1f} to {original_snr.max():.1f}"
            )
            print(f"Validation - Timestep range: {t.min()} to {t.max()}")
            print(
                f"Validation - Target SNR from timestep: {target_snr_from_timestep.min():.1f} to {target_snr_from_timestep.max():.1f}"
            )

        # Encode original noisy signal
        z_noisy = self.encode(x_noisy)
        self.val_latents.append(z_noisy.view(batch_size, -1).detach().cpu())

        # Denoise using the UNet
        if self.predict_noise:
            # Predict the noise in latent space
            cond = torch.full_like(class_labels, 11)
            predicted_noise = self.forward(z_noisy, t, cond)
            # Remove predicted noise to get clean latent
            z_denoised = z_noisy - predicted_noise
        else:
            # Directly predict clean latent
            cond = torch.full_like(class_labels, 11)
            z_denoised = self.forward(z_noisy, t, cond)

        # Classification on denoised latent
        x_denoised = self.decode(z_denoised)
        logits, _ = self.rfnet(x_denoised, return_features=True)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == class_labels).float().mean()

        # Also get classification on original noisy signal for comparison
        logits_original, _ = self.rfnet(x_noisy, return_features=True)
        preds_original = torch.argmax(logits_original, dim=1)
        acc_original = (preds_original == class_labels).float().mean()

        # Core losses that we can always compute
        arcface_loss = self.compute_arcface_alignment_loss(z_denoised, class_labels)
        classification_loss = F.cross_entropy(logits, class_labels)
        classification_loss_original = F.cross_entropy(logits_original, class_labels)

        # Try to compute reconstruction losses, but handle gracefully
        try:
            recon_losses = self.compute_reconstruction_losses(
                x_denoised, x_noisy, x_noisy, is_paired=False
            )
            reconstruction_loss = recon_losses.get(
                "total_reconstruction", torch.tensor(0.0, device=self.device)
            )
        except Exception as e:
            print(f"Warning: Could not compute reconstruction losses: {e}")
            # Fallback: simple MSE between denoised and original
            reconstruction_loss = F.mse_loss(x_denoised, x_noisy)
            recon_losses = {"total_reconstruction": reconstruction_loss}

        # Compute total validation loss
        total_loss = (
            self.arc_base * arcface_loss
            + self.class_base * classification_loss
            + 0.1
            * reconstruction_loss  # Reduce weight since we don't have ground truth
        )

        # Logging
        self.log("val/loss", total_loss, prog_bar=True)
        self.log("val/acc_denoised", acc, prog_bar=True)
        self.log("val/acc_original", acc_original, prog_bar=True)
        self.log("val/acc_improvement", acc - acc_original, prog_bar=True)
        self.log("val/classification_loss", classification_loss)
        self.log("val/classification_loss_original", classification_loss_original)
        self.log("val/arcface_loss", arcface_loss)

        # Log reconstruction components if available
        for key, value in recon_losses.items():
            if key != "total_reconstruction":
                self.log(f"val/{key}", value)
        self.log("val/reconstruction_loss", reconstruction_loss)

        # Store predictions for analysis
        self.val_preds.append(preds_original.detach().cpu())  # Original predictions
        self.val_labels.append(class_labels.detach().cpu())
        self.val_snrs.append(original_snr.detach().cpu())
        self.val_denoised_preds.append(preds.detach().cpu())  # Denoised predictions

        # === Visualization: Add synthetic AWGN for comparison ===
        if len(self.viz_examples["clean"]) < 10 and batch_idx < 3:
            # Select interesting samples (lowest, highest, and median SNR)
            snr_indices = torch.argsort(original_snr)
            viz_indices = [snr_indices[0], snr_indices[-1]]
            if batch_size > 2:
                viz_indices.append(snr_indices[batch_size // 2])

            for viz_idx in viz_indices[:10]:  # Limit to 3 per batch
                # Store original noisy input as "clean" reference
                self.viz_examples["clean"].append(
                    x_noisy[viz_idx : viz_idx + 1].detach().cpu()
                )

                # Add additional synthetic AWGN for visualization comparison
                current_timestep = t[viz_idx].item()

                # Add more noise by sampling a higher timestep (lower SNR)
                if current_timestep < self.n_steps - 50:
                    additional_degradation_t = torch.randint(
                        current_timestep + 10,
                        min(current_timestep + 100, self.n_steps),
                        (1,),
                        device=self.device,
                    )
                else:
                    additional_degradation_t = torch.tensor(
                        [self.n_steps - 1], device=self.device
                    )

                # Add synthetic noise for visualization
                x_single = x_noisy[viz_idx : viz_idx + 1]
                snr_single = original_snr[viz_idx : viz_idx + 1]

                x_plus_awgn, _ = self.awgn_scheduler.add_awgn_noise(
                    x_single, snr_single, additional_degradation_t
                )
                self.viz_examples["noisy"].append(x_plus_awgn.detach().cpu())

                # Store denoised output
                self.viz_examples["denoised"].append(
                    x_denoised[viz_idx : viz_idx + 1].detach().cpu()
                )
                self.viz_examples["labels"].append(
                    class_labels[viz_idx : viz_idx + 1].detach().cpu()
                )
                self.viz_examples["original_snr"].append(
                    original_snr[viz_idx : viz_idx + 1].detach().cpu()
                )
                self.viz_examples["timesteps"].append(
                    t[viz_idx : viz_idx + 1].detach().cpu()
                )

        return total_loss

    def on_validation_epoch_end(self):
        if hasattr(self, "val_preds") and len(self.val_preds) > 0:
            # Concatenate predictions
            original_preds = torch.cat(self.val_preds).cpu().numpy()
            denoised_preds = (
                torch.cat(self.val_denoised_preds).cpu().numpy()
                if hasattr(self, "val_denoised_preds")
                and len(self.val_denoised_preds) > 0
                else None
            )
            all_labels = torch.cat(self.val_labels).cpu().numpy()
            all_snrs = torch.cat(self.val_snrs).cpu().numpy()
            # Concatenate latent representations if they exist
            if hasattr(self, "val_latents") and len(self.val_latents) > 0:
                # Store the concatenated tensor
                latents_tensor = torch.cat(self.val_latents)
                # Replace the list with the tensor (temporary)
                self.val_latents = latents_tensor
                # Create comprehensive analysis plots
                self._plot_validation_analysis(
                    denoised_preds, denoised_preds, all_labels, all_snrs
                )
                # Create denoising visualizations if we have collected examples
                self._create_denoising_visualizations()
                # Add ArcFace prototype visualization
                # self.visualize_arcface_prototypes()
                # Reset val_latents to an empty list
                self.val_latents = []
            else:
                print(
                    "Warning: No latent representations collected, skipping ArcFace visualization"
                )
            # Clear stored predictions
            self.val_preds.clear()
            self.val_labels.clear()
            self.val_snrs.clear()
            self.val_denoised_preds.clear()

    def _create_denoising_visualizations(self):
        """Create all visualization plots from collected examples"""
        # Limit to a reasonable number of examples
        num_examples = min(len(self.viz_examples["clean"]), 5)

        # Process each example
        for i in range(num_examples):
            # Get the data for this example - make sure to convert to float32
            x_clean = (
                self.viz_examples["clean"][i].to(self.device).float()
            )  # Explicitly use float
            x_noisy = self.viz_examples["noisy"][i].to(self.device).float()
            x_denoised = self.viz_examples["denoised"][i].to(self.device).float()
            class_labels = self.viz_examples["labels"][i].to(self.device)
            target_snr = self.viz_examples["original_snr"][i].to(self.device)

            # Encode for latent space visualization
            with torch.no_grad():
                z_clean = self.encode(x_clean)
                z_noisy = self.encode(x_noisy)
                z_denoised = self.encode(x_denoised)

            # 2. Visualize denoising process
            self.visualize_denoising_process(
                x_clean,
                x_noisy,
                x_denoised,
                z_clean,
                z_noisy,
                z_denoised,
                target_snr,
                class_labels,
                max_samples=1,
            )

        # Clear the examples after visualization
        for key in self.viz_examples:
            self.viz_examples[key] = []

    def _plot_validation_analysis(self, original_preds, denoised_preds, labels, snrs):
        """Plot confusion matrix and SNR vs accuracy analysis using actual label names"""
        # Convert label_names to a Python list to avoid OmegaConf issues
        if not isinstance(self.label_names, list):
            try:
                label_names = [
                    self.label_names[str(i)]
                    if isinstance(self.label_names, dict)
                    else self.label_names[i]
                    for i in range(self.num_classes)
                ]
            except:
                # Fallback if we can't extract names
                label_names = [f"Class {i}" for i in range(self.num_classes)]
        else:
            label_names = self.label_names

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Confusion Matrix for original predictions
        cm = confusion_matrix(labels, original_preds, labels=range(self.num_classes))
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Confusion Matrix (Original)")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("True")

        # Rotate x-axis labels for better readability
        plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha="right")

        # Plot 2: Accuracy vs SNR
        snr_bins = np.arange(-20, 25, 5)
        accuracies = []

        for i in range(len(snr_bins) - 1):
            mask = (snrs >= snr_bins[i]) & (snrs < snr_bins[i + 1])
            if mask.sum() > 0:
                acc = (original_preds[mask] == labels[mask]).mean()
                accuracies.append(acc)
            else:
                accuracies.append(0)

        axes[0, 1].plot(snr_bins[:-1] + 2.5, accuracies, "b-o")
        axes[0, 1].set_xlabel("SNR (dB)")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Classification Accuracy vs SNR")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.05)

        # Plot 3: Per-class accuracy with actual label names
        class_accuracies = []
        for class_id in range(self.num_classes):
            mask = labels == class_id
            if mask.sum() > 0:
                acc = (original_preds[mask] == labels[mask]).mean()
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0)

        # Use actual label names for x-axis
        axes[1, 0].bar(range(self.num_classes), class_accuracies)
        axes[1, 0].set_xlabel("Modulation Type")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Per-Class Accuracy")
        axes[1, 0].set_xticks(range(self.num_classes))
        axes[1, 0].set_xticklabels(label_names, rotation=45, ha="right")

        # Plot 4: SNR distribution
        axes[1, 1].hist(snrs, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 1].set_xlabel("SNR (dB)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("SNR Distribution")
        axes[1, 1].axvline(
            self.get_curriculum_snr_threshold(),
            color="red",
            linestyle="--",
            label=f"Current Threshold: {self.get_curriculum_snr_threshold():.1f} dB",
        )
        axes[1, 1].legend()

        # Add epoch information to the figure
        if hasattr(self, "current_epoch"):
            fig.suptitle(
                f"Validation Analysis - Epoch {self.current_epoch}", fontsize=16, y=0.98
            )

        plt.tight_layout()

        # Log to wandb
        if self.logger and hasattr(self.logger, "experiment"):
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            self.logger.experiment.log({"validation_analysis": wandb.Image(img)})

        plt.close()

    def visualize_denoising_process(
        self,
        x_clean,
        x_noisy,
        x_denoised,
        z_clean,
        z_noisy,
        z_denoised,
        target_snr,
        class_labels=None,
        max_samples=2,
    ):
        """
        Visualize the denoising process in both signal and latent space using t-SNE.

        Args:
            x_clean: Clean signals [batch, 2, length]
            x_noisy: Noisy signals [batch, 2, length]
            x_denoised: Denoised signals [batch, 2, length]
            z_clean: Clean latents [batch, channels, length]
            z_noisy: Noisy latents [batch, channels, length]
            z_denoised: Denoised latents [batch, channels, length]
            target_snr: Target SNR values after noise addition [batch]
            class_labels: Optional class labels for signals [batch]
            max_samples: Maximum number of samples to visualize
        """
        batch_size = x_clean.shape[0]
        num_samples = min(batch_size, max_samples)

        # Create figure with multiple rows (one per sample) and 3 columns
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

        # If only one sample, expand axes dimensions
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Get sample data
            clean = x_clean[i].detach().cpu().numpy()  # [2, length]
            noisy = x_noisy[i].detach().cpu().numpy()  # [2, length]
            denoised = x_denoised[i].detach().cpu().numpy()  # [2, length]
            targ_snr = target_snr[i].item()

            # Get class label if provided
            label_str = ""
            if class_labels is not None:
                label_idx = class_labels[i].item()
                # Get actual label name
                label_str = f" - {self.label_names[int(label_idx)]}"

            time_axis = np.arange(clean.shape[1])

            # Plot time-domain signals
            axes[i, 0].plot(time_axis, clean[0], "b-", label="Clean I", alpha=0.9)
            axes[i, 0].plot(time_axis, clean[1], "b--", label="Clean Q", alpha=0.9)
            axes[i, 0].plot(time_axis, noisy[0], "r-", label="Noisy I", alpha=0.5)
            axes[i, 0].plot(time_axis, noisy[1], "r--", label="Noisy Q", alpha=0.5)
            axes[i, 0].plot(time_axis, denoised[0], "g-", label="Denoised I", alpha=0.7)
            axes[i, 0].plot(
                time_axis, denoised[1], "g--", label="Denoised Q", alpha=0.7
            )
            axes[i, 0].set_title(f"Time Domain (SNR: {targ_snr:.1f}dB){label_str}")
            axes[i, 0].legend(loc="upper right")
            axes[i, 0].grid(True, alpha=0.3)

            # Plot I/Q constellations
            axes[i, 1].scatter(
                clean[0], clean[1], s=4, c="blue", alpha=0.7, label="Clean"
            )
            axes[i, 1].scatter(
                noisy[0], noisy[1], s=2, c="red", alpha=0.3, label="Noisy"
            )
            axes[i, 1].scatter(
                denoised[0], denoised[1], s=3, c="green", alpha=0.5, label="Denoised"
            )
            axes[i, 1].set_title(f"I/Q Constellation{label_str}")
            axes[i, 1].set_xlim(-2, 2)
            axes[i, 1].set_ylim(-2, 2)
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend(loc="upper right")

            try:
                # Plot latent space visualizations using t-SNE instead of PCA
                from sklearn.manifold import TSNE

                # Reshape latents for t-SNE
                z_clean_flat = (
                    z_clean[i].detach().cpu().numpy().reshape(1, -1)
                )  # [1, features]
                z_noisy_flat = (
                    z_noisy[i].detach().cpu().numpy().reshape(1, -1)
                )  # [1, features]
                z_denoised_flat = (
                    z_denoised[i].detach().cpu().numpy().reshape(1, -1)
                )  # [1, features]

                # Stack for t-SNE
                z_combined = np.vstack(
                    [z_clean_flat, z_noisy_flat, z_denoised_flat]
                )  # [3, features]

                if z_combined.shape[0] > 1:  # Need at least 2 samples for t-SNE
                    # For single examples, we'll create copies to allow t-SNE to work
                    if z_combined.shape[0] < 4:
                        # Add small noise to create multiple versions of each point
                        noise_scale = 1e-4
                        z_expanded = []
                        for z in z_combined:
                            z_expanded.append(z)  # Original
                            for _ in range(3):  # 3 noisy copies
                                z_expanded.append(
                                    z + np.random.normal(0, noise_scale, z.shape)
                                )
                        z_combined = np.vstack(z_expanded)

                    # Apply t-SNE with appropriate perplexity (lower for fewer samples)
                    perplexity = min(
                        5, z_combined.shape[0] - 1
                    )  # Perplexity must be less than n_samples
                    tsne = TSNE(
                        n_components=2,
                        random_state=42,
                        perplexity=perplexity,
                        learning_rate="auto",
                        init="pca",
                    )
                    z_2d = tsne.fit_transform(z_combined)

                    # Only keep the first 3 points (original clean, noisy, denoised)
                    z_2d = z_2d[:3]

                    # Plot t-SNE latent space
                    axes[i, 2].scatter(
                        z_2d[0, 0],
                        z_2d[0, 1],
                        s=100,
                        c="blue",
                        marker="o",
                        label="Clean",
                    )
                    axes[i, 2].scatter(
                        z_2d[1, 0],
                        z_2d[1, 1],
                        s=100,
                        c="red",
                        marker="x",
                        label="Noisy",
                    )
                    axes[i, 2].scatter(
                        z_2d[2, 0],
                        z_2d[2, 1],
                        s=100,
                        c="green",
                        marker="+",
                        label="Denoised",
                    )

                    # Add arrows to show denoising direction
                    axes[i, 2].arrow(
                        z_2d[1, 0],
                        z_2d[1, 1],
                        z_2d[2, 0] - z_2d[1, 0],
                        z_2d[2, 1] - z_2d[1, 1],
                        head_width=0.1,
                        head_length=0.1,
                        fc="black",
                        ec="black",
                        alpha=0.7,
                    )

                    # Add another arrow from noisy to clean for comparison
                    axes[i, 2].arrow(
                        z_2d[1, 0],
                        z_2d[1, 1],
                        z_2d[0, 0] - z_2d[1, 0],
                        z_2d[0, 1] - z_2d[1, 1],
                        head_width=0.1,
                        head_length=0.1,
                        fc="blue",
                        ec="blue",
                        alpha=0.3,
                        linestyle="--",
                    )

                    axes[i, 2].set_title(f"Latent Space (t-SNE){label_str}")
                    axes[i, 2].grid(True, alpha=0.3)
                    axes[i, 2].legend(loc="upper right")
                else:
                    # Fallback if t-SNE doesn't work
                    axes[i, 2].text(
                        0.5,
                        0.5,
                        "t-SNE requires multiple points",
                        ha="center",
                        va="center",
                        transform=axes[i, 2].transAxes,
                    )
                    axes[i, 2].set_title("Latent Space (t-SNE unavailable)")

            except Exception as e:
                # Fallback visualization if t-SNE fails
                print(f"t-SNE failed, using simple representation: {e}")
                axes[i, 2].text(
                    0.5,
                    0.5,
                    f"t-SNE Error: {str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=axes[i, 2].transAxes,
                )
                axes[i, 2].set_title("Latent Space (t-SNE failed)")

            # Calculate and display metrics
            mse_noisy = np.mean((clean - noisy) ** 2)
            mse_denoised = np.mean((clean - denoised) ** 2)
            improvement = (mse_noisy - mse_denoised) / mse_noisy * 100

            # Add text annotations with metrics
            axes[i, 1].text(
                0.05,
                0.05,
                f"MSE Noisy: {mse_noisy:.4f}\nMSE Denoised: {mse_denoised:.4f}\nImprovement: {improvement:.1f}%",
                transform=axes[i, 1].transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        plt.tight_layout()

        # Convert to wandb Image
        if self.logger:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            img = Image.open(buf)
            self.logger.experiment.log(
                {"denoising_process_visualization": wandb.Image(img)}
            )

        plt.close(fig)
        return fig

    def visualize_arcface_prototypes(self):
        """
        Visualize ArcFace class prototypes and sample projections using t-SNE.
        Shows how latent embeddings are distributed relative to class centers.
        """
        if not hasattr(self, "arcface_centers") or not hasattr(self, "val_latents"):
            return

        try:
            # Create figure with 2 subplots (not 3)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Get ArcFace centers (prototypes)
            centers = self.arcface_centers.centers.detach().cpu().numpy()

            # Ensure val_latents is a tensor
            if isinstance(self.val_latents, list):
                if len(self.val_latents) == 0:
                    print("No latent samples available for visualization")
                    return
                embeddings = torch.cat(self.val_latents)
            else:
                embeddings = self.val_latents

            # Ensure val_labels is a tensor
            if isinstance(self.val_labels, list):
                if len(self.val_labels) == 0:
                    print("No labels available for visualization")
                    return
                labels = torch.cat(self.val_labels).cpu().numpy()
            else:
                labels = self.val_labels.cpu().numpy()

            # Handle labels shape - squeeze if needed
            if labels.ndim > 1:
                labels = labels.squeeze()

            # Normalize latent embeddings
            embeddings_norm = F.normalize(embeddings, p=2, dim=1).cpu().numpy()

            # Set up colors and class names
            unique_labels = np.unique(labels)
            num_classes = len(self.label_names)
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

            # 1. Plot class prototypes on unit circle (first 2 dimensions)
            prototype_2d = centers[:, :2]  # First 2 dimensions
            prototype_2d = prototype_2d / np.maximum(
                np.linalg.norm(prototype_2d, axis=1, keepdims=True), 1e-10
            )

            for i in range(num_classes):
                # Get actual label name
                label_name = self.label_names[i]

                # Plot class prototype
                axes[0].scatter(
                    prototype_2d[i, 0],
                    prototype_2d[i, 1],
                    s=200,
                    marker="*",
                    c=[colors[i]],
                    label=label_name,
                    edgecolors="black",
                    linewidths=1,
                    alpha=0.9,
                    zorder=10,
                )

                # Add text label
                axes[0].text(
                    prototype_2d[i, 0] * 1.1,
                    prototype_2d[i, 1] * 1.1,
                    label_name,
                    color=colors[i],
                    fontweight="bold",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

                # Draw lines from origin to prototype
                axes[0].plot(
                    [0, prototype_2d[i, 0]],
                    [0, prototype_2d[i, 1]],
                    color=colors[i],
                    linestyle="--",
                    alpha=0.5,
                )

            # Draw unit circle
            circle = plt.Circle(
                (0, 0), 1, fill=False, color="gray", linestyle="-", alpha=0.8
            )
            axes[0].add_artist(circle)

            # Draw axes
            axes[0].axhline(y=0, color="k", linestyle=":", alpha=0.3)
            axes[0].axvline(x=0, color="k", linestyle=":", alpha=0.3)

            # Set limits and title
            axes[0].set_xlim(-1.2, 1.2)
            axes[0].set_ylim(-1.2, 1.2)
            axes[0].set_aspect("equal")
            axes[0].set_title("ArcFace Class Prototypes on Unit Circle")
            axes[0].grid(True, alpha=0.3)

            # 2. Use t-SNE to visualize prototypes and embeddings together
            from sklearn.manifold import TSNE

            # Combine prototypes and embeddings for t-SNE
            combined_data = np.vstack([centers, embeddings_norm])
            combined_labels = np.concatenate(
                [
                    np.arange(num_classes),  # Class indices for prototypes
                    labels,  # Class labels for embeddings
                ]
            )

            # Create indicator for prototypes vs samples
            is_prototype = np.concatenate(
                [
                    np.ones(num_classes),  # 1 for prototypes
                    np.zeros(len(labels)),  # 0 for embeddings
                ]
            )

            # Apply t-SNE
            perplexity = min(
                30, len(combined_data) // 5
            )  # Lower perplexity for smaller datasets
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                learning_rate="auto",
                init="pca",
            )
            combined_2d = tsne.fit_transform(combined_data)

            # Separate prototype and sample points
            prototype_points = combined_2d[:num_classes]
            sample_points = combined_2d[num_classes:]

            # Plot prototypes and samples in t-SNE space
            for i in range(num_classes):
                # Plot class prototype
                axes[1].scatter(
                    prototype_points[i, 0],
                    prototype_points[i, 1],
                    s=200,
                    marker="*",
                    c=[colors[i]],
                    edgecolors="black",
                    linewidths=1,
                    alpha=1.0,
                    zorder=10,
                )

                # Add prototype label
                axes[1].text(
                    prototype_points[i, 0],
                    prototype_points[i, 1] + 0.5,
                    self.label_names[i],
                    color=colors[i],
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

                # Plot embeddings for this class
                mask = labels == i
                if mask.sum() > 0:
                    axes[1].scatter(
                        sample_points[mask, 0],
                        sample_points[mask, 1],
                        s=30,
                        c=[colors[i]],
                        alpha=0.5,
                        label=self.label_names[i],
                    )

            axes[1].set_title("t-SNE: ArcFace Embeddings and Prototypes")
            axes[1].grid(True, alpha=0.3)

            # Add a single legend for the t-SNE plot
            handles, labels = axes[1].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[1].legend(
                by_label.values(),
                by_label.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.1),
                ncol=3,
                fontsize=9,
            )

            # Add epoch information
            if hasattr(self, "current_epoch"):
                fig.suptitle(
                    f"ArcFace Prototype Analysis - Epoch {self.current_epoch}",
                    fontsize=14,
                    y=0.98,
                )

            # Add spacing margin
            plt.tight_layout()

            # Log to wandb
            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.log(
                    {"arcface_embeddings_tsne": wandb.Image(fig)}
                )

            plt.close(fig)

            # Create additional visualization: Angular distribution between samples and their prototypes
            self._plot_angular_distribution()

        except Exception as e:
            print(f"Error in ArcFace prototype visualization: {e}")
            import traceback

            traceback.print_exc()
            plt.close("all")

    def _plot_angular_distribution(self):
        """
        Plot the distribution of angular distances between embeddings and their class prototypes.
        This shows how closely samples align with their class centers.
        """
        try:
            # Get ArcFace centers
            centers = self.arcface_centers.centers.detach().cpu()

            # Ensure val_latents is a tensor
            if isinstance(self.val_latents, list):
                if len(self.val_latents) == 0:
                    print("No latent samples available for visualization")
                    return
                embeddings = torch.cat(self.val_latents)
            else:
                embeddings = self.val_latents

            # Ensure val_labels is a tensor
            if isinstance(self.val_labels, list):
                if len(self.val_labels) == 0:
                    print("No labels available for visualization")
                    return
                labels = torch.cat(self.val_labels)
            else:
                labels = self.val_labels

            # Handle labels shape - squeeze if needed
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            # Normalize embeddings
            centers_norm = F.normalize(centers, p=2, dim=1)
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)

            # Calculate angular distances to matching class centers
            cos_similarities = []
            angular_dists = []
            class_names = []  # Store class names for each sample
            snrs = []

            # Process each embedding individually
            for i in range(len(embeddings_norm)):
                if i >= len(labels):
                    print(
                        f"Warning: More embeddings ({len(embeddings_norm)}) than labels ({len(labels)})"
                    )
                    break

                # Get label as Python integer
                label_idx = labels[i].item()

                # Get center for this class
                class_center = centers_norm[label_idx]
                embedding = embeddings_norm[i]

                # Get the actual modulation name
                class_name = self.label_names[label_idx]
                class_names.append(class_name)

                # Compute cosine similarity
                cos_sim = F.cosine_similarity(
                    embedding.unsqueeze(0), class_center.unsqueeze(0)
                ).item()
                cos_similarities.append(cos_sim)

                # Convert to angle in degrees
                angle = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi
                angular_dists.append(angle)

                # Get SNR if available
                if (
                    hasattr(self, "val_snrs")
                    and isinstance(self.val_snrs, list)
                    and len(self.val_snrs) > 0
                ):
                    if i < len(torch.cat(self.val_snrs)):
                        snrs.append(torch.cat(self.val_snrs)[i].item())
                elif hasattr(self, "val_snrs") and torch.is_tensor(self.val_snrs):
                    if i < len(self.val_snrs):
                        snrs.append(self.val_snrs[i].item())

            # Skip visualization if we don't have any valid data
            if len(angular_dists) == 0:
                print("No valid angular distances computed")
                return

            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: Histogram of angular distances
            axes[0].hist(
                angular_dists, bins=30, alpha=0.7, color="blue", edgecolor="black"
            )
            axes[0].axvline(
                x=np.mean(angular_dists),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(angular_dists):.2f}°",
            )
            axes[0].set_xlabel("Angular Distance to Class Prototype (degrees)")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Distribution of Angular Distances")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Angular distance vs SNR (if available)
            if len(snrs) > 0:
                # Create array for colormap based on class
                unique_classes = np.unique(class_names)
                class_to_idx = {name: i for i, name in enumerate(unique_classes)}
                color_indices = np.array([class_to_idx[name] for name in class_names])

                scatter = axes[1].scatter(
                    snrs, angular_dists, alpha=0.6, c=color_indices, cmap="tab20"
                )

                # Add legend for modulation types
                legend_elements = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=plt.cm.tab20(i / len(unique_classes)),
                        label=name,
                        markersize=8,
                    )
                    for i, name in enumerate(unique_classes)
                ]

                axes[1].legend(
                    handles=legend_elements,
                    title="Modulation",
                    loc="upper right",
                    fontsize=8,
                )

                axes[1].set_xlabel("SNR (dB)")
                axes[1].set_ylabel("Angular Distance (degrees)")
                axes[1].set_title("Angular Distance vs SNR by Modulation Type")
                axes[1].grid(True, alpha=0.3)

                # Fit and plot trend line
                if len(snrs) > 2:  # Need at least 3 points for meaningful trend
                    z = np.polyfit(snrs, angular_dists, 1)
                    p = np.poly1d(z)
                    snr_range = np.linspace(min(snrs), max(snrs), 100)
                    axes[1].plot(
                        snr_range,
                        p(snr_range),
                        "r--",
                        alpha=0.8,
                        label=f"Trend: {z[0]:.4f} deg/dB",
                    )
                    axes[1].legend(
                        handles=legend_elements
                        + [
                            plt.Line2D(
                                [0],
                                [0],
                                linestyle="--",
                                color="r",
                                label=f"Trend: {z[0]:.4f} deg/dB",
                            )
                        ],
                        title="Modulation",
                        loc="upper right",
                        fontsize=8,
                    )
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "No SNR data available",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                )

            plt.tight_layout()

            # Log to wandb
            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.log(
                    {"arcface_angular_distribution": wandb.Image(fig)}
                )

            plt.close(fig)

        except Exception as e:
            print(f"Error in angular distribution plot: {e}")
            import traceback

            traceback.print_exc()
            plt.close("all")

    def configure_optimizers(self):
        # Single optimizer for all components
        optimizer = torch.optim.AdamW(
            [
                {"params": self.unet.parameters()},
                {
                    "params": self.rfnet.parameters(),
                    "lr": self.learning_rate * 2,
                },  # Higher LR for classifier
            ],
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.1,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

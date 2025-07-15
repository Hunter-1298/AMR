import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
from torch.optim import AdamW
from typing import Dict, Tuple, Optional, List
import wandb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import math


class DDPMScheduler(nn.Module):
    """Standard DDPM scheduler for AWGN noise"""

    def __init__(
        self,
        n_steps: int = 100,
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
        unet,
        classifier,
        signal_length: int = 1024,
        learning_rate: float = 1e-4,
        num_diffusion_steps: int = 100,
        beta_schedule: str = "cosine",
        # Core loss weights
        noise_loss_weight: float = 0.0,
        signal_loss_weight: float = 0.0,
        phase_loss_weight: float = 0.0,
        perceptual_loss_weight: float = 5.0,  # NEW: VGG/perceptual loss weight
        classification_weight: float = 0.0,  # Just for monitoring
        # Training settings
        diffusion_only_epochs: int = 0,
        min_snr_training: float = 0.0,
        max_snr_training: float = 30.0,
        # Classifier guidance
        use_classifier_guidance: bool = False,
        base_guidance_scale: float = 1.0,
        # Perceptual loss settings
        use_perceptual_loss: bool = True,
        perceptual_layers=['conv', 'transformer_input', 'transformer_output'],  # All three
        perceptual_weights: Dict[str, float] = None,
        # Classifier freezing
        freeze_classifier: bool = True,
        # Model settings
        num_classes: int = 3,
        perceptual_loss_type: str = "hierarchical",  # "simple", "hierarchical", "advanced"
        log_loss_types: bool = False,  # Enable detailed loss type logging
        label_names: List[str] = ["QPSK", "8PSK", "16PSK"],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet", "classifier"])
        self.automatic_optimization = False

        # Core components
        self.unet = unet
        self.classifier = classifier
        self.ddpm_scheduler = DDPMScheduler(
            n_steps=num_diffusion_steps, schedule=beta_schedule
        )
        self.perceptual_loss_type = perceptual_loss_type
        self.log_loss_types = log_loss_types

        # Loss weights
        self.noise_loss_weight = noise_loss_weight
        self.signal_loss_weight = signal_loss_weight
        self.phase_loss_weight = phase_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.classification_weight = classification_weight

        # Training settings
        self.diffusion_only_epochs = diffusion_only_epochs
        self.min_snr_training = min_snr_training
        self.max_snr_training = max_snr_training

        # Classifier guidance
        self.use_classifier_guidance = use_classifier_guidance
        self.base_guidance_scale = base_guidance_scale

        # Perceptual loss settings
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_layers = perceptual_layers
        self.perceptual_weights = perceptual_weights or {
            'conv': 0.4,
            'transformer_input': 0.3,
            'transformer_output': 0.3
        }

        # Classifier settings
        self.freeze_classifier = freeze_classifier
        self.num_classes = num_classes
        self.label_names = label_names

        # Freeze classifier if requested
        if self.freeze_classifier:
            self._freeze_classifier()

    def _freeze_classifier(self):
        """Freeze all classifier parameters"""
        for param in self.classifier.parameters():
            param.requires_grad = False
        print(f"✓ Classifier frozen: {sum(p.numel() for p in self.classifier.parameters())} parameters")

    def signal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple MSE loss in complex domain"""
        return F.mse_loss(pred, target)

    def phase_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Phase-aware loss for PSK signals"""
        # Convert to complex
        pred_complex = torch.complex(pred[:, 0], pred[:, 1])
        target_complex = torch.complex(target[:, 0], target[:, 1])

        # Normalize to unit circle (PSK property)
        pred_normalized = pred_complex / (torch.abs(pred_complex) + 1e-8)
        target_normalized = target_complex / (torch.abs(target_complex) + 1e-8)

        # Cosine similarity loss (1 - cos_sim)
        cos_sim = torch.real(pred_normalized * torch.conj(target_normalized))
        return (1 - cos_sim).mean()

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical perceptual loss with different loss functions for different feature levels

        Args:
            pred: [B, 2, T] predicted signal
            target: [B, 2, T] target signal
        """
        if not self.use_perceptual_loss:
            return torch.tensor(0.0, device=pred.device)

        # Access the underlying classifier model
        if hasattr(self.classifier, 'classifier'):
            actual_classifier = self.classifier.classifier
        else:
            actual_classifier = self.classifier

        if not hasattr(actual_classifier, 'get_feature_layers'):
            return self._simple_feature_loss(pred, target)

        # Extract features from predicted signal
        pred_features = actual_classifier.get_feature_layers(pred)

        # Extract features from target signal (no gradients needed)
        with torch.no_grad():
            target_features = actual_classifier.get_feature_layers(target)

        total_loss = 0.0
        total_weight = 0.0

        for layer_name in self.perceptual_layers:
            if layer_name in pred_features and layer_name in target_features:
                pred_feat = pred_features[layer_name]
                target_feat = target_features[layer_name]
                weight = self.perceptual_weights.get(layer_name, 1.0)

                # Apply different loss functions based on layer type
                if layer_name == 'conv':
                    # L1 loss for structural/spatial features (encourages sparsity and sharp edges)
                    loss = F.l1_loss(pred_feat, target_feat)

                elif layer_name == 'transformer_input':
                    # L2 loss for mid-level features (smooth optimization)
                    loss = F.mse_loss(pred_feat, target_feat)

                elif layer_name == 'transformer_output':
                    # Cosine similarity for high-level semantic features
                    # Flatten features to [B, -1] for cosine similarity
                    pred_flat = pred_feat.view(pred_feat.size(0), -1)
                    target_flat = target_feat.view(target_feat.size(0), -1)

                    # Normalize features
                    pred_norm = F.normalize(pred_flat, p=2, dim=-1)
                    target_norm = F.normalize(target_flat, p=2, dim=-1)

                    # Cosine similarity loss (1 - cosine_similarity)
                    cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
                    loss = (1 - cosine_sim).mean()

                else:
                    # Default: L2 loss for any other layers
                    loss = F.mse_loss(pred_feat, target_feat)

                # Log individual layer losses
                self.log(f"perceptual_{layer_name}", loss)

                # Add to total loss
                weighted_loss = weight * loss
                total_loss += weighted_loss
                total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            final_loss = total_loss / total_weight
        else:
            final_loss = torch.tensor(0.0, device=pred.device)

        return final_loss

    def _compute_classifier_guidance(self, signal: torch.Tensor, labels: torch.Tensor,
                                    timesteps: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute classifier guidance in isolated context"""
        try:
            # Enable gradients only for input signal
            signal_guided = signal.clone().detach().requires_grad_(True)

            # Temporarily enable classifier gradients for computation only
            for param in self.classifier.parameters():
                param.requires_grad_(True)

            self.classifier.train()  # Temporarily enable train mode for gradients

            # Forward pass - get only logits, not features
            logits = self.classifier(signal_guided, return_features=False)
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

            # Compute gradients
            grad = torch.autograd.grad(
                outputs=target_log_probs.sum(),
                inputs=signal_guided,
                create_graph=False,
                retain_graph=False,
            )[0]

            # Process guidance
            grad_norm = torch.norm(grad.view(batch_size, -1), dim=1, keepdim=True).unsqueeze(2)
            grad_normalized = grad / (grad_norm + 1e-8)

            timestep_scale = 1.0 - (timesteps.float() / self.ddpm_scheduler.n_steps)
            guidance_scale = self.base_guidance_scale * timestep_scale.unsqueeze(1).unsqueeze(2)

            guidance_adjustment = (guidance_scale * grad_normalized).detach()

        except Exception as e:
            print(f"Guidance computation failed: {e}")
            guidance_adjustment = torch.zeros_like(signal)

        finally:
            # Always restore frozen state
            for param in self.classifier.parameters():
                param.requires_grad_(False)
            self.classifier.eval()

        return guidance_adjustment

    def training_step(self, batch, batch_idx):
        # Get optimizer
        opt_unet = self.optimizers()
        sch_unet = self.lr_schedulers()

        # Unpack batch
        clean_signals, corrupted_signals, labels, snrs = batch
        device = clean_signals.device

        # Filter for training SNR range
        snr_mask = (snrs >= self.min_snr_training) & (snrs <= self.max_snr_training)
        if not snr_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        clean_signals = clean_signals[snr_mask]
        corrupted_signals = corrupted_signals[snr_mask]
        labels = labels[snr_mask]
        snrs = snrs[snr_mask]
        batch_size = clean_signals.shape[0]

        # Sample timesteps and add noise
        timesteps = self.ddpm_scheduler.sample_timesteps(batch_size, device)
        noisy_signals, true_noise = self.ddpm_scheduler.add_noise(clean_signals, timesteps)

        # ===========================
        # FORWARD PASS
        # ===========================

        # Single forward pass with zero conditioning
        zero_condition = torch.zeros_like(clean_signals)
        noise_pred, signal_pred = self.unet(
            noisy_signals,
            timesteps,
            context=zero_condition,
            return_both=True
        )

        # ===========================
        # CLASSIFIER GUIDANCE (if enabled)
        # ===========================

        signal_pred_final = signal_pred

        if self.use_classifier_guidance and self.current_epoch >= self.diffusion_only_epochs:
            guidance_adjustment = self._compute_classifier_guidance(
                signal_pred.detach(),
                labels,
                timesteps,
                batch_size
            )
            signal_pred_final = signal_pred + guidance_adjustment

        # ===========================
        # COMPUTE LOSSES
        # ===========================

        # 1. Noise prediction loss
        noise_loss = F.mse_loss(noise_pred, true_noise)

        # 2. Signal reconstruction loss
        signal_loss = self.signal_loss(signal_pred_final, clean_signals)

        # 3. Phase loss
        phase_loss = self.phase_loss(signal_pred_final, clean_signals)

        # 4. Perceptual loss using classifier features
        perceptual_loss = self.perceptual_loss(signal_pred_final, clean_signals)

        # 5. Classification loss (for monitoring only)
        classification_loss = torch.tensor(0.0, device=device)
        if self.current_epoch >= self.diffusion_only_epochs:
            with torch.no_grad():
                self.classifier.eval()
                class_logits_final = self.classifier(signal_pred, return_features=False)
                classification_loss = F.cross_entropy(class_logits_final, labels)

                preds = class_logits_final.argmax(dim=-1)
                acc = (preds == labels).float().mean()
                self.log("train_acc", acc, prog_bar=True)

        # Total loss
        total_loss = (
            self.noise_loss_weight * noise_loss +
            self.signal_loss_weight * signal_loss +
            self.phase_loss_weight * phase_loss +
            self.perceptual_loss_weight * perceptual_loss +
            self.classification_weight * classification_loss  # Usually 0 for monitoring
        )

        # ===========================
        # OPTIMIZATION
        # ===========================

        opt_unet.zero_grad()
        self.manual_backward(total_loss)

        # Clear any accidental gradients on classifier
        for param in self.classifier.parameters():
            if param.grad is not None:
                param.grad = None

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=0.5)

        opt_unet.step()
        sch_unet.step()

        # ===========================
        # LOGGING
        # ===========================

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_noise_loss", noise_loss)
        self.log("train_signal_loss", signal_loss)
        self.log("train_phase_loss", phase_loss)
        self.log("train_perceptual_loss", perceptual_loss)
        self.log("train_class_loss", classification_loss)

        # Log guidance statistics
        if self.use_classifier_guidance and self.current_epoch >= self.diffusion_only_epochs:
            signal_change = torch.norm((signal_pred_final - signal_pred).view(batch_size, -1), dim=1).mean()
            self.log("train_guidance_change", signal_change)

        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr)

        # Log perceptual loss breakdown by layer
        if self.use_perceptual_loss and len(self.perceptual_layers) > 1:
            with torch.no_grad():
                pred_features = self.classifier.get_feature_layers(signal_pred_final)
                target_features = self.classifier.get_feature_layers(clean_signals)

                for layer_name in self.perceptual_layers:
                    if layer_name in pred_features and layer_name in target_features:
                        layer_loss = F.mse_loss(pred_features[layer_name], target_features[layer_name])
                        self.log(f"train_perceptual_{layer_name}", layer_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        clean_signals, corrupted_signals, labels, snrs = batch
        device = clean_signals.device
        batch_size = clean_signals.shape[0]

        # Sample timesteps and add noise
        timesteps = self.ddpm_scheduler.sample_timesteps(batch_size, device)
        noisy_signals, true_noise = self.ddpm_scheduler.add_noise(clean_signals, timesteps)

        with torch.no_grad():
            # Single forward pass
            zero_condition = torch.zeros_like(clean_signals)
            noise_pred, signal_pred = self.unet(
                noisy_signals,
                timesteps,
                context=zero_condition,
                return_both=True
            )

        # Compute losses
        noise_loss = F.mse_loss(noise_pred, true_noise)
        signal_loss = self.signal_loss(signal_pred, clean_signals)
        phase_loss = self.phase_loss(signal_pred, clean_signals)
        perceptual_loss = self.perceptual_loss(signal_pred, clean_signals)

        classification_loss = torch.tensor(0.0, device=device)
        if self.current_epoch >= self.diffusion_only_epochs:
            class_logits = self.classifier(signal_pred, return_features=False)
            classification_loss = F.cross_entropy(class_logits, labels)

            # Accuracy
            preds = class_logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
            self.log("val_acc", acc, prog_bar=True)

            # Per-SNR accuracy
            snr_ranges = [(-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30)]
            for snr_min, snr_max in snr_ranges:
                mask = (snrs >= snr_min) & (snrs < snr_max)
                if mask.any():
                    snr_acc = (preds[mask] == labels[mask]).float().mean()
                    self.log(f"val_acc_snr_{snr_min}to{snr_max}", snr_acc)

        total_loss = (
            self.noise_loss_weight * noise_loss +
            self.signal_loss_weight * signal_loss +
            self.phase_loss_weight * phase_loss +
            self.perceptual_loss_weight * perceptual_loss +
            self.classification_weight * classification_loss
        )

        # Logging
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_noise_loss", noise_loss)
        self.log("val_signal_loss", signal_loss)
        self.log("val_phase_loss", phase_loss)
        self.log("val_perceptual_loss", perceptual_loss)
        self.log("val_class_loss", classification_loss)

        # Log perceptual loss breakdown
        if self.use_perceptual_loss and len(self.perceptual_layers) > 1:
            pred_features = self.classifier.get_feature_layers(signal_pred)
            target_features = self.classifier.get_feature_layers(clean_signals)

            for layer_name in self.perceptual_layers:
                if layer_name in pred_features and layer_name in target_features:
                    layer_loss = F.mse_loss(pred_features[layer_name], target_features[layer_name])
                    self.log(f"val_perceptual_{layer_name}", layer_loss)

        # Store for visualization
        if batch_idx == 0:
            self.val_data = {
                "clean_signals": clean_signals[:4].detach().cpu(),
                "signal_pred": signal_pred[:4].detach().cpu(),
                "corrupted_signals": corrupted_signals[:4].detach().cpu(),
                "labels": labels[:4].detach().cpu(),
                "snrs": snrs[:4].detach().cpu(),
            }

        return total_loss

    def configure_optimizers(self):
        # Verify only UNet parameters are being optimized
        unet_params = list(self.unet.parameters())
        trainable_params = [p for p in unet_params if p.requires_grad]

        print(f"Optimizing {len(trainable_params)} UNet parameters")
        print(f"Total UNet parameters: {sum(p.numel() for p in trainable_params)}")

        optimizer = AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20000,
            T_mult=2,
            eta_min=self.hparams.learning_rate * 0.2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # Keep existing visualization methods...
    def on_validation_epoch_end(self):
        """Create visualizations at the end of validation"""
        if hasattr(self, "val_data") and self.val_data is not None:
            self._create_combined_visualization()

    def _create_combined_visualization(self):
        """Create visualization showing original → noisy → denoised progression"""
        try:
            if self.current_epoch < self.diffusion_only_epochs:
                # Diffusion only: 3 rows (original, noisy, denoised)
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                title_suffix = "Diffusion Only"
            else:
                # Diffusion + classification: 3 rows (original, noisy, denoised)
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                title_suffix = "Diffusion + Classification"

            # ==========================================
            # CONSTELLATION PLOTS - SIGNAL PROGRESSION
            # ==========================================
            for i in range(min(4, len(self.val_data["clean_signals"]))):
                original_clean = self.val_data["clean_signals"][i]  # Original clean signal
                noisy_signal = self.val_data["corrupted_signals"][i]  # Noisy version
                denoised_signal = self.val_data["signal_pred"][i]     # Our denoised output
                true_label = self.val_data["labels"][i].item()
                snr = self.val_data["snrs"][i].item()

                # Convert to complex for constellation plots
                original_complex = torch.complex(original_clean[0], original_clean[1])
                noisy_complex = torch.complex(noisy_signal[0], noisy_signal[1])
                denoised_complex = torch.complex(denoised_signal[0], denoised_signal[1])

                # Row 1: Original clean signals
                axes[0, i].scatter(
                    original_complex.real.numpy(),
                    original_complex.imag.numpy(),
                    alpha=0.8, s=20, c="green"
                )
                axes[0, i].set_title(f"Original {self.label_names[true_label]}\n(Ground Truth)")
                axes[0, i].set_xlim(-2.5, 2.5)
                axes[0, i].set_ylim(-2.5, 2.5)
                axes[0, i].grid(True, alpha=0.3)
                axes[0, i].set_aspect("equal")
                axes[0, i].set_xlabel("In-Phase")
                axes[0, i].set_ylabel("Quadrature")

                # Row 2: Noisy signals (what we receive)
                axes[1, i].scatter(
                    noisy_complex.real.numpy(),
                    noisy_complex.imag.numpy(),
                    alpha=0.6, s=20, c="red"
                )

                # Compute noise level
                noise_mse = F.mse_loss(noisy_signal, original_clean).item()

                axes[1, i].set_title(f"Noisy Signal\nSNR: {snr:.1f} dB | MSE: {noise_mse:.4f}")
                axes[1, i].set_xlim(-2.5, 2.5)
                axes[1, i].set_ylim(-2.5, 2.5)
                axes[1, i].grid(True, alpha=0.3)
                axes[1, i].set_aspect("equal")
                axes[1, i].set_xlabel("In-Phase")
                axes[1, i].set_ylabel("Quadrature")

                # Row 3: Denoised signals (our output)
                axes[2, i].scatter(
                    denoised_complex.real.numpy(),
                    denoised_complex.imag.numpy(),
                    alpha=0.6, s=20, c="blue"
                )

                # Compute denoising performance metrics
                denoised_mse = F.mse_loss(denoised_signal, original_clean).item()
                improvement = ((noise_mse - denoised_mse) / noise_mse * 100) if noise_mse > 0 else 0

                # Estimate SNR improvement
                original_power = torch.mean(torch.abs(original_complex) ** 2).item()
                noise_power_before = noise_mse
                noise_power_after = denoised_mse

                snr_before = 10 * torch.log10(torch.tensor(original_power / (noise_power_before + 1e-10)))
                snr_after = 10 * torch.log10(torch.tensor(original_power / (noise_power_after + 1e-10)))
                snr_improvement = snr_after - snr_before

                axes[2, i].set_title(f"Denoised Signal\nMSE: {denoised_mse:.4f} | ↑{improvement:.1f}%\nSNR Gain: +{snr_improvement:.1f} dB")
                axes[2, i].set_xlim(-2.5, 2.5)
                axes[2, i].set_ylim(-2.5, 2.5)
                axes[2, i].grid(True, alpha=0.3)
                axes[2, i].set_aspect("equal")
                axes[2, i].set_xlabel("In-Phase")
                axes[2, i].set_ylabel("Quadrature")

            # Add row labels with arrows showing progression
            fig.text(0.02, 0.83, 'Original\nClean', ha='center', va='center',
                    rotation=90, fontsize=12, fontweight='bold', color='green')
            fig.text(0.02, 0.5, 'Received\nNoisy', ha='center', va='center',
                    rotation=90, fontsize=12, fontweight='bold', color='red')
            fig.text(0.02, 0.17, 'Denoised\nOutput', ha='center', va='center',
                    rotation=90, fontsize=12, fontweight='bold', color='blue')

            plt.suptitle(f"Signal Processing Pipeline - Epoch {self.current_epoch} ({title_suffix})",
                        fontsize=16, y=0.95)
            plt.tight_layout()
            plt.subplots_adjust(left=0.08, top=0.9, bottom=0.1)  # Make room for labels and arrows

            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.log({"denoising_pipeline": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()

    def _create_snr_performance_subplot(self, ax):
        """Create SNR vs accuracy subplot for denoised signal classification"""
        try:
            snr_ranges = [(-20, -15), (-15, -10), (-10, -5), (-5, 0),
                        (0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
            snr_centers = [(low + high) / 2 for low, high in snr_ranges]

            # Try to get accuracies from logged metrics
            accuracies = []
            trainer_logs = self.trainer.logged_metrics if hasattr(self.trainer, 'logged_metrics') else {}

            for snr_min, snr_max in snr_ranges:
                key = f"val_acc_snr_{snr_min}to{snr_max}"
                if key in trainer_logs:
                    acc = trainer_logs[key].item() if hasattr(trainer_logs[key], 'item') else float(trainer_logs[key])
                    accuracies.append(acc)
                else:
                    accuracies.append(0.0)  # Default if no data

            # Plot denoised signal classification performance
            ax.plot(snr_centers, accuracies, 'b-o', linewidth=3, markersize=8,
                    label="Denoised Signal Classification", color='blue')

            # Add reference lines
            ax.axhline(y=1/3, color='gray', linestyle=':', alpha=0.7, label="Chance Level (33%)")
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label="Good Performance (80%)")

            # Highlight different SNR regions
            ax.axvspan(-20, -10, alpha=0.1, color='red', label='Challenging SNR')
            ax.axvspan(-10, 0, alpha=0.1, color='orange', label='Moderate SNR')
            ax.axvspan(0, 20, alpha=0.1, color='green', label='Good SNR')

            ax.set_xlabel("SNR (dB)", fontsize=12)
            ax.set_ylabel("Classification Accuracy", fontsize=12)
            ax.set_title(f"Denoised Signal Classification vs SNR - Epoch {self.current_epoch}", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='lower right')
            ax.set_ylim(0, 1.05)
            ax.set_xlim(-22, 32)

            # Add performance annotations
            if len(accuracies) > 0:
                max_acc = max(accuracies)
                avg_acc = sum(accuracies) / len(accuracies)
                ax.text(0.02, 0.98, f"Max Acc: {max_acc:.3f}\nAvg Acc: {avg_acc:.3f}",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        except Exception as e:
            print(f"Error in SNR performance subplot: {e}")
            # Fallback: create empty plot with error message
            ax.text(0.5, 0.5, f"SNR Plot Error: {str(e)}", transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
            ax.set_title("SNR Performance Analysis (Error)")



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

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
        n_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
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
            betas = torch.clamp(betas, 0, 0.999)
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
        learning_rate: float = 3e-4,
        num_diffusion_steps: int = 1000,
        beta_schedule: str = "cosine",
        # Loss weights
        noise_loss_weight: float = 1.0,
        signal_loss_weight: float = 1.0,
        classification_weight: float = 100.0,  # Higher weight since classifier is pretrained
        # Training schedule
        diffusion_only_epochs: int = 0,
        min_snr_training: float = 0.0,  # Only train on SNR >= 0 dB
        # Classifier settings
        num_classes: int = 3,
        label_names: List[str] = ["QPSK", "8PSK", "16PSK"],
        # Frozen classifier settings
        keep_classifier_frozen: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet", "classifier"])
        self.automatic_optimization = False  # Manual optimization

        # Core components
        self.unet = unet
        self.classifier = classifier
        self.ddpm_scheduler = DDPMScheduler(
            n_steps=num_diffusion_steps, schedule=beta_schedule
        )

        # Loss functions
        self.noise_mse_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()

        # Loss weights
        self.noise_loss_weight = noise_loss_weight
        self.signal_loss_weight = signal_loss_weight
        self.classification_weight = classification_weight

        # Training schedule
        self.diffusion_only_epochs = diffusion_only_epochs
        self.min_snr_training = min_snr_training

        # Classifier settings
        self.num_classes = num_classes
        self.label_names = label_names
        self.keep_classifier_frozen = keep_classifier_frozen

        # Freeze classifier immediately if requested
        if self.keep_classifier_frozen:
            self._freeze_classifier()

    def _freeze_classifier(self):
        """Freeze all classifier parameters"""
        for param in self.classifier.parameters():
            param.requires_grad = False
        print("✓ Classifier parameters frozen - weights will not update")

    def _unfreeze_classifier(self):
        """Unfreeze classifier parameters (for future use)"""
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.keep_classifier_frozen = False
        print("✓ Classifier parameters unfrozen - weights will update")

    def complex_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss in complex domain"""
        pred_complex = torch.complex(pred[:, 0], pred[:, 1])
        target_complex = torch.complex(target[:, 0], target[:, 1])
        return torch.mean(torch.abs(pred_complex - target_complex) ** 2)

    def training_step(self, batch, batch_idx):
        # Get UNet optimizer only (classifier is frozen)
        if self.keep_classifier_frozen:
            opt_unet = self.optimizers()
            sch_unet = self.lr_schedulers()
        else:
            # Future: when classifier is unfrozen
            opt_unet, opt_classifier = self.optimizers()
            sch_unet, sch_classifier = self.lr_schedulers()

        # Unpack batch
        clean_signals, corrupted_signals, labels, snrs = batch

        batch_size = clean_signals.shape[0]
        device = clean_signals.device

        # Filter for SNR >= min_snr_training
        snr_mask = snrs >= self.min_snr_training
        if not snr_mask.any():
            # Skip this batch if no samples meet SNR criteria
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Filter batch
        clean_signals = clean_signals[snr_mask]
        corrupted_signals = corrupted_signals[snr_mask]
        labels = labels[snr_mask]
        snrs = snrs[snr_mask]
        filtered_batch_size = clean_signals.shape[0]

        # Sample random timesteps
        timesteps = self.ddpm_scheduler.sample_timesteps(filtered_batch_size, device)

        # Add noise to clean signals
        noisy_signals, true_noise = self.ddpm_scheduler.add_noise(clean_signals, timesteps)

        # =========================
        # DIFFUSION FORWARD PASSES
        # =========================

        # PASS 1: Forward with zero conditioning
        zero_condition = torch.zeros_like(clean_signals)
        noise_pred_1, signal_pred_1 = self.unet(
            noisy_signals,
            timesteps,
            context=zero_condition,
            return_both=True
        )

        # PASS 2: Forward with signal prediction as conditioning
        noise_pred_2, signal_pred_2 = self.unet(
            noisy_signals,
            timesteps,
            context=signal_pred_1,  # Gradients flow through
            return_both=True
        )

        # =========================
        # COMPUTE ALL LOSSES
        # =========================

        # Diffusion losses (from second pass)
        noise_loss = self.noise_mse_loss(noise_pred_2, true_noise)
        signal_loss = self.complex_mse_loss(signal_pred_2, clean_signals)

        # Optional: small first pass loss for better self-conditioning
        first_pass_loss = 0.1 * self.complex_mse_loss(signal_pred_1, clean_signals)

        # Combined diffusion loss
        diffusion_loss = (
            self.noise_loss_weight * noise_loss +
            self.signal_loss_weight * signal_loss +
            first_pass_loss
        )

        # Classification loss (starts after diffusion-only epochs)
        classification_loss = torch.tensor(0.0, device=device)
        classification_metrics = {}

        if self.current_epoch >= self.diffusion_only_epochs:
            # Set classifier to train mode for proper forward pass
            # (needed for dropout/batchnorm even when frozen)
            self.classifier.train()

            # Forward through frozen classifier - gradients still flow to UNet
            class_logits = self.classifier(signal_pred_2)  # No detach - end-to-end gradients
            classification_loss = self.classification_loss(class_logits, labels)

            # Compute classification metrics
            with torch.no_grad():
                _, preds = torch.max(class_logits, 1)
                acc = (preds == labels).float().mean()
                classification_metrics = {
                    "train_class_acc": acc,
                    "train_classification_loss": classification_loss
                }

        # =========================
        # SINGLE COMBINED LOSS
        # =========================

        total_loss = diffusion_loss + self.classification_weight * classification_loss

        # =========================
        # OPTIMIZATION (UNet only)
        # =========================

        # Zero gradients - only UNet since classifier is frozen
        if self.keep_classifier_frozen:
            opt_unet.zero_grad()
        else:
            opt_unet.zero_grad()
            opt_classifier.zero_grad()

        # Single backward pass through entire graph
        self.manual_backward(total_loss)

        # Clip gradients - only UNet matters since classifier is frozen
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        if not self.keep_classifier_frozen:
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)

        # Step optimizers - only UNet since classifier is frozen
        opt_unet.step()
        sch_unet.step()

        if not self.keep_classifier_frozen:
            opt_classifier.step()
            sch_classifier.step()

        # =========================
        # LOGGING
        # =========================

        # Log main metrics
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_diffusion_loss", diffusion_loss)
        self.log("train_noise_loss", noise_loss)
        self.log("train_signal_loss", signal_loss)
        self.log("train_first_pass_loss", first_pass_loss)
        self.log("train_filtered_batch_size", float(filtered_batch_size))

        # Log classification metrics if in classification phase
        for metric_name, metric_value in classification_metrics.items():
            self.log(metric_name, metric_value, prog_bar=(metric_name == "train_class_acc"))

        # Log training phase and classifier status
        if self.current_epoch < self.diffusion_only_epochs:
            self.log("train_phase", 1.0)  # Diffusion only
        else:
            self.log("train_phase", 2.0)  # Diffusion + Classification

        self.log("classifier_frozen", float(self.keep_classifier_frozen))

        # Log loss components for debugging
        if self.current_epoch >= self.diffusion_only_epochs:
            self.log("train_classification_weight", self.classification_weight)
            self.log("train_weighted_class_loss", self.classification_weight * classification_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch (validate on ALL samples, regardless of SNR)
        clean_signals, corrupted_signals, labels, snrs = batch

        batch_size = clean_signals.shape[0]
        device = clean_signals.device

        # Sample random timesteps
        timesteps = self.ddpm_scheduler.sample_timesteps(batch_size, device)

        # Add noise
        noisy_signals, true_noise = self.ddpm_scheduler.add_noise(clean_signals, timesteps)

        # =========================
        # DIFFUSION VALIDATION
        # =========================

        # PASS 1: Zero conditioning
        zero_condition = torch.zeros_like(clean_signals)
        with torch.no_grad():
            noise_pred_1, signal_pred_1 = self.unet(
                noisy_signals,
                timesteps,
                context=zero_condition,
                return_both=True
            )

        # PASS 2: Self-conditioning
        with torch.no_grad():
            noise_pred_2, signal_pred_2 = self.unet(
                noisy_signals,
                timesteps,
                context=signal_pred_1,
                return_both=True
            )

        # Compute diffusion losses (from second pass)
        noise_loss = self.noise_mse_loss(noise_pred_2, true_noise)
        signal_loss = self.complex_mse_loss(signal_pred_2, clean_signals)
        diffusion_loss = self.noise_loss_weight * noise_loss + self.signal_loss_weight * signal_loss

        # Improvement metrics
        signal_loss_1 = self.complex_mse_loss(signal_pred_1, clean_signals)
        signal_improvement = signal_loss_1 - signal_loss

        # =========================
        # CLASSIFICATION VALIDATION
        # =========================

        classification_loss = torch.tensor(0.0, device=device)
        val_class_acc = torch.tensor(0.0, device=device)
        preds = None
        class_logits = None

        if self.current_epoch >= self.diffusion_only_epochs:
            # Set classifier to eval mode for validation
            self.classifier.eval()

            with torch.no_grad():
                # Classify denoised signals
                class_logits = self.classifier(signal_pred_2)
                classification_loss = self.classification_loss(class_logits, labels)

                # Classification metrics
                _, preds = torch.max(class_logits, 1)
                val_class_acc = (preds == labels).float().mean()

                # Per-SNR accuracy analysis
                snr_ranges = [
                    (-20, -15), (-15, -10), (-10, -5), (-5, 0),
                    (0, 5), (5, 10), (10, 15), (15, 20), (20, 30)
                ]
                for snr_min, snr_max in snr_ranges:
                    snr_mask = (snrs >= snr_min) & (snrs < snr_max)
                    if snr_mask.any():
                        snr_acc = (preds[snr_mask] == labels[snr_mask]).float().mean()
                        self.log(f"val_acc_snr_{snr_min}to{snr_max}", snr_acc)
                        # Also log count for debugging
                        self.log(f"val_count_snr_{snr_min}to{snr_max}", float(snr_mask.sum()))

        total_loss = diffusion_loss + self.classification_weight * classification_loss

        # =========================
        # VALIDATION LOGGING
        # =========================

        # Main metrics
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_noise_loss", noise_loss)
        self.log("val_signal_loss", signal_loss)
        self.log("val_signal_improvement", signal_improvement)

        if self.current_epoch >= self.diffusion_only_epochs:
            self.log("val_class_loss", classification_loss)
            self.log("val_class_acc", val_class_acc, prog_bar=True)

        # Store data for visualization (first batch only)
        if batch_idx == 0:
            self.val_data = {
                "clean_signals": clean_signals[:4].detach().cpu(),
                "noisy_signals": noisy_signals[:4].detach().cpu(),
                "signal_pred_1": signal_pred_1[:4].detach().cpu(),
                "signal_pred_2": signal_pred_2[:4].detach().cpu(),
                "labels": labels[:4].detach().cpu(),
                "snrs": snrs[:4].detach().cpu(),
                "timesteps": timesteps[:4].detach().cpu(),
            }

            if self.current_epoch >= self.diffusion_only_epochs and preds is not None:
                self.val_data["predictions"] = preds[:4].detach().cpu()
                self.val_data["class_logits"] = class_logits[:4].detach().cpu()

        return total_loss

    def configure_optimizers(self):
        """Configure optimizer only for UNet since classifier is frozen"""

        if self.keep_classifier_frozen:
            # Only optimize UNet parameters
            optimizer = AdamW(
                self.unet.parameters(),  # Only UNet params
                lr=self.hparams.learning_rate,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
            )

            # Scheduler for UNet only
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

        else:
            # Future: when classifier is unfrozen, return both optimizers
            opt_unet = AdamW(self.unet.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
            opt_classifier = AdamW(self.classifier.parameters(), lr=1e-4, weight_decay=1e-4)  # Lower LR for pretrained

            sch_unet = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_unet, T_0=7988, T_mult=1, eta_min=self.hparams.learning_rate * 0.01)
            sch_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_classifier, T_0=7988, T_mult=1, eta_min=1e-6)

            return [opt_unet, opt_classifier], [
                {"scheduler": sch_unet, "interval": "step", "frequency": 1},
                {"scheduler": sch_classifier, "interval": "step", "frequency": 1}
            ]

    def on_validation_epoch_end(self):
        """Create visualizations"""
        if hasattr(self, "val_data") and self.val_data is not None:
            self._create_combined_visualization()

    def _create_combined_visualization(self):
        """Create visualization showing both denoising and classification with SNR analysis"""
        try:
            if self.current_epoch < self.diffusion_only_epochs:
                # Only show denoising
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                title_suffix = "Diffusion Only"
                show_classification = False
            else:
                # Show denoising + classification + SNR analysis
                fig = plt.figure(figsize=(20, 16))
                gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.2], width_ratios=[1, 1, 1, 1, 1, 1])

                # Top 2 rows: constellation plots (4 examples each)
                axes = []
                for row in range(2):
                    row_axes = []
                    for col in range(4):
                        ax = fig.add_subplot(gs[row, col])
                        row_axes.append(ax)
                    axes.append(row_axes)

                # Bottom row: SNR performance plot (spans full width)
                snr_ax = fig.add_subplot(gs[2, :])

                title_suffix = "Diffusion + Classification"
                show_classification = True

            # ==========================================
            # CONSTELLATION PLOTS (same as before)
            # ==========================================
            for i in range(min(4, len(self.val_data["clean_signals"]))):
                clean = self.val_data["clean_signals"][i]
                signal_pred_1 = self.val_data["signal_pred_1"][i]
                signal_pred_2 = self.val_data["signal_pred_2"][i]
                true_label = self.val_data["labels"][i].item()
                snr = self.val_data["snrs"][i].item()

                # Convert to complex for constellation plots
                clean_complex = torch.complex(clean[0], clean[1])
                pred1_complex = torch.complex(signal_pred_1[0], signal_pred_1[1])
                pred2_complex = torch.complex(signal_pred_2[0], signal_pred_2[1])

                if show_classification:
                    # Row 1: Clean signal
                    axes[0][i].scatter(
                        clean_complex.real.numpy(),
                        clean_complex.imag.numpy(),
                        alpha=0.6, s=20, c="green"
                    )
                    axes[0][i].set_title(f"Clean {self.label_names[true_label]}\nSNR: {snr:.1f} dB")
                    axes[0][i].set_xlim(-2, 2)
                    axes[0][i].set_ylim(-2, 2)
                    axes[0][i].grid(True, alpha=0.3)
                    axes[0][i].set_aspect("equal")

                    # Row 2: Denoised + Classification
                    axes[1][i].scatter(
                        pred2_complex.real.numpy(),
                        pred2_complex.imag.numpy(),
                        alpha=0.6, s=20, c="blue"
                    )

                    if "predictions" in self.val_data:
                        pred_label = self.val_data["predictions"][i].item()
                        correct = true_label == pred_label
                        status = "✓" if correct else "✗"
                        color = "green" if correct else "red"

                        mse2 = F.mse_loss(signal_pred_2, clean).item()
                        axes[1][i].set_title(
                            f"{status} Pred: {self.label_names[pred_label]}\n"
                            f"MSE: {mse2:.4f}",
                            color=color
                        )
                    else:
                        mse2 = F.mse_loss(signal_pred_2, clean).item()
                        axes[1][i].set_title(f"Denoised (MSE: {mse2:.4f})")

                    axes[1][i].set_xlim(-2, 2)
                    axes[1][i].set_ylim(-2, 2)
                    axes[1][i].grid(True, alpha=0.3)
                    axes[1][i].set_aspect("equal")
                else:
                    # Original 3-row layout for diffusion only
                    # Row 1: Clean signal
                    axes[0, i].scatter(
                        clean_complex.real.numpy(),
                        clean_complex.imag.numpy(),
                        alpha=0.6, s=20, c="green"
                    )
                    axes[0, i].set_title(f"Clean {self.label_names[true_label]}\nSNR: {snr:.1f} dB")
                    axes[0, i].set_xlim(-2, 2)
                    axes[0, i].set_ylim(-2, 2)
                    axes[0, i].grid(True, alpha=0.3)
                    axes[0, i].set_aspect("equal")

                    # Rows 2-3: First and second pass predictions
                    axes[1, i].scatter(pred1_complex.real.numpy(), pred1_complex.imag.numpy(),
                                    alpha=0.6, s=20, c="orange")
                    mse1 = F.mse_loss(signal_pred_1, clean).item()
                    axes[1, i].set_title(f"Pass 1 (MSE: {mse1:.4f})")
                    axes[1, i].set_xlim(-2, 2)
                    axes[1, i].set_ylim(-2, 2)
                    axes[1, i].grid(True, alpha=0.3)
                    axes[1, i].set_aspect("equal")

                    axes[2, i].scatter(pred2_complex.real.numpy(), pred2_complex.imag.numpy(),
                                    alpha=0.6, s=20, c="blue")
                    mse2 = F.mse_loss(signal_pred_2, clean).item()
                    improvement = ((mse1 - mse2) / mse1 * 100) if mse1 > 0 else 0
                    axes[2, i].set_title(f"Pass 2 (MSE: {mse2:.4f}, +{improvement:.1f}%)")
                    axes[2, i].set_xlim(-2, 2)
                    axes[2, i].set_ylim(-2, 2)
                    axes[2, i].grid(True, alpha=0.3)
                    axes[2, i].set_aspect("equal")

            # ==========================================
            # SNR PERFORMANCE PLOT (NEW!)
            # ==========================================
            if show_classification and "predictions" in self.val_data:
                self._create_snr_performance_subplot(snr_ax)

            plt.suptitle(f"Training Results - Epoch {self.current_epoch} ({title_suffix})", fontsize=16)
            plt.tight_layout()

            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.log({"training_progress": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in visualization: {e}")

    def _create_snr_performance_subplot(self, ax):
        """Create SNR vs accuracy subplot for denoised signal classification"""
        try:
            # Collect all validation data from this epoch
            # Note: This assumes you have stored comprehensive validation results
            # You might need to modify validation_step to collect all results

            # For now, we'll use the logged metrics from the current validation epoch
            # In a full implementation, you'd want to collect all val data

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

    def forward(self, x):
        return self.classifier(x)

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
    for RadioML 2018.01A.
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

        # Stage C: Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(trans_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2, 1024]  real and imag as two channels
        Returns:
            logits: [B, num_classes]
        """
        # Conv feature extraction
        y = self.conv_extractor(x)  # [B, C, 128]
        # Prepare for transformer: -> [B, 128, C]
        y = y.permute(0, 2, 1)
        # Project to transformer dimension
        y = self.input_proj(y)  # [B, 128, trans_dim]
        # Add positional encodings
        y = self.pos_enc(y)
        # Transformer encoder
        y = self.transformer(y)  # [B, 128, trans_dim]
        # Back to [B, trans_dim, 128]
        y = y.permute(0, 2, 1)
        # Pool over time
        y = self.pool(y).squeeze(-1)  # [B, trans_dim]
        # Classifier
        logits = self.classifier(y)  # [B, num_classes]
        return logits

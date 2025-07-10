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


class SimplifiedReconstructionLoss(nn.Module):
    """Complex MSE + Phase Angle Loss + Complex Correlation Loss + Amplitude Penalty for phase preservation"""

    def __init__(
        self,
        complex_mse_weight=1.0,
        phase_weight=2.0,
        corr_weight=1.0,
        amp_weight=0.3,
        align_global_phase=False,
    ):
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

        total = (
            self.complex_mse_weight * mse
            + self.phase_weight * phase
            + self.corr_weight * corr
            + self.amp_weight * amp
        )

        return total, {
            "total": total.item(),
            "mse": mse.item(),
            "phase": phase.item(),
            "corr": corr.item(),
            "amp": amp.item(),
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

        phase_offset = torch.angle(
            torch.mean(pred_c * torch.conj(target_c), dim=1, keepdim=True)
        )
        phase_corr = torch.exp(-1j * phase_offset)

        aligned = pred_c * phase_corr
        return torch.stack([aligned.real, aligned.imag], dim=1)


class PSKDenoisingClassifier(L.LightningModule):
    """PSK Denoising Classifier with Curriculum Learning and Joint Guidance"""

    def __init__(
        self,
        unet,
        signal_length: int = 1024,
        learning_rate: float = 1e-3,
        num_diffusion_steps: int = 1000,
        beta_schedule: str = "cosine",
        sample_rate: float = 1e6,
        # Loss weights
        complex_mse_weight: float = 1.0,
        phase_weight: float = 2.5,
        corr_weight: float = 1.5,
        amp_weight: float = 0.3,
        align_global_phase: bool = True,
        classification_weight: float = 2.0,
        # Curriculum learning parameters
        classifier_only_epochs: int = 5,
        curriculum_schedule: Optional[Dict] = None,
        # Feature matching
        classifier_feature_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet"])
        self.automatic_optimization = False

        self.label_names = ["QPSK", "8PSK", "16PSK"]
        self.num_classes = 3

        # Core components
        self.unet = unet
        self.ddpm_scheduler = DDPMScheduler(
            n_steps=num_diffusion_steps, schedule=beta_schedule
        )

        # Loss functions
        self.reconstruction_loss = SimplifiedReconstructionLoss(
            complex_mse_weight=complex_mse_weight,
            phase_weight=phase_weight,
            corr_weight=corr_weight,
            amp_weight=amp_weight,
            align_global_phase=align_global_phase,
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Classifier
        self.classifier = HybridConvTransformer(
            in_ch=2,
            num_classes=self.num_classes,
        )

        # Curriculum learning
        self.classifier_only_epochs = classifier_only_epochs
        if curriculum_schedule is None:
            self.curriculum_schedule = {
                0: 0,  # Epochs 0-9: SNR >= 10 dB (classifier only)
                5: -10,  # Epochs 15-19: SNR >= 0 dB
                10: -20,  # Epochs 20-24: SNR >= -5 dB
            }
        else:
            self.curriculum_schedule = curriculum_schedule

        # Weights
        self.classification_weight = classification_weight
        self.classifier_feature_weight = classifier_feature_weight

    def get_current_min_snr(self) -> float:
        """Get minimum SNR for current epoch based on curriculum"""
        current_epoch = self.current_epoch
        min_snr = 10
        for epoch_threshold, snr_threshold in sorted(self.curriculum_schedule.items()):
            if current_epoch >= epoch_threshold:
                min_snr = snr_threshold
            else:
                break
        return min_snr

    def snr_to_timestep(self, snr_db: torch.Tensor) -> torch.Tensor:
        """Map SNR values to DDPM timesteps"""
        timesteps = torch.zeros_like(snr_db, dtype=torch.float32)

        # SNR >= 10 dB: timesteps 0-20 (minimal denoising)
        mask_very_clean = snr_db >= 10
        if mask_very_clean.any():
            timesteps[mask_very_clean] = 20 * (1 - (snr_db[mask_very_clean] - 10) / 20)

        # SNR [0, 10) dB: timesteps 20-200 (light denoising)
        mask_moderate = (snr_db >= 0) & (snr_db < 10)
        if mask_moderate.any():
            timesteps[mask_moderate] = 20 + 180 * (10 - snr_db[mask_moderate]) / 10

        # SNR [-10, 0) dB: timesteps 200-600 (moderate denoising)
        mask_noisy = (snr_db >= -10) & (snr_db < 0)
        if mask_noisy.any():
            timesteps[mask_noisy] = 200 + 400 * (-snr_db[mask_noisy]) / 10

        # SNR [-20, -10) dB: timesteps 600-950 (heavy denoising)
        mask_very_noisy = (snr_db >= -20) & (snr_db < -10)
        if mask_very_noisy.any():
            normalized = (snr_db[mask_very_noisy] + 20) / 10
            timesteps[mask_very_noisy] = 950 - 350 * normalized

        # SNR < -20 dB: timestep 950 (maximum denoising)
        mask_extreme = snr_db < -20
        timesteps[mask_extreme] = 950

        return timesteps.round().long().clamp(0, self.ddpm_scheduler.n_steps - 1)

    def get_classifier_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from classifier for feature matching"""
        y = self.classifier.conv_extractor(x)
        y = y.permute(0, 2, 1)
        y = self.classifier.input_proj(y)
        y = self.classifier.pos_enc(y)
        y = self.classifier.transformer(y)
        y = y.permute(0, 2, 1)
        y = self.classifier.pool(y).squeeze(-1)
        return y

    def get_current_min_snr_at_epoch(self, epoch: int) -> float:
        """Helper to get min SNR at specific epoch"""
        min_snr = 10
        for epoch_threshold, snr_threshold in sorted(self.curriculum_schedule.items()):
            if epoch >= epoch_threshold:
                min_snr = snr_threshold
            else:
                break
        return min_snr

    def adaptive_denoise_signal(
        self, corrupted_signal: torch.Tensor, snr_db: torch.Tensor, num_steps: int = 10
    ) -> torch.Tensor:
        """Simple SNR-aware denoising for inference"""
        self.unet.eval()

        batch_size = corrupted_signal.shape[0]
        device = corrupted_signal.device

        starting_timesteps = self.snr_to_timestep(snr_db)
        clean_mask = starting_timesteps < 10
        if clean_mask.all():
            return corrupted_signal

        current_signal = corrupted_signal.clone()

        with torch.no_grad():
            for step in range(num_steps):
                process_mask = starting_timesteps > 10
                if not process_mask.any():
                    break

                progress = step / (num_steps - 1) if num_steps > 1 else 1.0
                current_timesteps = torch.zeros(
                    batch_size, device=device, dtype=torch.long
                )

                for i in range(batch_size):
                    if process_mask[i]:
                        start_t = starting_timesteps[i].item()
                        decay_rate = 2.0
                        current_timesteps[i] = int(
                            start_t * math.exp(-decay_rate * progress)
                        )
                    else:
                        current_timesteps[i] = 0

                if process_mask.any():
                    predicted_clean = self.unet(current_signal, current_timesteps)

                    base_alpha = 0.05
                    timestep_factor = (
                        current_timesteps.float() / self.ddpm_scheduler.n_steps
                    )
                    alpha = base_alpha + 0.15 * timestep_factor
                    alpha = alpha.view(-1, 1, 1)

                    process_mask_expanded = process_mask.view(-1, 1, 1)
                    current_signal = torch.where(
                        process_mask_expanded,
                        (1 - alpha) * current_signal + alpha * predicted_clean,
                        current_signal,
                    )

        return current_signal

    def training_step(self, batch, batch_idx):
        opt_denoise = self.optimizers()
        scheduler_denoise = self.lr_schedulers()

        synced_signals, corrupted_signals, labels, snrs = batch
        device = synced_signals.device

        # Get current minimum SNR based on curriculum
        min_snr = self.get_current_min_snr()

        # Filter batch based on curriculum
        curriculum_mask = snrs >= min_snr
        if not curriculum_mask.any():
            return torch.tensor(0.0, device=device)

        # Filter samples
        curr_synced = synced_signals[curriculum_mask]
        curr_corrupted = corrupted_signals[curriculum_mask]
        curr_labels = labels[curriculum_mask]
        curr_snrs = snrs[curriculum_mask].float()

        total_loss = torch.tensor(0.0, device=device)

        # PHASE 1: CLASSIFIER ONLY
        if self.current_epoch < self.classifier_only_epochs:
            high_snr_mask = curr_snrs >= 0
            if high_snr_mask.any():
                corrupted_clean = curr_corrupted[high_snr_mask]
                clean_labels = curr_labels[high_snr_mask]

                class_logits = self.classifier(corrupted_clean)
                classification_loss = self.classification_loss(
                    class_logits, clean_labels
                )
                total_loss = classification_loss

                _, preds = torch.max(class_logits, 1)
                acc = (preds == clean_labels).float().mean()
                self.log("train_classifier_only_acc", acc, prog_bar=True)
                self.log("train_phase", 1.0)

        # PHASE 2: DENOISING + CLASSIFICATION
        else:
            # CORRECT TRAINING NOISE ASSIGNMENT: More noise to higher SNR
            training_timesteps = 500 + 15 * curr_snrs  # PLUS sign for correct behavior
            training_timesteps = torch.clamp(training_timesteps, 50, 950).long()

            # Add randomness for robustness
            noise_range = 100
            random_offset = torch.randint(
                -noise_range // 2,
                noise_range // 2,
                training_timesteps.shape,
                device=device,
            )
            training_timesteps = torch.clamp(
                training_timesteps + random_offset, 50, 950
            )

            # Add noise according to training schedule
            noisy_signals, _ = self.ddpm_scheduler.add_noise(
                curr_corrupted, training_timesteps
            )

            # Denoise using the SAME timesteps
            predicted_clean = self.unet(noisy_signals, training_timesteps)

            # 1. Reconstruction loss ONLY for high SNR signals
            high_snr_mask = curr_snrs >= 10
            recon_loss = torch.tensor(0.0, device=device)

            if high_snr_mask.any():
                high_snr_pred = predicted_clean[high_snr_mask]
                high_snr_target = curr_synced[high_snr_mask]
                recon_loss, recon_components = self.reconstruction_loss(
                    high_snr_pred, high_snr_target
                )

                # Log reconstruction components
                for name, loss_val in recon_components.items():
                    self.log(f"train_recon_{name}", loss_val)

            # 2. Classification loss for ALL signals
            class_logits = self.classifier(predicted_clean)
            class_loss = self.classification_loss(class_logits, curr_labels)

            # Optional: Feature matching
            if self.classifier_feature_weight > 0:
                with torch.no_grad():
                    clean_features = self.get_classifier_features(curr_synced)
                denoised_features = self.get_classifier_features(predicted_clean)
                feature_loss = F.mse_loss(denoised_features, clean_features)
                class_loss += self.classifier_feature_weight * feature_loss

            # Combine losses
            total_loss = recon_loss + self.classification_weight * class_loss

            # Log metrics including timestep verification
            _, preds = torch.max(class_logits, 1)
            overall_acc = (preds == curr_labels).float().mean()
            self.log("train_denoised_acc", overall_acc, prog_bar=True)
            self.log("train_reconstruction_loss", recon_loss, prog_bar=True)
            self.log("train_classification_loss", class_loss)

            # Log timestep statistics to verify correct noise assignment
            self.log("train_avg_timestep", training_timesteps.float().mean())

            if high_snr_mask.any():
                high_timesteps = training_timesteps[high_snr_mask].float().mean()
                self.log(
                    "train_high_snr_avg_timesteps", high_timesteps
                )  # Should be ~800
                high_snr_acc = (
                    (preds[high_snr_mask] == curr_labels[high_snr_mask]).float().mean()
                )
                self.log("train_high_snr_acc", high_snr_acc)

            low_snr_mask = curr_snrs < 0
            if low_snr_mask.any():
                low_timesteps = training_timesteps[low_snr_mask].float().mean()
                self.log(
                    "train_low_snr_avg_timesteps", low_timesteps
                )  # Should be ~200-350
                low_snr_acc = (
                    (preds[low_snr_mask] == curr_labels[low_snr_mask]).float().mean()
                )
                self.log("train_low_snr_acc", low_snr_acc)

            self.log("train_phase", 2.0)

        # Backpropagation
        opt_denoise.zero_grad()
        self.manual_backward(total_loss)

        if self.current_epoch >= self.classifier_only_epochs:
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)

        opt_denoise.step()
        scheduler_denoise.step()

        # Logging
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_min_snr_threshold", min_snr)

        return total_loss

    def validation_step(self, batch, batch_idx):
        synced_signals, corrupted_signals, labels, snrs = batch
        device = synced_signals.device
        batch_size = len(labels)
        # Phase 2: DEPLOYMENT TESTING - Denoise corrupted signals directly
        with torch.no_grad():
            # Directly denoise the corrupted signals (realistic deployment scenario)
            denoised_signals = self.inference_denoise_signal(
                corrupted_signals,  # Use original corrupted signals from dataset
                snrs,
                num_steps=10,
            )
            denoised_signals_for_viz = denoised_signals

        # Classify the denoised signals
        class_logits = self.classifier(denoised_signals)

        # ONLY FOR VISUALIZATION: Add noise to show the training process
        if batch_idx == 0:  # First batch only for visualization
            val_timesteps = 500 + 15 * snrs.float()
            val_timesteps = torch.clamp(val_timesteps, 50, 950).long()
            noisy_signals_for_viz, _ = self.ddpm_scheduler.add_noise(
                corrupted_signals, val_timesteps
            )
        else:
            noisy_signals_for_viz = corrupted_signals  # Use original for other batches

        # Compute classification metrics
        classification_loss = self.classification_loss(class_logits, labels)
        _, predicted_classes = torch.max(class_logits, 1)
        accuracy = (predicted_classes == labels).float().mean()

        # Log metrics
        self.log("val_loss", classification_loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

        # Accuracy by SNR range (deployment performance analysis)
        snr_ranges = [
            (-20, -15),
            (-15, -10),
            (-10, -5),
            (-5, 0),
            (0, 5),
            (5, 10),
            (10, 15),
            (15, 20),
            (20, 30),
        ]
        for snr_min, snr_max in snr_ranges:
            mask = (snrs >= snr_min) & (snrs < snr_max)
            if mask.any():
                range_acc = (predicted_classes[mask] == labels[mask]).float().mean()
                self.log(f"val_deploy_acc_{snr_min}to{snr_max}dB", range_acc)

        # Log performance improvement over raw corrupted signals
        if self.current_epoch >= self.classifier_only_epochs:
            # Test raw corrupted signal performance for comparison
            with torch.no_grad():
                raw_logits = self.classifier(corrupted_signals)
                _, raw_preds = torch.max(raw_logits, 1)
                raw_accuracy = (raw_preds == labels).float().mean()

                improvement = accuracy - raw_accuracy
                self.log("val_denoising_improvement", improvement, prog_bar=True)
                self.log("val_raw_corrupted_acc", raw_accuracy)

        # Store data for visualization (first batch only)
        if batch_idx == 0:
            self.val_data = {
                "original_signals": synced_signals[:4].detach().cpu(),
                "corrupted_signals": corrupted_signals[:4].detach().cpu(),
                "noisy_signals": noisy_signals_for_viz[:4].detach().cpu(),
                "denoised_signals": denoised_signals_for_viz[:4].detach().cpu(),
                "labels": labels[:4].detach().cpu(),
                "snrs": snrs[:4].detach().cpu(),
                "predicted_classes": predicted_classes[:4].detach().cpu(),
                "class_logits": class_logits[:4].detach().cpu(),
            }

        return classification_loss

    def _create_deployment_performance_plot(self):
        """Create a plot showing deployment performance vs baseline"""
        try:
            # Collect validation metrics from the current epoch
            trainer_logs = self.trainer.logged_metrics

            snr_ranges = [
                (-20, -15),
                (-15, -10),
                (-10, -5),
                (-5, 0),
                (0, 5),
                (5, 10),
                (10, 15),
                (15, 20),
                (20, 30),
            ]
            snr_centers = [(low + high) / 2 for low, high in snr_ranges]

            # Extract deployment accuracies
            deploy_accs = []
            for snr_min, snr_max in snr_ranges:
                key = f"val_deploy_acc_{snr_min}to{snr_max}dB"
                if key in trainer_logs:
                    deploy_accs.append(trainer_logs[key].item())
                else:
                    deploy_accs.append(None)

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot deployment performance
            valid_deploy = [
                (x, y) for x, y in zip(snr_centers, deploy_accs) if y is not None
            ]
            if valid_deploy:
                ax.plot(
                    *zip(*valid_deploy),
                    "b-o",
                    label="Deployment Performance (Denoised)",
                    linewidth=2,
                    markersize=8,
                )

            # Add baseline performance if available
            if "val_raw_corrupted_acc" in trainer_logs:
                raw_acc = trainer_logs["val_raw_corrupted_acc"].item()
                ax.axhline(
                    y=raw_acc,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Raw Corrupted Baseline ({raw_acc:.3f})",
                )

            # Add improvement metric if available
            if "val_denoising_improvement" in trainer_logs:
                improvement = trainer_logs["val_denoising_improvement"].item()
                ax.text(
                    0.02,
                    0.98,
                    f"Avg Improvement: {improvement:.3f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
                )

            ax.set_xlabel("SNR (dB)", fontsize=12)
            ax.set_ylabel("Classification Accuracy", fontsize=12)
            ax.set_title(
                f"Deployment Performance Analysis - Epoch {self.current_epoch}",
                fontsize=14,
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(-22, 32)

            # Add chance level
            ax.axhline(
                y=1 / 3, color="gray", linestyle=":", alpha=0.5, label="Chance Level"
            )

            # Highlight critical SNR regions
            ax.axvspan(-20, -10, alpha=0.1, color="red", label="Critical SNR Range")
            ax.axvspan(-10, 0, alpha=0.1, color="orange", label="Challenging SNR Range")
            ax.axvspan(0, 20, alpha=0.1, color="green", label="Good SNR Range")

            plt.tight_layout()

            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.log({"deployment_performance": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in deployment performance plot: {e}")

    def _create_constellation_visualization(self):
        """Create constellation plots showing realistic deployment pipeline"""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            for i in range(min(2, len(self.val_data["original_signals"]))):
                # Get data
                original = self.val_data["original_signals"][i]
                corrupted = self.val_data["corrupted_signals"][
                    i
                ]  # Real channel-corrupted signal
                noisy = self.val_data["noisy_signals"][i]  # For visualization only
                denoised = self.val_data["denoised_signals"][
                    i
                ]  # Denoised from corrupted
                true_label = self.val_data["labels"][i].item()
                pred_label = self.val_data["predicted_classes"][i].item()
                snr = self.val_data["snrs"][i].item()

                # Convert to complex
                original_complex = torch.complex(original[0], original[1])
                corrupted_complex = torch.complex(corrupted[0], corrupted[1])
                noisy_complex = torch.complex(noisy[0], noisy[1])
                denoised_complex = torch.complex(denoised[0], denoised[1])

                # Plot 1: Original clean signal
                axes[i, 0].scatter(
                    original_complex.real,
                    original_complex.imag,
                    alpha=0.6,
                    s=20,
                    c="green",
                )
                axes[i, 0].set_title(f"Original Clean\n{self.label_names[true_label]}")
                axes[i, 0].set_xlim(-1.5, 1.5)
                axes[i, 0].set_ylim(-1.5, 1.5)
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 0].set_aspect("equal")

                # Plot 2: Channel corrupted (real received signal)
                axes[i, 1].scatter(
                    corrupted_complex.real,
                    corrupted_complex.imag,
                    alpha=0.6,
                    s=20,
                    c="red",
                )
                axes[i, 1].set_title(f"Channel Corrupted\nSNR: {snr:.1f} dB")
                axes[i, 1].set_xlim(-1.5, 1.5)
                axes[i, 1].set_ylim(-1.5, 1.5)
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].set_aspect("equal")

                # Plot 3: Denoised signal (deployment output)
                axes[i, 2].scatter(
                    denoised_complex.real,
                    denoised_complex.imag,
                    alpha=0.6,
                    s=20,
                    c="blue",
                )
                axes[i, 2].set_title(f"Denoised\n(Deployment Output)")
                axes[i, 2].set_xlim(-1.5, 1.5)
                axes[i, 2].set_ylim(-1.5, 1.5)
                axes[i, 2].grid(True, alpha=0.3)
                axes[i, 2].set_aspect("equal")

                # Plot 4: Classification result
                class_probs = F.softmax(self.val_data["class_logits"][i], dim=0)
                axes[i, 3].bar(range(3), class_probs.cpu().numpy(), alpha=0.7)
                axes[i, 3].set_xticks(range(3))
                axes[i, 3].set_xticklabels(self.label_names, rotation=45)
                axes[i, 3].set_ylabel("Probability")

                # Highlight prediction
                correct = true_label == pred_label
                color = "green" if correct else "red"
                axes[i, 3].axvline(pred_label, color=color, linewidth=3, alpha=0.7)

                status = "✓" if correct else "✗"
                axes[i, 3].set_title(
                    f"{status} Predicted: {self.label_names[pred_label]}"
                )

            plt.suptitle(
                f"Deployment Pipeline - Epoch {self.current_epoch}", fontsize=16
            )
            plt.tight_layout()

            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.log({"deployment_pipeline": wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in constellation visualization: {e}")

    def inference_denoise_signal(
        self, corrupted_signal: torch.Tensor, snr_db: torch.Tensor, num_steps: int = 10
    ) -> torch.Tensor:
        """
        Inference denoising with SNR-aware timestep scheduling
        Clean signals (high SNR) get minimal or no denoising
        """
        self.unet.eval()

        batch_size = corrupted_signal.shape[0]
        device = corrupted_signal.device

        # Define SNR thresholds
        CLEAN_THRESHOLD = 10.0  # dB - signals above this are considered clean

        # Early exit for all clean signals
        if (snr_db >= CLEAN_THRESHOLD).all():
            return corrupted_signal  # Return unchanged

        # Calculate timesteps based on SNR
        # Lower SNR → Higher timesteps (more denoising)
        inference_timesteps = torch.zeros_like(snr_db, dtype=torch.float32)

        # Clean signals (SNR >= 10 dB): timestep = 0 (no denoising)
        # Good signals (5 <= SNR < 10 dB): timesteps 50-150 (minimal denoising)
        # Moderate signals (0 <= SNR < 5 dB): timesteps 150-400
        # Noisy signals (-10 <= SNR < 0 dB): timesteps 400-700
        # Very noisy signals (SNR < -10 dB): timesteps 700-950

        # Simple piecewise linear mapping
        for i in range(batch_size):
            snr = snr_db[i].item()
            if snr >= 10:
                inference_timesteps[i] = 0
            elif snr >= 5:
                inference_timesteps[i] = 150 - 20 * (
                    snr - 5
                )  # 150 at SNR=5, 50 at SNR=10
            elif snr >= 0:
                inference_timesteps[i] = 400 - 50 * snr  # 400 at SNR=0, 150 at SNR=5
            elif snr >= -10:
                inference_timesteps[i] = 400 - 30 * snr  # 400 at SNR=0, 700 at SNR=-10
            else:
                inference_timesteps[i] = 700 - 25 * (
                    snr + 10
                )  # 700 at SNR=-10, 950 at SNR=-20

        starting_timesteps = inference_timesteps.long().clamp(
            0, self.ddpm_scheduler.n_steps - 1
        )

        # Skip signals that don't need denoising
        process_mask = starting_timesteps > 0
        if not process_mask.any():
            return corrupted_signal

        current_signal = corrupted_signal.clone()

        with torch.no_grad():
            for step in range(num_steps):
                if not process_mask.any():
                    break

                progress = step / (num_steps - 1) if num_steps > 1 else 1.0

                # Calculate current timesteps with exponential decay
                current_timesteps = torch.zeros(
                    batch_size, device=device, dtype=torch.long
                )
                for i in range(batch_size):
                    if process_mask[i]:
                        start_t = starting_timesteps[i].item()
                        decay_rate = 2.5
                        current_timesteps[i] = int(
                            start_t * math.exp(-decay_rate * progress)
                        )

                # Get denoising prediction only for signals that need it
                if process_mask.any():
                    predicted_clean = self.unet(current_signal, current_timesteps)

                    # Simple adaptive step size
                    timestep_factor = (
                        current_timesteps.float() / self.ddpm_scheduler.n_steps
                    )
                    alpha = 0.1 + 0.2 * timestep_factor  # Range: 0.1 to 0.3
                    alpha = alpha.view(-1, 1, 1)

                    # Only update signals that need processing
                    process_mask_expanded = process_mask.view(-1, 1, 1)
                    current_signal = torch.where(
                        process_mask_expanded,
                        (1 - alpha) * current_signal + alpha * predicted_clean,
                        current_signal,  # Keep original for clean signals
                    )

        return current_signal

    def on_validation_epoch_end(self):
        """Create visualizations"""
        if hasattr(self, "val_data") and self.val_data is not None:
            self._create_constellation_visualization()
            self._create_deployment_performance_plot()

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        params = list(self.unet.parameters()) + list(self.classifier.parameters())

        optimizer = AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": "OneCycleLR",
            },
        }


class PSKBaselineClassifier(L.LightningModule):
    """Baseline PSK Classifier without denoising - for performance comparison"""

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

        self.classifier = classifier
        self.num_classes = num_classes
        self.label_names = label_names

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # For tracking metrics
        self.validation_step_outputs = []

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        opt_denoise = self.optimizers()
        scheduler_denoise = self.lr_schedulers()

        synced_signals, corrupted_signals, labels, snrs = batch
        device = synced_signals.device

        # Get current minimum SNR based on curriculum
        min_snr = self.get_current_min_snr()

        # Filter batch based on curriculum
        curriculum_mask = snrs >= min_snr
        if not curriculum_mask.any():
            return torch.tensor(0.0, device=device)

        # Filter samples
        curr_synced = synced_signals[curriculum_mask]
        curr_corrupted = corrupted_signals[curriculum_mask]
        curr_labels = labels[curriculum_mask]
        curr_snrs = snrs[curriculum_mask].float()

        total_loss = torch.tensor(0.0, device=device)

        # PHASE 1: CLASSIFIER ONLY
        if self.current_epoch < self.classifier_only_epochs:
            high_snr_mask = curr_snrs >= 10
            if high_snr_mask.any():
                corrupted_clean = curr_corrupted[high_snr_mask]
                clean_labels = curr_labels[high_snr_mask]

                class_logits = self.classifier(corrupted_clean)
                classification_loss = self.classification_loss(
                    class_logits, clean_labels
                )
                total_loss = classification_loss

                _, preds = torch.max(class_logits, 1)
                acc = (preds == clean_labels).float().mean()
                self.log("train_classifier_only_acc", acc, prog_bar=True)
                self.log("train_phase", 1.0)

        # PHASE 2: DENOISING + CLASSIFICATION
        else:
            # TRAINING NOISE ASSIGNMENT: More noise to higher SNR
            # High SNR → High timesteps (more noise during training)
            # Low SNR → Low timesteps (less noise during training)
            training_timesteps = (
                500 + 15 * curr_snrs
            )  # Increased multiplier for more dramatic effect
            training_timesteps = torch.clamp(training_timesteps, 50, 950).long()

            # Add randomness for robustness
            noise_range = 100
            random_offset = torch.randint(
                -noise_range // 2,
                noise_range // 2,
                training_timesteps.shape,
                device=device,
            )
            training_timesteps = torch.clamp(
                training_timesteps + random_offset, 50, 950
            )

            # Add noise according to training schedule
            noisy_signals, _ = self.ddpm_scheduler.add_noise(
                curr_corrupted, training_timesteps
            )

            # Denoise using the SAME timesteps used for noise addition
            predicted_clean = self.unet(noisy_signals, training_timesteps)

            # LOSS COMPUTATION

            # 1. Reconstruction loss ONLY for high SNR signals
            high_snr_mask = curr_snrs >= 10
            recon_loss = torch.tensor(0.0, device=device)

            if high_snr_mask.any():
                high_snr_pred = predicted_clean[high_snr_mask]
                high_snr_target = curr_synced[high_snr_mask]
                recon_loss, recon_components = self.reconstruction_loss(
                    high_snr_pred, high_snr_target
                )

                # Log reconstruction components
                for name, loss_val in recon_components.items():
                    self.log(f"train_recon_{name}", loss_val)

            # 2. Classification loss for ALL signals
            class_logits = self.classifier(predicted_clean)
            class_loss = self.classification_loss(class_logits, curr_labels)

            # Optional: Feature matching for all signals
            if self.classifier_feature_weight > 0:
                with torch.no_grad():
                    clean_features = self.get_classifier_features(curr_synced)
                denoised_features = self.get_classifier_features(predicted_clean)
                feature_loss = F.mse_loss(denoised_features, clean_features)
                class_loss += self.classifier_feature_weight * feature_loss

            # Combine losses
            total_loss = recon_loss + self.classification_weight * class_loss

            # Log metrics
            _, preds = torch.max(class_logits, 1)
            overall_acc = (preds == curr_labels).float().mean()
            self.log("train_denoised_acc", overall_acc, prog_bar=True)
            self.log("train_reconstruction_loss", recon_loss, prog_bar=True)
            self.log("train_classification_loss", class_loss)

            # Log timestep statistics
            self.log("train_avg_timestep", training_timesteps.float().mean())
            if high_snr_mask.any():
                high_timesteps = training_timesteps[high_snr_mask].float().mean()
                high_snr_acc = (
                    (preds[high_snr_mask] == curr_labels[high_snr_mask]).float().mean()
                )
                self.log("train_high_snr_timesteps", high_timesteps)
                self.log("train_high_snr_acc", high_snr_acc)

            low_snr_mask = curr_snrs < 0
            if low_snr_mask.any():
                low_timesteps = training_timesteps[low_snr_mask].float().mean()
                low_snr_acc = (
                    (preds[low_snr_mask] == curr_labels[low_snr_mask]).float().mean()
                )
                self.log("train_low_snr_timesteps", low_timesteps)
                self.log("train_low_snr_acc", low_snr_acc)

            self.log("train_phase", 2.0)

        # Backpropagation
        opt_denoise.zero_grad()
        self.manual_backward(total_loss)

        if self.current_epoch >= self.classifier_only_epochs:
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)

        opt_denoise.step()
        scheduler_denoise.step()

        # Logging
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_min_snr_threshold", min_snr)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        _, synced_signals, labels, snrs = batch
        # synced_signals, original_unsynced_signals, labels, snrs = batch

        # Classify the clean synchronized signals
        logits = self.classifier(synced_signals)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == labels).float().mean()

        # Calculate average confidence
        confidence = torch.max(probs, dim=1)[0].mean()

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_confidence", confidence)

        # Per-class accuracy
        for i, class_name in enumerate(self.label_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == labels[class_mask]).float().mean()
                self.log(f"val_acc_{class_name}", class_acc)

        # SNR-based accuracy analysis
        snr_ranges = [(-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30)]
        for snr_min, snr_max in snr_ranges:
            snr_mask = (snrs >= snr_min) & (snrs < snr_max)
            if snr_mask.sum() > 0:
                snr_acc = (preds[snr_mask] == labels[snr_mask]).float().mean()
                self.log(f"val_acc_snr_{snr_min}to{snr_max}", snr_acc)

        # Store for epoch-end analysis
        self.validation_step_outputs.append(
            {
                "preds": preds.detach().cpu(),
                "labels": labels.detach().cpu(),
                "snrs": snrs.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
        )

        return loss

    def on_validation_epoch_end(self):
        """Calculate and log confusion matrix and other epoch-level metrics"""
        if not self.validation_step_outputs:
            return

        # Aggregate all predictions and labels
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        all_snrs = torch.cat([x["snrs"] for x in self.validation_step_outputs])

        # Calculate confusion matrix
        num_classes = len(self.label_names)
        confusion_matrix = torch.zeros(num_classes, num_classes)

        for t, p in zip(all_labels, all_preds):
            confusion_matrix[t.long(), p.long()] += 1

        # Normalize confusion matrix
        confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True)

        # Log confusion matrix
        if self.logger and hasattr(self.logger, "experiment"):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(confusion_matrix.numpy(), cmap="Blues")

            # Add labels
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(self.label_names)
            ax.set_yticklabels(self.label_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix - Epoch {self.current_epoch}")

            # Add text annotations
            for i in range(num_classes):
                for j in range(num_classes):
                    text = ax.text(
                        j,
                        i,
                        f"{confusion_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

            plt.colorbar(im)
            plt.tight_layout()

            self.logger.experiment.log({"confusion_matrix": wandb.Image(fig)})
            plt.close(fig)

        # Clear outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.classifier.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
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

import torch
import numpy as np
import torch.nn as nn
import wandb
import torch.nn.functional as F
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns


class LatentClassifier(pl.LightningModule):
    """
    Classifier that works with the Latent Diffusion + ArcFace setup.

    This classifier:
    1. Uses the pre-trained ArcFace encoder to get latent representations
    2. Uses the latent diffusion model to denoise the latents at various noise levels
    3. Classifies the denoised latents using RFNet or a simple classifier head
    """

    def __init__(
        self,
        diffusion,  # Your LatentDiffusion model
        encoder,
        classifier_head,   # RFNet or simple classifier
        learning_rate=1e-4,
        num_classes=11,
        label_names=None,
        beta=0.25,
        fine_tune_diffusion=False,
        classifier_free=False,
        # SNR scheduling parameters
        min_snr=-20.0,
        max_snr=20.0,
        # Training strategy
        use_curriculum=True,
        curriculum_start_snr=10.0,
        curriculum_end_snr=-20.0,
        curriculum_epochs=50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['latent_diffusion', 'classifier_head'])

        # Store models
        self.latent_diffusion = diffusion
        self.awgn_scheduler = self.latent_diffusion.awgn_scheduler
        self.classifier_head = classifier_head
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.label_names = label_names or [f"Class_{i}" for i in range(num_classes)]
        self.classifier_free = classifier_free

        # SNR parameters
        self.min_snr = min_snr
        self.max_snr = max_snr

        # Curriculum learning
        self.use_curriculum = use_curriculum
        self.curriculum_start_snr = curriculum_start_snr
        self.curriculum_end_snr = curriculum_end_snr
        self.curriculum_epochs = curriculum_epochs

        # Get diffusion parameters
        self.n_steps = self.latent_diffusion.n_steps

        # Freeze diffusion by default
        if not fine_tune_diffusion:
            for param in self.latent_diffusion.parameters():
                param.requires_grad = False

        # Accuracy tracking
        self.reset_accuracy_tracking()

    def reset_accuracy_tracking(self):
        """Reset accuracy tracking for new epoch"""
        # SNR-based accuracy: 20 bins from -20 to 18 dB
        self.snr_accuracy = [(0, 0) for _ in range(20)]
        # Modulation-based accuracy
        self.mod_accuracy = [(0, 0) for _ in range(self.num_classes)]

    def get_curriculum_snr_threshold(self):
        """Get current SNR threshold for curriculum learning"""
        if not self.use_curriculum:
            return self.curriculum_end_snr

        progress = min(self.current_epoch / self.curriculum_epochs, 1.0)
        threshold = (
            self.curriculum_start_snr * (1 - progress) +
            self.curriculum_end_snr * progress
        )
        return threshold

    def snr_to_timestep(self, snr):
        """Use the same SNR to timestep mapping as the latent diffusion model"""
        return self.awgn_scheduler.snr_to_timestep(snr)

    def forward(self, x, snr=None, timestep=None):
        """Forward pass through the classifier"""
        # 1. Encode to latent space using ArcFace encoder
        z_noisy = self.latent_diffusion.encode(x)  # [batch, 32, 8]

        # 2. Determine timestep for denoising
        if timestep is not None:
            t = timestep
        elif snr is not None:
            # Use the SAME mapping as latent diffusion
            t = self.snr_to_timestep(snr.float())
        else:
            # Default: sample random timesteps
            batch_size = z_noisy.shape[0]
            t = torch.randint(0, self.n_steps, (batch_size,), device=z_noisy.device)

        # 3. Denoise using latent diffusion
        z_denoised = self.denoise_latent(z_noisy, t)

        # 4. Classify denoised latents
        logits = self.classifier_head(z_denoised)

        return logits, z_denoised, t

    def denoise_latent(self, z_noisy, t, class_embedding=None):
        """
        Denoise latent representation using the diffusion model
        """
        with torch.set_grad_enabled(self.training and not self.hparams.fine_tune_diffusion):
            if self.latent_diffusion.predict_noise:
                # Model predicts noise
                predicted_noise = self.latent_diffusion.forward(z_noisy, t, class_embedding)
                z_denoised = z_noisy - predicted_noise
            else:
                # Model predicts clean latent directly
                z_denoised = self.latent_diffusion.forward(z_noisy, t, class_embedding)

        return z_denoised

    def training_step(self, batch, batch_idx):
        x, labels, snr = batch
        labels = labels.long()

        # Apply curriculum learning filter
        if self.use_curriculum:
            snr_threshold = self.get_curriculum_snr_threshold()
            curriculum_mask = snr.squeeze() >= snr_threshold

            if not curriculum_mask.any():
                # Skip batch if no samples meet curriculum criteria
                self.log("curriculum_snr_threshold", snr_threshold, prog_bar=True)
                self.log("curriculum_samples_skipped", len(x))
                return None

            # Filter batch
            x = x[curriculum_mask]
            labels = labels[curriculum_mask]
            snr = snr[curriculum_mask]

        batch_size = x.shape[0]

        # Add small random jitter to timesteps during training for exploration
        base_timestep = self.snr_to_timestep(snr.float())
        if self.training:
            jitter = torch.randint(
                -self.n_steps // 20,
                self.n_steps // 20,
                (batch_size,),
                device=x.device
            )
            t = (base_timestep + jitter).clamp(0, self.n_steps - 1)
        else:
            t = base_timestep

        # Forward pass
        logits, z_denoised, _ = self.forward(x, timestep=t)

        # Classification loss
        cls_loss = self.criterion(logits, labels)

        # Calculate accuracy
        acc = self.calculate_accuracy(logits, labels, snr)

        # Logging
        log_dict = {
            "train/loss": cls_loss,
            "train/acc": acc,
            "train/avg_timestep": t.float().mean(),
            "train/timestep_std": t.float().std(),
            "train/avg_snr": snr.float().mean(),
            "train/batch_size": float(batch_size),
        }

        if self.use_curriculum:
            log_dict["train/curriculum_threshold"] = self.get_curriculum_snr_threshold()

        self.log_dict(log_dict, prog_bar=True)

        return cls_loss

    def validation_step(self, batch, batch_idx):
        x, labels, snr = batch
        labels = labels.long()

        # Forward pass - use SNR-based timestep mapping
        logits, z_denoised, t = self.forward(x, snr=snr)

        # Classification loss
        val_loss = self.criterion(logits, labels)

        # Calculate accuracy and update tracking
        val_acc = self.calculate_accuracy(logits, labels, snr, update_tracking=True)

        # Logging
        self.log_dict({
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val_loss": val_loss,     # For checkpoint callback
            "val_acc": val_acc,       # For checkpoint callback
            "val/avg_timestep": t.float().mean(),
            "val/avg_snr": snr.float().mean(),
        }, prog_bar=True)

        # Cache first batch for visualization - ENSURE DEVICE CONSISTENCY
        if batch_idx == 0:
            with torch.no_grad():
                # Get original encoded latents (no denoising)
                z_original = self.latent_diffusion.encode(x)

                # Store on CPU to avoid device issues in callback
                self.example_batch = {
                    # Original data - store on CPU
                    'x': x.detach().cpu(),
                    'labels': labels.detach().cpu(),
                    'snr': snr.detach().cpu(),

                    # Timestep information
                    't': t.detach().cpu(),

                    # Latent representations - store on CPU
                    'z_original': z_original.detach().cpu(),
                    'z_denoised': z_denoised.detach().cpu(),

                    # Classification results - store on CPU
                    'logits': logits.detach().cpu(),
                    'predictions': torch.argmax(logits, dim=1).detach().cpu(),

                    # Metadata
                    'label_names': self.label_names,
                    'num_classes': self.num_classes,
                    'current_epoch': self.current_epoch if hasattr(self, 'current_epoch') else 0,
                }

        return val_loss

    def calculate_accuracy(self, logits, labels, snr, update_tracking=False):
        """Calculate accuracy and optionally update tracking statistics"""
        predicted_classes = torch.argmax(logits, dim=1)
        correct = (predicted_classes == labels).float()
        acc = correct.mean().item()

        if update_tracking:
            # Update SNR-based accuracy tracking
            snr_indices = ((snr.squeeze() + 20) / 2).long().clamp(0, 19)

            for idx in range(20):
                mask = (snr_indices == idx)
                if mask.any():
                    total = mask.sum().item()
                    correct_count = (mask & (predicted_classes == labels)).sum().item()
                    curr_correct, curr_total = self.snr_accuracy[idx]
                    self.snr_accuracy[idx] = (curr_correct + correct_count, curr_total + total)

            # Update modulation-based accuracy tracking
            for mod_idx in range(self.num_classes):
                mask = (labels == mod_idx)
                if mask.any():
                    total = mask.sum().item()
                    correct_count = (mask & (predicted_classes == labels)).sum().item()
                    curr_correct, curr_total = self.mod_accuracy[mod_idx]
                    self.mod_accuracy[mod_idx] = (curr_correct + correct_count, curr_total + total)

        return acc

    def on_train_epoch_end(self):
        """Log training epoch metrics"""
        if self.use_curriculum:
            current_threshold = self.get_curriculum_snr_threshold()
            progress = min(self.current_epoch / self.curriculum_epochs, 1.0) * 100

            print(f"Epoch {self.current_epoch}: SNR threshold = {current_threshold:.1f} dB "
                  f"(Progress: {progress:.1f}%)")

    def on_validation_epoch_end(self):
        """Create visualizations and log metrics at the end of validation"""
        # 1. Create accuracy plots
        self._create_accuracy_plots()

        # 3. Create confusion matrix
        self._create_confusion_matrix()

        # 4. Reset accuracy tracking for next epoch
        self.reset_accuracy_tracking()

    def _create_accuracy_plots(self):
        """Create SNR vs accuracy and per-modulation accuracy plots"""
        try:
            # SNR accuracy plot
            snr_values = [-20 + (i * 2) for i in range(20)]
            snr_accuracies = []

            for correct, total in self.snr_accuracy:
                acc = correct / total if total > 0 else 0
                snr_accuracies.append(acc)

            # Modulation accuracy plot
            mod_accuracies = []
            for correct, total in self.mod_accuracy:
                acc = correct / total if total > 0 else 0
                mod_accuracies.append(acc)

            # Create wandb tables and plots
            snr_data = [[snr, acc] for snr, acc in zip(snr_values, snr_accuracies)]
            snr_table = wandb.Table(columns=["SNR", "Accuracy"], data=snr_data)

            mod_data = [[self.label_names[i], acc] for i, acc in enumerate(mod_accuracies)]
            mod_table = wandb.Table(columns=["Modulation", "Accuracy"], data=mod_data)

            # Log to wandb
            wandb.log({
                "accuracy/snr_vs_accuracy": wandb.plot.line(
                    snr_table, "SNR", "Accuracy", title="Classification Accuracy vs SNR"
                ),
                "accuracy/modulation_accuracy": wandb.plot.bar(
                    mod_table, "Modulation", "Accuracy", title="Per-Modulation Accuracy"
                )
            })

        except Exception as e:
            print(f"Error creating accuracy plots: {e}")


    def _create_confusion_matrix(self):
        """Create confusion matrix from example batch"""
        try:
            if not hasattr(self, 'example_batch'):
                return

            example_batch = self.example_batch
            true_labels = example_batch['labels'].numpy()
            predictions = example_batch['predictions'].numpy()

            # Create confusion matrix
            cm = confusion_matrix(true_labels, predictions, labels=range(self.num_classes))
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=self.label_names,
                yticklabels=self.label_names,
                ax=ax
            )

            ax.set_title('Confusion Matrix (Normalized)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Log to wandb
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            img = wandb.Image(Image.open(buf))
            plt.close()

            wandb.log({
                "visualization/confusion_matrix": img
            })

        except Exception as e:
            print(f"Error creating confusion matrix: {e}")

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Separate parameters based on what we're training
        if self.hparams.fine_tune_diffusion:
            # If fine-tuning diffusion, use different learning rates
            optimizer = torch.optim.AdamW([
                {'params': self.classifier_head.parameters(), 'lr': self.learning_rate},
                {'params': self.latent_diffusion.parameters(), 'lr': self.learning_rate * 0.1}  # Lower LR for pre-trained model
            ], weight_decay=1e-4)
        else:
            # Only train classifier head
            optimizer = torch.optim.AdamW(
                self.classifier_head.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4
            )

        # OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.05,  # 5% warmup
            anneal_strategy="cos",
            final_div_factor=100,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

import torch
import numpy as np
import torch.nn as nn
import wandb
import torch.nn.functional as F
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import io
from PIL import Image


class TimestepPredictor(nn.Module):
    def __init__(self, in_channels=32, hidden_dim=1024, n_steps=500):
        super().__init__()
        self.n_steps = n_steps
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim//4, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, n_steps)
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.final(features)    # [B, n_steps]
        return logits

class LatentClassifier(pl.LightningModule):
    def __init__(self, diffusion, encoder, classifier_head, learning_rate, num_classes, label_names,beta=0.25, fine_tune_diffusion=False, classifier_free=False):
        super().__init__()
        self.save_hyperparameters(ignore=['diffusion', 'encoder', 'classifier_head'])

        # Store models
        self.diffusion = diffusion
        self.encoder = encoder
        self.classifier_head = classifier_head
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.label_names = label_names
        self.classifier_free = classifier_free
        self.beta = beta
        self.timestep_predictor = TimestepPredictor(in_channels=32, n_steps=diffusion.n_steps)

        # Add scheduling parameters (fixed typo here)
        self.schedule_factor = 0.2  # Weight for mixture distribution
        self.n_steps = diffusion.n_steps

        # Freeze diffusion by default
        if not fine_tune_diffusion:
            for param in self.diffusion.parameters():
                param.requires_grad = False

        # Accuracy tracking
        self.snr_accuracy = [(0,0) for _ in range(20)]
        self.mod_accuracy = [(0,0) for _ in range(self.num_classes)]

    def forward(self, x):
        # run the model through the diffusion process and classifiy
        return self.classifier_head(x)

    def training_step(self, batch, batch_idx):
        x, context, snr = batch
        z = self.diffusion.encode(x).float()
        pred_class, class_logits = self.diffusion.embedding_conditioner(z)

        B = z.shape[0]

        # Map SNR to timesteps - convert SNR to float first
        snr = snr.float()  # Convert to float to avoid dtype issues
        min_snr, max_snr = -20.0, 20.0  # SNR Range
        normalized_snr = (snr - min_snr) / (max_snr - min_snr)
        normalized_snr = torch.clamp(normalized_snr, 0.0, 1.0)
        x = normalized_snr * 2 - 1  # Map to [-1, 1]
        transformed = 1.0 / (1.0 + torch.exp(x / self.beta))  #
        t = (transformed * (self.n_steps - 1)).long()
        t = t.view(B)

        # Add small random jitter during training for exploration
        if self.training:
            # Add random offset of up to ±5% of total steps
            jitter = torch.randint(-self.n_steps//20, self.n_steps//20, (B,), device=t.device)
            t = (t + jitter).clamp(0, self.n_steps - 1)

        # Diffusion & classification
        if self.classifier_free:
            classifier_free = torch.full_like(context, 11, device=self.device)
            pred_noise = self.diffusion(z, t, classifier_free)
        else:
            pred_noise = self.diffusion(z, t, pred_class)

        a = self.diffusion.sqrt_alpha_bar[t].view(-1, 1, 1).float()
        am1 = self.diffusion.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1).float()
        denoised = (z - am1*pred_noise)/a
        logits_cls = self.classifier_head(denoised)

        # Classification loss
        cls_loss = self.criterion(logits_cls, context)

        #timestep predictor
        # t_logits = self.timestep_predictor(denoised)
        # t_target = t
        # timestep_loss = F.cross_entropy(t_logits, t_target)
        # t_pred = torch.argmax(t_logits, dim=1)
        # t_acc = (t_pred == t_target).float().mean()
        # self.log("train/timestep_loss", timestep_loss)
        # self.log("train/timestep_acc", t_acc)


        # Logging - use float everywhere
        self.log_dict({
            "train/loss": cls_loss,
            "train/avg_t": t.float().mean(),
            "train/t_std": t.float().std(),
            "train/avg_snr": snr.float().mean(),
            "train/acc": self.accuracy(logits_cls, context)
        }, prog_bar=True)

        return cls_loss


    def validation_step(self, batch, batch_idx):
        x, context, snr = batch
        context = context.long()
        assert torch.all(context < self.num_classes), f"Invalid context values: {context.max()}"

        # 1) Encode
        z = self.diffusion.encode(x).float()
        pred_class, class_logits = self.diffusion.embedding_conditioner(z)
        B = z.shape[0]

        # 2) Map SNR to timesteps - convert SNR to float first
        snr = snr.float()  # Convert to float to avoid dtype issues
        min_snr, max_snr = -20.0, 20.0  # Typical SNR range in dB
        normalized_snr = (snr - min_snr) / (max_snr - min_snr)
        x = normalized_snr * 2 - 1  # Map to [-1, 1]
        transformed = 1.0 / (1.0 + torch.exp(x / self.beta))  #beta = Temperature, controls tails and steepness
        t = (transformed * (self.n_steps - 1)).long()
        t = t.view(B)

        # Diffusion & classification
        if self.classifier_free:
            classifier_free = torch.full_like(context, 11, device=self.device)
            pred_noise = self.diffusion(z, t, classifier_free)
        else:
            pred_noise = self.diffusion(z, t, pred_class)
        a_t = self.diffusion.sqrt_alpha_bar[t].view(-1, 1, 1).float()
        am1_t = self.diffusion.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1).float()
        denoised_z = (z - am1_t * pred_noise) / a_t

        # 5) Classify
        logits_cls = self.classifier_head(denoised_z)
        pred_probs = F.softmax(logits_cls, dim=-1)
        assert logits_cls.shape[1] == self.num_classes, f"Wrong pred shape: {logits_cls.shape}"

        #timestep predictor
        # t_logits = self.timestep_predictor(denoised_z)
        # t_target = t
        # timestep_loss = F.cross_entropy(t_logits, t_target)
        # t_pred = torch.argmax(t_logits, dim=1)
        # t_acc = (t_pred == t_target).float().mean()
        # self.log("val/timestep_loss", timestep_loss)
        # self.log("val/timestep_acc", t_acc)

        # 6) Loss & metrics
        val_loss = self.criterion(logits_cls, context)
        val_acc = self.accuracy(logits_cls, context, snr)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

        # 7) Cache example batch and associated computed values
        if batch_idx == 0:
            self.example_batch = {
                'x': x.detach().cpu(),
                'z': z.detach().cpu(),
                'context': context.detach().cpu(),
                'snr': snr.detach().cpu(),
                't': t.detach().cpu(),
                'denoised_z': denoised_z.detach().cpu(),
                'pred': logits_cls.detach().cpu()
            }

        return val_loss


    def configure_optimizers(self):  # pyright: ignore
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        # OneCycleLR automatically determines warm-up and decay schedules
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,  # Peak LR at warmup end
            total_steps=int(
                self.trainer.estimated_stepping_batches
            ),  # Total training steps
            pct_start=0.05,  # 5% of training is used for warm-up
            anneal_strategy="cos",  # Cosine decay after warmup
            final_div_factor=100,  # Reduce LR by 10x at the end
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def accuracy(self, pred, y, snr=None):
        # Ensure inputs are properly formatted
        y = y.long()  # Ensure labels are long type
        predicted_classes = torch.argmax(pred, dim=1)

        # Verify shapes
        assert predicted_classes.shape == y.shape, f"Shape mismatch: pred={predicted_classes.shape}, y={y.shape}"

        # Ensure values are within valid range
        assert torch.all(y < self.num_classes), f"Labels must be < {self.num_classes}"

        correct = (predicted_classes == y).float()  # Use float for mean calculation
        acc = correct.mean().item()

        if snr is not None:
            # Convert SNR values to indices (from -20:20:2 to 0:20)
            snr_indices = ((snr + 20) / 2).long().clamp(0, 19)  # Add clamp to ensure valid indices

            # For each SNR index, update total and correct counts
            for idx in range(20):
                mask = (snr_indices == idx)
                total = mask.sum().item()
                correct_count = (mask & (predicted_classes == y)).sum().item()
                curr_correct, curr_total = self.snr_accuracy[idx]
                self.snr_accuracy[idx] = (curr_correct + correct_count, curr_total + total)

            # Track modulation-based accuracy
            for mod_idx in range(self.num_classes):
                mask = (y == mod_idx)
                total = mask.sum().item()
                correct_count = (mask & (predicted_classes == y)).sum().item()
                curr_correct, curr_total = self.mod_accuracy[mod_idx]
                self.mod_accuracy[mod_idx] = (curr_correct + correct_count, curr_total + total)

        return acc

    def on_train_epoch_end(self):

        # Create SNR accuracy line plot
        snr_values = [-20 + (i * 2) for i in range(20)]  # -20 to 18 in steps of 2
        accuracies = []
        for correct, total in self.snr_accuracy:
            acc = correct / total if total > 0 else 0
            accuracies.append(acc)
        # Create data for line plot
        snr_data = [[snr, acc] for snr, acc in zip(snr_values, accuracies)]
        snr_table = wandb.Table(columns=["SNR", "Accuracy"], data=snr_data)

        # Create Mod Accuracy Bar chart
        mod_accuracies = []
        for correct, total in self.mod_accuracy:
            acc = correct / total if total > 0 else 0
            mod_accuracies.append(acc)
        # Create data for bar
        mod_data = [[f"{self.label_names[i]}", acc] for i, acc in enumerate(mod_accuracies)]
        mod_table = wandb.Table(columns=["Modulation", "Accuracy"], data=mod_data)

        wandb.log({
            "Mod Accuracy": wandb.plot.bar(
                mod_table, "Modulation", "Accuracy", title="Modulation Accuracy"
            ),
            "snr_accuracy": wandb.plot.line(
                snr_table,
                "SNR",
                "Accuracy",
                title="Accuracy vs SNR"
            )
        })

        # Reset for next epoch
        self.epoch_batch_count = 0
        self.snr_accuracy = [(0,0) for _ in range(20)]
        self.mod_accuracy = [(0,0) for _ in range(self.num_classes)]

    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch.
        Creates clear visualization of SNR to timestep mapping with current beta value.
        """
        try:
            # Only log for certain epochs to avoid too many plots
            current_epoch = self.current_epoch
            max_epochs = self.trainer.max_epochs

            if current_epoch == 0 or current_epoch == max_epochs - 1 or current_epoch % 10 == 0:
                if hasattr(self, 'example_batch'):
                    example_batch = self.example_batch

                    # Get SNR and timestep data
                    snr = example_batch['snr']
                    t = example_batch['t']

                    # Create a clean, focused plot
                    fig = plt.figure(figsize=(10, 6))

                    # Plot the actual data points
                    plt.scatter(
                        snr.numpy(),
                        t.numpy(),
                        color='blue',
                        alpha=0.7,
                        s=40,
                        label='Data points'
                    )

                    # Generate and plot the sigmoid mapping function
                    test_snr = np.linspace(-20, 20, 100)
                    norm_snr = (test_snr + 20) / 40
                    x = norm_snr * 2 - 1
                    sigmoid = 1.0 / (1.0 + np.exp(x / self.beta))
                    test_t = sigmoid * (self.n_steps - 1)
                    plt.plot(test_snr, test_t, 'r-', linewidth=2, alpha=0.8,
                            label=f'Sigmoid (β={self.beta:.3f})')

                    # Add linear mapping for comparison
                    linear_t = (1 - norm_snr) * (self.n_steps - 1)
                    plt.plot(test_snr, linear_t, 'g--', linewidth=1.5, alpha=0.6,
                            label='Linear mapping')

                    # Style the plot
                    plt.xlabel('SNR (dB)', fontsize=12)
                    plt.ylabel('Timestep', fontsize=12)
                    plt.title(f'SNR to Timestep Mapping (β={self.beta:.3f}, Epoch {current_epoch})',
                            fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.legend(frameon=True, fontsize=10)

                    # Add shaded regions to indicate SNR quality
                    plt.axvspan(-20, -10, alpha=0.1, color='red', label='_nolegend_')
                    plt.axvspan(-10, 0, alpha=0.1, color='orange', label='_nolegend_')
                    plt.axvspan(0, 10, alpha=0.1, color='yellow', label='_nolegend_')
                    plt.axvspan(10, 20, alpha=0.1, color='green', label='_nolegend_')

                    # Add annotations for SNR regions
                    plt.text(-18, self.n_steps * 0.1, "Very low SNR", rotation=90,
                            fontsize=8, alpha=0.7)
                    plt.text(-8, self.n_steps * 0.1, "Low SNR", rotation=90,
                            fontsize=8, alpha=0.7)
                    plt.text(2, self.n_steps * 0.1, "Medium SNR", rotation=90,
                            fontsize=8, alpha=0.7)
                    plt.text(12, self.n_steps * 0.1, "High SNR", rotation=90,
                            fontsize=8, alpha=0.7)

                    # Adjust y-axis to show full timestep range
                    plt.ylim(0, self.n_steps)

                    # Save the figure to buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150)
                    buf.seek(0)

                    # Create wandb Image
                    img = wandb.Image(Image.open(buf))
                    plt.close()

                    # Log the image with consistent key for run coloring
                    self.logger.experiment.log({
                        # Use a consistent key across runs for color differentiation
                        "snr_timestep_mapping": img,
                        "beta_value": self.beta,
                        "epoch": current_epoch
                    }, step=self.global_step)

                    # Also create a simple scatter plot for WandB's interactive features
                    data = [[s.item(), t.item()] for s, t in zip(snr, t)]
                    table = wandb.Table(columns=["SNR", "Timestep"], data=data)

                    wandb.log({
                        "snr_timestep_interactive": wandb.plot.scatter(
                            table,
                            "SNR",
                            "Timestep",
                            title=f"SNR to Timestep (β={self.beta:.3f})"
                        ),
                        "beta_value": self.beta,
                        "epoch": current_epoch
                    })

        except Exception as e:
            print(f"Error in validation epoch end: {str(e)}")
            import traceback
            traceback.print_exc()

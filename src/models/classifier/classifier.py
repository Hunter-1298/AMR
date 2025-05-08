import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import io


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
    def __init__(self, diffusion, encoder, classifier_head, learning_rate, num_classes, label_names, fine_tune_diffusion=False, classifier_free=False):
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
        self.classifier_fre = classifier_free
        # self.timestep_predictor = TimestepPredictor(in_channels=32, n_steps=diffusion.n_steps)

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
        B = z.shape[0]

        # Map SNR to timesteps - convert SNR to float first
        snr = snr.float()  # Convert to float to avoid dtype issues
        min_snr, max_snr = -20.0, 20.0  # Typical SNR range in dB
        normalized_snr = (snr - min_snr) / (max_snr - min_snr)
        normalized_snr = torch.clamp(normalized_snr, 0.0, 1.0)

        # Inverse mapping: high SNR → low timestep, low SNR → high timestep
        t = ((1.0 - normalized_snr) * (self.n_steps - 1)).long()
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
            pred_noise = self.diffusion(z, t, context)
        a = self.diffusion.sqrt_alpha_bar[t].view(-1, 1, 1).float()
        am1 = self.diffusion.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1).float()
        denoised = (z - am1*pred_noise)/a
        logits_cls = self.classifier_head(denoised)

        # Classification loss
        cls_loss = self.criterion(logits_cls, context)

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
        B = z.shape[0]

        # 2) Map SNR to timesteps - convert SNR to float first
        snr = snr.float()  # Convert to float to avoid dtype issues
        min_snr, max_snr = -20.0, 20.0  # Typical SNR range in dB
        normalized_snr = (snr - min_snr) / (max_snr - min_snr)
        normalized_snr = torch.clamp(normalized_snr, 0.0, 1.0)

        # Inverse mapping: high SNR → low timestep, low SNR → high timestep
        t = ((1.0 - normalized_snr) * (self.n_steps - 1)).long()
        t = t.view(B)

        # Diffusion & classification
        if self.classifier_free:
            classifier_free = torch.full_like(context, 11, device=self.device)
            pred_noise = self.diffusion(z, t, classifier_free)
        else:
            pred_noise = self.diffusion(z, t, context)
        a_t = self.diffusion.sqrt_alpha_bar[t].view(-1, 1, 1).float()
        am1_t = self.diffusion.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1).float()
        denoised_z = (z - am1_t * pred_noise) / a_t

        # 5) Classify
        logits_cls = self.classifier_head(denoised_z)
        pred_probs = F.softmax(logits_cls, dim=-1)
        assert logits_cls.shape[1] == self.num_classes, f"Wrong pred shape: {logits_cls.shape}"

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
        Creates basic visualizations of validation metrics only for first and last epoch.
        """
        try:
            # Only log for first and last epoch
            current_epoch = self.current_epoch
            max_epochs = self.trainer.max_epochs

            if current_epoch == 0 or current_epoch == max_epochs - 1:
                if hasattr(self, 'example_batch'):
                    example_batch = self.example_batch

                    # Create basic scatter plot of timesteps vs SNR
                    data = []
                    snr = example_batch['snr']
                    t = example_batch['t']
                    context = example_batch['context']

                    for s, time_step, c in zip(snr, t, context):
                        data.append([
                            s.item(),
                            time_step.item(),
                            self.label_names[c.item()]
                        ])

                    # Create and log wandb table
                    table = wandb.Table(
                        columns=["SNR", "Timestep", "Modulation"],
                        data=data
                    )

                    # Log simple scatter plot with epoch information in title
                    epoch_label = "First" if current_epoch == 0 else "Final"
                    wandb.log({
                        f"Timestep vs SNR ({epoch_label} Epoch)": wandb.plot.scatter(
                            table,
                            "SNR",
                            "Timestep",
                            title=f"Timestep Selection vs SNR - {epoch_label} Epoch"
                        )
                    })

        except Exception as e:
            print(f"Error in validation epoch end: {str(e)}")
            import traceback
            traceback.print_exc()

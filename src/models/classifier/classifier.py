import torch
import wandb
import torch.nn.functional as F
import lightning.pytorch as pl

class LatentClassifier(pl.LightningModule):
    def __init__(self, diffusion, encoder, classifier_head, learning_rate,num_classes,label_names, fine_tune_diffusion=False):
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

        # Freeze diffusion by default
        if not fine_tune_diffusion:
            for param in self.diffusion.parameters():
                param.requires_grad = False


        # going to store [correct/total]
        self.snr_accuracy = [(0,0) for _ in range(20)]
        # going to store [correct/total]
        self.mod_accuracy = [(0,0) for _ in range(self.num_classes)]

    def forward(self, x):
        # run the model through the diffusion process and classifiy
        return self.classifier_head(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning training step

        Args:
            batch: Batch of data containing (x, context, _)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Unpack batch
        x, context, snr = batch

        # Sample random timesteps
        t = torch.randint(0, self.diffusion.n_steps, (x.shape[0],), device=self.device).long()

        # Encode input to latent space
        z = self.diffusion.encode(x)

        # Calculate diffusion loss - currently prediciting noise distribution
        noise_loss, predicted_noise = self.diffusion.p_losses(z, t, context)

        pred = self(predicted_noise)
        loss = self.criterion(pred,context)
        train_acc = self.accuracy(pred, context)

        # Log loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning training step

        Args:
            batch: Batch of data containing (x, context, _)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Unpack batch
        x, context, snr = batch

        # Sample random timesteps
        t = torch.randint(0, self.diffusion.n_steps, (x.shape[0],), device=self.device).long()

        # Encode input to latent space
        z = self.diffusion.encode(x)

        # Calculate diffusion loss - currently prediciting noise distribution
        noise_loss, predicted_noise = self.diffusion.p_losses(z, t, context)

        pred = self(predicted_noise)
        val_loss = self.criterion(pred,context)
        val_acc = self.accuracy(pred, context, snr)

        # Log loss
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc)

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
        # Get predicted class by taking argmax along class dimension
        predicted_classes = torch.argmax(pred, dim=1)
        # Compare predictions with ground truth and calculate accuracy
        correct = (predicted_classes == y).sum().item()
        acc = correct / y.size(0)

        # Get binary mask of correct predictions
        correct_mask = (predicted_classes == y).long()

        # Called only on validation steps
        if snr is not None:
            # Convert SNR values to indices (from -20:20:2 to 0:20)
            snr_indices = ((snr + 20) / 2).long()

            # For each SNR index, update total and correct counts
            for idx in range(20):
                mask = (snr_indices == idx)
                total = mask.sum().item()
                correct = (mask & correct_mask.bool()).sum().item()
                curr_correct, curr_total = self.snr_accuracy[idx]
                self.snr_accuracy[idx] = (curr_correct + correct, curr_total + total)

            # Track modulation-based accuracy
            for mod_idx in range(self.num_classes):
                mask = (y == mod_idx)
                total = mask.sum().item()
                correct = (mask & correct_mask.bool()).sum().item()
                curr_correct, curr_total = self.mod_accuracy[mod_idx]
                self.mod_accuracy[mod_idx] = (curr_correct + correct, curr_total + total)

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

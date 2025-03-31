import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from lightning.pytorch.loggers import WandbLogger
import wandb
from typing import List


class Router(L.LightningModule):
    """
    Router Module for Mixture of Experts Architecture

    This class implements a router network that dynamically assigns input data to different expert models.
    It uses a neural network to compute routing weights that determine how much each expert contributes
    to the final prediction.

    Args:
        cfg (DictConfig): Configuration object containing:
            - num_experts (int): Number of expert models (default: 5)
            - input_dim (int): Input dimension size (default: 256)
            - hidden_dim (int): Hidden layer dimension size (default: 128)
            - optimizer (dict): Optimizer configuration
            - experts (list): List of expert model configurations

    The router network consists of a simple MLP that outputs routing weights for each expert.
    These weights are used to combine the outputs of individual experts into a final prediction.
    """

    def __init__(
        self,
        label_names: List[str],
        input_dim: int,
        hidden_dim: int,
        feature_dim: int,
        num_classes: int,
        lr: float,
        load_balance_experts: bool,
        experts: List[nn.Module],
        classifier: nn.Module,
    ):
        super().__init__()
        # Get parameters from config
        self.label_names = label_names
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_experts = len(experts)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.load_balance = load_balance_experts
        self.num_classes = num_classes

        # Router network to determine expert weights
        self.router_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, self.num_experts),
        )

        # Store the experts directly from the config
        self.experts = nn.ModuleList(experts)

        # Create data for wandb bar chart
        self.expert_names = [
            f"Expert {self.experts[i].__str__().split('(')[0]}"
            for i in range(len(self.experts))
        ]

        # Initialize accumulator for expert usage across batches
        self.epoch_expert_usage = torch.zeros(self.num_experts)
        self.epoch_batch_count = 0

        # going to store [correct/total]
        self.snr_accuracy = [(0,0) for _ in range(20)]

        # going to store [correct/total]
        self.mod_accuracy = [(0,0) for _ in range(self.num_classes)]

        # Use the classifier passed from config
        self.classifier = classifier

    def forward(self, x):
        batch_size = x.size(0)

        # Get routing weights for each expert [batch_size, num_experts]
        # Flatten x to be a single dim for linear layer
        routing_weights = F.softmax(
            self.router_network(x.reshape(x.size(0), -1)), dim=-1
        )
        assert routing_weights.shape == (batch_size, self.num_experts), (
            "routing weights are incorrect shape"
        )

        # Apply each expert and combine results according to routing weights
        expert_outputs = []
        for expert in self.experts:
            # expert_output shape: [batch_size, feature_dim]
            expert_output = expert(x)
            expert_outputs.append(expert_output)

        # Stack results: [batch_size, num_experts, feature_dim]
        stacked_results = torch.stack(expert_outputs, dim=1)
        assert stacked_results.shape == (
            batch_size,
            self.num_experts,
            self.feature_dim,
        ), "experts should return hidden_dim size"

        # Reshape routing_weights for broadcasting: [batch_size,num_experts] -> [batch_size, num_experts, 1]
        weights = routing_weights.unsqueeze(-1)

        # Scale expert outputs based on router weights: [batch_size,experts, features] * [batch_size, num_experts,1] -> [batch_size, experts, features]
        # The sum along the experts to get the total weights
        combined_output = (stacked_results * weights).sum(dim=1)

        # Final classification - accepts hidden_dim * experts
        prediction = self.classifier(combined_output)

        return prediction, routing_weights

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        pred, routing_weights = self(x)

        # Calculate loss - pred, truth
        classification_loss = self.criterion(pred, y)
        train_acc = self.accuracy(pred,y)

        # Add routing loss to encourage load balancing
        # load_balancing_loss = self._compute_load_balancing_loss(routing_weights)

        # Combined loss
        # total_loss = classification_loss + load_balancing_loss

        # Log expert usage instead of balancing for now
        expert_usage = routing_weights.mean(dim=0)

        # Accumulate expert usage for epoch-end logging
        self.epoch_expert_usage += expert_usage.detach().to("cpu")
        self.epoch_batch_count += 1

        if self.load_balance:
            load_balancing_loss = self._compute_load_balancing_loss(expert_usage)
            total_loss = classification_loss + load_balancing_loss
            self.log("total_loss", total_loss, on_epoch=True, on_step=False)
        # Use self.log properly with name and value as separate arguments
        self.log("train_loss", classification_loss, on_epoch=True, on_step=False)
        self.log("train_acc", train_acc, on_epoch=True, on_step=False)

        # self.log('train_total_loss', total_loss)
        # self.log('train_load_balancing_loss', load_balancing_loss)
        return total_loss if self.load_balance else classification_loss

    def validation_step(self, batch, batch_idx):
        x, y, snr = batch
        pred, routing_weights = self(x)
        val_loss = self.criterion(pred, y)
        val_acc = self.accuracy(pred,y,snr)

        # These run every epoch by default
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)

        # self.log('val_total_loss', total_loss)
        # self.log('val_load_balancing_loss', load_balancing_loss)

    # Load balancing for future use cases maybe?
    def _compute_load_balancing_loss(self, expert_usage):
        # Ideal uniform distribution
        target_usage = torch.ones_like(expert_usage) / self.num_experts

        # Get current learning rate from optimizer
        balance_lr = self.optimizers().param_groups[0]['lr']

        # KL divergence loss to encourage uniform expert usage
        load_balancing_loss = F.kl_div(expert_usage.log(), target_usage, reduction='batchmean')

        return load_balancing_loss * balance_lr

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
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # reduce lr when val_loss plateaus after patience epochs
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=1, verbose=True
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self):
        # Calculate average expert usage over the epoch
        avg_expert_usage = self.epoch_expert_usage / self.epoch_batch_count

        # Convert tensor to list for wandb visualization
        usage_data = avg_expert_usage.tolist()

        # Create data for bar chart
        data = [[name, usage] for name, usage in zip(self.expert_names, usage_data)]
        table = wandb.Table(columns=["Expert", "Usage"], data=data)

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
            "Expert Usage": wandb.plot.bar(
                table, "Expert", "Usage", title="Expert Usage"
            ),
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
        self.epoch_expert_usage = torch.zeros(self.num_experts)
        self.epoch_batch_count = 0
        self.snr_accuracy = [(0,0) for _ in range(20)]
        self.mod_accuracy = [(0,0) for _ in range(self.num_classes)]


import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
import matplotlib.pyplot as plt
import wandb
from sklearn.manifold import TSNE  # Add this import


class LatentContrastiveEncoder(L.LightningModule):
    def __init__(self, label_names, encoder, projection_dim=128, temperature=0.5, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.learning_rate = learning_rate
        self.label_names = label_names
        self.temperature = temperature

        # Initialize encoder
        self.encoder = encoder

        # Get encoder output dimension
        # Assuming encoder has a latent_dim attribute, otherwise use default
        # latent_dim = getattr(encoder, 'latent_dim', encoder.fc.out_features)
        latent_dim = 2048

        # Add projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, projection_dim)
        )

    def encode(self, x):
        """Get latent representation from encoder"""
        return self.encoder(x)

    def project(self, features):
        """Project latent features to contrastive space"""
        #flatten feature vector first
        features = features.view(features.shape[0], -1)
        return self.projection_head(features)

    def forward(self, x):
        """Full forward pass: encode and project"""
        z = self.encode(x)
        return z, self.project(z)

    def nt_xent_loss(self, features, labels):
        """
        NT-Xent loss using labels to identify positive pairs
        Args:
            features: Projected features [batch_size, proj_dim]
            labels: Class labels [batch_size]
        """
        # Normalize features
        # L2 norm with F.normalize -- make easch row vector have length of 1
        features = F.normalize(features, dim=1)

        # Get batch size
        batch_size = features.size(0)

        # Compute similarity matrix - equilivant to cosine similarity becuase we L2 normalized already
        # determines how similar each signal is with each other, entry (i,j) is how similiar signals i and j ar
        sim_matrix = torch.matmul(features, features.T)

        # Create mask for positive pairs (same modulation class)
        # Create a binary matrix where labels_i == labels_j, unsqueezse to have [batch_size,1] and [1, batch_size]
        labels = labels.squeeze()
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # Remove self-similarity from positives (diagonal)
        # created identiy matrix of batch_size, that we invert to mask the identity so we dont use the same signal as positive pairs
        mask_no_self = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
        pos_mask = pos_mask * mask_no_self.float()

        # Temperature-scaled logits
        logits = sim_matrix / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Calculate log probability - convert similarity into probability
        exp_logits = torch.exp(logits) * mask_no_self.float()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Calculate positive pair loss
        # For each anchor, we find all positive pairs and average their loss
        pos_pairs_per_sample = pos_mask.sum(1)
        mask_has_positives = (pos_pairs_per_sample > 0)

        # Average loss for anchors with positives
        if mask_has_positives.sum() > 0:
            # Compute mean log-likelihood for positive pairs
            mean_log_prob_pos = (pos_mask * log_prob).sum(1)[mask_has_positives] / pos_pairs_per_sample[mask_has_positives]
            # Final NT-Xent loss
            loss = -mean_log_prob_pos.mean()
        else:
            loss = torch.tensor(0.0, device=self.device)  # Return tensor instead of scalar 0

        return loss

    def triplet_loss(self, features, labels, snrs, margin=2.0):
        """
        Modified triplet loss to ensure we're getting challenging pairs
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        batch_size = features.size(0)

        # Compute distance matrix
        distance_matrix = 1.0 - torch.matmul(features, features.T)

        # Create masks
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        pos_mask = pos_mask & (~torch.eye(batch_size, device=self.device, dtype=torch.bool))
        neg_mask = ~(labels.unsqueeze(0) == labels.unsqueeze(1))

        # Get all pairwise distances
        pos_dists = distance_matrix * pos_mask.float()
        neg_dists = distance_matrix * neg_mask.float()

        # Replace invalid distances (0 from mask) with appropriate values
        pos_dists[pos_mask == 0] = float('-inf')  # For max operation
        neg_dists[neg_mask == 0] = float('inf')   # For min operation

        # For each anchor, get hardest positive and negative
        hardest_pos_dists, pos_indices = pos_dists.max(dim=1)  # Furthest positive
        hardest_neg_dists, neg_indices = neg_dists.min(dim=1)  # Closest negative

        # Only consider valid triplets where:
        # 1. We have both positive and negative samples
        # 2. The negative is closer than (positive - margin), making it a hard case
        valid_mask = (hardest_pos_dists > float('-inf')) & (hardest_neg_dists < float('inf'))

        # Additional mask for hard triplets
        hard_triplet_mask = hardest_neg_dists < (hardest_pos_dists + margin)
        valid_mask = valid_mask & hard_triplet_mask

        # Compute triplet loss only for valid, hard triplets
        triplet_loss = F.relu(hardest_pos_dists - hardest_neg_dists + margin)

        if valid_mask.sum() > 0:
            # Add some regularization to prevent collapse
            loss = triplet_loss[valid_mask].mean()

            # Optional: Add L2 regularization to prevent feature collapse
            l2_reg = 0.01 * torch.norm(features, p=2, dim=1).mean()
            loss = loss + l2_reg

            return loss
        else:
            # If no valid triplets, return small positive value to prevent collapse
            return torch.tensor(0.1, device=self.device, requires_grad=True)

    def training_step(self, batch, batch_idx):
        x, labels, snr = batch

        # Get encoded representation and projected features
        z, projected = self(x)

        # Compute contrastive loss
        loss = self.triplet_loss(projected, labels, snrs=snr)
        # loss = self.nt_xent_loss(projected, labels)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels, snr = batch

        # Get encoded representation and projected features
        z, projected = self(x)

        # Compute contrastive loss
        loss = self.triplet_loss(projected, labels, snrs=snr)
        # loss = self.nt_xent_loss(projected, labels)



        self.log("val_loss", loss, prog_bar=True)

        # Save batch for visualization
        if batch_idx == 0:
            self.val_latents = z.detach().cpu()
            self.val_labels = labels.detach().cpu()
            self.val_snrs = snr.detach().cpu()

        return loss

    def on_validation_epoch_end(self):
        """Visualize latent space with t-SNE"""
        if not hasattr(self, 'val_latents'):
            return

        # Flatten the latent representations if they're 3D
        latents_2d = self.val_latents.view(self.val_latents.size(0), -1)  # [batch_size, features]

        # Create t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        z_tsne = tsne.fit_transform(latents_2d)

        # Create plot
        plt.figure(figsize=(10, 8))

        # Get unique labels and assign colors
        unique_labels = torch.unique(self.val_labels)
        # Fix: use a properly defined colormap
        cmap = plt.cm.get_cmap('tab20')
        colors = cmap(torch.linspace(0, 1, len(unique_labels)).numpy())

        # Plot each class with its own color
        for i, label in enumerate(unique_labels):
            mask = self.val_labels.squeeze() == label
            plt.scatter(
                z_tsne[mask, 0], z_tsne[mask, 1],
                c=[colors[i]],
                label=self.label_names[int(label)],
                alpha=0.7
            )

        plt.legend(fontsize=12)
        plt.title("t-SNE Visualization of Latent Space", fontsize=16)
        plt.xlabel("t-SNE dimension 1", fontsize=14)
        plt.ylabel("t-SNE dimension 2", fontsize=14)
        plt.tight_layout()

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({"latent_tsne": wandb.Image(plt)})

        plt.close()

        # Create SNR-based visualization
        # Get unique labels and SNRs
        unique_snrs = torch.unique(self.val_snrs)
        unique_labels = torch.unique(self.val_labels)

        num_mods = len(unique_labels)  # Show all modulations
        rows = (num_mods + 2) // 3  # Calculate needed rows (ceil(num_mods/3))

        # Create figure with adjusted layout
        fig = plt.figure(figsize=(15, 5*rows))  # Adjust figure height based on rows
        # Add space for colorbar by adjusting the gridspec
        gs = plt.GridSpec(rows, 4, figure=fig, width_ratios=[1, 1, 1, 0.1])  # rows x 4 grid with narrow last column for colorbar

        # Initialize scatter for colorbar
        scatter = None

        # Plot each modulation with SNR
        for i in range(num_mods):
            label = unique_labels[i]
            # Create subplot in rows x 3 grid (excluding the last column reserved for colorbar)
            ax = fig.add_subplot(gs[i // 3, i % 3])  # Integer division and modulo for layout

            # Get points for this modulation
            mask = self.val_labels.squeeze() == label
            if not torch.any(mask):
                continue

            mod_z = z_tsne[mask]
            mod_snrs = self.val_snrs[mask].numpy()

            # Scatter plot with SNR as color
            scatter = ax.scatter(
                mod_z[:, 0], mod_z[:, 1],
                c=mod_snrs,
                cmap='viridis',
                alpha=0.8,
                vmin=min(unique_snrs).item(),
                vmax=max(unique_snrs).item()
            )

            ax.set_title(f"{self.label_names[int(label)]}")

        # Add colorbar in the reserved space if scatter exists
        if scatter is not None:
            cbar_ax = fig.add_subplot(gs[:, -1])  # Use the last column for colorbar
            plt.colorbar(scatter, cax=cbar_ax, label='SNR (dB)')

        fig.suptitle("t-SNE by Modulation with SNR as Color", fontsize=16, y=1.02)
        plt.tight_layout()

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({"latent_snr": wandb.Image(fig)})

        plt.close()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-1
        )
        # Fix the type error with total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.05,
            anneal_strategy="cos",
            final_div_factor=1000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

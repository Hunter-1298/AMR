import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
import matplotlib.pyplot as plt
import wandb
from sklearn.manifold import TSNE
import copy


class MoCoV3Encoder(L.LightningModule):
    def __init__(self, label_names, encoder, projection_dim=256, hidden_dim=256, learning_rate=1e-3, temperature=0.2, momentum=0.996):
        # https://arxiv.org/pdf/2104.02057 - MoCoV3
        # MoCoV3 paper uses hidden_dim=4096
        # using large batch size ~ 4096 in order to ignore queue from MoCoV2
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.learning_rate = learning_rate
        self.label_names = label_names
        self.temperature = temperature
        self.momentum = momentum

        # Create online and target networks
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)

        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Get encoder output dimension [32 x 64] from our encoder that we flatten
        latent_dim = 2048

        # Projection head (same for both networks)
        self.online_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.target_projector = copy.deepcopy(self.online_projector)
        for param in self.target_projector.parameters():
            param.requires_grad = False

        # Prediction head (only for online network)
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def encode(self, x):
        """Get latent representation from encoder (for downstream tasks)"""
        return self.online_encoder(x)

    @torch.no_grad()
    def _update_target_network(self):
        """Update target network using momentum encoder"""
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = target_params.data * self.momentum + \
                                online_params.data * (1 - self.momentum)

        for online_params, target_params in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_params.data = target_params.data * self.momentum + \
                                online_params.data * (1 - self.momentum)

    def forward_target(self, x):
        """Get target network representations"""
        with torch.no_grad():
            target_features = self.target_encoder(x)
            target_features_flat = target_features.view(target_features.shape[0], -1)
            target_projections = self.target_projector(target_features_flat)
        return target_projections

    def forward_online(self, x):
        """Get online network representations"""
        online_features = self.online_encoder(x)
        online_features_flat = online_features.view(online_features.shape[0], -1)
        online_projections = self.online_projector(online_features_flat)
        online_predictions = self.predictor(online_projections)
        return online_predictions

    def forward(self, x):
        """Standard forward pass for inference and feature extraction"""
        return self.encode(x)

    def info_nce_loss(self, query, key, batch_size):
        """
        InfoNCE loss implementation

        Args:
            query: Online network output [N, dim]
            key: Target network output [N, dim]
            batch_size: Batch size N
        """
        # Normalize embeddings - same as L2 norm, assuming cos = 1
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        # Compute logits
        # positive logits: Nx1 -> computes similiary between i from query and i from key -> equilivant to dot product
        # gets doit product between signal[i] from query and signal[i] from key, whcih are different views of the signal
        l_pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        # negative logits: NxN -> computes pairwise similiarites between all pairs i,j between query and key
        l_neg = torch.einsum('nc,ck->nk', [query, key.T])

        # Remove self-contrast cases - removes positive on diagonals from our negative matrcies
        mask = torch.eye(batch_size, device=query.device)
        l_neg = l_neg.masked_fill(mask == 1, float('-inf'))

        # Logits: Nx(1+N-1)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature scaling
        logits /= self.temperature

        # Labels: positives are the 0th
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        # Calculate loss
        loss = F.cross_entropy(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        # Unpack batch
        (x1, x2), labels, snrs = batch
        batch_size = x1.shape[0]

        # Get online predictions
        q1 = self.forward_online(x1)  # queries: queries from online network
        q2 = self.forward_online(x2)  # queries: queries from online network

        # Get target projections
        with torch.no_grad():
            k1 = self.forward_target(x1)  # keys: targets from momentum network
            k2 = self.forward_target(x2)  # keys: targets from momentum network

        # Compute loss both ways (symmetric loss)
        loss_1 = self.info_nce_loss(q1, k2, batch_size)
        loss_2 = self.info_nce_loss(q2, k1, batch_size)

        # Average both loss terms
        loss = 0.5 * (loss_1 + loss_2)

        # Update target network
        self._update_target_network()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        (x1, x2), labels, snrs = batch
        batch_size = x1.shape[0]

        # Get online predictions
        q1 = self.forward_online(x1)
        q2 = self.forward_online(x2)

        # Get target projections
        with torch.no_grad():
            k1 = self.forward_target(x1)
            k2 = self.forward_target(x2)

        # Compute loss both ways (symmetric loss)
        loss_1 = self.info_nce_loss(q1, k2, batch_size)
        loss_2 = self.info_nce_loss(q2, k1, batch_size)

        # Average both loss terms
        loss = 0.5 * (loss_1 + loss_2)

        self.log("val_loss", loss, prog_bar=True)

        # Save latent vectors for visualization
        # Use the first view only for visualization to avoid duplicates
        if batch_idx == 0:
            # Extract features before projection for visualization
            with torch.no_grad():
                latent_features = self.online_encoder(x1)

            self.val_latents = latent_features.detach().cpu()
            self.val_labels = labels.detach().cpu()
            self.val_snrs = snrs.detach().cpu()

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
        plt.title("t-SNE Visualization of MoCo Latent Space", fontsize=16)
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

        fig.suptitle("MoCo t-SNE by Modulation with SNR as Color", fontsize=16, y=1.02)
        plt.tight_layout()

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({"latent_snr": wandb.Image(fig)})

        plt.close()

    def configure_optimizers(self):
        # Start with learning rate of 0.01 as specified in the paper
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-1
        )

        # Use ReduceLROnPlateau scheduler that will halve the learning rate
        # if validation loss doesn't improve for 5 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # halve the learning rate
            patience=5,  # wait 5 epochs for improvement
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitor validation loss
                "interval": "epoch",
                "frequency": 1,
            },
        }

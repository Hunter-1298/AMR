import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.manifold import TSNE

class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution for I/Q processing"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # x shape: [batch, 2, length] where dim=1 is [I, Q]
        i_channel = x[:, 0:1]  # Real part
        q_channel = x[:, 1:2]  # Imaginary part

        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_part = self.conv_real(i_channel) - self.conv_imag(q_channel)
        imag_part = self.conv_real(q_channel) + self.conv_imag(i_channel)

        return torch.cat([real_part, imag_part], dim=1)

class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for RF signals"""
    def __init__(self, in_channels=2):
        super().__init__()
        # Different kernel sizes to capture different temporal patterns
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(in_channels, 64, kernel_size=3),
            self._make_conv_block(in_channels, 64, kernel_size=7),
            self._make_conv_block(in_channels, 64, kernel_size=15),
            self._make_conv_block(in_channels, 64, kernel_size=31)
        ])

    def _make_conv_block(self, in_ch, out_ch, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            ComplexConv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch * 2),  # *2 because complex output has I/Q
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        features = []
        for block in self.conv_blocks:
            features.append(block(x))
        return torch.cat(features, dim=1)  # Concatenate along channel dimension

class FrequencyDomainProcessor(nn.Module):
    """Process frequency domain features"""
    def __init__(self, signal_length=128):
        super().__init__()
        self.signal_length = signal_length
        # Process magnitude and phase separately
        self.mag_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.phase_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: [batch, 2, length] - I/Q channels
        batch_size = x.shape[0]

        # Convert to complex tensor
        complex_signal = torch.complex(x[:, 0], x[:, 1])  # [batch, length]

        # FFT
        fft_signal = torch.fft.fft(complex_signal, dim=-1)

        # Extract magnitude and phase
        magnitude = torch.abs(fft_signal).unsqueeze(1)  # [batch, 1, length]
        phase = torch.angle(fft_signal).unsqueeze(1)    # [batch, 1, length]

        # Process magnitude and phase
        mag_features = self.mag_processor(magnitude)
        phase_features = self.phase_processor(phase)

        return torch.cat([mag_features, phase_features], dim=1)

class AttentionBlock(nn.Module):
    """Self-attention for important feature selection"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, length = x.shape

        # Generate query, key, value
        q = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C//8]
        k = self.key(x).view(batch_size, -1, length)                      # [B, C//8, L]
        v = self.value(x).view(batch_size, -1, length).permute(0, 2, 1)   # [B, L, C]

        # Attention weights
        attention = torch.bmm(q, k)  # [B, L, L]
        attention = F.softmax(attention, dim=-1)

        # Apply attention
        out = torch.bmm(attention, v)  # [B, L, C]
        out = out.permute(0, 2, 1).view(batch_size, channels, length)

        return self.gamma * out + x

class RFEncoder(nn.Module):
    """Encoder for RF I/Q signals"""
    def __init__(self, signal_length=128, latent_dim=256):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim

        # Multi-scale time domain features
        self.time_features = MultiScaleFeatureExtractor(in_channels=2)

        # Frequency domain features
        self.freq_features = FrequencyDomainProcessor(signal_length)

        # Combine features (256 from time + 128 from freq = 384 channels)
        combined_channels = 256 + 128

        # Attention mechanism
        self.attention = AttentionBlock(combined_channels)

        # Encoder layers with residual connections
        self.encoder_layers = nn.Sequential(
            nn.Conv1d(combined_channels, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv1d(512, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv1d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Calculate the size after convolutions
        self.encoded_size = self._get_encoded_size()

        # Final latent representation
        self.to_latent = nn.Sequential(
            nn.Linear(self.encoded_size, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def _get_encoded_size(self):
        """Calculate the size after convolutions"""
        with torch.no_grad():
            x = torch.randn(1, 2, self.signal_length)
            time_feat = self.time_features(x)
            freq_feat = self.freq_features(x)
            combined = torch.cat([time_feat, freq_feat], dim=1)
            encoded = self.encoder_layers(combined)
            return encoded.numel()

    def forward(self, x):
        # Extract time and frequency domain features
        time_features = self.time_features(x)
        freq_features = self.freq_features(x)

        # Combine features
        combined_features = torch.cat([time_features, freq_features], dim=1)

        # Apply attention
        attended_features = self.attention(combined_features)

        # Encode
        encoded = self.encoder_layers(attended_features)

        # Flatten and get latent representation
        encoded_flat = encoded.view(encoded.shape[0], -1)
        latent = self.to_latent(encoded_flat)

        return latent

class RFDecoder(nn.Module):
    """Decoder to reconstruct RF I/Q signals"""
    def __init__(self, latent_dim=256, signal_length=128):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim

        # Calculate initial size for decoder
        self.init_size = signal_length // 8  # After 3 upsampling layers

        # From latent to initial feature map
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoder layers
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose1d(128, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Final layer to get I/Q channels
            nn.Conv1d(64, 2, kernel_size=5, padding=2),
            nn.Tanh()  # Assuming normalized input
        )

    def forward(self, latent):
        # From latent to feature map
        x = self.from_latent(latent)
        x = x.view(x.shape[0], 128, self.init_size)

        # Decode
        reconstructed = self.decoder_layers(x)

        # Ensure correct output size
        if reconstructed.shape[-1] != self.signal_length:
            reconstructed = F.interpolate(reconstructed, size=self.signal_length, mode='linear', align_corners=False)

        return reconstructed

class ArcFaceLoss(nn.Module):
    """ArcFace loss for better angular separation"""
    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)

        # Get target cosine values
        target_cosine = cosine[torch.arange(len(labels)), labels]

        # Add margin to target
        target_theta = torch.acos(torch.clamp(target_cosine, -1 + 1e-7, 1 - 1e-7))
        target_theta_margin = target_theta + self.margin
        target_cosine_margin = torch.cos(target_theta_margin)

        # Replace target values with margin-adjusted values
        logits = cosine * 1.0
        logits[torch.arange(len(labels)), labels] = target_cosine_margin

        # Scale logits
        logits *= self.scale

        return F.cross_entropy(logits, labels)

class RFEncoderDecoder(L.LightningModule):
    """Complete encoder-decoder with ArcFace loss"""
    def __init__(self, label_names, signal_length=128, latent_dim=256, learning_rate=1e-3,
                 arcface_margin=0.5, arcface_scale=64, reconstruction_weight=1.0, arcface_weight=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.label_names = label_names
        self.num_classes = len(label_names)
        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.arcface_weight = arcface_weight

        # Networks
        self.encoder = RFEncoder(signal_length, latent_dim)
        self.decoder = RFDecoder(latent_dim, signal_length)

        # Loss functions
        self.arcface_loss = ArcFaceLoss(latent_dim, self.num_classes, arcface_margin, arcface_scale)
        self.reconstruction_loss = nn.MSELoss()

    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)

    def decode(self, latent):
        """Reconstruct from latent"""
        return self.decoder(latent)

    def forward(self, x):
        """Full forward pass"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def training_step(self, batch, batch_idx):
        # For training, we expect single signals with labels
        x, labels, snrs = batch

        # Forward pass
        latent, reconstructed = self.forward(x)

        # Compute losses
        recon_loss = self.reconstruction_loss(reconstructed, x)
        arcface_loss = self.arcface_loss(latent, labels.squeeze())

        # Combined loss
        total_loss = (self.reconstruction_weight * recon_loss +
                     self.arcface_weight * arcface_loss)

        # Logging
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_arcface_loss", arcface_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, labels, snrs = batch

        # Forward pass
        latent, reconstructed = self.forward(x)

        # Compute losses
        recon_loss = self.reconstruction_loss(reconstructed, x)
        arcface_loss = self.arcface_loss(latent, labels.squeeze())

        # Combined loss
        total_loss = (self.reconstruction_weight * recon_loss +
                     self.arcface_weight * arcface_loss)

        # Logging
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_arcface_loss", arcface_loss)

        # Save for visualization
        if batch_idx == 0:
            self.val_latents = latent.detach().cpu()
            self.val_labels = labels.detach().cpu()
            self.val_snrs = snrs.detach().cpu()
            self.val_original = x.detach().cpu()
            self.val_reconstructed = reconstructed.detach().cpu()

        return total_loss

    def on_validation_epoch_end(self):
        """Visualize latent space and reconstructions"""
        if not hasattr(self, 'val_latents'):
            return

        # t-SNE visualization
        self._plot_tsne()
        self._plot_reconstructions()

    def _plot_tsne(self):
        """Create t-SNE plot of latent space"""
        tsne = TSNE(n_components=2, random_state=42)
        z_tsne = tsne.fit_transform(self.val_latents.numpy())

        plt.figure(figsize=(10, 8))
        unique_labels = torch.unique(self.val_labels)
        cmap = plt.cm.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = self.val_labels.squeeze() == label
            plt.scatter(
                z_tsne[mask, 0], z_tsne[mask, 1],
                c=[colors[i]], label=self.label_names[int(label)],
                alpha=0.7
            )

        plt.legend()
        plt.title("t-SNE Visualization of ArcFace Latent Space")
        plt.tight_layout()

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({"latent_tsne": wandb.Image(plt)})

        plt.close()

    def _plot_reconstructions(self):
        """Plot original vs reconstructed signals"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for i in range(min(4, len(self.val_original))):
            # Original I/Q
            axes[0, i].plot(self.val_original[i, 0].numpy(), label='I', alpha=0.7)
            axes[0, i].plot(self.val_original[i, 1].numpy(), label='Q', alpha=0.7)
            axes[0, i].set_title(f'Original - {self.label_names[int(self.val_labels[i])]}')
            axes[0, i].legend()

            # Reconstructed I/Q
            axes[1, i].plot(self.val_reconstructed[i, 0].numpy(), label='I', alpha=0.7)
            axes[1, i].plot(self.val_reconstructed[i, 1].numpy(), label='Q', alpha=0.7)
            axes[1, i].set_title('Reconstructed')
            axes[1, i].legend()

        plt.tight_layout()

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({"reconstructions": wandb.Image(fig)})

        plt.close()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

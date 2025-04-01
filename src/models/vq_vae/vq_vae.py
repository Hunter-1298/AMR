import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class RFVQVAE(L.LightningModule):
    def __init__(self, encoder, decoder, input_dim=2, embedding_dim=64, codebook_size=512, commitment_cost=0.25, learning_rate=1e-4, iq_window=1, epochs=10):
        super().__init__()
        self.save_hyperparameters()
        
        self.iq_window = iq_window

        self.input_dim = input_dim
        
        # Encoder (IQ data → latent space)
        self.encoder = encoder       

        # Vector quantizer
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # controls how strongly encoder output is pulled to stayu closer to codebook
        self.commitment_cost = commitment_cost
        
        # Decoder
        self.decoder = decoder         

        self.learning_rate = learning_rate

    def vector_quantize(self, z):
        # z shape: [batch_size, channels, num_windows, embedding_dim]
        # Reshape to [batch_size * num_windows, embedding_dim]
        batch_size, channels, num_windows, embedding_dim = z.shape
        flat_z = z.transpose(2, 3).reshape(-1, embedding_dim)
        
        # Calculate distances for each window independently
        distances = (flat_z.unsqueeze(1) - self.codebook.weight.unsqueeze(0)) ** 2
        distances = distances.sum(dim=-1)
        
        # Get index of nearest embedding for each window
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize using codebook
        quantized = self.codebook(encoding_indices)
        
        # Reshape back to match input dimensions
        quantized = quantized.view(batch_size, channels, num_windows, -1)
        
        # Calculate VQ loss
        q_latent_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized, z.detach())
        vq_loss = q_latent_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, encoding_indices, q_latent_loss, commitment_loss, vq_loss

    
    def forward(self, x):
        # get the latent space encoding
        batch_size, channels, iq = x.shape
        assert iq % self.iq_window == 0, "IQ data not divisible into windows"
        x = x.view(batch_size, channels, iq//self.iq_window, self.iq_window) 
        encoded_x = self.encoder(x)

        #quantize into codebook
        quantized, indices, latent_loss, commitment_loss, vq_loss = self.vector_quantize(encoded_x)
        
        # reconstruct the input from the latent space
        reconstructed = self.decoder(quantized)

        return reconstructed, quantized, indices, latent_loss, commitment_loss, vq_loss
    
    def training_step(self, batch, batch_idx):
        x, _, _ = batch  # Ignore labels and SNR for VQVAE training
        
        reconstructed, _, _, latent_loss, commitment_loss, vq_loss = self(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)
        
        # Total loss
        total_loss = recon_loss + vq_loss
        
        # Log metrics
        self.log("train_loss", total_loss, on_epoch=True, on_step=False)
        self.log("reconstruction_loss", recon_loss, on_epoch=True, on_step=False)
        self.log("latent_loss", latent_loss, on_epoch=True, on_step=False)
        self.log("commitment_loss", commitment_loss, on_epoch=True, on_step=False)
        self.log("vq_loss", vq_loss, on_epoch=True, on_step=False)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        
        reconstructed, _, _, latent_loss, commitment_loss, vq_loss = self(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)
        
        # Total loss
        total_loss = recon_loss + vq_loss
        
        # Log metrics
        self.log("val_loss", total_loss, on_epoch=True, on_step=False)
        self.log("val_reconstruction_loss", recon_loss, on_epoch=True, on_step=False)
        self.log("val_latent_loss", latent_loss, on_epoch=True, on_step=False)
        self.log("val_commitment_loss", commitment_loss, on_epoch=True, on_step=False)
        self.log("val_vq_loss", vq_loss, on_epoch=True, on_step=False)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
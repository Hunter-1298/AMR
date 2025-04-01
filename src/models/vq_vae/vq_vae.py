import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class RFVQVAE(L.LightningModule):
    def __init__(self, encoder, decoder,quantizer, input_dim=2, embedding_dim=64, codebook_size=512, commitment_cost=0.25, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        # Create parallel networks to learn amplitude and phase separatly
        self.encoder_amp = encoder
        self.encoder_phase = encoder

        self.vq_amp = quantizer
        self.vq_phase = quantizer

        self.decoder_amp = decoder
        self.decoder_phase = decoder


    def forward(self, amp, phase):
        z_amp = self.encoder_amp(amp)
        # Encode
        z_phase = self.encoder_phase(phase)

        # Quantize
        q_amp, amp_indices, loss_vq_amp = self.vq_amp(z_amp)
        q_phase, phase_indices, loss_vq_phase = self.vq_phase(z_phase)

        # Decode
        amp_recon = self.decoder_amp(q_amp)
        phase_recon = self.decoder_phase(q_phase)

        # Losses
        recon_loss = F.mse_loss(amp_recon, amp) + F.mse_loss(phase_recon, phase)
        vq_loss = loss_vq_amp + loss_vq_phase

        return amp_recon, phase_recon, amp_indices, phase_indices, recon_loss + vq_loss

    def training_step(self, batch, batch_idx):
        x, _, _ = batch  # Ignore labels and SNR for VQVAE training
        
        # Split x into amplitude and phase components
        amp, phase = x[:, 0:1], x[:, 1:2]  # Assuming x has shape [batch_size, 2, ...]
        
        # Call forward with both components
        amp_recon, phase_recon, amp_indices, phase_indices, total_loss = self(amp, phase)
        
        # Log metrics
        self.log("train_loss", total_loss, on_epoch=True, on_step=False)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        
        # Split x into amplitude and phase components
        amp, phase = x[:, 0:1], x[:, 1:2]  # Assuming x has shape [batch_size, 2, ...]
        
        # Call forward with both components
        amp_recon, phase_recon, amp_indices, phase_indices, total_loss = self(amp, phase)
        
        # Log metrics
        self.log("val_loss", total_loss, on_epoch=True, on_step=False)
        
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
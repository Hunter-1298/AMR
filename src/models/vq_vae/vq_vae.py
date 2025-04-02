import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import wandb
import matplotlib.pyplot as plt

class RFVQVAE(L.LightningModule):
    def __init__(self, encoder_amp, encoder_phase,
                 decoder_amp, decoder_phase,
                 quantizer_amp, quantizer_phase,
                 input_dim=2, embedding_dim=64,
                 codebook_size=512, commitment_cost=0.25,
                 learning_rate=1e-4
                 ):

        super().__init__()
        self.learning_rate = learning_rate

        # Create parallel networks to learn amplitude and phase separately
        self.encoder_amp = encoder_amp
        self.encoder_phase = encoder_phase 

        # Create separate quantizer instances
        self.vq_amp = quantizer_amp
        self.vq_phase = quantizer_phase

        # Create separate decoder instances
        self.decoder_amp = decoder_amp
        self.decoder_phase = decoder_phase

        
        
    # get samples from the rfml dataset and encode after training
    def encode(self, samples):
        amp = self.encoder_amp(samples[:,[0],:])
        phase = self.encoder_phase(samples[:,[1],:])
        return torch.stack((amp, phase), dim=1)


    def forward(self, amp, phase):
        # Encode
        # [32,1,128] -> [32,32,64]
        z_amp = self.encoder_amp(amp)
        z_phase = self.encoder_phase(phase)

        # Quantize
        q_amp, amp_indices, loss_vq_amp = self.vq_amp(z_amp)
        q_phase, phase_indices, loss_vq_phase = self.vq_phase(z_phase)

        # Decode
        amp_recon = self.decoder_amp(q_amp)
        phase_recon = self.decoder_phase(q_phase)

        amp_recon_loss = F.mse_loss(amp_recon, amp)  * 50
        phase_recon_loss = F.mse_loss(phase_recon, phase)  

        # Losses
        recon_loss = amp_recon_loss + phase_recon_loss
        vq_loss = loss_vq_amp + loss_vq_phase

        return amp_recon, phase_recon, amp_indices, phase_indices, recon_loss + vq_loss, vq_loss, recon_loss, amp_recon_loss, phase_recon_loss

    def training_step(self, batch, batch_idx):
        x, _, _ = batch  # Ignore labels and SNR for VQVAE training
        
        # Split x into amplitude and phase components
        amp, phase = x[:, 0:1], x[:, 1:2]  # Assuming x has shape [batch_size, 2, ...]
        
        # Call forward with both components
        _, _, amp_indices, phase_indices, total_loss, _, _, _, _ = self(amp, phase)
        
        # Log metrics
        self.log("train_loss", total_loss, on_epoch=True, on_step=False)
        
        return total_loss


    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        
        # Split x into amplitude and phase components
        amp, phase = x[:, 0:1], x[:, 1:2]  # Assuming x has shape [batch_size, 2, ...]
        
        # Call forward with both components
        amp_recon, phase_recon, amp_indices, phase_indices, total_loss, vq_loss, recon_loss, amp_recon_loss, phase_recon_loss = self(amp, phase)
        
        # Log metrics
        self.log("val_loss", total_loss, on_epoch=True, on_step=False)
        self.log("recon_loss", recon_loss, on_epoch=True, on_step=False)
        self.log("amp_recon_loss", amp_recon_loss, on_epoch=True, on_step=False)
        self.log("phase_recon_loss", phase_recon_loss, on_epoch=True, on_step=False)
        self.log("vq_loss", vq_loss, on_epoch=True, on_step=False)

        # Store indices for epoch-end histogram - create new list at start of epoch
        if batch_idx == 0:
            self.epoch_phase_indices = phase_indices.detach().cpu()
            self.epoch_amp_indices= amp_indices.detach().cpu()
            self.example_amp = amp[0].detach().cpu()
            self.example_phase = phase[0].detach().cpu()
            self.example_amp_recon = amp_recon[0].detach().cpu()
            self.example_phase_recon = phase_recon[0].detach().cpu()
        
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

    def on_train_epoch_end(self):
        # Convert list of tensors to a single tensor using stack instead of cat
        all_amp_indices = self.epoch_amp_indices.flatten().numpy()
        all_phase_indices = self.epoch_phase_indices.flatten().numpy()
        
        # Create signal plots
        amp_fig = self._create_comparison_plot(
            self.example_amp.numpy().flatten(),
            self.example_amp_recon.numpy().flatten(),
            "Amplitude Comparison"
        )
        
        phase_fig = self._create_comparison_plot(
            self.example_phase.numpy().flatten(),
            self.example_phase_recon.numpy().flatten(),
            "Phase Comparison"
        )
        
        # Log everything to wandb - remove np_histogram parameter
        self.logger.experiment.log({
            "train/amp_codebook_hist": wandb.Histogram(all_amp_indices),
            "train/phase_codebook_hist": wandb.Histogram(all_phase_indices),
            "train/amp_comparison": amp_fig,
            "train/phase_comparison": phase_fig,
            "epoch": self.current_epoch
        })
        
        # Clear the stored indices
        self.epoch_amp_indices = []
        self.epoch_phase_indices = []

    def _create_comparison_plot(self, original, reconstructed, title):
        """Helper function to create comparison plots"""
        
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(original))
        ax.plot(x, original, label='Original', alpha=0.7)
        ax.plot(x, reconstructed, label='Reconstructed', alpha=0.7)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return wandb.Image(fig)
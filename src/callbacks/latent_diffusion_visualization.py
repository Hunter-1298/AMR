import torch
import matplotlib.pyplot as plt
import lightning as L
import wandb
import numpy as np
import imageio

class DiffusionVisualizationCallback(L.Callback):
    """Callback for visualizing the diffusion process"""

    def __init__(self, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    # def on_validation_epoch_end(self, trainer, pl_module):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Create visualizations at the end of validation epoch"""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return


        # Use the stored batch from the model
        if hasattr(pl_module, 'example_batch'):
            batch = pl_module.example_batch

            # 1. Visualize forward and reverse diffusion
            forward_fig, reverse_fig = self.create_diffusion_visualizations(pl_module, batch)

            # Log to wandb
            trainer.logger.experiment.log({
                "diffusion/forward": wandb.Image(forward_fig),
                "diffusion/reverse": wandb.Image(reverse_fig),
                "epoch": trainer.current_epoch
            })

            plt.close(forward_fig)
            plt.close(reverse_fig)

        else:
            print("No example batch found for visualization")

    def create_diffusion_visualizations(self, model, batch):
        """Create forward and reverse diffusion visualizations"""
        x, context, _ = batch
        x = x[:1].to(model.device)  # Just use first example

        # Select timesteps to visualize
        n_steps = 5
        timesteps = torch.linspace(0, model.n_steps-1, n_steps).long().to(model.device)

        # 1. Forward Diffusion (Adding Noise)
        forward_fig, forward_axes = plt.subplots(1, n_steps, figsize=(n_steps*3, 3))

        with torch.no_grad():
            z = model.encode(x)
            for i, t in enumerate(timesteps):
                # Apply noise
                noisy_z, _ = model.q_sample(z, t.unsqueeze(0))
                # Decode to signal space
                decoded = model.encoder.decoder(noisy_z)

                # Plot
                forward_axes[i].plot(decoded[0, 0].cpu().numpy(), label='Real')
                forward_axes[i].plot(decoded[0, 1].cpu().numpy(), label='Imag')
                forward_axes[i].set_title(f"t={t.item()}")

        forward_axes[0].legend()
        forward_fig.suptitle("Forward Diffusion (Adding Noise)")
        forward_fig.tight_layout()

        # 2. Reverse Diffusion (Denoising)
        reverse_fig, reverse_axes = plt.subplots(1, n_steps, figsize=(n_steps*3, 3))

        with torch.no_grad():
            # Start with noisy latent
            noisy_z, _ = model.q_sample(z, torch.tensor([model.n_steps-1], device=model.device))
            current_z = noisy_z.clone()

            for i, t in enumerate(reversed(timesteps)):
                # Predict noise
                predicted_noise = model(current_z, t.unsqueeze(0))

                # Denoising step
                alpha_t = model.alpha[t]
                alpha_bar_t = model.alpha_bar[t]
                beta_t = model.beta[t]

                # Estimate clean latent (x_0) from noisy latent (x_t) using the DDPM formula:
                # In the forward process: x_t = sqrt(α̅_t) * x_0 + sqrt(1-α̅_t) * ε
                # Solving for x_0: x_0 = (x_t - sqrt(1-α̅_t) * ε) / sqrt(α̅_t)
                # where ε is the predicted noise from the UNet model
                # More stable formula with clipping
                pred_original = (current_z - (1-alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()

                # Add clipping to prevent extreme values
                pred_original = torch.clamp(pred_original, -10.0, 10.0)

                # For visualization, we can directly use the predicted clean latent
                decoded = model.encoder.decoder(pred_original)

                # Plot
                reverse_axes[i].plot(decoded[0, 0].cpu().numpy(), label='Real' if i==0 else None)
                reverse_axes[i].plot(decoded[0, 1].cpu().numpy(), label='Imag' if i==0 else None)
                reverse_axes[i].set_title(f"t={t.item()}")

                # Update for next step (if not the last step)
                if i < n_steps - 1:
                    next_t = timesteps[-(i+2)]
                    noise = torch.randn_like(current_z)
                    current_z = alpha_t.sqrt() * pred_original

        reverse_axes[0].legend()
        reverse_fig.suptitle("Reverse Diffusion (Denoising)")
        reverse_fig.tight_layout()

        return forward_fig, reverse_fig

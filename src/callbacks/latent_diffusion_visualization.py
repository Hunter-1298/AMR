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
            denoising_fig = self.create_diffusion_visualizations(pl_module, batch)

            # Log to wandb
            trainer.logger.experiment.log({
                "diffusion/forward": wandb.Image(denoising_fig),
                "epoch": trainer.current_epoch
            })

            plt.close(denoising_fig)

        else:
            print("No example batch found for visualization")

    def create_diffusion_visualizations(self, model, batch):
        """Create forward and reverse diffusion visualizations with direct comparison"""
        x, context, _ = batch
        x = x[:1].to(model.device)  # Just use first example

        # Select timesteps to visualize
        n_steps = 5
        timesteps = torch.linspace(0, model.n_steps-1, n_steps).long().to(model.device)

        # Create figure with two rows
        fig, axes = plt.subplots(2, n_steps, figsize=(n_steps*3, 6))

        # Store noisy latents for each timestep
        noisy_latents = []

        with torch.no_grad():
            z = model.encode(x)

            # 1. Forward Diffusion (Adding Noise) - Top row
            for i, t in enumerate(timesteps):
                # Apply noise
                noisy_z, _ = model.q_sample(z, t.unsqueeze(0))
                noisy_latents.append(noisy_z.clone())  # Store for later denoising

                # Decode to signal space
                decoded = model.encoder.decoder(noisy_z)

                # Plot
                axes[0, i].plot(decoded[0, 0].cpu().numpy(), label='Real' if i==0 else None)
                axes[0, i].plot(decoded[0, 1].cpu().numpy(), label='Imag' if i==0 else None)
                axes[0, i].set_title(f"Noisy t={t.item()}")
                # axes[0, i].set_ylim(-1.5, 1.5)

            # Add legend to first plot
            axes[0, 0].legend()

            # 2. Reverse Diffusion (Denoising) - Bottom row
            # Now we denoise each noisy latent independently
            for i, t in enumerate(timesteps):
                # Get the corresponding noisy latent
                noisy_z = noisy_latents[i]

                # Predict noise
                predicted_noise = model(noisy_z, t.unsqueeze(0))

                # Stable formula to predict x0 directly
                #x₀ = (x_t - sqrt(1-α̅ₜ) * ε_θ(x_t, t)) / sqrt(α̅ₜ)
                # Where:
                # - `α̅ₜ` is alpha_bar_t (cumulative product of alphas up to timestep t)
                # - `ε_θ` is the predicted noise from our model
                alpha_bar_t = model.alpha_bar[t]

                # Proper formula using alpha_bar_t
                pred_original = (noisy_z - torch.sqrt(1-alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

                # Decode to see result
                denoised = model.encoder.decoder(pred_original)

                # Plot
                axes[1, i].plot(denoised[0, 0].cpu().numpy(), label='Real' if i==0 else None)
                axes[1, i].plot(denoised[0, 1].cpu().numpy(), label='Imag' if i==0 else None)
                axes[1, i].set_title(f"Denoised from t={t.item()}")
                # axes[1, i].set_ylim(-1.5, 1.5)  # Fixed y-axis scale

            # Add legend to first denoised plot
            axes[1, 0].legend()

        # Add row labels
        fig.text(0.02, 0.75, 'Forward\n(Adding Noise)', ha='center', va='center', rotation=90, fontsize=12)
        fig.text(0.02, 0.25, 'Reverse\n(Denoising)', ha='center', va='center', rotation=90, fontsize=12)

        fig.suptitle("Diffusion Process: \n Top: Forward Diffusion Process \n Bottom: Reverse Diffusion Process")
        fig.tight_layout()

        return fig

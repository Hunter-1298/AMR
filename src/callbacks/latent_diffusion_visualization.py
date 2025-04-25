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

# One shot denoising apporach
    # def create_diffusion_visualizations(self, model, batch):
    #     """Create forward and reverse diffusion visualizations with direct comparison"""
    #     x, context, _ = batch
    #     x = x[:1].to(model.device)  # Just use first example

    #     # Select timesteps to visualize
    #     n_steps = 5
    #     timesteps = torch.linspace(0, model.n_steps-1, n_steps).long().to(model.device)

    #     # Create figure with two rows
    #     fig, axes = plt.subplots(2, n_steps, figsize=(n_steps*3, 6))

    #     # Store noisy latents for each timestep
    #     noisy_latents = []

    #     with torch.no_grad():
    #         z = model.encode(x)

    #         # 1. Forward Diffusion (Adding Noise) - Top row
    #         for i, t in enumerate(timesteps):
    #             # Apply noise
    #             noisy_z, _ = model.q_sample(z, t.unsqueeze(0))
    #             noisy_latents.append(noisy_z.clone())  # Store for later denoising

    #             # Decode to signal space
    #             decoded = model.encoder.decoder(noisy_z)

    #             # Plot
    #             axes[0, i].plot(decoded[0, 0].cpu().numpy(), label='Real' if i==0 else None)
    #             axes[0, i].plot(decoded[0, 1].cpu().numpy(), label='Imag' if i==0 else None)
    #             axes[0, i].set_title(f"Noisy t={t.item()}")
    #             # axes[0, i].set_ylim(-1.5, 1.5)

    #         # Add legend to first plot
    #         axes[0, 0].legend()

    #         # 2. Reverse Diffusion (Denoising) - Bottom row
    #         # Now we denoise each noisy latent independently
    #         for i, t in enumerate(timesteps):
    #             # Get the corresponding noisy latent
    #             import pdb; pdb.set_trace()
    #             noisy_z = noisy_latents[i]

    #             # Predict noise
    #             predicted_noise = model(noisy_z, t.unsqueeze(0))

    #             # Stable formula to predict x0 directly
    #             #x₀ = (x_t - sqrt(1-α̅ₜ) * ε_θ(x_t, t)) / sqrt(α̅ₜ)
    #             # Where:
    #             # - `α̅ₜ` is alpha_bar_t (cumulative product of alphas up to timestep t)
    #             # - `ε_θ` is the predicted noise from our model
    #             alpha_bar_t = model.alpha_bar[t]

    #             # Proper formula using alpha_bar_t
    #             pred_original = (noisy_z - torch.sqrt(1-alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

    #             # Decode to see result
    #             denoised = model.encoder.decoder(pred_original)

    #             # Plot
    #             axes[1, i].plot(denoised[0, 0].cpu().numpy(), label='Real' if i==0 else None)
    #             axes[1, i].plot(denoised[0, 1].cpu().numpy(), label='Imag' if i==0 else None)
    #             axes[1, i].set_title(f"Denoised from t={t.item()}")
    #             # axes[1, i].set_ylim(-1.5, 1.5)  # Fixed y-axis scale

    #         # Add legend to first denoised plot
    #         axes[1, 0].legend()

    #     # Add row labels
    #     fig.text(0.02, 0.75, 'Forward\n(Adding Noise)', ha='center', va='center', rotation=90, fontsize=12)
    #     fig.text(0.02, 0.25, 'Reverse\n(Denoising)', ha='center', va='center', rotation=90, fontsize=12)

    #     fig.suptitle("Diffusion Process: \n Top: Forward Diffusion Process \n Bottom: Reverse Diffusion Process")
    #     fig.tight_layout()

    #     return fig

    def create_diffusion_visualizations(self, model, batch):
        """Create forward and reverse diffusion visualizations with SNR tracking"""
        import ipdb; ipdb.set_trace()
        x, context, snr = batch
        x = x[:1].to(model.device)  # Just use first example

        # Select timesteps to visualize
        n_steps = 5
        timesteps = torch.linspace(0, model.n_steps-1, n_steps).long().to(model.device)

        # Calculate theoretical SNR for each timestep
        # SNR = α̅ₜ/(1-α̅ₜ) expressed in dB
        snr_values = []
        for t in timesteps:
            alpha_bar_t = model.alpha_bar[t].item()
            snr_linear = alpha_bar_t / (1 - alpha_bar_t)  # Linear SNR
            snr_db = 10 * np.log10(snr_linear)  # SNR in decibels
            snr_values.append(snr_db)

        # Create figure with two rows for diffusion process and one additional row for SNR
        fig = plt.figure(figsize=(n_steps*3, 9))
        gs = fig.add_gridspec(3, n_steps, height_ratios=[3, 3, 1.5])

        # Create axes for each subplot
        axes_top = [fig.add_subplot(gs[0, i]) for i in range(n_steps)]
        axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(n_steps)]

        # Create one wide subplot for SNR
        ax_snr = fig.add_subplot(gs[2, :])

        # Store noisy latents for each timestep
        noisy_latents = []

        with torch.no_grad():
            z = model.encode(x)

            # Plot the clean signal first for reference
            clean_signal = model.encoder.decoder(z)
            clean_real = clean_signal[0, 0].cpu().numpy()
            clean_imag = clean_signal[0, 1].cpu().numpy()

            # Calculate clean signal energy for actual SNR measurement
            clean_energy = (clean_real**2 + clean_imag**2).mean()

            # Track actual measured SNR values
            measured_snr_forward = []
            measured_snr_reverse = []

            # 1. Forward Diffusion (Adding Noise) - Top row
            for i, t in enumerate(timesteps):
                # Apply noise
                noisy_z, noise = model.q_sample(z, t.unsqueeze(0))
                noisy_latents.append(noisy_z.clone())

                # Decode to signal space
                decoded = model.encoder.decoder(noisy_z)
                decoded_real = decoded[0, 0].cpu().numpy()
                decoded_imag = decoded[0, 1].cpu().numpy()

                # Calculate actual measured SNR for this noisy version
                noise_real = decoded_real - clean_real
                noise_imag = decoded_imag - clean_imag
                noise_energy = (noise_real**2 + noise_imag**2).mean()

                if noise_energy > 0:
                    measured_snr = clean_energy / noise_energy
                    measured_snr_db = 10 * np.log10(measured_snr)
                else:
                    measured_snr_db = float('inf')  # Infinite SNR if no noise

                measured_snr_forward.append(measured_snr_db)

                # Plot with SNR in title
                axes_top[i].plot(decoded_real, label='Real' if i==0 else None)
                axes_top[i].plot(decoded_imag, label='Imag' if i==0 else None)
                axes_top[i].set_title(f"Noisy t={t.item()}\nTheoretical SNR: {snr_values[i]:.1f} dB\nMeasured SNR: {measured_snr_db:.1f} dB")

            # Add legend to first plot
            axes_top[0].legend()

            # 2. Reverse Diffusion (Denoising) - Bottom row
            for i, t_start in enumerate(timesteps):
                # Get the corresponding noisy latent
                noisy_z = noisy_latents[i].clone()

                # Choose denoising approach based on timestep
                if t_start < 5:
                    # One-shot denoising for very small timesteps
                    t_tensor = t_start.unsqueeze(0)
                    predicted_noise = model(noisy_z, t_tensor)
                    alpha_bar_t = model.alpha_bar[t_start]

                    denoised_z = (noisy_z - torch.sqrt(1-alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                else:
                    # DDIM iterative denoising for larger timesteps
                    num_inference_steps = 10  # 10 steps is usually sufficient
                    current_z = noisy_z.clone()
                    steps = torch.linspace(t_start.item(), 0, num_inference_steps).round().long().to(model.device)

                    for j in range(len(steps)-1):
                        t_current = steps[j]
                        t_next = steps[j+1]

                        if t_current == t_next:
                            continue

                        t_tensor = torch.full((1,), t_current, device=model.device).long()
                        predicted_noise = model(current_z, t_tensor)

                        alpha_bar_current = model.alpha_bar[t_current]
                        alpha_bar_next = model.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0).to(model.device)

                        pred_x0 = (current_z - torch.sqrt(1-alpha_bar_current) * predicted_noise) / torch.sqrt(alpha_bar_current)
                        current_z = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(1-alpha_bar_next) * predicted_noise

                    denoised_z = current_z

                # Decode to signal space and calculate SNR
                denoised = model.encoder.decoder(denoised_z)
                denoised_real = denoised[0, 0].cpu().numpy()
                denoised_imag = denoised[0, 1].cpu().numpy()

                # Calculate measured SNR after denoising
                denoise_error_real = denoised_real - clean_real
                denoise_error_imag = denoised_imag - clean_imag
                error_energy = (denoise_error_real**2 + denoise_error_imag**2).mean()

                if error_energy > 0:
                    denoised_snr = clean_energy / error_energy
                    denoised_snr_db = 10 * np.log10(denoised_snr)
                else:
                    denoised_snr_db = float('inf')  # Perfect denoising

                measured_snr_reverse.append(denoised_snr_db)

                # Plot
                axes_bottom[i].plot(denoised_real, label='Real' if i==0 else None)
                axes_bottom[i].plot(denoised_imag, label='Imag' if i==0 else None)
                axes_bottom[i].set_title(f"Denoised from t={t_start.item()}\nTheoretical SNR: {snr_values[i]:.1f} dB\nRecovered SNR: {denoised_snr_db:.1f} dB")

            # Add legend to first denoised plot
            axes_bottom[0].legend()

            # 3. Plot SNR vs Timestep
            t_values = [t.item() for t in timesteps]

            # Plot theoretical SNR curve
            all_timesteps = np.linspace(0, model.n_steps-1, 100)
            all_snr_values = []
            for t in all_timesteps:
                t_idx = int(t)
                alpha_bar_t = model.alpha_bar[t_idx].item()
                snr_linear = alpha_bar_t / (1 - alpha_bar_t)
                snr_db = 10 * np.log10(snr_linear)
                all_snr_values.append(snr_db)

            ax_snr.plot(all_timesteps, all_snr_values, '-', color='gray', alpha=0.5, label='Theoretical SNR')

            # Plot measured SNR points
            ax_snr.plot(t_values, measured_snr_forward, 'ro', label='Measured SNR (Noisy)')
            ax_snr.plot(t_values, measured_snr_reverse, 'go', label='Recovered SNR (Denoised)')

            ax_snr.set_xlabel('Timestep (t)')
            ax_snr.set_ylabel('SNR (dB)')
            ax_snr.set_title('Signal-to-Noise Ratio vs. Timestep')
            ax_snr.legend()
            ax_snr.grid(True)

            # Log scale for y-axis to better visualize the SNR range
            ax_snr.set_yscale('symlog')  # Symmetric log scale to handle negative values

        # Add row labels
        fig.text(0.02, 0.77, 'Forward\n(Adding Noise)', ha='center', va='center', rotation=90, fontsize=12)
        fig.text(0.02, 0.5, 'Reverse\n(Denoising)', ha='center', va='center', rotation=90, fontsize=12)

        fig.suptitle("Diffusion Process with SNR Analysis", fontsize=14)
        fig.tight_layout()

        return fig

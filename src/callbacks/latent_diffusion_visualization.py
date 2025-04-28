import torch
import matplotlib.pyplot as plt
import lightning as L
import wandb
import numpy as np
import imageio

class DiffusionVisualizationCallback(L.Callback):
    """Callback for visualizing the diffusion process"""

    def __init__(self, every_n_epochs=1, create_animation=False):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.create_animation = create_animation

    def on_validation_epoch_end(self, trainer, pl_module):
        """Create visualizations at the end of validation epoch"""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Use the stored batch from the model
        if hasattr(pl_module, 'example_batch'):
            batch = pl_module.example_batch

            # 1. Visualize forward and reverse diffusion
            denoising_fig = self.create_diffusion_visualizations(pl_module, batch)

            logs = {
                "diffusion/forward": wandb.Image(denoising_fig),
                "epoch": trainer.current_epoch
            }

            # 2. Create diffusion animation only in validation mode
            if self.create_animation:
                gif_path = self.create_diffusion_animation(pl_module, batch)
                logs["diffusion/animation"] = wandb.Video(gif_path, fps=5, format="gif")

            # Log to wandb
            trainer.logger.experiment.log(logs)

            plt.close(denoising_fig)
        else:
            print("No example batch found for visualization")


    def create_diffusion_visualizations(self, model, batch):
        """Create diffusion visualizations with log magnitude plots and common y-axis"""
        x, context, _ = batch
        x = x[:1].to(model.device)

        # Encode to latent
        with torch.no_grad():
            z = model.encode(x)

        # Calculate base reconstruction error in signal domain
        with torch.no_grad():
            x_recon = model.decode(z)
            base_sig_pow = x.pow(2).mean().cpu().item()
            base_err = (x_recon - x).pow(2).mean().cpu().item()
            base_snr_db = 10 * np.log10(base_sig_pow / (base_err + 1e-8))

        # Timesteps for visualization
        n_steps = 5
        timesteps = torch.linspace(0, model.n_steps - 1, n_steps).long().to(model.device)

        # Create figure with 4 rows - make it even taller to accommodate row titles
        fig = plt.figure(figsize=(n_steps*3, 14))  # Further increased height

        # Create a GridSpec with more space between rows
        gs = fig.add_gridspec(4, n_steps, height_ratios=[3, 2, 3, 2], hspace=0.6)  # Increased spacing further

        # Create axes for all plots
        axes_top_time = [fig.add_subplot(gs[0, i]) for i in range(n_steps)]
        axes_top_freq = [fig.add_subplot(gs[1, i]) for i in range(n_steps)]
        axes_bottom_time = [fig.add_subplot(gs[2, i]) for i in range(n_steps)]
        axes_bottom_freq = [fig.add_subplot(gs[3, i]) for i in range(n_steps)]

        forward_snr = [base_snr_db]
        reverse_snr = []
        noisy_latents = []

        # Get clean signal for reference
        clean_signal = x_recon[0].cpu().numpy()
        clean_mag = np.sqrt(clean_signal[0]**2 + clean_signal[1]**2)

        # Calculate FFT of clean signal for reference
        clean_fft = np.fft.fft(clean_mag)
        clean_fft_mag = np.abs(clean_fft)
        clean_fft_log = 20 * np.log10(clean_fft_mag + 1e-8)  # Log magnitude in dB

        # Determine common y-axis limits for time domain
        all_signals = [clean_mag]  # Start with clean signal

        # First pass to gather all signals for y-axis determination
        with torch.no_grad():
            for t in timesteps:
                # Forward process
                z_t, _ = model.q_sample(z, t.unsqueeze(0))
                decoded = model.decode(z_t)[0].cpu().numpy()
                forward_mag = np.sqrt(decoded[0]**2 + decoded[1]**2)
                all_signals.append(forward_mag)

                # One-shot reverse process for y-axis determination
                eps_pred = model(z_t, t.unsqueeze(0))
                a_bar_tensor = model.alpha_bar[t]
                z_pred = (z_t - torch.sqrt(1 - a_bar_tensor) * eps_pred) / (torch.sqrt(a_bar_tensor) + 1e-8)
                decoded = model.decode(z_pred)[0].cpu().numpy()
                reverse_mag = np.sqrt(decoded[0]**2 + decoded[1]**2)
                all_signals.append(reverse_mag)

        # Calculate common y-axis limits
        all_signals_array = np.concatenate([s.reshape(-1) for s in all_signals])
        y_min = np.min(all_signals_array) * 0.9
        y_max = np.max(all_signals_array) * 1.1

        # Determine common y-axis limits for frequency domain
        freq_domain_min = -60  # dB
        freq_domain_max = np.max(clean_fft_log) * 1.1

        # Forward diffusion
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_val = t.cpu().item()

                # Add noise
                z_t, noise = model.q_sample(z, t.unsqueeze(0))
                noisy_latents.append(z_t.clone())

                # For SNR calculation
                a_bar_tensor = model.alpha_bar[t]
                a_bar_display = a_bar_tensor.cpu().item()

                # Calculate SNR
                sig_pow = z.pow(2).mean().cpu().item()
                noi_pow = noise.pow(2).mean().cpu().item() * (1 - a_bar_display) / a_bar_display
                snr_db = 10 * np.log10(sig_pow / (noi_pow + 1e-8))
                forward_snr.append(snr_db)

                # Decode and prepare for plotting
                decoded = model.decode(z_t)[0].cpu().numpy()
                decoded_mag = np.sqrt(decoded[0]**2 + decoded[1]**2)

                # Calculate FFT for frequency domain plot
                fft_result = np.fft.fft(decoded_mag)
                fft_mag = np.abs(fft_result)
                fft_log = 20 * np.log10(fft_mag + 1e-8)  # Log magnitude in dB

                # Time domain plot - magnitude only
                axes_top_time[i].plot(clean_mag, 'k--', alpha=0.5, label='Clean' if i==0 else None)
                axes_top_time[i].plot(decoded_mag, 'b-', label='Noisy' if i==0 else None)
                axes_top_time[i].set_title(f"t={t_val:.0f}, SNR: {snr_db:.1f} dB")
                axes_top_time[i].set_ylim(y_min, y_max)  # Common y-axis
                axes_top_time[i].grid(True)

                # Frequency domain plot (log magnitude)
                freq = np.fft.fftfreq(len(decoded_mag))
                pos_freq_idx = np.where(freq >= 0)[0]  # Only show positive frequencies
                axes_top_freq[i].plot(freq[pos_freq_idx], clean_fft_log[pos_freq_idx], 'k--', alpha=0.5, label='Clean' if i==0 else None)
                axes_top_freq[i].plot(freq[pos_freq_idx], fft_log[pos_freq_idx], 'b-', label='Noisy' if i==0 else None)
                axes_top_freq[i].set_title(f"t={t_val:.0f}")
                axes_top_freq[i].set_ylim(freq_domain_min, freq_domain_max)  # Common y-axis
                axes_top_freq[i].grid(True)

            # Add legend to first time plot
            axes_top_time[0].legend()
            axes_top_freq[0].legend()

        # Reverse diffusion
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_val = t.cpu().item()
                z_t = noisy_latents[i]
                current_noisy_snr = forward_snr[i+1]

                # Denoise
                if t < 5:
                    eps_pred = model(z_t, t.unsqueeze(0))
                    a_bar_tensor = model.alpha_bar[t]
                    z_pred = (z_t - torch.sqrt(1 - a_bar_tensor) * eps_pred) / (torch.sqrt(a_bar_tensor) + 1e-8)
                else:
                    steps = torch.linspace(t.cpu().item(), 0, 10).round().long().to(model.device)
                    z_pred = z_t.clone()
                    for j in range(len(steps)-1):
                        t_cur, t_next = steps[j], steps[j+1]
                        eps = model(z_pred, t_cur.unsqueeze(0))
                        a_cur_tensor = model.alpha_bar[t_cur]
                        a_next_tensor = model.alpha_bar[t_next] if t_next>0 else torch.tensor(1.0, device=model.device)
                        x0_pred = (z_pred - torch.sqrt(1-a_cur_tensor)*eps) / (torch.sqrt(a_cur_tensor)+1e-8)
                        z_pred = torch.sqrt(a_next_tensor)*x0_pred + torch.sqrt(1-a_next_tensor)*eps

                # Calculate SNR
                mse = ((z_pred - z) ** 2).mean().cpu().item()
                sig_pow = z.pow(2).mean().cpu().item()
                denoise_snr = 10 * np.log10(sig_pow / (mse + 1e-8))
                reverse_snr.append(denoise_snr)

                # Get noisy signal for reference (from forward process)
                noisy_decoded = model.decode(z_t)[0].cpu().numpy()
                noisy_mag = np.sqrt(noisy_decoded[0]**2 + noisy_decoded[1]**2)

                # Decode denoised signal
                decoded = model.decode(z_pred)[0].cpu().numpy()
                decoded_mag = np.sqrt(decoded[0]**2 + decoded[1]**2)

                # Calculate FFT for frequency domain plot
                fft_result = np.fft.fft(decoded_mag)
                fft_mag = np.abs(fft_result)
                fft_log = 20 * np.log10(fft_mag + 1e-8)  # Log magnitude in dB

                # Calculate FFT for noisy signal
                noisy_fft = np.fft.fft(noisy_mag)
                noisy_fft_mag = np.abs(noisy_fft)
                noisy_fft_log = 20 * np.log10(noisy_fft_mag + 1e-8)

                # Time domain plot - magnitude only
                axes_bottom_time[i].plot(clean_mag, 'k--', alpha=0.5, label='Clean' if i==0 else None)
                axes_bottom_time[i].plot(noisy_mag, 'b-', alpha=0.5, label='Noisy' if i==0 else None)
                axes_bottom_time[i].plot(decoded_mag, 'r-', label='Denoised' if i==0 else None)
                axes_bottom_time[i].set_title(f"t={t_val:.0f}, SNR: {denoise_snr:.1f} dB")
                axes_bottom_time[i].set_ylim(y_min, y_max)  # Common y-axis
                axes_bottom_time[i].grid(True)

                # Frequency domain plot (log magnitude)
                freq = np.fft.fftfreq(len(decoded_mag))
                pos_freq_idx = np.where(freq >= 0)[0]  # Only show positive frequencies
                axes_bottom_freq[i].plot(freq[pos_freq_idx], clean_fft_log[pos_freq_idx], 'k--', alpha=0.5, label='Clean' if i==0 else None)
                axes_bottom_freq[i].plot(freq[pos_freq_idx], noisy_fft_log[pos_freq_idx], 'b-', alpha=0.5, label='Noisy' if i==0 else None)
                axes_bottom_freq[i].plot(freq[pos_freq_idx], fft_log[pos_freq_idx], 'r-', label='Denoised' if i==0 else None)
                axes_bottom_freq[i].set_title(f"t={t_val:.0f}")
                axes_bottom_freq[i].set_ylim(freq_domain_min, freq_domain_max)  # Common y-axis
                axes_bottom_freq[i].grid(True)

            # Add legend to first denoised plot
            axes_bottom_time[0].legend()
            axes_bottom_freq[0].legend()

        # Add main title
        fig.suptitle("Latent Diffusion Process", fontsize=16, y=0.98)

        # First draw to make sure all plot positions are finalized
        fig.canvas.draw()

        # Calculate row positions - get both top and bottom positions of the plots
        row1_top = axes_top_time[0].get_position().y1  # Top of first row
        row1_bottom = axes_top_time[0].get_position().y0  # Bottom of first row

        row2_top = axes_top_freq[0].get_position().y1  # Top of second row
        row2_bottom = axes_top_freq[0].get_position().y0  # Bottom of second row

        row3_top = axes_bottom_time[0].get_position().y1  # Top of third row
        row3_bottom = axes_bottom_time[0].get_position().y0  # Bottom of third row

        row4_top = axes_bottom_freq[0].get_position().y1  # Top of fourth row
        row4_bottom = axes_bottom_freq[0].get_position().y0  # Bottom of fourth row

        # Calculate positions halfway between rows, but closer to the row they describe
        row1_pos = 0.91  # Keep first row title at the top
        row2_pos = (row1_bottom + row2_top -.025) / 2 # Between row 1 and 2
        row3_pos = (row2_bottom + row3_top -.025) / 2  # Between row 2 and 3
        row4_pos = (row3_bottom + row4_top -.025) / 2 # Between row 3 and 4

        # Add row subtitles with adjusted positions
        fig.text(0.5, row1_pos, "Forward Diffusion: Time Domain (Adding Noise)",
                 ha='center', fontsize=12, fontweight='bold')

        fig.text(0.5, row2_pos, "Forward Diffusion: Frequency Domain (Log Magnitude)",
                 ha='center', fontsize=12, fontweight='bold')

        fig.text(0.5, row3_pos, "Reverse Diffusion: Time Domain (Denoising)",
                 ha='center', fontsize=12, fontweight='bold')

        fig.text(0.5, row4_pos, "Reverse Diffusion: Frequency Domain (Log Magnitude)",
                 ha='center', fontsize=12, fontweight='bold')

        # Use a larger rect value to leave more space between elements
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        return fig


    def create_diffusion_animation(self, model, batch):
        """Create an animated GIF showing forward diffusion followed by reverse diffusion with pause at end"""
        x, context, original_snr = batch
        x = x[:1].to(model.device)  # Just use first example

        # Number of frames for each process
        n_frames = 15
        timesteps = torch.linspace(0, model.n_steps-1, n_frames).long().to(model.device)

        # Create directory for frames
        import os
        os.makedirs('media/diffusion_animation', exist_ok=True)

        all_frames = []  # List to store all image frames

        with torch.no_grad():
            # Get original clean signal
            z = model.encode(x)
            clean_signal = model.decode(z)
            clean_real = clean_signal[0, 0].cpu().numpy()
            clean_imag = clean_signal[0, 1].cpu().numpy()

            # Calculate signal magnitude
            clean_magnitude = np.sqrt(clean_real**2 + clean_imag**2)

            # Forward diffusion: generate all noisy versions
            noisy_magnitudes = []
            noisy_latents = []

            for t in timesteps:
                noisy_z, _ = model.q_sample(z, t.unsqueeze(0))
                noisy_latents.append(noisy_z.clone())

                noisy_signal = model.decode(noisy_z)
                noisy_real = noisy_signal[0, 0].cpu().numpy()
                noisy_imag = noisy_signal[0, 1].cpu().numpy()

                noisy_magnitude = np.sqrt(noisy_real**2 + noisy_imag**2)
                noisy_magnitudes.append(noisy_magnitude)

            # Reverse diffusion: generate all denoised versions
            denoised_magnitudes = []

            # Start from the most noisy sample (last timestep)
            current_z = noisy_latents[-1].clone()

            # Use same timesteps but in reverse order
            for i in range(len(timesteps)-1, -1, -1):
                t = timesteps[i]

                # Denoising
                if t < 5:
                    t_tensor = t.unsqueeze(0)
                    predicted_noise = model(current_z, t_tensor)
                    alpha_bar_t = model.alpha_bar[t]
                    current_z = (current_z - torch.sqrt(1-alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                else:
                    t_tensor = torch.full((1,), t, device=model.device).long()
                    predicted_noise = model(current_z, t_tensor)

                    alpha_bar_t = model.alpha_bar[t]
                    pred_x0 = (current_z - torch.sqrt(1-alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

                    t_next = timesteps[i-1] if i > 0 else torch.tensor(0).to(model.device)
                    alpha_bar_next = model.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0).to(model.device)

                    current_z = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(1-alpha_bar_next) * predicted_noise

                # Decode
                denoised = model.decode(current_z)
                denoised_real = denoised[0, 0].cpu().numpy()
                denoised_imag = denoised[0, 1].cpu().numpy()

                denoised_magnitude = np.sqrt(denoised_real**2 + denoised_imag**2)
                denoised_magnitudes.append(denoised_magnitude)

            # Reverse the denoising arrays to match forward timesteps order
            denoised_magnitudes = denoised_magnitudes[::-1]

            # PART 1: Create frames for forward diffusion process
            for i in range(len(timesteps)):
                fig, ax = plt.subplots(figsize=(10, 6))

                # Always show original signal as reference
                ax.plot(clean_magnitude, 'k--', alpha=0.4, label='Original Signal')

                # Show forward diffusion in blue
                ax.plot(noisy_magnitudes[i], 'b-', linewidth=2.5, label='Noisy Signal')

                # Add timestep and phase information
                t = timesteps[i].item()
                progress = ((i+1) / len(timesteps)) * 100

                ax.set_title(f'Forward Diffusion Process - Adding Noise\n'
                        f'Timestep {t} ({progress:.0f}% complete)')

                ax.legend()
                ax.grid(True)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Signal Magnitude')

                # Set fixed y-limits from -5 to 5
                ax.set_ylim(0,12)

                plt.tight_layout()
                frame_path = f'media/diffusion_animation/frame_{i:03d}.png'
                plt.savefig(frame_path)
                plt.close(fig)

                # Read the saved image
                frame = imageio.imread(frame_path)

                # Add the same frame multiple times to slow down the animation
                for _ in range(5):  # Add each frame 5 times
                    all_frames.append(frame)

            # PART 2: Create frames for reverse diffusion process
            # Start from the most noisy state
            for i in range(len(timesteps)):
                fig, ax = plt.subplots(figsize=(10, 6))

                # Always show original signal as reference
                ax.plot(clean_magnitude, 'k--', alpha=0.4, label='Original Signal')

                # Show the most noisy signal (final forward state) in blue
                ax.plot(noisy_magnitudes[-1], 'b-', alpha=0.5, label='Most Noisy Signal')

                # Show the denoising progress in red
                reverse_idx = len(timesteps) - 1 - i  # Start from end and work backward
                ax.plot(denoised_magnitudes[reverse_idx], 'r-', linewidth=2.5, label='Denoised Signal')

                # Add timestep and phase information
                t = timesteps[reverse_idx].item()
                progress = ((i+1) / len(timesteps)) * 100

                ax.set_title(f'Reverse Diffusion Process - Removing Noise\n'
                        f'Timestep {t} ({progress:.0f}% complete)')

                ax.legend()
                ax.grid(True)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Signal Magnitude')

                # Set fixed y-limits from -5 to 5
                ax.set_ylim(0,12)

                plt.tight_layout()
                frame_path = f'media/diffusion_animation/frame_{i+len(timesteps):03d}.png'
                plt.savefig(frame_path)
                plt.close(fig)

                # Read the saved image
                frame = imageio.imread(frame_path)

                # Add the same frame multiple times to slow down the animation
                for _ in range(5):  # Add each frame 5 times
                    all_frames.append(frame)

            # PART 3: Create a final comparison frame with long pause
            fig, ax = plt.subplots(figsize=(10, 6))

            # Show original signal as reference
            ax.plot(clean_magnitude, 'k--', alpha=0.4, label='Original Signal')

            # Show the denoised signal (final result)
            ax.plot(denoised_magnitudes[0], 'r-', linewidth=2.5, label='Final Denoised Signal')

            # Add comparison info
            ax.set_title('Diffusion Process Complete\nComparing Original vs Denoised Signal')

            ax.legend()
            ax.grid(True)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Signal Magnitude')
            ax.set_ylim(0,12)

            plt.tight_layout()
            final_frame_path = f'media/diffusion_animation/frame_final.png'
            plt.savefig(final_frame_path)
            plt.close(fig)

            final_frame = imageio.imread(final_frame_path)

            # Add many copies of the final frame to create a long pause
            for _ in range(30):  # 30 copies for a very long pause
                all_frames.append(final_frame)

            # Create the GIF
            gif_path = 'media/diffusion_animation/diffusion_process.gif'
            imageio.mimsave(
                gif_path,
                all_frames,
                duration=0.1,  # Shorter duration since we're duplicating frames
                loop=0        # Loop indefinitely
            )

            return gif_path

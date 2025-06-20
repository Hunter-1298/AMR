import torch
import matplotlib.pyplot as plt
import lightning as L
import wandb
import numpy as np
import imageio
import os
from sklearn.manifold import TSNE


class DiffusionTSNEVisualizationCallback(L.Callback):
    """Callback for visualizing how class structure evolves in latent space during latent diffusion"""

    def __init__(self, every_n_epochs=1, create_animation=False, label_names=None):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.create_animation = create_animation
        self.label_names = label_names

    def on_validation_epoch_end(self, trainer, pl_module):
        """Create t-SNE visualizations at the end of validation epoch"""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Use the stored batch from the model
        if hasattr(pl_module, "example_batch"):
            batch = pl_module.example_batch

            # Create visualizations
            try:
                # Visualize latent space evolution during diffusion
                tsne_fig = self.create_latent_diffusion_visualizations(pl_module, batch)

                # Visualize ArcFace embedding space
                arcface_fig = self.create_arcface_visualizations(pl_module, batch)

                logs = {
                    "latent_diffusion/tsne_evolution": wandb.Image(tsne_fig),
                    "arcface/embedding_space": wandb.Image(arcface_fig),
                    "epoch": trainer.current_epoch,
                }

                # Create diffusion animation if requested
                if self.create_animation:
                    gif_path = self.create_latent_diffusion_animation(pl_module, batch)
                    if gif_path:
                        logs["latent_diffusion/evolution_animation"] = wandb.Video(gif_path, fps=5, format="gif")

                # Log to wandb
                trainer.logger.experiment.log(logs)

                plt.close(tsne_fig)
                plt.close(arcface_fig)

            except Exception as e:
                print(f"Error creating t-SNE visualizations: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No example batch found for visualization")

    def create_latent_diffusion_visualizations(self, model, batch):
        """Create t-SNE visualizations of latent space evolution during diffusion"""
        # Unpack batch based on classifier structure
        if len(batch) == 3:
            x, labels, snr = batch['x'], batch['labels'], batch['snr']
        else:
            print("Unexpected batch structure in example_batch")
            return None

        batch_size = x.shape[0]

        # Get label names
        label_names = self.label_names or getattr(model, 'label_names', None)
        if not label_names:
            label_names = [f"Class_{i}" for i in range(model.num_classes)]

        # Convert labels to numpy
        if isinstance(labels, torch.Tensor):
            class_labels = labels.cpu().numpy()
        else:
            class_labels = labels

        # Number of unique classes
        num_classes = len(np.unique(class_labels))

        # Number of diffusion steps to visualize
        n_steps = 6

        # Create timesteps from high noise to low noise
        max_timestep = model.latent_diffusion.n_steps - 1
        timesteps = torch.linspace(max_timestep, 0, n_steps).long()

        # Lists to store latents for each diffusion stage
        all_latents = []
        all_stages = []

        with torch.no_grad():
            # 1. Encode to latent space using ArcFace encoder
            z_clean = model.latent_diffusion.encode(x)  # [batch, 32, 8]

            # Store clean latents
            all_latents.append(z_clean.reshape(batch_size, -1).cpu().numpy())
            all_stages.append("Clean Latent")

            # 2. Forward diffusion: Add noise to latents
            for i, t in enumerate(timesteps):
                if t > 0:
                    # Add AWGN noise to simulate different SNR levels
                    target_snr = model.latent_diffusion.awgn_scheduler.timestep_to_snr(t.unsqueeze(0))
                    current_snr = snr.float()

                    # Only add noise if target SNR is lower than current
                    if target_snr < current_snr:
                        # Simulate adding noise to get to target SNR
                        noise_power = self._calculate_noise_power(z_clean, current_snr, target_snr)
                        noise = torch.randn_like(z_clean) * torch.sqrt(noise_power)
                        z_noisy = z_clean + noise
                    else:
                        z_noisy = z_clean

                    all_latents.append(z_noisy.reshape(batch_size, -1).cpu().numpy())
                    all_stages.append(f"Noisy t={t.item():.0f} (SNR={target_snr.item():.1f}dB)")

            # 3. Reverse diffusion: Denoise latents
            z_current = z_noisy.clone() if 'z_noisy' in locals() else z_clean.clone()

            for i, t in enumerate(reversed(timesteps[1:])):  # Skip t=0 as it's the clean state
                # Use the latent diffusion model to denoise
                t_tensor = t.unsqueeze(0).repeat(batch_size).to(z_current.device)

                # Get class embeddings for conditioning
                class_embedding = model.latent_diffusion.arcface_centers[class_labels]

                # Denoise using UNet
                if model.latent_diffusion.predict_noise:
                    predicted_noise = model.latent_diffusion.forward(z_current, t_tensor, class_embedding)
                    z_denoised = z_current - predicted_noise
                else:
                    z_denoised = model.latent_diffusion.forward(z_current, t_tensor, class_embedding)

                z_current = z_denoised

                target_snr = model.latent_diffusion.awgn_scheduler.timestep_to_snr(t_tensor)
                all_latents.append(z_denoised.reshape(batch_size, -1).cpu().numpy())
                all_stages.append(f"Denoised t={t.item():.0f} (SNR={target_snr.mean().item():.1f}dB)")

        # Apply t-SNE to all latents together for consistency
        all_latents_array = np.vstack(all_latents)
        print(f"Applying t-SNE to {all_latents_array.shape[0]} latent points of dimension {all_latents_array.shape[1]}")

        perplexity = min(30, batch_size - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        latents_2d = tsne.fit_transform(all_latents_array)

        # Split back into separate arrays for each stage
        latents_by_stage = {}
        start_idx = 0
        for stage in all_stages:
            latents_by_stage[stage] = latents_2d[start_idx:start_idx + batch_size]
            start_idx += batch_size

        # Create visualization
        n_forward = len([s for s in all_stages if "Noisy" in s]) + 1  # +1 for clean
        n_reverse = len([s for s in all_stages if "Denoised" in s])

        max_cols = max(n_forward, n_reverse, 3)
        fig, axes = plt.subplots(2, max_cols, figsize=(max_cols * 3, 8), dpi=150)
        fig.suptitle("Latent Space Evolution During Diffusion", fontsize=16)

        # Handle case where we only have one row
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        # Create discrete colormap
        cmap = plt.cm.get_cmap('tab20', num_classes)

        # Row 1: Forward process (clean to noisy)
        forward_stages = [s for s in all_stages if "Clean" in s or "Noisy" in s]
        for i, stage in enumerate(forward_stages):
            if i < max_cols:
                ax = axes[0, i] if axes.ndim > 1 else axes[i]
                scatter = ax.scatter(
                    latents_by_stage[stage][:, 0],
                    latents_by_stage[stage][:, 1],
                    c=class_labels,
                    cmap=cmap,
                    s=60,
                    alpha=0.8,
                    vmin=0,
                    vmax=num_classes - 1
                )
                ax.set_title(stage, fontsize=10)
                ax.grid(True, alpha=0.3)

        # Row 2: Reverse process (noisy to clean)
        reverse_stages = [s for s in all_stages if "Denoised" in s]
        for i, stage in enumerate(reverse_stages):
            if i < max_cols:
                ax = axes[1, i] if axes.ndim > 1 else axes[i]
                ax.scatter(
                    latents_by_stage[stage][:, 0],
                    latents_by_stage[stage][:, 1],
                    c=class_labels,
                    cmap=cmap,
                    s=60,
                    alpha=0.8,
                    vmin=0,
                    vmax=num_classes - 1
                )
                ax.set_title(stage, fontsize=10)
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(max(len(forward_stages), len(reverse_stages)), max_cols):
            if axes.ndim > 1:
                axes[0, i].set_visible(False)
                axes[1, i].set_visible(False)

        # Add row labels
        if axes.ndim > 1:
            fig.text(0.02, 0.75, 'Noise Addition', ha='left', va='center', rotation='vertical', fontsize=12)
            fig.text(0.02, 0.25, 'Denoising', ha='left', va='center', rotation='vertical', fontsize=12)

        plt.tight_layout(rect=(0.03, 0, 1, 0.95))
        return fig

    def create_arcface_visualizations(self, model, batch):
        """Create t-SNE visualization of ArcFace embedding space"""
        # Unpack batch
        if len(batch) == 3:
            x, labels, snr = batch['x'], batch['labels'], batch['snr']
        else:
            return None

        # Get label names
        label_names = self.label_names or getattr(model, 'label_names', None)
        if not label_names:
            label_names = [f"Class_{i}" for i in range(model.num_classes)]

        # Convert labels to numpy
        if isinstance(labels, torch.Tensor):
            class_labels = labels.cpu().numpy()
        else:
            class_labels = labels

        num_classes = len(np.unique(class_labels))

        with torch.no_grad():
            # Get ArcFace embeddings
            embeddings = model.latent_diffusion.arcface_centers.get_raw_embeddings(x)
            embeddings_norm = model.latent_diffusion.arcface_centers.get_arcface_embeddings(x)

            # Get class prototypes
            prototypes = model.latent_diffusion.arcface_centers.get_centers()

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Prepare data for t-SNE
        all_embeddings = torch.cat([embeddings_norm, prototypes], dim=0)
        all_labels = np.concatenate([class_labels, np.arange(num_classes)])
        is_prototype = np.concatenate([np.zeros(len(class_labels)), np.ones(num_classes)])

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) // 4))
        embeddings_2d = tsne.fit_transform(all_embeddings.cpu().numpy())

        # Split embeddings and prototypes
        sample_embeddings_2d = embeddings_2d[is_prototype == 0]
        prototype_embeddings_2d = embeddings_2d[is_prototype == 1]

        # Create discrete colormap
        cmap = plt.cm.get_cmap('tab20', num_classes)

        # Plot 1: t-SNE of embeddings and prototypes
        for i in range(num_classes):
            # Plot sample embeddings
            mask = class_labels == i
            if mask.sum() > 0:
                axes[0].scatter(
                    sample_embeddings_2d[mask, 0],
                    sample_embeddings_2d[mask, 1],
                    c=[cmap(i)],
                    s=40,
                    alpha=0.6,
                    label=label_names[i]
                )

            # Plot prototype
            axes[0].scatter(
                prototype_embeddings_2d[i, 0],
                prototype_embeddings_2d[i, 1],
                c=[cmap(i)],
                s=200,
                marker='*',
                edgecolors='black',
                linewidth=1,
                alpha=0.9
            )

        axes[0].set_title('ArcFace Embeddings + Prototypes (t-SNE)')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Angular distances to prototypes
        with torch.no_grad():
            # Calculate angular distances to class prototypes
            angular_distances = []
            for i, label in enumerate(class_labels):
                embedding = embeddings_norm[i]
                prototype = prototypes[label]

                # Calculate cosine similarity and convert to angle
                cos_sim = torch.cosine_similarity(embedding.unsqueeze(0), prototype.unsqueeze(0))
                angle = torch.acos(torch.clamp(cos_sim, -1, 1)) * 180 / np.pi
                angular_distances.append(angle.item())

        # Create scatter plot colored by SNR
        scatter = axes[1].scatter(
            snr.cpu().numpy() if isinstance(snr, torch.Tensor) else snr,
            angular_distances,
            c=class_labels,
            cmap=cmap,
            s=60,
            alpha=0.7,
            vmin=0,
            vmax=num_classes - 1
        )

        axes[1].set_xlabel('SNR (dB)')
        axes[1].set_ylabel('Angular Distance to Prototype (degrees)')
        axes[1].set_title('Angular Distance vs SNR')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_latent_diffusion_animation(self, model, batch):
        """Create animation of latent space evolution during diffusion"""
        try:
            # Create directory for frames
            os.makedirs("media/latent_diffusion_animation", exist_ok=True)

            # This is a simplified version - you can expand it based on your needs
            # For now, just return None to avoid errors
            return None

        except Exception as e:
            print(f"Error creating animation: {e}")
            return None

    def _calculate_noise_power(self, signal, current_snr_db, target_snr_db):
        """Calculate noise power needed to achieve target SNR"""
        # Calculate signal power
        signal_power = torch.mean(signal ** 2, dim=(1, 2), keepdim=True)

        # Convert SNRs to linear scale
        current_snr_linear = 10 ** (current_snr_db.view(-1, 1, 1) / 10)
        target_snr_linear = 10 ** (target_snr_db.view(-1, 1, 1) / 10)

        # Calculate current noise power (assuming signal = clean + noise)
        current_total_power = signal_power
        clean_signal_power = current_total_power * current_snr_linear / (1 + current_snr_linear)

        # Calculate required noise power for target SNR
        target_noise_power = clean_signal_power / target_snr_linear

        # Calculate additional noise power needed
        current_noise_power = current_total_power - clean_signal_power
        additional_noise_power = target_noise_power - current_noise_power

        return torch.clamp(additional_noise_power, min=1e-10)


# Updated callback for the classifier
class ClassifierTSNECallback(L.Callback):
    """Callback for visualizing classifier behavior with t-SNE focused on denoised signal latent dimensions"""

    def __init__(self, every_n_epochs=2, label_names=None):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.label_names = label_names

    def on_validation_epoch_end(self, trainer, pl_module):
        """Create t-SNE visualizations at the end of validation epoch"""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        if not hasattr(pl_module, "example_batch"):
            print("No example batch found for t-SNE visualization")
            return

        try:
            # Create latent-focused visualizations
            denoised_latent_evolution_fig = self.create_denoised_latent_evolution_viz(pl_module)
            snr_latent_relationship_fig = self.create_snr_latent_relationship_viz(pl_module)

            # Log to wandb
            logs = {
                "latent_analysis/denoised_evolution": wandb.Image(denoised_latent_evolution_fig),
                "latent_analysis/snr_relationship": wandb.Image(snr_latent_relationship_fig),
                "epoch": trainer.current_epoch,
            }

            trainer.logger.experiment.log(logs)

            # Close figures
            plt.close(denoised_latent_evolution_fig)

        except Exception as e:
            print(f"Error creating latent t-SNE visualizations: {e}")
            import traceback
            traceback.print_exc()

    def create_denoised_latent_evolution_viz(self, model):
        """Show how denoised signal latent representations evolve across noise levels"""
        batch = model.example_batch

        # Get data and move to correct device
        labels = batch['labels'].numpy()
        x = batch['x'].to(model.device)  # FIXED: Ensure x is on the right device

        # Get label names
        label_names = self.label_names or batch.get('label_names', [f"Class_{i}" for i in range(batch['num_classes'])])
        num_classes = len(np.unique(labels))

        with torch.no_grad():
            # Create a range of noise levels for comprehensive analysis
            noise_levels = [0, 200, 400, 600, 800]  # Reduced for faster processing

            all_denoised_latents = []
            all_labels_extended = []
            all_noise_levels = []

            for noise_level in noise_levels:
                try:
                    # Get denoised signals at this noise level
                    t_level = torch.full((len(x),), noise_level, device=model.device, dtype=torch.long)
                    logits, z_denoised, _ = model.forward(x, timestep=t_level)

                    # Decode the denoised latents back to signal space
                    x_denoised = model.latent_diffusion.decode(z_denoised)

                    # Re-encode the denoised signals to get their latent representations
                    z_denoised_signal_latent = model.latent_diffusion.encode(x_denoised)

                    # Flatten for t-SNE
                    z_flat = z_denoised_signal_latent.reshape(len(z_denoised_signal_latent), -1).cpu().numpy()

                    all_denoised_latents.append(z_flat)
                    all_labels_extended.extend(labels)
                    all_noise_levels.extend([noise_level] * len(labels))

                except Exception as e:
                    print(f"Error processing noise level {noise_level}: {e}")
                    continue

        if len(all_denoised_latents) == 0:
            return self._create_error_figure("No valid data for denoised latent evolution")

        # Combine all latents for consistent t-SNE
        all_latents_array = np.vstack(all_denoised_latents)
        all_labels_array = np.array(all_labels_extended)
        all_noise_array = np.array(all_noise_levels)

        print(f"Applying t-SNE to {all_latents_array.shape[0]} denoised signal latent points")

        # Apply t-SNE
        perplexity = min(20, len(all_latents_array) // 8)  # Reduced perplexity
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300)
        latents_2d = tsne.fit_transform(all_latents_array)

        # Create visualization
        n_levels = len(set(all_noise_levels))
        fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 6))
        if n_levels == 1:
            axes = [axes]

        cmap = plt.cm.get_cmap('tab20', num_classes)

        for i, noise_level in enumerate(sorted(set(all_noise_levels))):
            # Get data for this noise level
            level_mask = all_noise_array == noise_level
            level_latents = latents_2d[level_mask]
            level_labels = all_labels_array[level_mask]

            # Create scatter plot
            scatter = axes[i].scatter(
                level_latents[:, 0],
                level_latents[:, 1],
                c=level_labels,
                cmap=cmap,
                s=60,
                alpha=0.7,
                vmin=0,
                vmax=num_classes - 1
            )

            # Convert noise level to equivalent SNR for title
            if hasattr(model.latent_diffusion, 'awgn_scheduler'):
                try:
                    target_snr = model.latent_diffusion.awgn_scheduler.timestep_to_snr(
                        torch.tensor([noise_level], device=model.device)
                    ).item()
                    axes[i].set_title(f'Noise Level {noise_level}\n(≈{target_snr:.1f} dB SNR)')
                except:
                    axes[i].set_title(f'Noise Level {noise_level}')
            else:
                axes[i].set_title(f'Noise Level {noise_level}')

            axes[i].grid(True, alpha=0.3)

        plt.suptitle('Denoised Signal Latent Representations Across Noise Levels', fontsize=16)
        plt.tight_layout()
        return fig


    def create_snr_latent_relationship_viz(self, model):
        """Analyze relationship between input SNR and denoised signal latent quality"""
        batch = model.example_batch

        labels = batch['labels'].numpy()
        x = batch['x'].to(model.device)
        snr = batch['snr'].numpy()

        # Get label names
        label_names = self.label_names or batch.get('label_names', [f"Class_{i}" for i in range(batch['num_classes'])])
        num_classes = len(np.unique(labels))

        with torch.no_grad():
            try:
                # Use the original timestep mapping (SNR-based)
                t_original = model.snr_to_timestep(torch.tensor(snr, device=model.device).float())

                # Get original noisy latents (before denoising)
                z_noisy_original = model.latent_diffusion.encode(x)

                # Get denoised representations using original SNR mapping
                logits, z_denoised, _ = model.forward(x, timestep=t_original)
                x_denoised = model.latent_diffusion.decode(z_denoised)
                z_denoised_signal = model.latent_diffusion.encode(x_denoised)

                # Flatten latents for analysis
                z_noisy_flat = z_noisy_original.reshape(len(z_noisy_original), -1).cpu().numpy()
                z_denoised_flat = z_denoised_signal.reshape(len(z_denoised_signal), -1).cpu().numpy()

            except Exception as e:
                print(f"Error in SNR-latent relationship analysis: {e}")
                return self._create_error_figure("Error in SNR-latent relationship analysis")

        # Combine noisy and denoised latents for consistent t-SNE
        combined_latents = np.vstack([z_noisy_flat, z_denoised_flat])

        # Apply t-SNE for 2D visualization
        perplexity = min(20, len(combined_latents) // 4)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300)
        latents_2d = tsne.fit_transform(combined_latents)

        # Split back into noisy and denoised
        noisy_latents_2d = latents_2d[:len(labels)]
        denoised_latents_2d = latents_2d[len(labels):]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

        cmap = plt.cm.get_cmap('tab20', num_classes)

        # Plot 1: Noisy latents colored by class
        for class_idx in range(num_classes):
            mask = labels == class_idx
            if mask.sum() > 0:
                axes[0].scatter(
                    noisy_latents_2d[mask, 0], noisy_latents_2d[mask, 1],
                    c=[cmap(class_idx)], s=60, alpha=0.7,
                    label=label_names[class_idx]
                )

        axes[0].set_title('Original Noisy Signal Latents\n(Colored by Class)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Plot 2: Denoised latents with trajectory arrows
        # Plot denoised points
        for class_idx in range(num_classes):
            mask = labels == class_idx
            if mask.sum() > 0:
                axes[1].scatter(
                    denoised_latents_2d[mask, 0], denoised_latents_2d[mask, 1],
                    c=[cmap(class_idx)], s=60, alpha=0.8,
                    label=label_names[class_idx]
                )

        # Add trajectory arrows (sample a few for clarity)
        arrow_samples = min(20, len(labels))  # Show max 20 arrows to avoid clutter
        arrow_indices = np.linspace(0, len(labels)-1, arrow_samples, dtype=int)

        for idx in arrow_indices:
            # Draw arrow from noisy to denoised position
            dx = denoised_latents_2d[idx, 0] - noisy_latents_2d[idx, 0]
            dy = denoised_latents_2d[idx, 1] - noisy_latents_2d[idx, 1]

            axes[1].arrow(
                noisy_latents_2d[idx, 0], noisy_latents_2d[idx, 1],
                dx, dy,
                head_width=0.5, head_length=0.5,
                fc='gray', ec='gray', alpha=0.5, linewidth=1
            )

        axes[1].set_title('Denoised Signal Latents\n(with Denoising Trajectories)', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: SNR groupings of denoised latents
        # Create SNR bins for better visualization
        snr_bins = np.percentile(snr, [0, 20, 40, 60, 80, 100])  # Quintiles
        snr_bin_labels = [f'{snr_bins[i]:.1f} to {snr_bins[i+1]:.1f} dB' for i in range(len(snr_bins)-1)]
        snr_colors = plt.cm.plasma(np.linspace(0, 1, len(snr_bin_labels)))

        # Assign each sample to an SNR bin
        snr_bin_indices = np.digitize(snr, snr_bins) - 1
        snr_bin_indices = np.clip(snr_bin_indices, 0, len(snr_bin_labels) - 1)

        # Plot each SNR group with different colors and sizes
        for bin_idx, (bin_label, color) in enumerate(zip(snr_bin_labels, snr_colors)):
            mask = snr_bin_indices == bin_idx
            if mask.sum() > 0:
                # Use different marker sizes based on SNR (higher SNR = larger markers)
                marker_size = 40 + bin_idx * 15
                axes[2].scatter(
                    denoised_latents_2d[mask, 0],
                    denoised_latents_2d[mask, 1],
                    c=[color],
                    s=marker_size,
                    alpha=0.7,
                    label=bin_label,
                    edgecolors='black',
                    linewidth=0.5
                )

        axes[2].set_title('Denoised Signal Latents\n(Grouped by Input SNR)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Add metrics text box
        # Calculate average movement distance
        movement_distances = np.sqrt(
            (denoised_latents_2d[:, 0] - noisy_latents_2d[:, 0])**2 +
            (denoised_latents_2d[:, 1] - noisy_latents_2d[:, 1])**2
        )
        avg_movement = np.mean(movement_distances)

        # Calculate class separation improvement
        from scipy.spatial.distance import pdist, squareform

        # Calculate average within-class distances for noisy vs denoised
        noisy_separation = []
        denoised_separation = []

        for class_idx in range(num_classes):
            mask = labels == class_idx
            if mask.sum() > 1:
                noisy_class_points = noisy_latents_2d[mask]
                denoised_class_points = denoised_latents_2d[mask]

                noisy_distances = pdist(noisy_class_points)
                denoised_distances = pdist(denoised_class_points)

                noisy_separation.append(np.mean(noisy_distances))
                denoised_separation.append(np.mean(denoised_distances))

        if noisy_separation and denoised_separation:
            separation_improvement = (np.mean(noisy_separation) - np.mean(denoised_separation)) / np.mean(noisy_separation) * 100
        else:
            separation_improvement = 0

        # Add metrics text
        metrics_text = f"""Denoising Metrics:
    Avg Movement: {avg_movement:.2f}
    Class Separation: {separation_improvement:+.1f}%
    Samples: {len(labels)}"""

        fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.suptitle('Latent Space Evolution: Noisy → Denoised → SNR Grouped', fontsize=16)
        plt.tight_layout()
        return fig

    def _create_error_figure(self, error_message):
        """Create a figure displaying an error message"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"Visualization Error:\n{error_message}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

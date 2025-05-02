import torch
import matplotlib.pyplot as plt
import lightning as L
import wandb
import numpy as np
import imageio
import os
from sklearn.manifold import TSNE


class DiffusionTSNEVisualizationCallback(L.Callback):
    """Callback for visualizing how class structure evolves in latent space during diffusion"""

    def __init__(self, every_n_epochs=1, create_animation=False):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.create_animation = create_animation

    def on_validation_epoch_end(self, trainer, pl_module):
        """Create t-SNE visualizations at the end of validation epoch"""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Use the stored batch from the model
        if hasattr(pl_module, "example_batch"):
            batch = pl_module.example_batch

            # Visualize latent space with t-SNE
            tsne_fig = self.create_diffusion_visualizations(pl_module, batch)

            logs = {
                "latent/tsne_classes": wandb.Image(tsne_fig),
                "epoch": trainer.current_epoch,
            }

            # Create diffusion animation if requested
            if self.create_animation:
                gif_path = self.create_diffusion_animation(pl_module, batch)
                logs["latent/tsne_classes_animation"] = wandb.Video(gif_path, fps=5, format="gif")

            # Log to wandb
            trainer.logger.experiment.log(logs)

            plt.close(tsne_fig)
        else:
            print("No example batch found for visualization")

    def create_diffusion_visualizations(self, model, batch):
        """Create t-SNE visualizations of class distribution in latent space during diffusion"""
        x, context, _ = batch
        batch_size = x.shape[0]

        # Number of timesteps to visualize
        n_steps = 5
        timesteps = torch.linspace(0, model.n_steps - 1, n_steps).long().to(model.device)

        # Extract class labels from the context
        if context.dim() > 1 and context.shape[1] > 1:  # One-hot encoded
            class_labels = context.argmax(dim=1).cpu().numpy()
        else:  # Already indices
            class_labels = context.cpu().numpy()

        # Get label names if available
        label_names = getattr(model, 'label_names', None)
        if not label_names and hasattr(model, 'encoder') and hasattr(model.encoder, 'label_names'):
            label_names = model.encoder.label_names

        # Number of unique classes
        num_classes = len(np.unique(class_labels))

        # Lists to store latents for each diffusion stage
        all_latents = []
        all_stages = []
        saved_noisy_latents = {}  # Store noisy latents by timestep

        with torch.no_grad():
            # Get encoded latents (original)
            z = model.encode(x)

            # FORWARD DIFFUSION: Process each timestep including t=0 (original)
            for i, t in enumerate(torch.tensor([0] + timesteps.tolist()).to(model.device)):
                if t == 0:
                    # Original latents (no noise)
                    z_t = z.clone()
                else:
                    # Add noise according to timestep
                    z_t, _ = model.q_sample(z, t.unsqueeze(0).repeat(batch_size))
                    # Save this noisy state for exactly matching reverse diffusion
                    saved_noisy_latents[t.item()] = z_t.clone()

                # Store latents and stage label
                all_latents.append(z_t.cpu().numpy().reshape(batch_size, -1))
                all_stages.append(f"t={t.item()}")

            # REVERSE DIFFUSION: Start from noisiest state and work backwards
            # First get the noisiest state from saved forward states
            z_noisy = saved_noisy_latents[timesteps[-1].item()]

            # Then get all the reverse timesteps we need to process
            reverse_timesteps = timesteps[:-1].flip(0)  # Exclude the noisiest state (already included) and reverse

            for t in reverse_timesteps:
                # Apply reverse diffusion step
                if t < 5:
                    eps_pred = model(z_noisy, t.unsqueeze(0).repeat(batch_size), context)
                    a_bar_tensor = model.alpha_bar[t]
                    z_pred = (z_noisy - torch.sqrt(1 - a_bar_tensor) * eps_pred) / (
                        torch.sqrt(a_bar_tensor) + 1e-8
                    )
                else:
                    steps = torch.linspace(timesteps[-1].item(), t.item(), 10).round().long().to(model.device)
                    z_pred = z_noisy.clone()
                    for j in range(len(steps) - 1):
                        t_cur, t_next = steps[j], steps[j + 1]
                        eps = model(z_pred, t_cur.unsqueeze(0).repeat(batch_size), context)
                        a_cur_tensor = model.alpha_bar[t_cur]
                        a_next_tensor = (
                            model.alpha_bar[t_next]
                            if t_next > 0
                            else torch.tensor(1.0, device=model.device)
                        )
                        x0_pred = (z_pred - torch.sqrt(1 - a_cur_tensor) * eps) / (
                            torch.sqrt(a_cur_tensor) + 1e-8
                        )
                        z_pred = (
                            torch.sqrt(a_next_tensor) * x0_pred
                            + torch.sqrt(1 - a_next_tensor) * eps
                        )

                # Update for next iteration
                z_noisy = z_pred.clone()

                # Store latents and stage label
                all_latents.append(z_pred.cpu().numpy().reshape(batch_size, -1))
                all_stages.append(f"t={t.item()} (denoised)")

        # Combine all latents for a single t-SNE computation
        all_latents_array = np.vstack(all_latents)

        # Apply t-SNE to reduce dimensions to 2D
        print(f"Applying t-SNE to {all_latents_array.shape[0]} points of dimension {all_latents_array.shape[1]}")
        perplexity = min(30, batch_size-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        latents_2d = tsne.fit_transform(all_latents_array)

        # Split back into separate arrays for each stage
        latents_by_stage = {}
        start_idx = 0
        for stage in all_stages:
            latents_by_stage[stage] = latents_2d[start_idx:start_idx+batch_size]
            start_idx += batch_size

        # Create a 2-row grid with stages properly arranged
        # First row: Original → Forward diffusion (t=0 to t=max)
        # Second row: Most noisy → Reverse diffusion (t=max to t=min, excluding t=0)

        # Calculate number of columns needed
        n_cols = n_steps + 1  # Original + forward steps

        fig, axes = plt.subplots(2, n_cols, figsize=(max(n_cols * 3, 20), 10), dpi=150)
        fig.suptitle("Class Structure in Latent Space During Diffusion", fontsize=16)

        # Create discrete colormap for class labels
        cmap = plt.cm.get_cmap('tab10', num_classes)

        # Row 1: Forward diffusion (including original t=0)
        forward_stages = all_stages[:n_cols]
        for i, stage in enumerate(forward_stages):
            scatter = axes[0, i].scatter(
                latents_by_stage[stage][:, 0],
                latents_by_stage[stage][:, 1],
                c=class_labels,
                cmap=cmap,
                s=80,
                alpha=0.9,
                vmin=0,
                vmax=num_classes-1
            )
            t_val = int(stage.split('=')[1].split(' ')[0])
            axes[0, i].set_title(f"Forward t={t_val}")
            axes[0, i].grid(True)

        # Row 2: Most noisy state → Reverse diffusion
        # First plot is the most noisy state (already in forward row)
        most_noisy_stage = forward_stages[-1]
        axes[1, 0].scatter(
            latents_by_stage[most_noisy_stage][:, 0],
            latents_by_stage[most_noisy_stage][:, 1],
            c=class_labels,
            cmap=cmap,
            s=80,
            alpha=0.9,
            vmin=0,
            vmax=num_classes-1
        )
        t_val = int(most_noisy_stage.split('=')[1].split(' ')[0])
        axes[1, 0].set_title(f"Noisiest t={t_val}")
        axes[1, 0].grid(True)

        # Remaining plots are reverse diffusion stages
        reverse_stages = all_stages[n_cols:]
        for i, stage in enumerate(reverse_stages):
            axes[1, i+1].scatter(
                latents_by_stage[stage][:, 0],
                latents_by_stage[stage][:, 1],
                c=class_labels,
                cmap=cmap,
                s=80,
                alpha=0.9,
                vmin=0,
                vmax=num_classes-1
            )
            t_val = int(stage.split('=')[1].split(' ')[0])
            axes[1, i+1].set_title(f"Reverse t={t_val}")
            axes[1, i+1].grid(True)

        # Add colorbar with class names
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), ticks=range(num_classes))
        if label_names:
            cbar.ax.set_yticklabels(label_names)
            cbar.set_label('Signal Class')
        else:
            cbar.set_label('Class Label')

        # Add row labels
        fig.text(0.02, 0.75, 'Forward Diffusion', ha='left', va='center', rotation='vertical', fontsize=14)
        fig.text(0.02, 0.25, 'Reverse Diffusion', ha='left', va='center', rotation='vertical', fontsize=14)

        plt.tight_layout(rect=[0.03, 0, 1, 0.95])
        return fig

    def create_diffusion_animation(self, model, batch):
        """Create an enhanced animated GIF showing class structure evolution with reference points"""
        x, context, _ = batch
        batch_size = x.shape[0]

        # Create directory for frames
        os.makedirs("media/tsne_animation", exist_ok=True)
        all_frames = []  # List to store all image frames

        # Number of frames for each process
        n_frames = 15
        timesteps = torch.linspace(0, model.n_steps - 1, n_frames).long().to(model.device)

        # Extract class labels from the context
        if context.dim() > 1 and context.shape[1] > 1:  # One-hot encoded
            class_labels = context.argmax(dim=1).cpu().numpy()
        else:  # Already indices
            class_labels = context.cpu().numpy()

        # Get label names if available
        label_names = getattr(model, 'label_names', None)
        if not label_names and hasattr(model, 'encoder') and hasattr(model.encoder, 'label_names'):
            label_names = model.encoder.label_names

        # Number of unique classes
        num_classes = len(np.unique(class_labels))

        # Collect latents for all timesteps
        all_latents = []  # Store all latent representations
        stage_labels = []  # Store the diffusion stage for each latent set
        noisy_latents = []  # Store intermediate latents for reverse diffusion

        with torch.no_grad():
            # Get encoded latents (original)
            z = model.encode(x)
            all_latents.append(z.clone().cpu().numpy())
            stage_labels.append("Original")

            # Forward diffusion
            for i, t in enumerate(timesteps):
                z_t, _ = model.q_sample(z, t.unsqueeze(0).repeat(batch_size))
                noisy_latents.append(z_t.clone())

                all_latents.append(z_t.cpu().numpy())
                stage_labels.append(f"Forward t={t.item():.0f}")

            # Reverse diffusion starting from most noisy state
            z_noisy = noisy_latents[-1].clone()

            for i in range(len(timesteps)-1, -1, -1):
                t = timesteps[i]

                # Apply reverse diffusion step
                if t < 5:
                    eps_pred = model(z_noisy, t.unsqueeze(0).repeat(batch_size), context)
                    a_bar_tensor = model.alpha_bar[t]
                    z_pred = (z_noisy - torch.sqrt(1 - a_bar_tensor) * eps_pred) / (
                        torch.sqrt(a_bar_tensor) + 1e-8
                    )
                else:
                    steps = torch.linspace(t.cpu().item(), 0, 10).round().long().to(model.device)
                    z_pred = z_noisy.clone()
                    for j in range(len(steps) - 1):
                        t_cur, t_next = steps[j], steps[j + 1]
                        eps = model(z_pred, t_cur.unsqueeze(0).repeat(batch_size), context)
                        a_cur_tensor = model.alpha_bar[t_cur]
                        a_next_tensor = (
                            model.alpha_bar[t_next]
                            if t_next > 0
                            else torch.tensor(1.0, device=model.device)
                        )
                        x0_pred = (z_pred - torch.sqrt(1 - a_cur_tensor) * eps) / (
                            torch.sqrt(a_cur_tensor) + 1e-8
                        )
                        z_pred = (
                            torch.sqrt(a_next_tensor) * x0_pred
                            + torch.sqrt(1 - a_next_tensor) * eps
                        )

                z_noisy = z_pred.clone()
                all_latents.append(z_pred.cpu().numpy())
                stage_labels.append(f"Reverse t={t.item():.0f}")

        # Apply t-SNE to all latents together for consistency
        all_latents_array = np.vstack([latent.reshape(batch_size, -1) for latent in all_latents])
        print(f"Applying t-SNE for animation with {all_latents_array.shape[0]} points of dimension {all_latents_array.shape[1]}")
        perplexity = min(30, batch_size-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        latents_2d = tsne.fit_transform(all_latents_array)

        # Split back into frames
        frame_latents = np.split(latents_2d, len(stage_labels))

        # Store references to original and most noisy state
        original_2d = frame_latents[0]
        most_noisy_2d = frame_latents[n_frames]  # Last forward diffusion step

        # Get consistent axis limits
        x_min, x_max = latents_2d[:, 0].min(), latents_2d[:, 0].max()
        y_min, y_max = latents_2d[:, 1].min(), latents_2d[:, 1].max()

        # Add padding to axis limits
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # Create discrete colormap for class labels
        cmap = plt.cm.get_cmap('tab10', num_classes)

        # Create frames for animation
        for i, (latents, stage) in enumerate(zip(frame_latents, stage_labels)):
            fig, ax = plt.subplots(figsize=(10, 8))

            # Show original latents as reference points (small, transparent)
            if i > 0:  # Skip for the original frame itself
                ax.scatter(
                    original_2d[:, 0],
                    original_2d[:, 1],
                    c=class_labels,
                    cmap=cmap,
                    s=30,
                    alpha=0.2,
                    vmin=0,
                    vmax=num_classes-1,
                    marker='o',
                    label="Original"
                )

            # Show most noisy latents as reference points (if we're in reverse diffusion)
            if "Reverse" in stage:
                ax.scatter(
                    most_noisy_2d[:, 0],
                    most_noisy_2d[:, 1],
                    c=class_labels,
                    cmap=cmap,
                    s=30,
                    alpha=0.2,
                    vmin=0,
                    vmax=num_classes-1,
                    marker='s',  # Square marker to differentiate
                    label="Most Noisy"
                )

            # Create main scatter plot colored by class
            scatter = ax.scatter(
                latents[:, 0],
                latents[:, 1],
                c=class_labels,
                cmap=cmap,
                s=100,
                alpha=0.9,
                vmin=0,
                vmax=num_classes-1,
                label=stage
            )

            # Add title
            if "Forward" in stage:
                title = f"Class Structure in Latent Space\n{stage} - Adding Noise"
            elif "Reverse" in stage:
                title = f"Class Structure in Latent Space\n{stage} - Removing Noise"
            else:
                title = f"Class Structure in Latent Space\n{stage}"

            ax.set_title(title, fontsize=14)

            # Set consistent axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True)

            # Add legend
            if i > 0:
                ax.legend(loc='upper right')

            # Add colorbar
            cbar = plt.colorbar(scatter, ticks=range(num_classes))
            if label_names:
                cbar.ax.set_yticklabels(label_names)
                cbar.set_label('Signal Class')
            else:
                cbar.set_label('Class Label')

            # Add descriptive text about what's happening
            if "Forward" in stage:
                progress = int(((i-1) / (n_frames-1)) * 100) if i > 0 else 0
                ax.text(0.5, -0.1,
                    f"Forward Diffusion: {progress}% Complete\nClasses gradually mix as noise increases",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            elif "Reverse" in stage:
                reverse_idx = i - (n_frames + 1)  # Index in reverse process
                progress = int(((reverse_idx+1) / n_frames) * 100)
                ax.text(0.5, -0.1,
                    f"Reverse Diffusion: {progress}% Complete\nClasses gradually separate as noise is removed",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

            # Save the frame
            plt.tight_layout()
            frame_path = f"media/tsne_animation/class_frame_{i:03d}.png"
            plt.savefig(frame_path, dpi=100)
            plt.close(fig)

            # Read the saved image
            frame = imageio.imread(frame_path)

            # Add the same frame multiple times to control animation speed
            if i == 0:  # Original state - longer pause
                repeats = 10
            elif i == n_frames:  # Most noisy state - extended pause
                repeats = 20  # Long pause at most noisy state
            elif i == len(stage_labels)-1:  # Final denoised state - longer pause
                repeats = 15
            else:
                repeats = 3

            for _ in range(repeats):
                all_frames.append(frame)

        # Create the GIF
        gif_path = "media/tsne_animation/class_structure_evolution.gif"
        imageio.mimsave(
            gif_path,
            all_frames,
            duration=0.15,
            loop=0
        )

        return gif_path

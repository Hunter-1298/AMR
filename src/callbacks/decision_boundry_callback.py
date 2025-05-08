import torch
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import wandb
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import lightning.pytorch as pl
import matplotlib.animation as animation

class DecisionBoundaryVisualizationCallback(pl.Callback):
    def __init__(self, every_n_epochs=5, num_samples=500, create_animation=True, label_names=None):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.label_names = label_names
        self.num_samples = num_samples
        self.create_animation = create_animation
        self.frames = []  # Store frames for animation
        # Add static axis bounds
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

    def on_validation_epoch_end(self, trainer, pl_module):
        """Create visualizations at the end of validation epoch"""
        if trainer.current_epoch % self.every_n_epochs == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            self._create_decision_boundary_viz(trainer, pl_module)

    def _initialize_bounds(self, z_2d):
        """Initialize the static bounds for plotting"""
        margin = 1.0  # Add some margin around the data
        self.x_min = z_2d[:, 0].min() - margin
        self.x_max = z_2d[:, 0].max() + margin
        self.y_min = z_2d[:, 1].min() - margin
        self.y_max = z_2d[:, 1].max() + margin

    def _create_decision_boundary_viz(self, trainer, pl_module):
        if not hasattr(pl_module, 'example_batch'):
            print("Warning: No example batch available for visualization")
            return

        try:
            # Get the example batch dictionary
            example_batch = pl_module.example_batch

            # Get the already processed data
            denoised_z = example_batch['denoised_z']  # Get denoised latents
            context = example_batch['context']

            # Move tensors to device if needed
            denoised_z = denoised_z.to(pl_module.device)
            context = context.to(pl_module.device)

            # Use t-SNE to reduce latent space to 2D for visualization
            tsne = TSNE(n_components=2, random_state=42)
            z_2d = tsne.fit_transform(denoised_z.reshape(denoised_z.shape[0], -1).cpu().numpy())

            # Initialize bounds if not already set
            if self.x_min is None:
                self._initialize_bounds(z_2d)

            # Create figure with slightly wider figure to accommodate vertical legend
            plt.figure(figsize=(13, 10))

            # Create a grid for decision boundary visualization
            h = 0.1  # Step size in the mesh
            xx, yy = np.meshgrid(np.arange(self.x_min, self.x_max, h),
                            np.arange(self.y_min, self.y_max, h))

            # Create dummy latent vectors from mesh grid points
            grid_points = np.c_[xx.ravel(), yy.ravel()]

            # For each grid point, find nearest latent vector
            distances = distance.cdist(grid_points, z_2d)
            nearest_indices = np.argmin(distances, axis=1)

            # Get classifier predictions
            with torch.no_grad():
                logits = pl_module.classifier_head(denoised_z)
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

            # Use these predictions to colorize the grid
            grid_predictions = predictions[nearest_indices]
            grid_predictions = grid_predictions.reshape(xx.shape)

            # Create custom colormap
            num_classes = pl_module.num_classes
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
            custom_cmap = ListedColormap(colors)

            # Plot decision regions
            contour = plt.contourf(xx, yy, grid_predictions, alpha=0.3,
                            cmap=custom_cmap,
                            levels=np.arange(num_classes + 1) - 0.5)

            # Plot class examples
            scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1],
                                c=context.cpu().numpy(),
                                cmap=custom_cmap,
                                edgecolors='k',
                                s=50)

            # Get the label names to use
            label_names = self.label_names
            if label_names is None and hasattr(pl_module, 'label_names'):
                label_names = pl_module.label_names

            # Add class labels as vertical legend
            if label_names is not None:
                # Create legend elements for classes present in the data
                handles = []
                labels = []

                # Get unique class indices from the context
                unique_indices = torch.unique(context).cpu().numpy()

                for idx in unique_indices:
                    if idx < len(label_names):
                        handles.append(plt.Rectangle((0,0),1,1,
                                                facecolor=colors[idx],
                                                alpha=0.6,
                                                edgecolor='k'))
                        labels.append(label_names[idx])

                # Place legend on the right side, vertically aligned
                # Set ncol=1 for single column vertical legend
                plt.legend(handles=handles,
                        labels=labels,
                        title="Modulation Classes",
                        loc='center right',  # Right side, vertically centered
                        fontsize='small',    # Smaller font
                        framealpha=0.7,      # Slightly transparent frame
                        ncol=1,              # Single column for vertical layout
                        borderpad=1,         # Padding around legend border
                        labelspacing=0.5)    # Spacing between entries

            plt.xlim(self.x_min, self.x_max)
            plt.ylim(self.y_min, self.y_max)
            plt.title(f'Decision Boundaries (Epoch {trainer.current_epoch})')
            plt.xlabel('t-SNE dimension 1')
            plt.ylabel('t-SNE dimension 2')

            # Add grid for better readability
            plt.grid(alpha=0.2)

            # Save the figure with some extra width to accommodate the legend
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)

            # Log to wandb
            img = wandb.Image(Image.open(buf))
            trainer.logger.experiment.log({
                "decision_boundary": img,
                "current_epoch": trainer.current_epoch
            })

            # Store frame for animation
            if self.create_animation:
                self.frames.append(Image.open(buf))

        except Exception as e:
            print(f"Error in decision boundary visualization: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_fit_end(self, trainer, pl_module):
        # Create and log the animation at the end of training
        if self.create_animation and len(self.frames) > 1:
            self._create_and_log_animation(trainer.logger)

    def _create_and_log_animation(self, logger):
        """Create a GIF animation from stored frames and log to wandb"""
        if not self.frames:
            return

        print(f"Creating animation from {len(self.frames)} frames")

        # Save frames as GIF
        gif_path = "decision_boundary_animation.gif"
        self.frames[0].save(
            gif_path,
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=500,  # 500ms per frame
            loop=0  # Loop forever
        )

        # Log the animation to wandb
        logger.experiment.log({
            "decision_boundary_animation": wandb.Video(gif_path, fps=2, format="gif")
        })

        print(f"Animation saved to {gif_path} and logged to wandb")

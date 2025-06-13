import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
import torchmetrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class ArcFaceLinearProbe(L.LightningModule):
    """Linear probe specifically designed for ArcFace embeddings"""

    def __init__(
        self,
        encoder,
        num_classes,
        label_names,
        learning_rate=1e-3,
        use_normalized_embeddings=True,
        embedding_type='arcface',  # 'arcface', 'raw', or 'both'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['arcface_encoder'])

        self.arcface_encoder = encoder
        self.num_classes = num_classes
        self.label_names = label_names
        self.learning_rate = learning_rate
        self.use_normalized_embeddings = use_normalized_embeddings
        self.embedding_type = embedding_type

        # Freeze the encoder completely
        for param in self.arcface_encoder.parameters():
            param.requires_grad = False
        self.arcface_encoder.eval()

        # Determine embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 2, 128)
            if hasattr(self.arcface_encoder, 'get_arcface_embeddings'):
                arcface_emb = self.arcface_encoder.get_arcface_embeddings(dummy_input)
                self.embedding_dim = arcface_emb.shape[1]
            else:
                # Fallback: use encoder output and normalize
                encoder_output = self.arcface_encoder.encode(dummy_input)
                if isinstance(encoder_output, dict):
                    emb = encoder_output.get('signal_features', encoder_output.get('full_embedding'))
                else:
                    emb = encoder_output
                self.embedding_dim = emb.shape[1]

        print(f"ArcFace embedding dimension: {self.embedding_dim}")

        # Create classifier based on embedding type
        if embedding_type == 'both':
            # Use both normalized and raw embeddings
            classifier_input_dim = self.embedding_dim * 2
        else:
            classifier_input_dim = self.embedding_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(classifier_input_dim, num_classes)
        )

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

        # For analysis
        self.val_embeddings = []
        self.val_predictions = []
        self.val_targets = []
        self.val_snrs = []

    def get_embeddings(self, x):
        """Extract ArcFace embeddings"""
        with torch.no_grad():
            if hasattr(self.arcface_encoder, 'get_arcface_embeddings'):
                arcface_emb = self.arcface_encoder.get_arcface_embeddings(x)

                if self.embedding_type == 'arcface':
                    return arcface_emb
                elif self.embedding_type == 'raw':
                    raw_emb = self.arcface_encoder.get_raw_embeddings(x)
                    return raw_emb
                elif self.embedding_type == 'both':
                    raw_emb = self.arcface_encoder.get_raw_embeddings(x)
                    return torch.cat([arcface_emb, raw_emb], dim=1)

            else:
                # Fallback method
                encoder_output = self.arcface_encoder.encode(x)
                if isinstance(encoder_output, dict):
                    emb = encoder_output.get('signal_features', encoder_output.get('full_embedding'))
                else:
                    emb = encoder_output

                if self.use_normalized_embeddings:
                    emb = F.normalize(emb, p=2, dim=1)

                return emb

    def forward(self, x):
        """Forward pass"""
        embeddings = self.get_embeddings(x)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def training_step(self, batch, batch_idx):
        x, labels, snrs = batch
        labels = labels.squeeze().long()

        logits, embeddings = self(x)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        f1 = self.train_f1(preds, labels)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_f1', f1)

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels, snrs = batch
        labels = labels.squeeze().long()

        logits, embeddings = self(x)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)
        f1 = self.val_f1(preds, labels)

        # Store for analysis
        self.val_embeddings.append(embeddings.cpu())
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())
        self.val_snrs.extend(snrs.cpu().numpy().flatten())

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1)

        return loss

    def on_validation_epoch_end(self):
        """Comprehensive analysis of ArcFace embeddings"""
        if len(self.val_predictions) == 0:
            return

        try:
            # Convert to numpy
            y_pred = np.array(self.val_predictions)
            y_true = np.array(self.val_targets)
            snrs = np.array(self.val_snrs)
            embeddings = torch.cat(self.val_embeddings, dim=0).numpy()

            # 1. Classification Report
            self._log_classification_metrics(y_true, y_pred)

            # 2. Confusion Matrix
            self._plot_confusion_matrix(y_true, y_pred)

            # 3. SNR-based Analysis
            self._plot_snr_analysis(y_true, y_pred, snrs)

            # 4. ArcFace Embedding Analysis
            self._plot_arcface_analysis(embeddings, y_true, snrs)

            # 5. Angular Distance Analysis
            self._analyze_angular_distances(embeddings, y_true)

        except Exception as e:
            print(f"Error in validation analysis: {e}")
        finally:
            # Clear stored data
            self.val_embeddings = []
            self.val_predictions = []
            self.val_targets = []
            self.val_snrs = []

    def _log_classification_metrics(self, y_true, y_pred):
        """Log detailed classification metrics"""
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0
        )

        for i, class_name in enumerate(self.label_names):
            if class_name in report:
                self.log(f'val_precision_{class_name}', report[class_name]['precision'])
                self.log(f'val_recall_{class_name}', report[class_name]['recall'])
                self.log(f'val_f1_{class_name}', report[class_name]['f1-score'])

        if 'macro avg' in report:
            self.log('val_macro_precision', report['macro avg']['precision'])
            self.log('val_macro_recall', report['macro avg']['recall'])
            self.log('val_macro_f1', report['macro avg']['f1-score'])

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(12, 10))

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            ax=ax,
            cmap='Blues'
        )

        ax.set_title(f'ArcFace Linear Probe - Confusion Matrix\nEpoch {self.current_epoch}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({'arcface_confusion_matrix': wandb.Image(fig)})

        plt.close(fig)

    def _plot_snr_analysis(self, y_true, y_pred, snrs):
        """Analyze performance vs SNR"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Overall accuracy vs SNR
        snr_bins = np.arange(-20, 20, 2)
        accuracies = []
        bin_centers = []
        sample_counts = []

        for i in range(len(snr_bins) - 1):
            mask = (snrs >= snr_bins[i]) & (snrs < snr_bins[i + 1])
            if mask.sum() > 0:
                bin_acc = (y_true[mask] == y_pred[mask]).mean()
                accuracies.append(bin_acc)
                bin_centers.append((snr_bins[i] + snr_bins[i + 1]) / 2)
                sample_counts.append(mask.sum())

        ax1.plot(bin_centers, accuracies, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('ArcFace Linear Probe: Accuracy vs SNR')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Per-class accuracy at high SNR (>10dB)
        high_snr_mask = snrs > 10
        if high_snr_mask.sum() > 0:
            class_accuracies = []
            class_names_used = []

            for i, class_name in enumerate(self.label_names):
                class_mask = (y_true == i) & high_snr_mask
                if class_mask.sum() > 0:
                    class_acc = (y_true[class_mask] == y_pred[class_mask]).mean()
                    class_accuracies.append(class_acc)
                    class_names_used.append(class_name)

            if class_accuracies:
                bars = ax2.bar(range(len(class_accuracies)), class_accuracies)
                ax2.set_xlabel('Modulation Class')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Per-Class Accuracy (SNR > 10dB)')
                ax2.set_xticks(range(len(class_names_used)))
                ax2.set_xticklabels(class_names_used, rotation=45, ha='right')
                ax2.set_ylim(0, 1)

                # Color bars by performance
                for bar, acc in zip(bars, class_accuracies):
                    if acc > 0.9:
                        bar.set_color('green')
                    elif acc > 0.7:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')

        plt.tight_layout()

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({'arcface_snr_analysis': wandb.Image(fig)})

        plt.close(fig)

    def _plot_arcface_analysis(self, embeddings, y_true, snrs):
        """Analyze ArcFace embedding quality"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        unique_labels = np.unique(y_true)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        # 1. t-SNE visualization colored by class
        if len(embeddings) > 1:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
            embeddings_2d = tsne.fit_transform(embeddings)

            for i, label in enumerate(unique_labels):
                mask = y_true == label
                if mask.sum() > 0:
                    ax1.scatter(
                        embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=[colors[i]], label=self.label_names[int(label)],
                        alpha=0.7, s=20
                    )

            ax1.set_title('t-SNE: ArcFace Embeddings by Class')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. t-SNE colored by SNR
        if len(embeddings) > 1:
            scatter = ax2.scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=snrs, cmap='viridis', alpha=0.7, s=20
            )
            ax2.set_title('t-SNE: ArcFace Embeddings by SNR')
            plt.colorbar(scatter, ax=ax2, label='SNR (dB)')

        # 3. Embedding norm distribution
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        ax3.hist(embedding_norms, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(embedding_norms.mean(), color='red', linestyle='--',
                   label=f'Mean: {embedding_norms.mean():.3f}')
        ax3.axvline(1.0, color='green', linestyle='--', alpha=0.8,
                   label='Unit norm (ArcFace target)')
        ax3.set_xlabel('Embedding L2 Norm')
        ax3.set_ylabel('Count')
        ax3.set_title('ArcFace Embedding Norm Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Within-class vs between-class distances
        self._plot_distance_analysis(ax4, embeddings, y_true)

        plt.tight_layout()

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({'arcface_embedding_analysis': wandb.Image(fig)})

        plt.close(fig)

    def _plot_distance_analysis(self, ax, embeddings, y_true):
        """Analyze within-class vs between-class distances"""
        within_class_dists = []
        between_class_dists = []

        unique_labels = np.unique(y_true)

        # Sample for efficiency if too many points
        max_samples = 1000
        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings_sample = embeddings[indices]
            y_true_sample = y_true[indices]
        else:
            embeddings_sample = embeddings
            y_true_sample = y_true

        # Calculate distances
        for i in range(len(embeddings_sample)):
            for j in range(i + 1, len(embeddings_sample)):
                dist = np.linalg.norm(embeddings_sample[i] - embeddings_sample[j])

                if y_true_sample[i] == y_true_sample[j]:
                    within_class_dists.append(dist)
                else:
                    between_class_dists.append(dist)

        # Plot histograms
        if within_class_dists:
            ax.hist(within_class_dists, bins=30, alpha=0.5, label='Within-class', color='blue')
        if between_class_dists:
            ax.hist(between_class_dists, bins=30, alpha=0.5, label='Between-class', color='red')

        ax.set_xlabel('L2 Distance')
        ax.set_ylabel('Count')
        ax.set_title('Distance Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        if within_class_dists and between_class_dists:
            within_mean = np.mean(within_class_dists)
            between_mean = np.mean(between_class_dists)
            separation_ratio = between_mean / within_mean if within_mean > 0 else float('inf')

            ax.text(0.05, 0.95,
                   f'Within: {within_mean:.3f}\nBetween: {between_mean:.3f}\nRatio: {separation_ratio:.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _analyze_angular_distances(self, embeddings, y_true):
        """Analyze angular distances between class centroids"""
        unique_labels = np.unique(y_true)
        n_classes = len(unique_labels)

        if n_classes < 2:
            return

        # Calculate class centroids
        centroids = []
        class_names = []

        for label in unique_labels:
            mask = y_true == label
            if mask.sum() > 0:
                centroid = embeddings[mask].mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)  # Normalize
                centroids.append(centroid)
                class_names.append(self.label_names[int(label)])

        centroids = np.array(centroids)

        # Calculate angular distance matrix
        distance_matrix = np.zeros((len(centroids), len(centroids)))

        for i in range(len(centroids)):
            for j in range(len(centroids)):
                cos_sim = np.clip(np.dot(centroids[i], centroids[j]), -1.0, 1.0)
                angular_dist = np.arccos(cos_sim) * 180 / np.pi
                distance_matrix[i, j] = angular_dist

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(distance_matrix, cmap='viridis')

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_title('Angular Distances Between Class Centroids (degrees)')

        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f'{distance_matrix[i, j]:.1f}°',
                       ha='center', va='center',
                       color='white' if distance_matrix[i, j] > distance_matrix.max()/2 else 'black',
                       fontsize=8)

        plt.colorbar(im, ax=ax, label='Angular Distance (degrees)')
        plt.tight_layout()

        # Log minimum non-zero angular distance (class separability metric)
        non_diagonal = distance_matrix[np.eye(len(centroids)) == 0]
        min_angular_dist = non_diagonal.min() if len(non_diagonal) > 0 else 0
        mean_angular_dist = non_diagonal.mean() if len(non_diagonal) > 0 else 0

        self.log('arcface_min_angular_distance', min_angular_dist)
        self.log('arcface_mean_angular_distance', mean_angular_dist)

        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({'arcface_angular_distances': wandb.Image(fig)})

        plt.close(fig)

    def configure_optimizers(self):
        """Configure optimizer for linear probe"""
        optimizer = AdamW(
            self.classifier.parameters(),  # Only train classifier
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=True
            ),
            'monitor': 'val_acc',
            'interval': 'epoch',
            'frequency': 1,
            'strict': True
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

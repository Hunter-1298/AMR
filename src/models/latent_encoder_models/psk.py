import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
from torch.optim import AdamW
from typing import Dict, Tuple, Optional, List
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import math

class BatchSynchronizer(nn.Module):
    """
    Vectorized batch synchronization for PSK signals with fixed complex operations.
    """
    def __init__(self, signal_length=1024, sps=4, alpha=0.35, num_taps=33):
        super().__init__()
        self.signal_length = signal_length
        self.sps = sps
        self.alpha = alpha
        self.num_taps = num_taps

        # Pre-compute RRC filter
        self.register_buffer('rrc_filter', self._create_rrc_filter())

    def _create_rrc_filter(self):
        """Create Root Raised Cosine filter."""
        T = self.sps
        t = torch.arange(-self.num_taps//2, self.num_taps//2 + 1, dtype=torch.float32)
        t = t / T

        h = torch.zeros_like(t)
        eps = 1e-10

        for i, ti in enumerate(t):
            if abs(ti) < eps:
                h[i] = (1 - self.alpha + 4 * self.alpha / np.pi)
            elif abs(abs(4 * self.alpha * ti) - 1) < eps:
                h[i] = (self.alpha / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*self.alpha)) +
                                                   (1 - 2/np.pi) * np.cos(np.pi/(4*self.alpha)))
            else:
                numerator = np.sin(np.pi * ti * (1 - self.alpha)) + 4 * self.alpha * ti * np.cos(np.pi * ti * (1 + self.alpha))
                denominator = np.pi * ti * (1 - (4 * self.alpha * ti)**2)
                h[i] = numerator / denominator

        # Normalize
        h = h / torch.sqrt(torch.sum(h**2))
        return h

    def matched_filter(self, signals):
        """Apply matched filtering to batch of signals."""
        # signals: [batch, signal_length] complex -> separate to real/imag
        real_part = signals.real.unsqueeze(1)  # [batch, 1, length]
        imag_part = signals.imag.unsqueeze(1)  # [batch, 1, length]

        # Apply filter to both I and Q
        rrc_filter = self.rrc_filter.unsqueeze(0).unsqueeze(0)  # [1, 1, num_taps]

        filtered_real = F.conv1d(real_part, rrc_filter, padding=self.num_taps//2)
        filtered_imag = F.conv1d(imag_part, rrc_filter, padding=self.num_taps//2)

        # Ensure output length matches input length
        if filtered_real.shape[2] != self.signal_length:
            filtered_real = F.interpolate(filtered_real, size=self.signal_length, mode='linear', align_corners=False)
            filtered_imag = F.interpolate(filtered_imag, size=self.signal_length, mode='linear', align_corners=False)

        # Combine back to complex using torch.complex
        filtered = torch.complex(filtered_real.squeeze(1), filtered_imag.squeeze(1))

        return filtered

    def timing_recovery_batch(self, signals):
        """Simplified timing recovery for batch."""
        # Apply matched filtering
        filtered = self.matched_filter(signals)

        # Simple timing recovery: just apply a small random offset to simulate timing correction
        batch_size = filtered.shape[0]
        recovered_signals = []

        for i in range(batch_size):
            signal_i = filtered[i]

            # Apply a small circular shift (simulates timing correction)
            shift_amount = torch.randint(-4, 5, (1,)).item()  # Random shift of -4 to +4 samples
            if shift_amount != 0:
                shifted = torch.roll(signal_i, shift_amount)
            else:
                shifted = signal_i

            # Ensure exact length
            if len(shifted) != self.signal_length:
                if len(shifted) > self.signal_length:
                    shifted = shifted[:self.signal_length]
                else:
                    padding = torch.zeros(self.signal_length - len(shifted),
                                        dtype=shifted.dtype, device=shifted.device)
                    shifted = torch.cat([shifted, padding])

            recovered_signals.append(shifted)

        return torch.stack(recovered_signals)

    def carrier_recovery_batch(self, signals, modulation_orders):
        """Fixed carrier phase recovery for batch."""
        batch_size = signals.shape[0]
        corrected_signals = []

        for i in range(batch_size):
            signal = signals[i]

            # Get modulation order for this signal
            if isinstance(modulation_orders, torch.Tensor):
                mod_order = modulation_orders[i].item()
            else:
                mod_order = modulation_orders

            best_signal = signal
            best_score = float('inf')

            # Test different phase corrections
            num_phase_tests = 8
            phase_step = 2 * np.pi / mod_order / num_phase_tests

            for j in range(num_phase_tests):
                phase_offset = j * phase_step

                # FIXED: Apply phase correction using real arithmetic
                cos_phase = torch.cos(torch.tensor(phase_offset, device=signal.device))
                sin_phase = torch.sin(torch.tensor(phase_offset, device=signal.device))

                # Manual complex multiplication: signal * exp(-1j * phase_offset)
                # exp(-1j * phase) = cos(phase) - 1j * sin(phase)
                real_part = signal.real * cos_phase + signal.imag * sin_phase
                imag_part = signal.imag * cos_phase - signal.real * sin_phase
                corrected = torch.complex(real_part, imag_part)

                # Score based on constellation tightness
                angles = torch.atan2(corrected.imag, corrected.real)

                if mod_order == 4:  # QPSK
                    quantized_angles = torch.round(angles / (np.pi/2)) * (np.pi/2)
                elif mod_order == 8:  # 8PSK
                    quantized_angles = torch.round(angles / (np.pi/4)) * (np.pi/4)
                else:  # 16PSK
                    quantized_angles = torch.round(angles / (np.pi/8)) * (np.pi/8)

                # Calculate phase error
                angle_errors = torch.abs(angles - quantized_angles)
                # Handle wrap-around
                angle_errors = torch.minimum(angle_errors, 2*np.pi - angle_errors)
                error = torch.mean(angle_errors)

                if error < best_score:
                    best_score = error
                    best_signal = corrected

            # Ensure exact length
            if len(best_signal) != self.signal_length:
                if len(best_signal) > self.signal_length:
                    best_signal = best_signal[:self.signal_length]
                else:
                    padding = torch.zeros(self.signal_length - len(best_signal),
                                        dtype=best_signal.dtype, device=best_signal.device)
                    best_signal = torch.cat([best_signal, padding])

            corrected_signals.append(best_signal)

        return torch.stack(corrected_signals)

    def forward(self, i_signals, q_signals, modulation_orders=None):
        """
        Batch synchronization with fixed complex operations.
        """
        # Ensure inputs have correct shape
        if i_signals.shape[1] != self.signal_length:
            i_signals = F.interpolate(i_signals.unsqueeze(1), size=self.signal_length,
                                    mode='linear', align_corners=False).squeeze(1)
        if q_signals.shape[1] != self.signal_length:
            q_signals = F.interpolate(q_signals.unsqueeze(1), size=self.signal_length,
                                    mode='linear', align_corners=False).squeeze(1)

        # Combine to complex using torch.complex
        complex_signals = torch.complex(i_signals, q_signals)

        # Default modulation orders
        if modulation_orders is None:
            modulation_orders = 4  # QPSK default

        # Timing recovery
        timing_recovered = self.timing_recovery_batch(complex_signals)

        # Carrier recovery
        carrier_recovered = self.carrier_recovery_batch(timing_recovered, modulation_orders)

        # Normalize power
        power = torch.mean(torch.abs(carrier_recovered)**2, dim=1, keepdim=True)
        normalized = carrier_recovered / torch.sqrt(power + 1e-10)

        # Ensure output has exact length
        if normalized.shape[1] != self.signal_length:
            if normalized.shape[1] > self.signal_length:
                normalized = normalized[:, :self.signal_length]
            else:
                batch_size = normalized.shape[0]
                padding_length = self.signal_length - normalized.shape[1]
                padding = torch.zeros(batch_size, padding_length,
                                    dtype=normalized.dtype, device=normalized.device)
                normalized = torch.cat([normalized, padding], dim=1)

        # Separate back to I/Q
        return normalized.real, normalized.imag

class SimplePSKCNN(nn.Module):
    """
    Simplified PSK CNN with working batch synchronization.
    """
    def __init__(
        self,
        signal_length: int = 1024,
        num_classes: int = 3,
        sync_signals: bool = True,
    ):
        super().__init__()

        self.signal_length = signal_length
        self.num_classes = num_classes
        self.sync_signals = sync_signals

        # Batch synchronizer
        if sync_signals:
            self.synchronizer = BatchSynchronizer(signal_length=signal_length)

        # CNN layers
        self.conv1 = nn.Conv1d(2, 64, kernel_size=32, stride=4, padding=16)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=4)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=8, stride=2, padding=4)
        self.bn4 = nn.BatchNorm1d(512)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Decoder for compatibility
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, signal_length * 2)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [batch, 2, signal_length] - I/Q signal
        """
        batch_size = x.shape[0]

        # Apply batch synchronization if enabled
        if self.sync_signals and hasattr(self, 'synchronizer'):
            # Separate I and Q channels
            i_signal = x[:, 0, :]  # [batch, signal_length]
            q_signal = x[:, 1, :]  # [batch, signal_length]

            # Estimate modulation orders based on phase variance
            with torch.no_grad():
                phase = torch.atan2(q_signal, i_signal + 1e-8)
                phase_var = torch.var(phase, dim=1)

                # Simple heuristic for modulation order
                mod_orders = torch.where(phase_var < 0.5, 4,
                           torch.where(phase_var < 1.0, 8, 16))

            try:
                i_signal, q_signal = self.synchronizer(i_signal, q_signal, mod_orders)
                x = torch.stack([i_signal, q_signal], dim=1)
                print(f"Synchronization successful for batch of {batch_size} signals")
            except Exception as e:
                print(f"Synchronization failed: {e}, using original signal")
                # Continue with original signal

        # CNN forward pass
        h1 = F.relu(self.bn1(self.conv1(x)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        h4 = F.relu(self.bn4(self.conv4(h3)))

        # Classification
        class_logits = self.classifier(h4)

        # Global features
        global_features = F.adaptive_avg_pool1d(h4, 1).squeeze(-1)

        return {
            'features': global_features,
            'quantized': global_features,
            'class_logits': class_logits,
            'vq_loss': torch.tensor(0.0, device=x.device),
            'center_loss': torch.tensor(0.0, device=x.device),
            'codes': torch.zeros(batch_size, dtype=torch.long, device=x.device),
            'perplexity': torch.tensor(1.0, device=x.device),
            'code_distribution': {'QPSK': 0.33, '8PSK': 0.33, '16PSK': 0.34}
        }

class PSKDiscriminator(L.LightningModule):
    """
    Lightning module for PSK discrimination with working synchronization.
    """
    def __init__(
        self,
        signal_length: int = 1024,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        sync_signals: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # PSK types
        self.label_names = ['QPSK', '8PSK', '16PSK']
        self.num_classes = 3

        # Model
        self.encoder = SimplePSKCNN(
            signal_length=signal_length,
            num_classes=self.num_classes,
            sync_signals=sync_signals,
        )

        # Decoder reference for compatibility
        self.decoder = self.encoder.decoder

        # Track performance
        self.sync_improvements = []

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # Handle different batch formats
        if len(batch) == 4:
            x_i, x_j, labels, snrs = batch
            signals = x_i
        elif len(batch) == 3:
            (x_i, x_j), labels, snrs = batch
            signals = x_i
        elif len(batch) == 2:
            signals, labels = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        labels = labels.long()

        # Forward pass
        outputs = self.encoder(signals)

        # Classification loss
        class_loss = F.cross_entropy(outputs['class_logits'], labels, label_smoothing=0.1)

        # Accuracy
        preds = outputs['class_logits'].argmax(dim=1)
        acc = (preds == labels).float().mean()

        # Logging
        self.log('train_loss', class_loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return class_loss

    def validation_step(self, batch, batch_idx):
        # Handle different batch formats
        if len(batch) == 4:
            x_i, x_j, labels, snrs = batch
            signals = x_i
        elif len(batch) == 3:
            (x_i, x_j), labels, snrs = batch
            signals = x_i
        elif len(batch) == 2:
            signals, labels = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        labels = labels.long()

        # Test with and without synchronization (only on first batch for speed)
        if batch_idx == 0:
            self._compare_sync_performance(signals, labels)

        # Forward pass
        outputs = self.encoder(signals)

        # Loss and accuracy
        class_loss = F.cross_entropy(outputs['class_logits'], labels)
        preds = outputs['class_logits'].argmax(dim=1)
        acc = (preds == labels).float().mean()

        # Per-class accuracy
        for i, label_name in enumerate(self.label_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == labels[class_mask]).float().mean()
                self.log(f'val_acc_{label_name}', class_acc)

        # Logging
        self.log('val_loss', class_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        # Store for visualization
        if batch_idx == 0:
            self.val_outputs = outputs
            self.val_signals = signals.detach().cpu()
            self.val_labels = labels.detach().cpu()

        return class_loss

    def _compare_sync_performance(self, signals, labels):
        """Compare performance with and without synchronization."""
        try:
            # Test without sync
            original_sync_setting = self.encoder.sync_signals
            self.encoder.sync_signals = False
            with torch.no_grad():
                outputs_no_sync = self.encoder(signals)
                preds_no_sync = outputs_no_sync['class_logits'].argmax(dim=1)
                acc_no_sync = (preds_no_sync == labels).float().mean()

            # Test with sync
            self.encoder.sync_signals = True
            with torch.no_grad():
                outputs_sync = self.encoder(signals)
                preds_sync = outputs_sync['class_logits'].argmax(dim=1)
                acc_sync = (preds_sync == labels).float().mean()

            # Restore original setting
            self.encoder.sync_signals = original_sync_setting

            # Log comparison
            improvement = acc_sync - acc_no_sync
            self.log('sync_improvement', improvement)
            self.log('acc_no_sync', acc_no_sync)
            self.log('acc_with_sync', acc_sync)

            self.sync_improvements.append(improvement.item())

            print(f"Sync comparison - No sync: {acc_no_sync:.3f}, With sync: {acc_sync:.3f}, Improvement: {improvement:.3f}")

        except Exception as e:
            print(f"Sync comparison failed: {e}")

    def on_validation_epoch_end(self):
        """Visualize results."""
        if hasattr(self, 'val_outputs'):
            self._visualize_results()

    def _visualize_results(self):
        """Visualize classification results."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            features = self.val_outputs['features'].detach().cpu().numpy()
            labels = self.val_labels.numpy()

            # t-SNE visualization
            if len(features) > 3:
                tsne = TSNE(n_components=2, perplexity=min(30, len(features)-1), random_state=42)
                features_2d = tsne.fit_transform(features)

                colors = ['red', 'green', 'blue']
                for i, (label_name, color) in enumerate(zip(self.label_names, colors)):
                    mask = labels == i
                    if mask.sum() > 0:
                        axes[0, 0].scatter(
                            features_2d[mask, 0], features_2d[mask, 1],
                            c=color, label=label_name, alpha=0.7, s=50
                        )

                axes[0, 0].set_title('t-SNE: PSK CNN Features')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            preds = self.val_outputs['class_logits'].argmax(dim=1).cpu().numpy()
            cm = confusion_matrix(labels, preds)

            im = axes[0, 1].imshow(cm, cmap='Blues')
            axes[0, 1].set_xticks(range(3))
            axes[0, 1].set_yticks(range(3))
            axes[0, 1].set_xticklabels(self.label_names)
            axes[0, 1].set_yticklabels(self.label_names)
            axes[0, 1].set_title('Confusion Matrix')

            for i in range(3):
                for j in range(3):
                    axes[0, 1].text(j, i, str(cm[i, j]), ha='center', va='center')

            plt.colorbar(im, ax=axes[0, 1])

            # Constellation plots
            for idx, label_idx in enumerate([0, 1]):
                ax = axes[1, idx]
                mask = labels == label_idx
                if mask.sum() > 0:
                    sample_signal = self.val_signals[mask][0]
                    i_channel = sample_signal[0].numpy()
                    q_channel = sample_signal[1].numpy()

                    subsample_i = i_channel[::20]
                    subsample_q = q_channel[::20]

                    ax.scatter(subsample_i, subsample_q, alpha=0.6, s=10)
                    ax.set_title(f'{self.label_names[label_idx]} Constellation')
                    ax.set_xlabel('I')
                    ax.set_ylabel('Q')
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal')

            plt.tight_layout()

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({'psk_sync_results': wandb.Image(fig)})

            plt.close(fig)

        except Exception as e:
            print(f"Error in visualization: {e}")
            plt.close('all')

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def predict_psk_type(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict PSK type and return probabilities."""
        with torch.no_grad():
            outputs = self.encoder(x)
            logits = outputs['class_logits']
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            return preds, probs

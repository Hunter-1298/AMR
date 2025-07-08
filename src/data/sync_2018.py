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

class LearnableSynchronizer(nn.Module):
    """
    Learnable synchronization module with tunable parameters.
    """
    def __init__(self, signal_length=1024, num_phase_candidates=16, num_timing_offsets=8):
        super().__init__()
        self.signal_length = signal_length
        self.num_phase_candidates = num_phase_candidates
        self.num_timing_offsets = num_timing_offsets

        # Learnable phase correction network
        self.phase_estimator = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=64, stride=16, padding=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=8, padding=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Output phase correction in radians
            nn.Tanh()  # Constrain to [-1, 1], then scale to [-π, π]
        )

        # Learnable timing offset network
        self.timing_estimator = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=64, stride=16, padding=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=8, padding=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Output timing offset as fraction of symbol period
            nn.Tanh()  # Constrain to [-1, 1]
        )

        # Learnable matched filter (instead of fixed RRC)
        self.matched_filter = nn.Conv1d(2, 2, kernel_size=33, padding=16, groups=2)

        # Learnable PSK order classifier (helps with sync)
        self.psk_order_estimator = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=32, stride=8, padding=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # QPSK, 8PSK, 16PSK probabilities
            nn.Softmax(dim=1)
        )

        # Register modulation-specific phase step sizes
        self.register_buffer('psk_phase_steps', torch.tensor([np.pi/2, np.pi/4, np.pi/8]))  # QPSK, 8PSK, 16PSK

    def apply_phase_correction(self, signals, phase_corrections):
        """Apply learnable phase corrections."""
        batch_size = signals.shape[0]
        corrected_signals = []

        for i in range(batch_size):
            signal = signals[i]
            phase_offset = phase_corrections[i] * np.pi  # Scale from [-1,1] to [-π,π]

            # Manual complex multiplication for phase correction
            cos_phase = torch.cos(phase_offset)
            sin_phase = torch.sin(phase_offset)

            real_part = signal.real * cos_phase + signal.imag * sin_phase
            imag_part = signal.imag * cos_phase - signal.real * sin_phase
            corrected = torch.complex(real_part, imag_part)

            corrected_signals.append(corrected)

        return torch.stack(corrected_signals)

    def apply_timing_correction(self, signals, timing_offsets):
        """Apply learnable timing corrections."""
        batch_size = signals.shape[0]
        corrected_signals = []

        for i in range(batch_size):
            signal = signals[i]
            timing_offset = timing_offsets[i] * 8  # Scale to max ±8 samples

            # Apply fractional delay using interpolation
            if abs(timing_offset) > 0.1:  # Only apply if significant
                # Simple circular shift for integer part
                int_offset = int(timing_offset.round())
                if int_offset != 0:
                    signal = torch.roll(signal, int_offset.item())

            corrected_signals.append(signal)

        return torch.stack(corrected_signals)

    def constellation_quality_loss(self, signals, psk_probs):
        """Compute constellation quality loss for each PSK type."""
        batch_size = signals.shape[0]
        quality_losses = []

        for i in range(batch_size):
            signal = signals[i]
            probs = psk_probs[i]  # [3] probabilities for QPSK, 8PSK, 16PSK

            # Compute phase angles
            angles = torch.atan2(signal.imag, signal.real)

            # Compute quality for each PSK type
            type_losses = []
            for psk_idx, phase_step in enumerate(self.psk_phase_steps):
                # Quantize angles to nearest constellation point
                quantized_angles = torch.round(angles / phase_step) * phase_step

                # Phase error
                phase_errors = torch.abs(angles - quantized_angles)
                phase_errors = torch.minimum(phase_errors, 2*np.pi - phase_errors)

                # Weighted by probability of this PSK type
                weighted_error = torch.mean(phase_errors) * probs[psk_idx]
                type_losses.append(weighted_error)

            quality_losses.append(sum(type_losses))

        return torch.stack(quality_losses).mean()

    def forward(self, x):
        """
        x: [batch, 2, signal_length] - I/Q signal
        Returns: synchronized signal and auxiliary losses
        """
        batch_size = x.shape[0]

        # Estimate PSK order probabilities
        psk_probs = self.psk_order_estimator(x)  # [batch, 3]

        # Apply learnable matched filtering
        filtered = self.matched_filter(x)

        # Convert to complex for synchronization
        complex_signals = torch.complex(filtered[:, 0, :], filtered[:, 1, :])

        # Estimate phase corrections
        phase_corrections = self.phase_estimator(filtered).squeeze(-1)  # [batch]

        # Estimate timing corrections
        timing_corrections = self.timing_estimator(filtered).squeeze(-1)  # [batch]

        # Apply corrections
        timing_corrected = self.apply_timing_correction(complex_signals, timing_corrections)
        phase_corrected = self.apply_phase_correction(timing_corrected, phase_corrections)

        # Compute constellation quality loss
        quality_loss = self.constellation_quality_loss(phase_corrected, psk_probs)

        # Normalize power
        power = torch.mean(torch.abs(phase_corrected)**2, dim=1, keepdim=True)
        normalized = phase_corrected / torch.sqrt(power + 1e-10)

        # Convert back to I/Q
        synchronized_iq = torch.stack([normalized.real, normalized.imag], dim=1)

        return synchronized_iq, quality_loss, psk_probs, phase_corrections, timing_corrections

class EnhancedPSKCNN(nn.Module):
    """
    Enhanced PSK CNN with learnable synchronization and better 8PSK/16PSK discrimination.
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

        # Learnable synchronizer
        if sync_signals:
            self.synchronizer = LearnableSynchronizer(signal_length=signal_length)

        # Enhanced CNN with multi-scale features for better PSK discrimination
        # Scale 1: Fine details (good for 16PSK)
        self.conv1a = nn.Conv1d(2, 32, kernel_size=16, stride=2, padding=8)
        self.bn1a = nn.BatchNorm1d(32)

        self.conv2a = nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=4)
        self.bn2a = nn.BatchNorm1d(64)

        # Scale 2: Medium details (good for 8PSK)
        self.conv1b = nn.Conv1d(2, 32, kernel_size=32, stride=4, padding=16)
        self.bn1b = nn.BatchNorm1d(32)

        self.conv2b = nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=8)
        self.bn2b = nn.BatchNorm1d(64)

        # Scale 3: Coarse details (good for QPSK)
        self.conv1c = nn.Conv1d(2, 32, kernel_size=64, stride=8, padding=32)
        self.bn1c = nn.BatchNorm1d(32)

        self.conv2c = nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=16)
        self.bn2c = nn.BatchNorm1d(64)

        # Fusion layers
        self.fusion_conv = nn.Conv1d(192, 256, kernel_size=8, stride=2, padding=4)  # 64*3 = 192
        self.fusion_bn = nn.BatchNorm1d(256)

        self.final_conv = nn.Conv1d(256, 512, kernel_size=8, stride=2, padding=4)
        self.final_bn = nn.BatchNorm1d(512)

        # Attention mechanism for PSK order discrimination
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 512, kernel_size=1),
            nn.Sigmoid()
        )

        # Classification layers with PSK-specific heads
        self.global_pool = nn.AdaptiveAvgPool1d(16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Auxiliary PSK order classifier (helps with sync training)
        self.aux_psk_classifier = nn.Sequential(
            nn.Linear(512 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # Decoder for compatibility
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, signal_length * 2)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [batch, 2, signal_length] - I/Q signal
        """
        batch_size = x.shape[0]

        # Apply learnable synchronization
        sync_loss = torch.tensor(0.0, device=x.device)
        aux_psk_probs = None
        phase_corrections = None
        timing_corrections = None

        if self.sync_signals and hasattr(self, 'synchronizer'):
            try:
                x, sync_loss, aux_psk_probs, phase_corrections, timing_corrections = self.synchronizer(x)
            except Exception as e:
                print(f"Learnable synchronization failed: {e}, using original signal")

        # Multi-scale CNN feature extraction
        # Scale 1: Fine features
        h1a = F.relu(self.bn1a(self.conv1a(x)))
        h2a = F.relu(self.bn2a(self.conv2a(h1a)))

        # Scale 2: Medium features
        h1b = F.relu(self.bn1b(self.conv1b(x)))
        h2b = F.relu(self.bn2b(self.conv2b(h1b)))

        # Scale 3: Coarse features
        h1c = F.relu(self.bn1c(self.conv1c(x)))
        h2c = F.relu(self.bn2c(self.conv2c(h1c)))

        # Align feature maps for concatenation
        target_length = min(h2a.shape[2], h2b.shape[2], h2c.shape[2])

        if h2a.shape[2] != target_length:
            h2a = F.interpolate(h2a, size=target_length, mode='linear', align_corners=False)
        if h2b.shape[2] != target_length:
            h2b = F.interpolate(h2b, size=target_length, mode='linear', align_corners=False)
        if h2c.shape[2] != target_length:
            h2c = F.interpolate(h2c, size=target_length, mode='linear', align_corners=False)

        # Fuse multi-scale features
        fused = torch.cat([h2a, h2b, h2c], dim=1)  # [batch, 192, target_length]

        # Further processing
        h_fused = F.relu(self.fusion_bn(self.fusion_conv(fused)))
        h_final = F.relu(self.final_bn(self.final_conv(h_fused)))

        # Apply attention
        attention_weights = self.attention(h_final)
        h_attended = h_final * attention_weights

        # Global pooling and classification
        pooled_features = self.global_pool(h_attended)  # [batch, 512, 16]
        flattened = pooled_features.view(batch_size, -1)  # [batch, 512*16]

        class_logits = self.classifier(flattened)

        # Auxiliary PSK classification (for sync training)
        aux_class_logits = self.aux_psk_classifier(flattened)

        # Global features
        global_features = F.adaptive_avg_pool1d(h_attended, 1).squeeze(-1)

        return {
            'features': global_features,
            'quantized': global_features,
            'class_logits': class_logits,
            'aux_class_logits': aux_class_logits,
            'sync_loss': sync_loss,
            'aux_psk_probs': aux_psk_probs,
            'phase_corrections': phase_corrections,
            'timing_corrections': timing_corrections,
            'vq_loss': torch.tensor(0.0, device=x.device),
            'center_loss': torch.tensor(0.0, device=x.device),
            'codes': torch.zeros(batch_size, dtype=torch.long, device=x.device),
            'perplexity': torch.tensor(1.0, device=x.device),
            'code_distribution': {'QPSK': 0.33, '8PSK': 0.33, '16PSK': 0.34}
        }

class PSKDiscriminator(L.LightningModule):
    """
    Lightning module with learnable synchronization and enhanced PSK discrimination.
    """
    def __init__(
        self,
        signal_length: int = 1024,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        sync_signals: bool = True,
        sync_weight: float = 0.1,
        aux_weight: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # PSK types
        self.label_names = ['QPSK', '8PSK', '16PSK']
        self.num_classes = 3

        # Model
        self.encoder = EnhancedPSKCNN(
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

        # Main classification loss
        class_loss = F.cross_entropy(outputs['class_logits'], labels, label_smoothing=0.1)

        # Auxiliary PSK classification loss (helps sync training)
        aux_loss = F.cross_entropy(outputs['aux_class_logits'], labels)

        # Synchronization quality loss
        sync_loss = outputs['sync_loss']

        # Total loss
        total_loss = (class_loss +
                     self.hparams.aux_weight * aux_loss +
                     self.hparams.sync_weight * sync_loss)

        # Accuracy
        preds = outputs['class_logits'].argmax(dim=1)
        acc = (preds == labels).float().mean()

        # Auxiliary accuracy
        aux_preds = outputs['aux_class_logits'].argmax(dim=1)
        aux_acc = (aux_preds == labels).float().mean()

        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_class_loss', class_loss)
        self.log('train_aux_loss', aux_loss)
        self.log('train_sync_loss', sync_loss)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_aux_acc', aux_acc)

        return total_loss

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

        # Losses
        class_loss = F.cross_entropy(outputs['class_logits'], labels)
        aux_loss = F.cross_entropy(outputs['aux_class_logits'], labels)
        sync_loss = outputs['sync_loss']
        total_loss = (class_loss +
                     self.hparams.aux_weight * aux_loss +
                     self.hparams.sync_weight * sync_loss)

        # Accuracy
        preds = outputs['class_logits'].argmax(dim=1)
        acc = (preds == labels).float().mean()

        # Per-class accuracy (especially important for 8PSK vs 16PSK)
        for i, label_name in enumerate(self.label_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == labels[class_mask]).float().mean()
                self.log(f'val_acc_{label_name}', class_acc)

        # Confusion between 8PSK and 16PSK specifically
        psk8_mask = labels == 1  # 8PSK
        psk16_mask = labels == 2  # 16PSK

        if psk8_mask.sum() > 0 and psk16_mask.sum() > 0:
            # 8PSK classified as 16PSK
            psk8_as_16 = (preds[psk8_mask] == 2).float().mean()
            # 16PSK classified as 8PSK
            psk16_as_8 = (preds[psk16_mask] == 1).float().mean()

            self.log('val_8psk_as_16psk', psk8_as_16)
            self.log('val_16psk_as_8psk', psk16_as_8)

        # Logging
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_class_loss', class_loss)
        self.log('val_sync_loss', sync_loss)
        self.log('val_acc', acc, prog_bar=True)

        # Store for visualization
        if batch_idx == 0:
            self.val_outputs = outputs
            self.val_signals = signals.detach().cpu()
            self.val_labels = labels.detach().cpu()

        return total_loss

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

            print(f"Learnable Sync - No sync: {acc_no_sync:.3f}, With sync: {acc_sync:.3f}, Improvement: {improvement:.3f}")

        except Exception as e:
            print(f"Sync comparison failed: {e}")

    def configure_optimizers(self):
        # Different learning rates for sync and classification components
        sync_params = []
        classifier_params = []

        for name, param in self.named_parameters():
            if 'synchronizer' in name:
                sync_params.append(param)
            else:
                classifier_params.append(param)

        optimizer = AdamW([
            {'params': classifier_params, 'lr': self.hparams.learning_rate},
            {'params': sync_params, 'lr': self.hparams.learning_rate * 0.5}  # Lower LR for sync
        ], weight_decay=1e-4)

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

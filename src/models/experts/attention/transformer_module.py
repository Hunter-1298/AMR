import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import wandb
import matplotlib.pyplot as plt
import math
from .stacked_cross_attention import StackedBidirectionalCrossAttention

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to token embeddings to provide sequence position information.
    Uses the standard sinusoidal positional encoding from the Transformer paper.
    """
    def __init__(self, embed_dim, max_seq_length=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter but should be part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_length, embed_dim]
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, channels, seq_length, embed_dim]
                for our case, channels=2 (amplitude and phase)
        Returns:
            x with positional encoding added
        """
        # Apply positional encoding separately to each channel
        batch_size, channels, seq_length, embed_dim = x.shape
        
        # Get the appropriate positional encoding for this sequence length
        pos_encoding = self.pe[:, :seq_length, :]  # [1, seq_length, embed_dim]
        
        # Add positional encoding to each channel
        for i in range(channels):
            # x[:, i] shape: [batch_size, seq_length, embed_dim]
            x[:, i] = x[:, i] + pos_encoding
            
        return self.dropout(x)

class BidirectionalTransformer(L.LightningModule):
    def __init__(self, label_names, epochs, embed_dim=64, num_heads=8, num_layers=4, dropout=0.1,num_classes=11, learning_rate=1e-4):
        """
        PyTorch Lightning module for training a stacked bidirectional cross-attention transformer.
        
        Args:
            embed_dim (int): Embedding dimension per token.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout probability.
            learning_rate (float): Learning rate for optimizer.
        """

        super().__init__()
        self.save_hyperparameters()
        self.label_names = label_names
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.epochs = epochs

        # going to store [correct/total]
        self.snr_accuracy = [(0,0) for _ in range(20)]

        # going to store [correct/total]
        self.mod_accuracy = [(0,0) for _ in range(self.num_classes)]
        
        # Add positional encoding layer
        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_seq_length=64,  # Max sequence length expected (adjust if needed)
            dropout=dropout
        )
        
        # Create the stacked transformer
        self.transformer = StackedBidirectionalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the transformer.
        
        Args:
            x: Tokenized input tensor of shape [batch_size, 2, num_tokens, embed_dim]
               where 2 channels are amplitude and phase
        Returns:
            output: Processed tensor from transformer [batch_size, 2, num_tokens, embed_dim]
            logits: Classification logits [batch_size, num_classes]
        """
        # x = [batch_size, channels, tokens, embed_dim]
        
        # Ensure contiguous memory layout to prevent stride issues
        x = x.contiguous()
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer - already handles the correct input shape and will return the same
        output = self.transformer(x)
        
        # Ensure output is contiguous before further operations
        output = output.contiguous()
        
        # Use the first token's embedding for classification (similar to [CLS] in BERT) -global context represenations
        # Average over both amplitude and phase channels
        # Extract first token and ensure proper memory layout
        first_token = output[:, :, 0, :].contiguous()
        cls_token = first_token.mean(dim=1)  # [batch_size, embed_dim]
        
        # Apply classifier
        logits = self.classifier(cls_token)  # [batch_size, num_classes]
        
        return output, logits
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch  # x: tokenized input [batch_size, 2, num_tokens, embed_dim]
        
        # Apply transformer and classifier
        _, logits = self(x)
        
        # Classification loss only
        loss = F.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)
        train_acc = self.accuracy(pred,y)
 
        # Log metrics
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", train_acc, on_epoch=True, on_step=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, snr = batch
        
        # Apply transformer and classifier
        output, logits = self(x)
        
        # Classification loss only
        val_loss = F.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)
        val_acc = self.accuracy(pred,y,snr)
        
        # Log metrics
        self.log("val_loss", val_loss, on_epoch=True, on_step=False)
        self.log("val_acc", val_acc, on_epoch=True, on_step=False)
        
        
        return val_loss

        
    def accuracy(self, pred, y, snr=None):
        # Get predicted class by taking argmax along class dimension
        correct = (pred == y).sum().item()
        acc = correct / y.size(0)

        # Get binary mask of correct predictions
        correct_mask = (pred == y).long()

        # Called only on validation steps
        if snr is not None:
            # Convert SNR values to indices (from -20:20:2 to 0:20)
            snr_indices = ((snr + 20) / 2).long()
            
            # For each SNR index, update total and correct counts
            for idx in range(20):
                mask = (snr_indices == idx)
                total = mask.sum().item()
                correct = (mask & correct_mask.bool()).sum().item()
                curr_correct, curr_total = self.snr_accuracy[idx]
                self.snr_accuracy[idx] = (curr_correct + correct, curr_total + total)

            # Track modulation-based accuracy
            for mod_idx in range(self.num_classes):
                mask = (y == mod_idx)
                total = mask.sum().item()
                correct = (mask & correct_mask.bool()).sum().item()
                curr_correct, curr_total = self.mod_accuracy[mod_idx]
                self.mod_accuracy[mod_idx] = (curr_correct + correct, curr_total + total)

        return acc
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        # OneCycleLR automatically determines warm-up and decay schedules
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.learning_rate,  # Peak LR at warmup end
            total_steps=self.trainer.estimated_stepping_batches,  # Total training steps
            pct_start=0.1,  # 5% of training is used for warm-up
            anneal_strategy="cos",  # Cosine decay after warmup
            final_div_factor=10  # Reduce LR by 10x at the end
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_train_epoch_end(self):

        # Create SNR accuracy line plot
        snr_values = [-20 + (i * 2) for i in range(20)]  # -20 to 18 in steps of 2
        accuracies = []
        for correct, total in self.snr_accuracy:
            acc = correct / total if total > 0 else 0
            accuracies.append(acc)

        # Create data for line plot
        snr_data = [[snr, acc] for snr, acc in zip(snr_values, accuracies)]
        snr_table = wandb.Table(columns=["SNR", "Accuracy"], data=snr_data)
        
        # Create Mod Accuracy Bar chart
        mod_accuracies = []
        for correct, total in self.mod_accuracy:
            acc = correct / total if total > 0 else 0
            mod_accuracies.append(acc)
        # Create data for bar
        mod_data = [[f"{self.label_names[i]}", acc] for i, acc in enumerate(mod_accuracies)]
        mod_table = wandb.Table(columns=["Modulation", "Accuracy"], data=mod_data)
        
        wandb.log({
            "Mod Accuracy": wandb.plot.bar(
                mod_table, "Modulation", "Accuracy", title="Modulation Accuracy"
            ),
            "snr_accuracy": wandb.plot.line(
                snr_table, 
                "SNR", 
                "Accuracy",
                title="Accuracy vs SNR"
            )
        })

        # Reset for next epoch
        self.snr_accuracy = [(0,0) for _ in range(20)]
        self.mod_accuracy = [(0,0) for _ in range(self.num_classes)]
    
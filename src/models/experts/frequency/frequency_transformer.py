import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig


class FrequencyTransformer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Project across time dimension instead of channels
        # Keep I/Q relationship intact
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Position encoding for the hidden dimension
        self.pos_encoding = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, 2, 128]
        batch_size = x.shape[0]
        
        # Project time dimension to hidden_dim
        # Maintain I/Q as separate "tokens" for transformer
        x = self.input_projection(x)  # Shape: [batch_size, 2, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Process through transformer
        # Now transformer attends between I and Q across the hidden dimension
        x = self.transformer(x)  # Shape: [batch_size, 2, hidden_dim]
        
        # Global average pooling across I/Q
        x = x.mean(dim=1)  # Shape: [batch_size, hidden_dim]
        
        return self.output_layer(x)  # Shape: [batch_size, output_dim]
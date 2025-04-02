import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_attention import BidirectionalCrossAttention

class StackedBidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        """
        Stack of Bidirectional Cross-Attention layers for transformer-like behavior.
        
        Args:
            embed_dim (int): Embedding dimension per token.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of stacked bidirectional cross-attention layers.
            dropout (float): Dropout probability for attention.
        """
        super(StackedBidirectionalCrossAttention, self).__init__()
        
        # Create a stack of BidirectionalCrossAttention layers
        self.layers = nn.ModuleList([
            BidirectionalCrossAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization after each attention layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])
        
        # Feedforward networks for each layer
        self.feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization after each feedforward network
        self.ff_layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 2, num_tokens, embed_dim],
               where x[:, 0, :, :] are amplitude tokens and x[:, 1, :, :] are phase tokens.
        Returns:
            x: Tensor of the same shape [B, 2, num_tokens, embed_dim]
               after passing through all attention layers.
        """
        # Pass through each layer
        for i, (attention, norm, ff, ff_norm) in enumerate(zip(
            self.layers, self.layer_norms, self.feedforward, self.ff_layer_norms
        )):
            # First sublayer: Bidirectional Cross-Attention with residual connection
            residual = x
            
            # Apply attention
            attn_output = attention(x)
            
            # Apply layer normalization with residual connection
            # We need to apply LayerNorm to each modality (amp/phase) separately
            amp_norm = norm(attn_output[:, 0, :, :] + residual[:, 0, :, :])
            phase_norm = norm(attn_output[:, 1, :, :] + residual[:, 1, :, :])
            x = torch.stack([amp_norm, phase_norm], dim=1)
            
            # Second sublayer: Feedforward network with residual connection
            residual = x
            
            # Apply feedforward to each modality separately
            amp_ff = ff(x[:, 0, :, :])
            phase_ff = ff(x[:, 1, :, :])
            
            # Apply layer normalization with residual connection
            amp_ff_norm = ff_norm(amp_ff + residual[:, 0, :, :])
            phase_ff_norm = ff_norm(phase_ff + residual[:, 1, :, :])
            x = torch.stack([amp_ff_norm, phase_ff_norm], dim=1)
            
        return x 
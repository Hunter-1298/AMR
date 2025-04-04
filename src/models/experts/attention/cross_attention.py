import torch
import torch.nn as nn


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Bidirectional Cross-Attention module that enables amplitude and phase to attend to each other.

        Args:
            embed_dim (int): Embedding dimension per token.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability for attention.
        """
        super(BidirectionalCrossAttention, self).__init__()

        # Use nn.MultiheadAttention with batch_first=True so that input shape is [B, L, D]
        self.attn_amp_to_phase = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_phase_to_amp = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Projection layer to combine original and attended features
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 2, num_tokens, embed_dim],
               where x[:, 0, :, :] are amplitude tokens and x[:, 1, :, :] are phase tokens.
        Returns:
            fused: Tensor of the same shape [B, 2, num_tokens, embed_dim]
                   with fused amplitude and phase tokens.
        """
        # Ensure input is contiguous
        x = x.contiguous()

        # Separate the channels: amplitude and phase
        amplitude = x[:, 0, :, :].contiguous()  # Shape: [B, num_tokens, embed_dim]
        phase = x[:, 1, :, :].contiguous()  # Shape: [B, num_tokens, embed_dim]

        # Cross-attention: Amplitude queries Phase
        # Query: amplitude, Key/Value: phase
        attn_amp, _ = self.attn_amp_to_phase(amplitude, phase, phase)

        # Cross-attention: Phase queries Amplitude
        # Query: phase, Key/Value: amplitude
        attn_phase, _ = self.attn_phase_to_amp(phase, amplitude, amplitude)

        # Concatenate original and attended features, then project
        fused_amp = self.proj(torch.cat([amplitude, attn_amp.contiguous()], dim=-1))
        fused_phase = self.proj(torch.cat([phase, attn_phase.contiguous()], dim=-1))

        # Stack the fused representations back along the channel dimension
        fused = torch.stack(
            [fused_amp, fused_phase], dim=1
        )  # Shape: [B, 2, num_tokens, embed_dim]

        return fused.contiguous()


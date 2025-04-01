import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Decoder network that reconstructs input from latent representation
    
    Args:
        hidden_dim (int): Size of hidden dimension
        feature_dim (int): Size of feature dimension
    """
    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),  # upsample
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim//2, 2, kernel_size=3, padding=1)  # 2 channels for I/Q data
        )

    def forward(self, x):
        return self.decoder(x)
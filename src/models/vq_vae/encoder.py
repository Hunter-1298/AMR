import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder network that maps windowed IQ data to latent representation
    
    Args:
        hidden_dim (int): Size of hidden dimension
        feature_dim (int): Size of feature dimension (embedding_dim)
    """
    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            # Input shape: [batch_size, channels=2, window_size]
            nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim//2, feature_dim, kernel_size=3, padding=1),
            # Output shape: [batch_size, feature_dim, window_size//2]
            nn.AdaptiveAvgPool1d(1)  # Average pool to get one vector per window
            # Final output shape: [batch_size, feature_dim, 1]
        )
        self.projection = nn.Linear(2, feature_dim)

    def forward(self, x):
        # 1. Initial input shape: [32, 2, 128, 1]
        # [batch_size, channels, num_windows=128, iq_samples=1]
        batch_size, channels, num_windows, iq_samples = x.shape
        
        # 2. Reshape to process each window:
        x = x.squeeze(-1).transpose(1, 2)
        # After squeeze: [32, 2, 128]
        # After transpose: [32, 128, 2]
        # Now each of the 128 windows has its 2 channel values ready for processing
        
        # 3. Process each window independently:
        encoded = x.reshape(-1, 2)
        # Shape: [32*128, 2]
        # Each row represents one window's I/Q values
        
        # 4. Project to feature dimension using a linear layer instead of CNN
        encoded = self.projection(encoded)
        # Shape: [32*128, feature_dim]
        
        # 5. Reshape back to separate batch and windows:
        encoded = encoded.view(batch_size, num_windows, -1)
        # Shape: [32, 128, feature_dim]
        
        # 6. Final reshape for vector quantizer:
        encoded = encoded.transpose(1, 2).unsqueeze(1)
        # Shape: [32, 1, feature_dim, 128]
        # [batch_size, channels=1, feature_dim, num_windows]
        
        return encoded
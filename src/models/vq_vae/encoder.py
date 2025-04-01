import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder network that maps amplitude or phase signal to latent representation
    
    Args:
        hidden_dim (int): Size of hidden dimension
        feature_dim (int): Size of feature dimension (embedding_dim)
    """
    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Second conv block
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        
        # Final conv
        self.conv3 = nn.Conv1d(hidden_dim//2, feature_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # First conv block with residual
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = x + self.conv1(identity)  # Residual connection
        
        # Second conv block with residual
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + self.conv2(identity)  # Residual connection
        
        # Final conv
        x = self.conv3(x)
        return x
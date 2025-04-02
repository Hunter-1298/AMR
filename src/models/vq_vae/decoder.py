import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Decoder network that maps latent representation back to amplitude or phase signal
    
    Args:
        feature_dim (int): Size of input feature dimension (32)
        output_dim (int): Size of output sequence length
        output_channels (int): Number of output channels
        embedding_dim (int): Size of th e embedding dimension from quantizer (64)
    """
    def __init__(self, feature_dim, output_dim, output_channels, embedding_dim):
        super().__init__()
        hidden_dim = feature_dim // 2
        
        # First conv block
        self.conv1 = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # Shortcut for first block
        self.shortcut1 = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1) if feature_dim != hidden_dim else nn.Sequential()
        
        # Second conv block
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        # Shortcut for second block (needs both channel and spatial adjustment)
        self.shortcut2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=1, stride=2)
        
        # Final conv
        self.conv3 = nn.Conv1d(hidden_dim//2, output_channels, kernel_size=3, padding=1)

        self.proj = nn.Linear(32,output_dim)

    def forward(self, x):
        # First conv block with residual
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = x + self.shortcut1(identity)  # Proper residual connection
        
        # Second conv block with residual
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + self.shortcut2(identity)  # Proper residual connection
        
        # Final conv
        x = self.proj(F.relu(self.conv3(x)))
        return x
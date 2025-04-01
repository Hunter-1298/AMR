import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

class Frequency_CNN(nn.Module):
    """
    Frequency Domain CNN Expert Model
    
    This class implements a CNN architecture specialized for frequency domain feature extraction
    from IQ data. It first transforms the input IQ data to frequency domain using FFT,
    then processes it through CNN layers with residual connections.
    
    Args:
        hidden_dim (int): Hidden dimension size for intermediate layers
        feature_dim (int): Output feature dimension
    """
    def __init__(self, hidden_dim, feature_dim):
        super(Frequency_CNN, self).__init__()
        # Define CNN layers
        # Input: [batch_size, 2, 128]
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)  # Output: [batch_size, 32, 128]
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # Output: [batch_size, 64, 64] after pool
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)  # Output: [batch_size, hidden_dim, 32] after pool
        
        # Projection layers for residual connections to match dimensions
        self.proj1 = nn.Conv1d(2, 32, kernel_size=1)
        self.proj2 = nn.Conv1d(32, 64, kernel_size=1) 
        self.proj3 = nn.Conv1d(64, hidden_dim, kernel_size=1)
        
        # Pooling and activation
        self.pool = nn.MaxPool1d(2)  # Reduces sequence length by factor of 2
        self.relu = nn.ReLU()

        # Final projection to feature_dim
        self.final_proj = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, x):
        # Input shape: [batch_size, 2, 128]
        # Convert IQ data to frequency domain
        x_complex = torch.complex(x[:, 0], x[:, 1])  # Combine I/Q into complex
        x_freq = torch.fft.fft(x_complex, dim=1)  # Apply FFT
        x = torch.stack([x_freq.real, x_freq.imag], dim=1)  # Back to [batch_size, 2, 128]
        
        # First residual block
        identity = self.proj1(x)
        x = self.relu(self.conv1(x))  # [batch_size, 32, 128]
        x = x + identity
        x = self.pool(x)  # [batch_size, 32, 64]
        
        # Second residual block
        identity = self.proj2(x)
        x = self.relu(self.conv2(x))  # [batch_size, 64, 64]
        x = x + identity
        x = self.pool(x)  # [batch_size, 64, 32]
        
        # Third residual block
        identity = self.proj3(x)
        x = self.relu(self.conv3(x))  # [batch_size, hidden_dim, 32]
        x = x + identity
        x = self.pool(x)  # [batch_size, hidden_dim, 16]
        
        # Global average pooling - average across sequence dimension
        x = torch.mean(x, dim=2)  # [batch_size, hidden_dim]

        # Project to feature_dim dimensions
        x = self.final_proj(x)  # [batch_size, feature_dim]

        return x
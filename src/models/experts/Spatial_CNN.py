import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

class Spatial_CNN(nn.Module):
    """
    Spatial CNN Expert Model
    
    This class implements a CNN architecture specialized for spatial feature extraction.
    It serves as one of the expert models in the mixture of experts system.
    
    Args:
        cfg (DictConfig): Configuration object containing model parameters
    """
    def __init__(self, cfg: DictConfig):
        super(Spatial_CNN, self).__init__()
        # Define CNN layers
        # Input: [batch_size, 2, 128]
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)  # Output: [batch_size, 32, 128]
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # Output: [batch_size, 64, 64] after pool
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # Output: [batch_size, 128, 32] after pool
        
        # Pooling and activation
        self.pool = nn.MaxPool1d(2)  # Reduces sequence length by factor of 2
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input shape: [batch_size, 2, 128]
        x = self.relu(self.conv1(x))  # [batch_size, 32, 128]
        x = self.pool(x)  # [batch_size, 32, 64]
        
        x = self.relu(self.conv2(x))  # [batch_size, 64, 64]
        x = self.pool(x)  # [batch_size, 64, 32]
        
        x = self.relu(self.conv3(x))  # [batch_size, 128, 32]
        x = self.pool(x)  # [batch_size, 128, 16]
        
        # Global average pooling - average across sequence dimension
        x = torch.mean(x, dim=2)  # [batch_size, 128]
        return x
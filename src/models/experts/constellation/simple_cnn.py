import torch
import torch.nn as nn
import numpy as np

class Constellation_CNN(nn.Module):
    """
    CNN Expert Model for Constellation Diagram Feature Extraction
    
    Takes I/Q data as input and processes it through a CNN to extract features.
    
    Args:
        input_size (int): Number of input channels (2 for I/Q data)
        hidden_dim (int): Hidden dimension size for the classifier
        feature_dim (int): Output feature dimension
        grid_size (int): Size of the grid to project constellation points onto
    """
    def __init__(self, input_size=2, hidden_dim=256, feature_dim=128, grid_size=32):
        super(Constellation_CNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        
        # CNN layers
        self.features = nn.Sequential(
            # First conv block with 3x3 kernel
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # Second conv block with 3x3 kernel
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # Third conv block with 5x5 kernel
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier remains the same
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, feature_dim),
        )
    
    def iq_to_constellation(self, iq_data):
        """
        Convert I/Q data to a constellation diagram image
        
        Args:
            iq_data (torch.Tensor): [batch_size, 2, n_samples] I/Q data
            
        Returns:
            torch.Tensor: [batch_size, 1, grid_size, grid_size] constellation image
        """
        batch_size = iq_data.shape[0]
        device = iq_data.device
        
        # Initialize output tensor
        constellation_images = torch.zeros((batch_size, 1, self.grid_size, self.grid_size),
                                        device=device)
        
        # Scale factor to map I/Q values to grid indices
        scale = (self.grid_size // 2) / 2.0  # Assuming I/Q values are roughly in [-2, 2]
        offset = self.grid_size // 2  # Center offset
        
        for b in range(batch_size):
            # Get I/Q values
            i_data = iq_data[b, 0]
            q_data = iq_data[b, 1]
            
            # Scale and offset I/Q values to grid indices
            i_indices = (i_data * scale + offset).long().clamp(0, self.grid_size - 1)
            q_indices = (q_data * scale + offset).long().clamp(0, self.grid_size - 1)
            
            # Place points in the grid
            constellation_images[b, 0, q_indices, i_indices] = 1.0
            
        return constellation_images
        
    def forward(self, x):
        # Input shape: [batch_size, 2, n_samples]
        # Convert I/Q to constellation image
        x = self.iq_to_constellation(x)  # Shape: [batch_size, 1, grid_size, grid_size]
        
        # Process through CNN
        x = self.features(x)
        x = self.adaptive_pool(x)
        features = self.classifier(x)
        
        return features  # Shape: [batch_size, feature_dim]
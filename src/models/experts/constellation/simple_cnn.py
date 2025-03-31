import torch
import torch.nn as nn
import numpy as np

class Constellation_CNN(nn.Module):
    """
    CNN Expert Model for Constellation Diagram Feature Extraction
    
    Takes I/Q data as input, converts it to a density-windowed constellation diagram,
    then processes it through a CNN to extract features.
    
    Args:
        input_size (int): Number of input channels (2 for I/Q data)
        hidden_dim (int): Hidden dimension size for the classifier
        feature_dim (int): Output feature dimension
        window_size (int): Size of the density window grid
        density_thresholds (tuple): Thresholds for (low-medium, medium-high) density
    """
    def __init__(self, input_size=2, hidden_dim=256, feature_dim=128, 
                 window_size=32, density_thresholds=(5, 15)):
        super(Constellation_CNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.density_thresholds = density_thresholds
        
        # CNN layers with reduced channels
        self.features = nn.Sequential(
            # First conv block with 3x3 kernel
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, feature_dim),
        )
    
    def iq_to_density_window(self, iq_data):
        """
        Convert I/Q data to density-windowed constellation diagram
        
        Args:
            iq_data (torch.Tensor): [batch_size, 2, 128] I/Q data
            
        Returns:
            torch.Tensor: [batch_size, 3, window_size, window_size] density windows
        """
        batch_size = iq_data.shape[0]
        device = iq_data.device
        
        # Move to CPU for numpy processing
        iq_data_cpu = iq_data.cpu().numpy()
        
        # Initialize output tensor
        density_windows = torch.zeros((batch_size, 3, self.window_size, self.window_size),
                                    device=device)
        
        for b in range(batch_size):
            # Extract I and Q components
            i_data = iq_data_cpu[b, 0]
            q_data = iq_data_cpu[b, 1]
            
            # Create 2D histogram
            H, _, _ = np.histogram2d(
                i_data, q_data,
                bins=[self.window_size, self.window_size],
                range=[[-2, 2], [-2, 2]]
            )
            
            # Create density channels (blue: low, green: medium, yellow: high)
            low_th, high_th = self.density_thresholds
            low_density = (H <= low_th).astype(np.float32)
            med_density = ((H > low_th) & (H <= high_th)).astype(np.float32)
            high_density = (H > high_th).astype(np.float32)
            
            # Stack channels and convert to tensor
            density_window = torch.from_numpy(
                np.stack([high_density, med_density, low_density])
            ).to(device)
            
            density_windows[b] = density_window
            
        return density_windows
        
    def forward(self, x):
        # Input shape: [batch_size, 2, 128]
        # Convert I/Q to density window
        x = self.iq_to_density_window(x)  # Shape: [batch_size, 3, window_size, window_size]
        
        # Process through CNN
        x = self.features(x)
        x = self.adaptive_pool(x)
        features = self.classifier(x)
        
        return features  # Shape: [batch_size, feature_dim]
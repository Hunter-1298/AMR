import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Decoder network that maps latent representation back to amplitude or phase signal
    
    Args:
        latent_dim (int): Size of input latent dimension
        output_dim (int): Size of output dimension
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # First conv block
        intermediate_dim = (latent_dim + output_dim) // 2
        self.conv1 = nn.ConvTranspose1d(latent_dim, intermediate_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(intermediate_dim)
        
        # Second conv block
        self.conv2 = nn.ConvTranspose1d(intermediate_dim, intermediate_dim, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(intermediate_dim)
        
        # Final conv to output dimension
        self.conv3 = nn.ConvTranspose1d(intermediate_dim, output_dim, kernel_size=3, padding=1)

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
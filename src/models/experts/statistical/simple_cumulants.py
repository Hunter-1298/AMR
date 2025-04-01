import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

class Cumulants(nn.Module):
    """
    Statistical Expert Model using Signal Cumulants
    
    This class implements a statistical feature extractor that computes higher-order
    cumulants of the input IQ data up to the 8th order. These statistical features
    are then processed through an MLP to create a feature vector.
    
    Input shape: [batch_size, 2, 128] (I and Q channels)
    Output shape: [batch_size, feature_dim] (feature vector)
    """
    def __init__(self, hidden_dim, feature_dim):
        super(Cumulants, self).__init__()
        
        # Number of cumulant features per channel
        # 1st-8th order cumulants = 8 features per channel
        self.num_cumulant_features = 8 * 2  # 8 orders * 2 channels
        
        # MLP to process cumulant features
        self.mlp = nn.Sequential(
            nn.Linear(self.num_cumulant_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
    def compute_cumulants(self, x):
        """Compute cumulants up to 8th order for each channel"""
        # x shape: [batch_size, 2, 128]
        batch_size = x.shape[0]
        cumulants = []
        
        for channel in range(2):  # For both I and Q channels
            signal = x[:, channel, :]  # [batch_size, 128]
            
            # 1st order - mean
            c1 = torch.mean(signal, dim=1)
            
            # Center the signal
            centered = signal - c1.unsqueeze(1)
            
            # 2nd order - variance
            c2 = torch.mean(centered**2, dim=1)
            
            # 3rd order - skewness
            c3 = torch.mean(centered**3, dim=1)
            
            # 4th order - kurtosis
            c4 = torch.mean(centered**4, dim=1) - 3 * c2**2
            
            # 5th order
            c5 = torch.mean(centered**5, dim=1)
            
            # 6th order
            c6 = torch.mean(centered**6, dim=1)
            
            # 7th order
            c7 = torch.mean(centered**7, dim=1)
            
            # 8th order
            c8 = torch.mean(centered**8, dim=1)
            
            # Stack all cumulants for this channel
            channel_cumulants = torch.stack([c1, c2, c3, c4, c5, c6, c7, c8], dim=1)
            cumulants.append(channel_cumulants)
        
        # Combine I and Q cumulants
        # Shape: [batch_size, 16] (8 cumulants * 2 channels)
        return torch.cat(cumulants, dim=1)
    
    def forward(self, x):
        # Compute statistical features
        cumulant_features = self.compute_cumulants(x)
        
        # Normalize the cumulants for stability
        cumulant_features = torch.log1p(torch.abs(cumulant_features)) * torch.sign(cumulant_features)
        
        # Project to final feature space
        features = self.mlp(cumulant_features)
        
        return features
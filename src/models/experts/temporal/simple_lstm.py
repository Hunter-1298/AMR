import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

class Temporal_LSTM(nn.Module):
    """
    Temporal LSTM Expert Model
    
    This class implements an LSTM architecture specialized for temporal feature extraction.
    It serves as one of the expert models in the mixture of experts system.
    
    Args:
        input_size (int): Number of input channels (2 for I/Q data)
        hidden_dim (int): Hidden dimension size for LSTM
        num_layers (int): Number of LSTM layers
    """
    def __init__(self, input_size, hidden_dim, num_layers):
        super(Temporal_LSTM, self).__init__()
        
        # Input dimensions
        self.input_size = input_size  # Number of channels
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        # Note: Input shape needs to be [batch_size, seq_len, input_size]
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # Projection layer for residual connection
        self.proj = nn.Linear(self.input_size, self.hidden_size)
        
        # Final projection to 128 dimensions
        self.final_proj = nn.Linear(self.hidden_size, 128)
        
    def forward(self, x):
        # Input shape: [batch_size, 2, 128]
        # LSTM expects [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)  # Transform to [batch_size, 128, 2]
        
        # Create residual connection
        # Project input to match hidden size dimension
        residual = self.proj(x)  # [batch_size, 128, hidden_size]
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Add residual connection to LSTM output
        lstm_out = lstm_out + residual
        
        # Take mean across sequence dimension
        output = torch.mean(lstm_out, dim=1)  # [batch_size, hidden_size]
        
        # Project to 128 dimensions
        output = self.final_proj(output)  # [batch_size, 128]
        
        return output
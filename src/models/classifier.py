import torch.nn as nn
from omegaconf import DictConfig

class Classifier(nn.Module):
    """
    Classification head for the Mixture of Experts model.
    Takes the combined expert outputs and produces final classification predictions.
    
    Args:
        cfg (DictConfig): Configuration object containing:
            - input_dim (int): Input dimension from combined expert outputs
            - hidden_dim (int): Hidden layer dimension 
            - num_classes (int): Number of output classes
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)
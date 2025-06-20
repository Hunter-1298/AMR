import torch
import torch.nn as nn

class Conv1DHead(nn.Module):
    def __init__(self, input_channels=32, input_length=8, hidden_dim=128, num_classes=11, dropout=0.2):
        super().__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Convolutional layers to process the latent representation
        self.conv_net = nn.Sequential(
            # First conv block - reduce channels while maintaining length
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Second conv block - further processing
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Third conv block - prepare for global pooling
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )



    def forward(self, x):
        # Input shape: [batch, 32, 8]
        batch_size = x.shape[0]

        # Apply convolutional layers
        x = self.conv_net(x)  # [batch, hidden_dim*2, 8]

        # Global average pooling
        x = self.global_avg_pool(x)  # [batch, hidden_dim*2, 1]
        x = x.squeeze(-1)  # [batch, hidden_dim*2]

        # Classification
        logits = self.classifier(x)  # [batch, num_classes]

        return logits

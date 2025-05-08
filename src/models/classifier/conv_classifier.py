import torch
import torch.nn as nn

class Conv1DHead(nn.Module):
    def __init__(self, num_classes, in_channels=32, hidden_dim=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            # First conv block
            nn.Conv1d(in_channels, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),

            # Second conv block
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            # Third conv block
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim*2),
            nn.LeakyReLU(0.2),

            # Global pooling
            nn.AdaptiveAvgPool1d(1)  # → (B,hidden_dim*2,1)
        )

        # MLP head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 32, 64)
        x = self.conv_net(x)      # → (B,hidden_dim*2,1)
        return self.classifier(x)  # → (B,num_classes)

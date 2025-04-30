import torch
import torch.nn as nn

class Conv1DHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),      # → length 32
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # → (B,128,1)
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        # x: (B, 32, 64)
        x = self.conv_net(x)      # → (B,128,1)
        x = x.squeeze(-1)         # → (B,128)
        return self.classifier(x)

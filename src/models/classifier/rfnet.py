import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RFNet(nn.Module):
    """
    RFNet: Fast and efficient neural network for modulation classification
    Based on the original ITU Journal paper implementation

    Pure torch.nn.Module for integration with diffusion models
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 2,  # I/Q channels
        input_length: int = 128,
        dropout_rate: float = 0.2
    ):
        super(RFNet, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_length = input_length

        # First convolutional block
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second convolutional block
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Third convolutional block
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Fourth convolutional block
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(2)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Calculate flattened size after convolutions
        self._calculate_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc_out = nn.Linear(512, num_classes)

        # Initialize weights
        # self._initialize_weights()

    def _calculate_conv_output_size(self):
        """Calculate the size after all conv/pool operations"""
        # Simulate forward pass to get output size
        x = torch.randn(1, self.input_channels, self.input_length)

        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Conv block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        self.flattened_size = x.numel()

    # def _initialize_weights(self):
    #     """Initialize network weights using Xavier/He initialization"""
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through RFNet

        Args:
            x: Input tensor of shape [batch_size, 2, input_length] (I/Q channels)
            return_features: If True, return (logits, features) tuple

        Returns:
            logits: Class predictions [batch_size, num_classes]
            features: Feature vector [batch_size, 512] (if return_features=True)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # First FC layer
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)

        # Second FC layer (feature layer)
        features = self.fc2(x)
        features = self.bn_fc2(features)
        features = F.relu(features)
        features = self.dropout_fc2(features)

        # Output layer
        logits = self.fc_out(features)

        if return_features:
            return logits, features
        else:
            return logits


    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations without classification"""
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
        return features

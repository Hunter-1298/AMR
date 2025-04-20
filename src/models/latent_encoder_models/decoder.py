import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.upsample = None
        if stride != 1 or in_channels != out_channels:
            self.upsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.upsample is not None:
            identity = self.upsample(identity)
            
        out += identity
        out = self.relu(out)
        
        return out


class Decoder1D(nn.Module):
    def __init__(
        self,
        in_channels=32,  # Matches encoder's output channels
        out_channels=2,  # Original signal channels
        base_channels=256,
        num_blocks=[2, 2, 2],
    ):
        super(Decoder1D, self).__init__()
        
        # Initial projection from latent space
        self.initial_conv = nn.Conv1d(
            in_channels, base_channels // 8, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels // 8)
        self.relu = nn.ReLU(inplace=True)
        
        # Upsampling layers with residual blocks
        self.layer1 = self._make_layer(
            base_channels // 8, base_channels // 4, num_blocks[0]
        )
        self.upsample1 = nn.ConvTranspose1d(
            base_channels // 4, base_channels // 4,
            kernel_size=4, stride=2, padding=1, output_padding=0
        )
        
        self.layer2 = self._make_layer(
            base_channels // 4, base_channels // 2, num_blocks[1]
        )
        
        self.layer3 = self._make_layer(
            base_channels // 2, base_channels, num_blocks[2]
        )
        
        # Final convolution to get back to original number of channels
        self.final_conv = nn.Conv1d(
            base_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [BasicBlock1D(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial projection and processing
        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Upsampling path
        x = self.layer1(x)
        x = self.upsample1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        # Final convolution to get original channels
        x = self.final_conv(x)
        
        return x

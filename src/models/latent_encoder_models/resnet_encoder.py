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
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(
        self, in_channels=2, num_blocks=[2, 2, 2], base_channels=64, out_dim=128
    ):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks[0])
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, num_blocks[2], stride=2
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 4, out_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [BasicBlock1D(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x).squeeze(-1)
        z = self.fc(x)
        return z


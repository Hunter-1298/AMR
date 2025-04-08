import torch
import torch.nn as nn


class Decoder1D(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        output_channels=2,
        initial_channels=256,
        output_length=128,
    ):
        super(Decoder1D, self).__init__()

        # Project latent vector to a shape suitable for upsampling
        self.init_time = (
            output_length // 8
        )  # adjust depending on how many upsampling layers
        self.fc = nn.Linear(latent_dim, initial_channels * self.init_time)

        self.up_blocks = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose1d(
                    initial_channels,
                    initial_channels // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm1d(initial_channels // 2),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(
                    initial_channels // 2,
                    initial_channels // 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm1d(initial_channels // 4),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(
                    initial_channels // 4,
                    initial_channels // 8,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm1d(initial_channels // 8),
                nn.ReLU(),
            ),
        )

        self.output_conv = nn.Conv1d(
            initial_channels // 8, output_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), -1, self.init_time)  # [B, C, T]
        x = self.up_blocks(x)
        return self.output_conv(x)

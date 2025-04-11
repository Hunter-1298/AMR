import torch.nn as nn


class Decoder1D(nn.Module):
    def __init__(
        self,
        initial_channels=32,
        latent_dim=64,
        output_channels=2,
        output_length=128,
    ):
        super(Decoder1D, self).__init__()
        # takes in [batch_size, 32, 64]

        # Project latent vector to a shape suitable for upsampling
        self.init_time = output_length // 8

        # we can do this so we can be invariant on the encoding shape
        # adjust depending on how many upsampling layers
        # projects to a 512 dim  space
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

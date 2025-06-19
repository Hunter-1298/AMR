import torch
import torch.nn as nn
import torch.nn.functional as F


class MSC(nn.Module):
    """
    Multiscale Convolutional (MSC) layer: parallel depthwise convs with different kernel sizes,
    followed by pointwise convs to mix channels.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_sizes=(3, 5, 7)):
        super(MSC, self).__init__()
        assert out_ch % len(kernel_sizes) == 0, \
            "out_ch must be divisible by number of kernel sizes"
        branch_ch = out_ch // len(kernel_sizes)
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=padding, groups=in_ch, bias=False),
                    nn.Conv1d(in_ch, branch_ch, kernel_size=1, bias=False)
                )
            )
        self.bn = nn.BatchNorm1d(out_ch)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SCB(nn.Module):
    """
    Separable Convolution Block: depthwise conv -> pointwise conv -> BN -> ReLU
    (pooling removed to avoid zero-size issues); retain only dropout if desired.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dropout_rate: float = 0.0):
        super(SCB, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.dropout(x)
        return x


class RFNet(nn.Module):
    """
    Adapted RFNet for latent signals of shape (batch, 32, 8).
    Uses MSC + SCB layers (no pooling), global pooling, and minimal FC.

    Default in_channels=32, base_ch=96 (divisible by 3 for MSC), num_scb=3.
    """
    def __init__(
        self,
        num_classes: int = 11,
        in_channels: int = 32,
        base_ch: int = 96,  # divisible by 3 for MSC
        num_scb: int = 3,
        kernel_sizes_msc=(3, 5, 7),
        dropout_rate: float = 0.2
    ):
        super(RFNet, self).__init__()

        # MSC layer
        self.msc = MSC(in_ch=in_channels, out_ch=base_ch, kernel_sizes=kernel_sizes_msc)

        # build separable conv blocks without pooling
        scb_layers = []
        ch = base_ch
        for i in range(num_scb):
            nxt = ch * 2 if i > 0 else ch
            scb_layers.append(
                SCB(in_ch=ch, out_ch=nxt, kernel_size=3,
                    dropout_rate=dropout_rate)
            )
            ch = nxt
        self.scb_stack = nn.Sequential(*scb_layers)

        # Global pooling converts temporal dimension to 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final classifier
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # x: (batch, 32, 8)
        x = self.msc(x)                # -> (batch, base_ch, 8)
        x = self.scb_stack(x)         # -> (batch, ch, 8)
        x = self.global_pool(x)       # -> (batch, ch, 1)
        features = x.view(x.size(0), -1)  # -> (batch, ch)
        logits = self.fc(features)    # -> (batch, num_classes)
        if return_features:
            return logits, features
        return logits

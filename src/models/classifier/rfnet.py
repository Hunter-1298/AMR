import torch
import torch.nn as nn
import torch.nn.functional as F


class MSC(nn.Module):
    """
    Multiscale Convolutional (MSC) layer with groupnorm and ReLU.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_sizes=(3, 5, 7), num_groups=8):
        super().__init__()
        assert out_ch % len(kernel_sizes) == 0, (
            "out_ch must be divisible by number of kernel sizes"
        )
        branch_ch = out_ch // len(kernel_sizes)
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch,
                        in_ch,
                        kernel_size=k,
                        padding=padding,
                        groups=in_ch,
                        bias=False,
                    ),
                    nn.Conv1d(in_ch, branch_ch, kernel_size=1, bias=False),
                )
            )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SCB(nn.Module):
    """
    Separable Convolution Block with residual connection.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, dropout_rate=0.0, num_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.res_proj = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.res_proj(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.relu(x + residual)


class TemporalAttention(nn.Module):
    """
    Temporal attention layer to replace global pooling.
    """

    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.attn(x)  # (B, 1, T)
        return (x * weights).sum(dim=-1)  # (B, C)


class RFNet(nn.Module):
    """
    Refactored RFNet for RF signals of shape (batch, 2, 128).
    """

    def __init__(
        self,
        num_classes=11,
        in_channels=2,
        base_ch=96,
        num_scb=3,
        kernel_sizes_msc=(3, 5, 7),
        dropout_rate=0.2,
        use_positional_encoding=True,
        num_groups=8,
    ):
        super().__init__()

        # Optional learnable positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_enc = nn.Parameter(torch.randn(1, in_channels,128 ))

        # Initial MSC layer
        self.msc = MSC(
            in_ch=in_channels,
            out_ch=base_ch,
            kernel_sizes=kernel_sizes_msc,
            num_groups=num_groups,
        )

        # SCB stack
        scb_layers = []
        ch = base_ch
        for i in range(num_scb):
            out_ch = ch * 2 if i > 0 else ch
            scb_layers.append(
                SCB(
                    in_ch=ch,
                    out_ch=out_ch,
                    kernel_size=3,
                    dropout_rate=dropout_rate,
                    num_groups=num_groups,
                )
            )
            ch = out_ch
        self.scb_stack = nn.Sequential(*scb_layers)

        # Temporal Attention for global representation
        self.global_pool = TemporalAttention(ch)

        # Classifier head
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x, return_features=False):
        # Optional positional encoding
        if self.use_positional_encoding:
            x = x + self.pos_enc

        x = self.msc(x)  # -> (B, base_ch, 128)
        x = self.scb_stack(x)  # -> (B, ch, 128)
        features = self.global_pool(x)  # -> (B, ch)
        logits = self.fc(features)  # -> (B, num_classes)

        if return_features:
            return logits, features
        return logits

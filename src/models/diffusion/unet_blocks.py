# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [
        -0.01171875,
        -0.03515625,
        0.11328125,
        0.43359375,
        0.43359375,
        0.11328125,
        -0.03515625,
        -0.01171875,
    ],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}


def get_down_block(
    down_block_type: str,
    in_channels: int,
    out_channels: int,
    context_dim: int,
):
    if down_block_type == "DownBlock1D":
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels)
    elif down_block_type == "AttnDownBlock1D":
        return AttnDownBlock1D(
            out_channels=out_channels, in_channels=in_channels, context_dim=context_dim
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_mid_block(
    mid_block_type: str,
    in_channels: int,
    mid_channels: int,
    out_channels: int,
    context_dim: int,
):
    if mid_block_type == "UNetMidBlock1D":
        return UNetMidBlock1D(
            in_channels=in_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
            context_dim=context_dim
        )
    raise ValueError(f"{mid_block_type} does not exist.")


def get_up_block(
    up_block_type: str,
    in_channels: int,
    out_channels: int,
    context_dim: int,
):
    if up_block_type == "UpBlock1D":
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels)
    elif up_block_type == "AttnUpBlock1D":
        return AttnUpBlock1D(in_channels=in_channels, out_channels=out_channels, context_dim=context_dim)
    raise ValueError(f"{up_block_type} does not exist.")


#################### GLOBAL CLASSES for UP and DOWN Sampling #########################


class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.has_conv_skip = in_channels != out_channels

        if self.has_conv_skip:
            self.conv_skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.conv_1 = nn.Conv1d(in_channels, mid_channels, 5, padding=2)
        self.group_norm_1 = nn.GroupNorm(1, mid_channels)
        self.gelu_1 = nn.GELU()
        self.conv_2 = nn.Conv1d(mid_channels, out_channels, 5, padding=2)

        if not self.is_last:
            self.group_norm_2 = nn.GroupNorm(1, out_channels)
            self.gelu_2 = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = (
            self.conv_skip(hidden_states) if self.has_conv_skip else hidden_states
        )

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.group_norm_1(hidden_states)
        hidden_states = self.gelu_1(hidden_states)
        hidden_states = self.conv_2(hidden_states)

        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)

        output = hidden_states + residual
        return output


class SelfAttention1d(nn.Module):
    def __init__(self, in_channels: int, n_head: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        assert in_channels % n_head == 0, "in_channels must be divisible by n_head"

        self.in_channels = in_channels
        self.num_heads = n_head
        self.head_dim = in_channels // n_head

        # normalize over feature channels
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        # linear projections for Q, K, V over feature dim
        self.query = nn.Linear(in_channels, in_channels)
        self.key   = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

        # output projection
        self.out_proj = nn.Linear(in_channels, in_channels)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T] where
            B = batch size,
            C = in_channels (feature dimension),
            T = sequence length / spatial positions
        """
        residual = x
        B, C, T = x.size()

        # 1. Normalize features per channel
        x = self.norm(x)

        # 2. Prepare for attention: tokens = positions (T)
        #    shape -> [B, T, C]
        x = x.permute(0, 2, 1)

        # 3. Linear projections
        q = self.query(x)  # [B, T, C]
        k = self.key(x)
        v = self.value(x)

        # 4. Split into heads -> [B, T, H, D] then -> [B, H, T, D]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 5. Scaled dot-product attention over positions
        scale = self.head_dim ** -0.5
        attn_logits = (q * scale) @ (k * scale).transpose(-1, -2)  # [B, H, T, T]
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # 6. Attend to values -> [B, H, T, D]
        out = attn @ v

        # 7. Merge heads -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Final linear projection -> [B, T, C]
        out = self.out_proj(out)

        # 9. Back to original shape -> [B, C, T]
        out = out.permute(0, 2, 1)
        out = self.dropout(out)

        # 10. Residual connection
        return out + residual


class CrossAttention1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        context_dim: int,
        n_head: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        assert in_channels % n_head == 0, "in_channels must be divisible by n_head"

        self.in_channels = in_channels
        self.context_dim = context_dim
        self.num_heads = n_head
        self.head_dim = in_channels // n_head

        # normalize over feature channels
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        # query from hidden_states: use Conv1d on feature channels
        self.query = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        # key/value from context: project context_dim to in_channels
        self.key = nn.Linear(context_dim, in_channels)
        self.value = nn.Linear(context_dim, in_channels)

        # output projection back to in_channels
        self.out_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, C, T]
        context: [B, context_dim]
        Returns: [B, C, T]
        """
        residual = hidden_states
        B, C, T = hidden_states.size()

        # 1. Normalize features per channel
        x = self.norm(hidden_states)

        # 2. Compute queries: Conv1d over feature channels
        # hidden_states: [B, C, T] -> q: [B, C, T]
        q = self.query(x)
        # prepare for heads: [B, C, T] -> [B, T, C]
        q = q.permute(0, 2, 1)

        # 3. Prepare key/value from context
        # ensure context has shape [B, S, context_dim]
        B_ctx, ctx_dim = context.size()
        # 3. If context is [B, context_dim], make it [B, 1, context_dim]
        if context.dim() == 2:
            context = context.unsqueeze(1)  # [B, 1, context_dim]
        # Now context: [B, S=1, context_dim]
        # project to feature dimension: [B, S, C]
        k = self.key(context)
        v = self.value(context)

        S = context.size(1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]

        # 5. Scaled dot-product attention: [B, H, T, S]
        scale = self.head_dim ** -0.5
        attn_logits = (q * scale) @ (k * scale).transpose(-1, -2)
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # 6. Weighted sum: [B, H, T, D]
        out = attn @ v

        # 7. Merge heads -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Project back to channels and restore shape [B, C, T]
        out = out.permute(0, 2, 1)
        out = self.out_proj(out)
        out = self.dropout(out)

        # 9. Residual connection
        return out + residual


#######################################################################################


############### DOWN BLOCK CLASSES ##########################


class Downsample1d(nn.Module):
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (self.pad,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros(
            [hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]]
        )
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        weight[indices, indices] = kernel
        return F.conv1d(hidden_states, weight, stride=2)


class DownBlock1D(nn.Module):
    def __init__(
        self, out_channels: int, in_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.down(hidden_states)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        return hidden_states


class AttnDownBlock1D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        context_dim: int,
        mid_channels: Optional[int] = None,
        num_heads: int = 8,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        # Self-attention layers for 1D sequences
        self_attentions = [
            SelfAttention1d(mid_channels, n_head=num_heads),
            SelfAttention1d(mid_channels, n_head=num_heads),
            SelfAttention1d(out_channels, n_head=num_heads),
        ]

        # Cross-attention layers for 1D sequences
        cross_attentions = [
            CrossAttention1d(mid_channels, context_dim, n_head=num_heads),
            CrossAttention1d(mid_channels, context_dim, n_head=num_heads),
            CrossAttention1d(out_channels, context_dim, n_head=num_heads),
        ]

        self.self_attentions = nn.ModuleList(self_attentions)
        self.cross_attentions = nn.ModuleList(cross_attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.down(hidden_states)
        
        for resnet, self_attn, cross_attn in zip(
            self.resnets, self.self_attentions, self.cross_attentions
        ):
            hidden_states = resnet(hidden_states)
            hidden_states = self_attn(hidden_states)
            if temb is not None:
                hidden_states = cross_attn(hidden_states, temb)
            
        return hidden_states


######################## MIDDLE Bottleneck CLASES ########################
class UNetMidBlock1D(nn.Module):
    def __init__(
        self,
        mid_channels: int,
        in_channels: int,
        context_dim: int,  # Add context dimension parameter
        out_channels: Optional[int] = None,
        num_heads: int = 8,  # Add number of heads parameter
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        self.down = Downsample1d("cubic")
        self.up = Upsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        # Self attention layers
        self_attentions = [
            SelfAttention1d(mid_channels, n_head=num_heads),
            SelfAttention1d(mid_channels, n_head=num_heads),
            SelfAttention1d(mid_channels, n_head=num_heads),
            SelfAttention1d(mid_channels, n_head=num_heads),
            SelfAttention1d(mid_channels, n_head=num_heads),
            SelfAttention1d(out_channels, n_head=num_heads),
        ]

        # Add cross attention layers
        cross_attentions = [
            CrossAttention1d(mid_channels, context_dim, n_head=num_heads),
            CrossAttention1d(mid_channels, context_dim, n_head=num_heads),
            CrossAttention1d(mid_channels, context_dim, n_head=num_heads),
            CrossAttention1d(mid_channels, context_dim, n_head=num_heads),
            CrossAttention1d(mid_channels, context_dim, n_head=num_heads),
            CrossAttention1d(out_channels, context_dim, n_head=num_heads),
        ]

        self.attentions = nn.ModuleList(self_attentions)
        self.cross_attentions = nn.ModuleList(
            cross_attentions
        )  # Add cross attention module list
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    # [batch_size, channels, 8dim]
        hidden_states = self.down(hidden_states)
    # [batch_size, channels, 4dim]

        # Update forward pass to include both self and cross attention
        for resnet, self_attn, cross_attn in zip(
            self.resnets, self.attentions, self.cross_attentions
        ):
            hidden_states = resnet(hidden_states)
            hidden_states = self_attn(hidden_states)
            if temb is not None:
                hidden_states = cross_attn(hidden_states, temb)

        # upsample by 2
        hidden_states = self.up(hidden_states)

        return hidden_states


###################################################################################


######################## UP CLASSES ########################


class Upsample1d(nn.Module):
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros(
            [hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]]
        )
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        weight[indices, indices] = kernel
        return F.conv_transpose1d(
            hidden_states, weight, stride=2, padding=self.pad * 2 + 1
        )


class UpBlock1D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_state: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = torch.cat([hidden_states, res_hidden_state], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class AttnUpBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_dim: int,
        num_heads: int = 8,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        # We concatentate our reisudal connections, so 2x the input channels
        resnets = [
            ResConvBlock(2* in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        self_attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        # Cross-attention layers for 1D sequences
        cross_attentions = [
            CrossAttention1d(mid_channels, context_dim, n_head=num_heads),
            CrossAttention1d(mid_channels, context_dim, n_head=num_heads),
            CrossAttention1d(out_channels, context_dim, n_head=num_heads),
        ]

        self.self_attentions = nn.ModuleList(self_attentions)
        self.cross_attentions = nn.ModuleList(cross_attentions)
        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_state: torch.Tensor,
        temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # concatenate the residual connection 
        hidden_states = torch.cat([hidden_states, res_hidden_state], dim=1)
        
        for resnet, self_attn, cross_attn in zip(
            self.resnets, self.self_attentions, self.cross_attentions
        ):
            hidden_states = resnet(hidden_states)
            hidden_states = self_attn(hidden_states)
            if temb is not None:
                hidden_states = cross_attn(hidden_states, temb)
                
        # Upsample for next iteration
        return self.up(hidden_states)
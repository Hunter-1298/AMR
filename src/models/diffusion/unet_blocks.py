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
        self.channels = in_channels
        self.group_norm = nn.GroupNorm(1, num_channels=in_channels)
        self.num_heads = n_head

        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        self.value = nn.Linear(self.channels, self.channels)

        self.proj_attn = nn.Linear(self.channels, self.channels, bias=True)

        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch, channel_dim, seq = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        scale = 1 / math.sqrt(math.sqrt(key_states.shape[-1]))

        attention_scores = torch.matmul(
            query_states * scale, key_states.transpose(-1, -2) * scale
        )
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # compute attention output
        hidden_states = torch.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual

        return output


class CrossAttention1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        context_dim: int,
        n_head: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.channels = in_channels
        self.context_dim = context_dim
        self.num_heads = n_head

        self.group_norm = nn.GroupNorm(1, num_channels=in_channels)

        # For 1D sequences, we'll use Conv1d instead of Linear for query
        self.query = nn.Conv1d(in_channels, in_channels, 1)
        # Key and Value still use Linear as they operate on the context
        self.key = nn.Linear(self.context_dim, self.channels)
        self.value = nn.Linear(self.context_dim, self.channels)

        self.proj_attn = nn.Conv1d(in_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # For 1D data: (B, C, T) -> (B, H, T, D)
        if projection.dim() == 3:
            B, C, T = projection.shape
            projection = projection.transpose(1, 2)  # (B, T, C)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(
        self, hidden_states: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        batch, channel_dim, seq_len = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)

        # Process query while maintaining 1D structure
        query_states = self.query(hidden_states)  # (B, C, T)
        query_states = self.transpose_for_scores(query_states)  # (B, H, T, D)

        # Process context
        if context.dim() == 2:
            context = context.unsqueeze(1)  # Add sequence dimension if needed

        # Process key and value from context
        key_states = self.transpose_for_scores(self.key(context))  # (B, H, S, D)
        value_states = self.transpose_for_scores(self.value(context))  # (B, H, S, D)

        scale = 1 / math.sqrt(math.sqrt(key_states.shape[-1]))

        # Compute attention scores
        attention_scores = torch.matmul(
            query_states * scale, key_states.transpose(-1, -2) * scale
        )
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Apply attention
        hidden_states = torch.matmul(attention_probs, value_states)  # (B, H, T, D)

        # Reshape back to 1D sequence format
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)  # (B, T, C)
        hidden_states = hidden_states.transpose(1, 2)  # (B, C, T)

        # Final projection
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual
        return output


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
        return hidden_states, (hidden_states,)


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

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        # hidden_states shape: (B, C, T) where T is the sequence length
        hidden_states = self.down(hidden_states)

        for resnet, self_attn, cross_attn in zip(
            self.resnets, self.self_attentions, self.cross_attentions
        ):
            hidden_states = resnet(hidden_states)
            hidden_states = self_attn(hidden_states)
            if temb is not None:
                hidden_states = cross_attn(hidden_states, temb)

        return hidden_states, (hidden_states,)


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
        hidden_states = self.down(hidden_states)

        # Update forward pass to include both self and cross attention
        for resnet, self_attn, cross_attn in zip(
            self.resnets, self.attentions, self.cross_attentions
        ):
            hidden_states = resnet(hidden_states)
            hidden_states = self_attn(hidden_states)
            if temb is not None:
                hidden_states = cross_attn(hidden_states, temb)

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
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

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

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
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

        self.attentions = nn.ModuleList(attentions)
        self.cross_attentions = nn.ModuleList(cross_attentions)
        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        # hidden_states shape: (B, C, T) where T is the sequence length
        hidden_states = self.down(hidden_states)

        for resnet, self_attn, cross_attn in zip(
            self.resnets, self.self_attentions, self.cross_attentions
        ):
            hidden_states = resnet(hidden_states)
            hidden_states = self_attn(hidden_states)
            if temb is not None:
                hidden_states = cross_attn(hidden_states, temb)

        return hidden_states, (hidden_states,)

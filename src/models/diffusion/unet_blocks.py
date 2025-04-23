import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from .activations import get_activation

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
    num_layers: int,
    temb_channels: int,
    add_downsample: bool,
):
    if down_block_type == "DownResnetBlock1D":
        return DownResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    elif down_block_type == "AttnDownBlock1D":
        return AttnDownBlock1D(
            out_channels=out_channels,
            in_channels=in_channels,
            context_dim=context_dim,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_mid_block(
    mid_block_type: str,
    in_channels: int,
    mid_channels: int,
    out_channels: int,
    temb_channels: int,
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
    num_layers: int,
    temb_channels: int,
    add_upsample: bool,
):
    if up_block_type == "UpResnetBlock1D":
        return UpResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
        )
    elif up_block_type == "AttnUpBlock1D":
        return AttnUpBlock1D(in_channels=in_channels,
            out_channels=out_channels,
            context_dim=context_dim,
            temb_channels=temb_channels,
        )
    raise ValueError(f"{up_block_type} does not exist.")

def rearrange_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")

#################### GLOBAL CLASSES for UP and DOWN Sampling #########################

class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        n_groups (`int`, default `8`): Number of groups to separate the channels into.
        activation (`str`, defaults to `mish`): Name of the activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        n_groups: int = 8,
        activation: str = "mish",
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = get_activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        intermediate_repr = self.conv1d(inputs)
        intermediate_repr = rearrange_dims(intermediate_repr)
        intermediate_repr = self.group_norm(intermediate_repr)
        intermediate_repr = rearrange_dims(intermediate_repr)
        output = self.mish(intermediate_repr)
        return output

class ResidualTemporalBlock1D(nn.Module):
    """
    Residual 1D block with temporal convolutions.

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        embed_dim (`int`): Embedding dimension.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        activation (`str`, defaults `mish`): It is possible to choose the right activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        activation: str = "mish",
    ):
        super().__init__()
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)

        self.time_emb_act = get_activation(activation)
        self.time_emb = nn.Linear(embed_dim, out_channels)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        out = self.conv_in(inputs) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(inputs)

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
class Downsample1D(nn.Module):
    """A 1D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.channels
        return self.conv(inputs)



class AttnDownBlock1D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        context_dim: int,
        temb_channels: int,
        mid_channels: Optional[int] = None,
        add_downsample: bool = True,
        num_heads: int = 8,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True, padding=1)
        resnets = [
            ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels),
            ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels),
            ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels)
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
        hidden_states = self.downsample(hidden_states) #pyright: ignore

        for resnet, self_attn, cross_attn in zip(
            self.resnets, self.self_attentions, self.cross_attentions
        ):
            hidden_states = resnet(hidden_states)
            hidden_states = self_attn(hidden_states)
            if temb is not None:
                hidden_states = cross_attn(hidden_states, temb)

        return hidden_states

class DownResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        conv_shortcut: bool = False,
        temb_channels: int = 32,
        non_linearity: Optional[str] = None,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.add_downsample = add_downsample
        self.output_scale_factor = output_scale_factor

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.ModuleList(resnets)

        if non_linearity is None:
            self.nonlinearity = None
        else:
            self.nonlinearity = get_activation(non_linearity)

        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True, padding=1)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        output_states = ()

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        output_states += (hidden_states,)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states, output_states


######################## MIDDLE Bottleneck CLASES ########################
class UNetMidBlock1D(nn.Module):
    def __init__(
        self,
        mid_channels: int,
        in_channels: int,
        temb_channels: int,
        context_dim: int,  # Add context dimension parameter
        out_channels: Optional[int] = None,
        num_heads: int = 8,  # Add number of heads parameter
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        self.downsample = Downsample1D(out_channels, use_conv=True, padding=1)
        self.upsample = Upsample1D(out_channels, use_conv_transpose=True)

        resnets = [
            ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels),
            ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels),
            ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels)
        ]

        # Self attention layers
        self_attentions = [
            SelfAttention1d(mid_channels, n_head=num_heads),
            SelfAttention1d(mid_channels, n_head=num_heads),
            SelfAttention1d(out_channels, n_head=num_heads),
        ]

        # Add cross attention layers
        cross_attentions = [
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
        hidden_states = self.downsample(hidden_states)
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
        hidden_states = self.upsample(hidden_states)

        return hidden_states


###################################################################################


######################## UP CLASSES ########################
class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs

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
        temb_channels: int,
        num_heads: int = 8,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        # We concatentate our reisudal connections, so 2x the input channels
        resnets = [
            ResidualTemporalBlock1D(2 * in_channels, out_channels, embed_dim=temb_channels),
            ResidualTemporalBlock1D(2 * in_channels, out_channels, embed_dim=temb_channels),
            ResidualTemporalBlock1D(2 * in_channels, out_channels, embed_dim=temb_channels)
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
        self.upsample = Upsample1D(out_channels, use_conv_transpose=True)

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
        return self.upsample(hidden_states)

class UpResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        temb_channels: int = 32,
        non_linearity: Optional[str] = None,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm
        self.add_upsample = add_upsample
        self.output_scale_factor = output_scale_factor

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(2 * in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.ModuleList(resnets)

        if non_linearity is None:
            self.nonlinearity = None
        else:
            self.nonlinearity = get_activation(non_linearity)

        self.upsample = None
        if add_upsample:
            self.upsample = Upsample1D(out_channels, use_conv_transpose=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if res_hidden_states_tuple is not None:
            res_hidden_states = res_hidden_states_tuple[-1]
            hidden_states = torch.cat((hidden_states, res_hidden_states), dim=1)

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states

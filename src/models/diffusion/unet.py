import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union
import numpy as np
from embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps  # pyright: ignore


class Unet(nn.Module):
    """
    :param in_channels: is the number of channels in the input feature map
    :param out_channels: is the number of channels in the output feature map
    :param channels: is the base channel count for the model
    :param n_res_blocks: number of residual blocks at each level
    :param attention_levels: are the levels at which attention should be performed
    :param channel_multipliers: are the multiplicative factors for number of channels for each level
    :param n_heads: is the number of attention heads in the transformers
    :param tf_layers: is the number of transformer layers in the transformers
    :param d_cond: is the size of the conditional embedding in the transformers
    """

    def __init__(
        self,
        sample_size: int = 65536,
        sample_rate: Optional[int] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        block_out_channels: Tuple[int] = (32, 32, 64),  # pyright: ignore
        norm_num_groups: int = 8,
        time_embedding_type: str = "fourier",
        flip_sin_to_cos: bool = True,
        use_timestep_embedding: bool = False,
        freq_shift: float = 0.0,
        down_block_types: Tuple[str] = (
            "DownBlock1D",
            "AttnDownBlock1D",
            "AttnDownBlock1D",
        ),  # pyright: ignore
        up_block_types: Tuple[str] = ("AttnUpBlock1D", "AttnUpBlock1D", "UpBlock1D"),  # pyright: ignore
        mid_block_type: Tuple[str] = "UNetMidBlock1D",  # pyright: ignore
        out_block_type: str = None,  # pyright: ignore
        layers_per_block: int = 1,
        act_fn: str = None,  # pyright: ignore
    ):
        super(Unet, self).__init__()

        # time - gets the time embeddings, can also get the class embeddings
        # class CombinedTimestepLabelEmbeddings(nn.Module):
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=8,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0],
                flip_sin_to_cos=flip_sin_to_cos,
                downscale_freq_shift=freq_shift,
            )
            timestep_input_dim = block_out_channels[0]

        if use_timestep_embedding:
            time_embed_dim = block_out_channels[0] * 4
            self.time_mlp = TimestepEmbedding(
                in_channels=timestep_input_dim,  # pyright: ignore
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
                out_dim=block_out_channels[0],
            )

        # Get list of all of our modules
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None

        # down
        output_channel = in_channels
        #     "DownBlock1D",
        #     "AttnDownBlock1D",
        #     "AttnDownBlock1D",
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if i == 0:
                input_channel += extra_in_channels

            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_downsample=not is_final_block or downsample_each_block,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            embed_dim=block_out_channels[0],
            num_layers=layers_per_block,
            add_downsample=downsample_each_block,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        if out_block_type is None:
            final_upsample_channels = out_channels
        else:
            final_upsample_channels = block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = (
                reversed_block_out_channels[i + 1]
                if i < len(up_block_types) - 1
                else final_upsample_channels
            )

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = (
            norm_num_groups
            if norm_num_groups is not None
            else min(block_out_channels[0] // 4, 32)
        )
        self.out_block = get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=block_out_channels[0],
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=block_out_channels[-1] // 4,
        )

from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import GaussianFourierProjection, LabelEmbedding, TimestepEmbedding, Timesteps
from .unet_blocks import get_down_block, get_mid_block, get_up_block


class UNet1DModel(nn.Module):
    """
    A 1D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int`, *optional*): Default length of sample. Should be adaptable at runtime.
        in_channels (`int`, *optional*, defaults to 2): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 2): Number of channels in the output.
        extra_in_channels (`int`, *optional*, defaults to 0):
            Number of additional channels to be added to the input of the first down block. Useful for cases where the
            input data has more channels than what the model was initially designed for.
        time_embedding_type (`str`, *optional*, defaults to `"fourier"`): Type of time embedding to use.
        freq_shift (`float`, *optional*, defaults to 0.0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D")`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(32, 32, 64)`):
            Tuple of block output channels.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock1D"`): Block type for middle of UNet.
        out_block_type (`str`, *optional*, defaults to `None`): Optional output processing block of UNet.
        act_fn (`str`, *optional*, defaults to `None`): Optional activation function in UNet blocks.
        norm_num_groups (`int`, *optional*, defaults to 8): The number of groups for normalization.
        layers_per_block (`int`, *optional*, defaults to 1): The number of layers per block.
        downsample_each_block (`int`, *optional*, defaults to `False`):
            Experimental feature for using a UNet without upsampling.
    """

    def __init__(
        self,
        sample_size: int = 128,
        in_channels: int = 32,
        out_channels: int = 32,
        flip_sin_to_cos: bool = False,
        down_block_types: List[str] = [
            "DownBlock1D",
            "AttnDownBlock1D",
            "AttnDownBlock1D",
        ],  # pyright: ignore
        up_block_types: List[str] = ["AttnUpBlock1D", "AttnUpBlock1D", "UpBlock1D"],  # pyright: ignore
        mid_block_type: str = "UNetMidBlock1D",  # pyright: ignore
        block_out_channels: List[int] = [32, 32, 32],  # pyright: ignore
        num_attention_heads: int = 8,
        layers_per_block: int = 1,
        condition: bool = True,
        conditional: int = 11,
    ):
        super().__init__()

        # size of the input token dimensions
        self.sample_size = sample_size

        # Store output of bottleneck -- to visually see how we learn
        self.bottleneck_activations = None

        ############################### TIME EMBEDDINGS ######################################

        # initalize to a size that is large enough to hold semantically meaningful information
        timestep_input_dim = block_out_channels[0]
        time_embed_dim = block_out_channels[0] * 4

        # time_proj -> MLP projected timestep intervals
        self.time_proj = Timesteps(timestep_input_dim, flip_sin_to_cos)

        # time embeddings -> Higher dimensional time_proj to containe time_embed_dim features
        self.time_mlp = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # Set Conditional conditioning if we have it

        # class CombinedTimestepLabelEmbeddings(nn.Module):
        self.cond_embeddings = None
        if condition:
            self.cond_embeddings = (
                LabelEmbedding(conditional, time_embed_dim, 0.1)
            )
            self.cond_mlp = TimestepEmbedding(time_embed_dim, time_embed_dim)

        ######################################################################################

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None

        # down
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            # covert string to specific module we want
            down_block = get_down_block(
                down_block_type,
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block,
                embed_channels=time_embed_dim, #converts time_embed to channels to broadcast
                condition=condition,
                add_downsample=True,
                # context_dim=conditional_len,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            condition=condition,
            embed_channels=time_embed_dim, #converts time_embed to channels to broadcast
            # context_dim=conditional_len,
        )
        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        final_upsample_channels = out_channels
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
                embed_channels=time_embed_dim,
                condition=condition,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        context: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        r"""
        The [`UNet1DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_1d.UNet1DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_1d.UNet1DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_1d.UNet1DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # verify we have context or not
        # if hasattr(self, 'cond_embeddings') and context is None:
        #     raise ValueError("Context must be provided when using conditional model")
        # elif not hasattr(self, 'cond_embeddings') and context is not None:
        #     raise ValueError("Context cannot be provided when using unconditional model")

        # 1. time - batch_size timesteps x higher_dim
        timesteps = timestep
        timestep_embed = self.time_proj(timesteps)
        timestep_embed = self.time_mlp(timestep_embed.to(sample.dtype))
        # set conditonal embeddings - only applied in cross attention layers
        cond_embeddings = None
        if self.cond_embeddings:
            cond_embeddings = self.cond_embeddings(context)
            cond_embeddings = self.cond_mlp(cond_embeddings.to(sample.dtype))

        # 2. down
        # saving residual connections as tuple incase we want to save intermediate results
        down_block_res_samples = ()

        # original sample [64,32,64]
        # after downsample [64,32,32]
        # before first cross attention [64,32,32]
        for downsample_block in self.down_blocks:
            sample, res_sample = downsample_block(hidden_states=sample, temb=timestep_embed, context=cond_embeddings)
            down_block_res_samples+=res_sample

        # 3. mid
        if self.mid_block:
            sample = self.mid_block(hidden_states=sample, temb=timestep_embed, context=cond_embeddings)
            self.bottleneck_activations = sample.clone().detach()

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(sample, res_hidden_state=res_samples, temb=timestep_embed, context=cond_embeddings)

        # 5. post-process
        if self.out_block:
            sample = self.out_block(sample, timestep_embed)

        return sample

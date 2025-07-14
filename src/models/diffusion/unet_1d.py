from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import (
    GaussianFourierProjection,
    LabelEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from .unet_blocks import get_down_block, get_mid_block, get_up_block, get_out_block

class DualHeadPostProcessor(nn.Module):
    """
    Improved post-processing module with two specialized heads:
    1. Noise prediction head (standard diffusion)
    2. Clean signal prediction head (direct signal estimation)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_dim = hidden_dim or in_channels

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_channels),
        )

        # Shared feature refinement
        self.shared_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.shared_blocks.append(
                nn.Sequential(
                    nn.GroupNorm(8, in_channels),
                    nn.SiLU(),
                    nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.Dropout(dropout),
                )
            )

        # Noise prediction head - designed for residual learning
        self.noise_head = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1),
            # No activation - noise can be any value
        )

        # Clean signal prediction head - designed for direct signal estimation
        self.clean_head = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            # Additional refinement for clean signal
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

        # Learned scaling parameters for clean signal
        self.clean_scale = nn.Parameter(torch.ones(1, out_channels, 1))
        self.clean_bias = nn.Parameter(torch.zeros(1, out_channels, 1))

        # Residual connections
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        timestep_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Features from UNet backbone [B, C, L]
            timestep_embed: Time embeddings [B, time_embed_dim]

        Returns:
            Tuple of (noise_prediction, clean_prediction)
        """
        # Store residual
        residual = x

        # Add time conditioning
        time_features = self.time_proj(timestep_embed)
        time_features = time_features.unsqueeze(-1)  # [B, C, 1]
        x = x + time_features

        # Shared feature refinement with residual connections
        for block in self.shared_blocks:
            x = x + block(x)  # Residual connection

        # Generate predictions from specialized heads
        noise_pred = self.noise_head(x)
        clean_pred = self.clean_head(x)

        # Apply learned scaling to clean prediction for stability
        clean_pred = clean_pred * self.clean_scale + self.clean_bias

        # Add residual connection to noise prediction only
        # This helps with the typical diffusion formulation: x_t = x_0 + noise
        noise_pred = noise_pred + self.res_conv(residual)

        return noise_pred, clean_pred

class UNet1DModel(nn.Module):
    """
    A 1D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.
    Now includes dual-head post-processing for noise and clean signal prediction.
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
        ],
        up_block_types: List[str] = ["AttnUpBlock1D", "AttnUpBlock1D", "UpBlock1D"],
        mid_block_type: str = "UNetMidBlock1D",
        block_out_channels: List[int] = [32, 32, 32],
        num_attention_heads: int = 2,
        layers_per_block: int = 1,
        out_block: bool = True,
        condition: bool = False,
        conditional: int = 11,
        # New parameters for dual-head post-processing
        post_processor_type: str = "standard",  # "standard" or "adaptive"
        post_processor_layers: int = 2,
        post_processor_hidden_dim: Optional[int] = None,
        post_processor_dropout: float = 0.1,
    ):
        super().__init__()

        # size of the input token dimensions
        self.sample_size = sample_size
        self.out_block = out_block

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
        self.cond_embeddings = None
        if condition:
            self.cond_embeddings = LabelEmbedding(conditional, time_embed_dim, 0.8)
            self.cond_mlp = TimestepEmbedding(time_embed_dim, time_embed_dim)

        ######################################################################################

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

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
                embed_channels=time_embed_dim,
                condition=condition,
                add_downsample=True,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            condition=condition,
            embed_channels=time_embed_dim,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        final_upsample_channels = block_out_channels[0]  # Features before post-processing

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

        # Dual-head post-processor
        self.post_processor = DualHeadPostProcessor(
            in_channels=final_upsample_channels,
            out_channels=out_channels,
            time_embed_dim=time_embed_dim,
            hidden_dim=post_processor_hidden_dim,
            num_layers=post_processor_layers,
            dropout=post_processor_dropout,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        context: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_both: bool = False,
    ):
        """
        The UNet1DModel forward method with dual outputs.

        Args:
            sample: The noisy input tensor [batch_size, num_channels, sample_size]
            timestep: The number of timesteps to denoise an input
            context: Optional context (can be used for self-conditioning from previous predictions)
            return_dict: Whether to return dict or tuple
            return_both: Whether to return both noise and clean predictions

        Returns:
            If return_both=False: noise_prediction only (default)
            If return_both=True: (noise_prediction, clean_prediction)
        """

        # 1. time - batch_size timesteps x higher_dim
        timesteps = timestep
        timestep_embed = self.time_proj(timesteps)
        timestep_embed = self.time_mlp(timestep_embed.to(sample.dtype))

        # set conditional embeddings - only applied in cross attention layers
        cond_embeddings = None
        if self.cond_embeddings and context is not None:
            # cond_embeddings = self.cond_embeddings(context)
            # cond_embeddings = self.cond_mlp(cond_embeddings.to(sample.dtype))
            cond_embeddings = context

        # 2. down
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_sample = downsample_block(
                hidden_states=sample, temb=timestep_embed, context=cond_embeddings
            )
            down_block_res_samples += res_sample

        # 3. mid
        if self.mid_block:
            sample = self.mid_block(
                hidden_states=sample, temb=timestep_embed, context=cond_embeddings
            )
            self.bottleneck_activations = sample.clone().detach()

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(
                sample,
                res_hidden_state=res_samples,
                temb=timestep_embed,
                context=cond_embeddings,
            )

        # 5. post-process with dual heads
        if self.post_processor is not None:
            noise_pred, clean_pred = self.post_processor(sample, timestep_embed)
            if return_both:
                return noise_pred, clean_pred
            else:
                return noise_pred
        else:
            # Fallback if no post-processor
            return sample

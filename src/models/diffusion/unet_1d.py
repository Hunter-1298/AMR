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

class UNet1DModel(nn.Module):
    """
    UNet with sync parameter prediction from bottleneck
    """
    def __init__(
        self,
        in_channels: int = 2,
        sample_size: int = 1024,
        down_block_types: List[str] = ["DownResnetBlock1D", "AttnDownBlock1D", "AttnDownBlock1D"],
        block_out_channels: List[int] = [16, 32, 64],
        layers_per_block: int = 2,
        use_modulation_conditioning: bool = True,
        num_modulations: int = 3,  # QPSK, 8PSK, 16PSK
        num_attention_heads: int = 1,
        # Parameter ranges for normalization
        max_timing_offset: float = 1.0,     # normalized [-1, 1]
        max_freq_offset: float = 1.0,       # normalized [-1, 1]
        max_phase_offset: float = 1.0,      # normalized [-1, 1]
    ):
        super().__init__()

        self.sample_size = sample_size
        # Store as tensors with correct dtype
        self.register_buffer('max_timing_offset', torch.tensor(max_timing_offset, dtype=torch.float32))
        self.register_buffer('max_freq_offset', torch.tensor(max_freq_offset, dtype=torch.float32))
        self.register_buffer('max_phase_offset', torch.tensor(max_phase_offset, dtype=torch.float32))

        # Time embedding
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # Store dimensions
        self.time_embed_dim = time_embed_dim
        self.bottleneck_channels = block_out_channels[-1]

        # Modulation conditioning
        if use_modulation_conditioning:
            self.mod_embedding = nn.Embedding(num_modulations, time_embed_dim)
        else:
            self.mod_embedding = None

        # Down blocks
        self.down_blocks = nn.ModuleList([])

        for i, down_block_type in enumerate(down_block_types):
            if i == 0:
                input_channel = in_channels
                output_channel = block_out_channels[i]
            else:
                input_channel = block_out_channels[i-1]
                output_channel = block_out_channels[i]

            down_block = get_down_block(
                down_block_type=down_block_type,
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block,
                embed_channels=time_embed_dim,
                add_downsample=(i < len(down_block_types) - 1),
                condition=use_modulation_conditioning,
            )
            self.down_blocks.append(down_block)

        # Middle block
        self.mid_block = get_mid_block(
            mid_block_type="UNetMidBlock1D",
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            embed_channels=time_embed_dim,
            condition=use_modulation_conditioning,
        )

        # Parameter prediction network
        param_input_dim = self.bottleneck_channels + time_embed_dim

        self.param_predictor = nn.Sequential(
            nn.Linear(param_input_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

        # Individual parameter heads
        self.timing_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.freq_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.phase_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        sample: torch.Tensor,  # [B, 2, 1024]
        timestep: torch.Tensor,
        modulation: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """Forward pass - ONLY predicts parameters"""

        # Ensure input tensors have correct dtype
        sample = sample.float()  # Ensure float32
        timestep = timestep.long()  # Timesteps should be long
        if modulation is not None:
            modulation = modulation.long()  # Modulation indices should be long

        batch_size = sample.shape[0]

        # Time embedding
        t_emb = self.time_proj(timestep)
        t_emb = self.time_embedding(t_emb)

        # Modulation conditioning
        context = None
        if self.mod_embedding is not None and modulation is not None:
            mod_emb = self.mod_embedding(modulation)
            t_emb = t_emb + mod_emb

        # Down path
        x = sample
        for down_block in self.down_blocks:
            x, _ = down_block(hidden_states=x, temb=t_emb, context=context)

        # Bottleneck
        bottleneck_features = self.mid_block(hidden_states=x, temb=t_emb, context=context)

        # Parameter prediction
        pooled_features = F.adaptive_avg_pool1d(bottleneck_features, 1).squeeze(-1)
        combined_features = torch.cat([pooled_features, t_emb], dim=-1)

        # Ensure combined features are float32
        combined_features = combined_features.float()

        # Shared feature extraction
        shared_features = self.param_predictor(combined_features)

        # Individual parameter predictions (normalized to [-1, 1])
        timing_norm = self.timing_head(shared_features)
        freq_norm = self.freq_head(shared_features)
        phase_norm = self.phase_head(shared_features)

        # Scale to actual parameter ranges using registered buffers
        timing_offset = timing_norm * self.max_timing_offset
        freq_offset = freq_norm * self.max_freq_offset
        phase_offset = phase_norm * self.max_phase_offset

        if return_dict:
            return {
                'timing_offset': timing_offset,
                'freq_offset': freq_offset,
                'phase_offset': phase_offset,
                'timing_norm': timing_norm,
                'freq_norm': freq_norm,
                'phase_norm': phase_norm,
            }
        else:
            return timing_offset, freq_offset, phase_offset

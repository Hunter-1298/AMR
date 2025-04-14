from dataclasses import dataclass
from typing import Optional

@dataclass
class UNetConfig:
    sample_size: int = 32  # Latent space size
    in_channels: int = 4   # Latent channels
    out_channels: int = 4  # Latent channels
    layers_per_block: int = 2
    block_out_channels: tuple = (128, 256, 512, 512)
    down_block_types: tuple = (
        "DownBlock1D",
        "DownBlock1D",
        "DownBlock1D",
        "DownBlock1D",
    )
    up_block_types: tuple = (
        "UpBlock1D",
        "UpBlock1D",
        "UpBlock1D",
        "UpBlock1D",
    )
    mid_block_type: str = "MidBlock1D"
    norm_num_groups: int = 32
    time_embedding_type: str = "fourier"
    attention_head_dim: Optional[int] = 8
    attention_type: str = "self"

@dataclass
class VAEConfig:
    encoder: dict = {
        "in_channels": 1,  # Input channels
        "base_channels": 64,
        "num_blocks": 4,
        "latent_dim": 4,  # Latent channels
        "downsample_factor": 8  # Latent space size reduction
    }
    decoder: dict = {
        "in_channels": 4,  # Latent channels
        "base_channels": 64,
        "num_blocks": 4,
        "out_channels": 1,  # Output channels
        "upsample_factor": 8  # Latent space size expansion
    }

@dataclass
class LatentDiffusionConfig:
    unet_config: UNetConfig = UNetConfig()
    vae_config: VAEConfig = VAEConfig()
    n_steps: int = 1000
    linear_start: float = 0.0001
    linear_end: float = 0.02
    latent_scaling_factor: float = 0.18215 
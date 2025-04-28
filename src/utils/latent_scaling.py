import torch

def calculate_latent_scaling_factor(encoder, val_loader, num_batches=200):
    """Calculate the scaling factor to maintain unit variance in latent space (per-channel)"""
    latents = []
    encoder.eval()

    with torch.no_grad():
        for i, (x, _, _) in enumerate(val_loader):
            if i >= num_batches:
                break
            x = x.to(encoder.device)
            z = encoder.encode(x)
            latents.append(z.cpu())

    latents = torch.cat(latents, dim=0)  # Shape: [Total_Batch, 32, 64]

    # --- Option 2: Per-Channel Standard Deviation ---
    latents = latents.view(latents.size(0), latents.size(1), -1)  # Flatten spatial dims (batch, channels, spatial)
    std_per_channel = latents.std(dim=[0, 2])  # std across (batch + spatial dimensions) for each channel separately
    mean_std = std_per_channel.mean()  # Average std across channels

    scaling_factor = 1.0 / mean_std.item()

    print(f"Calculated scaling factor: {scaling_factor:.4f} from mean channel std: {mean_std.item():.4f}")
    return scaling_factor

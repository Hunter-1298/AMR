import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.beta = beta
        
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

    def forward(self, z_e):
        # Compute distances between inputs and codebook embeddings
        # Flatten [batch_size, kernels, embed_dim] -> [batch_size*kernels, embed_dim]
        z_e_flat = z_e.view(-1, self.embedding_dim)
        distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True) /
                    + torch.sum(self.embedding.weight**2, dim=1) /
                    - 2 * torch.matmul(z_e_flat, self.embedding.weight.T))
        
        # Get closest embedding index
        encoding_indices = torch.argmin(distances, dim=1).view(*z_e.shape[:-1])
        z_q = self.embedding(encoding_indices)

        # Compute VQ loss
        loss_vq = F.mse_loss(z_q.detach(), z_e) + self.beta * F.mse_loss(z_q, z_e.detach())

        # Replace gradients
        z_q = z_e + (z_q - z_e).detach()
        return z_q, encoding_indices, loss_vq
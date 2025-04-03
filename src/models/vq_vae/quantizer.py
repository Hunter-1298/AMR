import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    def __init__(self, codebook_size, embedding_dim, beta=0.25, decay=0.99, epsilon=1e-5):
        """
        Vector Quantizer with Exponential Moving Average updates for training only
        
        Args:
            codebook_size: Size of the codebook
            embedding_dim: Dimension of each embedding vector
            beta: Commitment cost coefficient
            decay: EMA decay factor (higher = slower updates)
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.beta = beta
        
        # Initialize embeddings
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)
        
        # EMA parameters - only needed during training
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.decay = decay
        self.epsilon = epsilon
        
        # Add update frequency to reduce overhead
        self.update_frequency = 10  # Update codebook every 10 batches
        self.batch_counter = 0

    def forward(self, z_e):
        # Compute distances between inputs and codebook embeddings
        z_e_flat = z_e.view(-1, self.embedding_dim).contiguous()
        
        # More efficient distance calculation
        z_e_norm = torch.sum(z_e_flat**2, dim=1, keepdim=True)
        embed_norm = torch.sum(self.embedding.weight**2, dim=1)
        distance_mat = z_e_norm + embed_norm - 2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        
        # Get closest embedding index
        encoding_indices = torch.argmin(distance_mat, dim=1)
        
        # Reshape encoding indices to match input shape
        encoding_indices_view = encoding_indices.view(*z_e.shape[:-1])
        
        # Get quantized vectors
        z_q = self.embedding(encoding_indices_view)
        
        # Compute commitment loss
        loss = self.beta * F.mse_loss(z_q.detach(), z_e)
        
        # EMA update with frequency control
        if self.training:
            self.batch_counter += 1
            if self.batch_counter % self.update_frequency == 0:
                with torch.no_grad():
                    # Create one-hot encodings for EMA update
                    encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z_e.device)
                    encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                    
                    # EMA update for cluster sizes - use in-place operations
                    n = encodings.sum(0)
                    self.ema_cluster_size.mul_(self.decay).add_(n, alpha=(1 - self.decay))
                    
                    # Laplace smoothing for unused codes
                    n_active = (self.ema_cluster_size > self.epsilon).sum()
                    if n_active < self.codebook_size:
                        self.ema_cluster_size.add_(self.epsilon)
                    
                    # EMA update for embedding weights - use in-place operations
                    embed_sum = torch.matmul(encodings.t(), z_e_flat)
                    self.ema_w.mul_(self.decay).add_(embed_sum, alpha=(1 - self.decay))
                    
                    # Normalize and update embedding weights - avoid division by zero
                    denom = self.ema_cluster_size.unsqueeze(1) + self.epsilon
                    normalized_weights = self.ema_w / denom
                    self.embedding.weight.data.copy_(normalized_weights)
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, encoding_indices_view, loss
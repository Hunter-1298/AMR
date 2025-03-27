import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from typing import List

class Router(L.LightningModule):

    """
    Router Module for Mixture of Experts Architecture

    This class implements a router network that dynamically assigns input data to different expert models.
    It uses a neural network to compute routing weights that determine how much each expert contributes
    to the final prediction.

    Args:
        cfg (DictConfig): Configuration object containing:
            - num_experts (int): Number of expert models (default: 5)
            - input_dim (int): Input dimension size (default: 256)
            - hidden_dim (int): Hidden layer dimension size (default: 128)
            - optimizer (dict): Optimizer configuration
            - experts (list): List of expert model configurations

    The router network consists of a simple MLP that outputs routing weights for each expert.
    These weights are used to combine the outputs of individual experts into a final prediction.
    """

    def __init__(self, input_dim: int, hidden_dim: int, lr: float, experts: List[nn.Module], classifier: nn.Module):
        super().__init__()
        # Get parameters from config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = len(experts)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr 
        
        # Router network to determine expert weights
        self.router_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_experts),
        )
        
        # Store the experts directly from the config
        self.experts = nn.ModuleList(experts)
        
        # Use the classifier passed from config
        self.classifier = classifier

    def forward(self, x):
        batch_size = x.size(0)

        # Get routing weights for each expert [batch_size, num_experts]
        # Flatten x to be a single dim for linear layer
        routing_weights = F.softmax(self.router_network(x.reshape(x.size(0),-1)), dim=-1)
        assert routing_weights.shape == (batch_size, self.num_experts), 'routing weights are incorrect shape'
        
        # Apply each expert and combine results according to routing weights
        expert_outputs = []
        for expert in self.experts:
            # expert_output shape: [batch_size, feature_dim]
            expert_output = expert(x)
            expert_outputs.append(expert_output)
        
        # Stack results: [batch_size, num_experts, feature_dim]
        stacked_results = torch.stack(expert_outputs, dim=1)
        assert stacked_results.shape == (batch_size, self.num_experts, self.hidden_dim), 'experts should return hidden_dim size'

        # Reshape routing_weights for broadcasting: [batch_size,num_experts] -> [batch_size, num_experts, 1]
        weights = routing_weights.unsqueeze(-1)
        
        # Scale expert outputs based on router weights: [batch_size,experts, features] * [batch_size, num_experts,1] -> [batch_size, experts, features]
        # The sum along the experts to get the total weights
        combined_output = (stacked_results * weights).sum(dim=1)
        
        # Final classification - accepts hidden_dim * experts
        prediction = self.classifier(combined_output)
        
        
        return prediction, routing_weights
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        pred, routing_weights = self(x)
        
        # Calculate loss - pred, truth
        classification_loss = self.criterion(pred, y)  
        
        # Add routing loss to encourage load balancing
        # load_balancing_loss = self._compute_load_balancing_loss(routing_weights)
        
        # Combined loss
        # total_loss = classification_loss + load_balancing_loss
        

        # Log expert usage instead of balancing for now
        expert_usage = routing_weights.mean(dim=0)
        for i, usage in enumerate(expert_usage):
            self.log(f'weighted_expert: {i}', usage)

        self.log('train_losss', classification_loss)
        # self.log('train_total_loss', total_loss)
        # self.log('train_load_balancing_loss', load_balancing_loss)
        return classification_loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        output, routing_weights = self(x)
        val_loss = self.criterion(output,y)  
        self.log('val_loss', val_loss)

        # load_balancing_loss = self._compute_load_balancing_loss(routing_weights)
        # total_loss = task_loss #+ load_balancing_loss
        
        # self.log('val_total_loss', total_loss)
        # self.log('val_load_balancing_loss', load_balancing_loss)
        
    # Load balancing for future use cases maybe?
    # def _compute_load_balancing_loss(self, routing_weights):
    #     # Compute fraction of routing weight per expert
    #     expert_usage = routing_weights.mean(dim=0)
    #     # Ideal uniform distribution
    #     target_usage = torch.ones_like(expert_usage) / self.num_experts
    #     # KL divergence loss to encourage uniform expert usage
    #     load_balancing_loss = F.kl_div(expert_usage.log(), target_usage, reduction='batchmean')
    #     return load_balancing_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
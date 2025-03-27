import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig


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

    def __init__(self, cfg: DictConfig):
        super(Router, self).__init__()
        # Get parameters from config
        self.num_experts = cfg.experts  # Get # of experts
        self.input_dim = cfg.Model.input_dim  # Get input dim size for router (2 * 128) = 256
        self.hidden_dim = cfg.Model.hidden_dim  # get hidden dim size for oruter
        self.criterion = nn.CrossEntropyLoss()  # 
        self.optimizer_config = cfg.optimizer  # Fix typo in optimizer
        
        # Router network to determine expert weights
        self.router_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_experts),
        )
        
        
        # Load expert models (these would be defined in other files)
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            expert = hydra.utils.instantiate(cfg.experts[i])
            self.experts.append(expert)

        self.classificer = hydra.utils.instantiate(cfg.Classifier)
        

    def forward(self, x):
        # Get routing weights for each expert
        routing_weights = F.softmax(self.router_network(x), dim=-1)
        
        # Apply each expert and combine results according to routing weights
        results = []
        for i, expert in enumerate(self.experts):
            # expert output will be a 128 feature vector
            expert_output = expert(x)
            results.append(expert_output)
            
        # Stack results and apply routing weights
        stacked_results = torch.stack(results, dim=1)  # [batch_size, num_experts, ...]

        # Reshape routing_weights for broadcasting
        weights = routing_weights.unsqueeze(-1)  
        
        # Weighted sum of expert outputs
        combined_output = (stacked_results * weights).sum(dim=1)
        
        # Pass this into the classificaiton output MLP
        
        return combined_output, routing_weights
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        output, routing_weights = self(x)
        
        # Calculate task loss 
        classification_loss = self.criterion(output, y)  
        
        # Add routing loss to encourage load balancing
        # load_balancing_loss = self._compute_load_balancing_loss(routing_weights)
        
        # Combined loss
        # total_loss = classification_loss + load_balancing_loss
        

        # Log expert usage instead of balancing for now
        expert_usage = routing_weights.mean(dim=0)
        for i, usage in enumerate(expert_usage):
            self.log(f'expert_{i}_usage', usage)

        self.log('train_losss', classification_loss)
        # self.log('train_total_loss', total_loss)
        # self.log('train_load_balancing_loss', load_balancing_loss)
        return classification_loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        output, routing_weights = self(x)
        loss = self.criterion(output,y)  
        self.log('loss', loss)

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
        optimizer =self.optimizer_config(params=self.parameters())
        return optimizer
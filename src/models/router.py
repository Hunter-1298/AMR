import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig

class Router(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Get parameters from config
        self.num_experts = cfg.get("num_experts", 5)  # Default to 5 experts
        self.input_dim = cfg.get("input_dim", 256)
        self.hidden_dim = cfg.get("hidden_dim", 128)
        self.optimizer_config = cfg.optimizer
        
        # Router network to determine expert weights
        self.router_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_experts),
        )
        
        # Load expert models (these would be defined in other files)
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            # This assumes cfg.experts[i] contains the config for each expert
            # The actual loading would depend on how your experts are defined
            expert = hydra.utils.instantiate(cfg.experts[i])
            self.experts.append(expert)

    def forward(self, x):
        # Get routing weights for each expert
        routing_weights = F.softmax(self.router_network(x), dim=-1)
        
        # Apply each expert and combine results according to routing weights
        results = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            results.append(expert_output)
            
        # Stack results and apply routing weights
        stacked_results = torch.stack(results, dim=1)  # [batch_size, num_experts, ...]
        # Reshape routing_weights for broadcasting
        weights = routing_weights.unsqueeze(-1)  # Add dims for broadcasting
        
        # Weighted sum of expert outputs
        combined_output = (stacked_results * weights).sum(dim=1)
        
        return combined_output, routing_weights
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        output, routing_weights = self(x)
        
        # Calculate task loss (depends on your specific task)
        task_loss = F.mse_loss(output, y)  # Example loss, adjust as needed
        
        # Add routing loss to encourage load balancing
        load_balancing_loss = self._compute_load_balancing_loss(routing_weights)
        
        # Combined loss
        total_loss = task_loss + load_balancing_loss
        
        self.log('train_task_loss', task_loss)
        self.log('train_load_balancing_loss', load_balancing_loss)
        self.log('train_total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        output, routing_weights = self(x)
        
        task_loss = F.mse_loss(output, y)  # Example loss, adjust as needed
        load_balancing_loss = self._compute_load_balancing_loss(routing_weights)
        total_loss = task_loss + load_balancing_loss
        
        self.log('val_task_loss', task_loss)
        self.log('val_load_balancing_loss', load_balancing_loss)
        self.log('val_total_loss', total_loss)
        
    def _compute_load_balancing_loss(self, routing_weights):
        # Compute fraction of routing weight per expert
        expert_usage = routing_weights.mean(dim=0)
        # Ideal uniform distribution
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        # KL divergence loss to encourage uniform expert usage
        load_balancing_loss = F.kl_div(expert_usage.log(), target_usage, reduction='batchmean')
        return load_balancing_loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer
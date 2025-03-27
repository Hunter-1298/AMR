from typing import List
import torch.nn as nn

class Router(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        experts: List[nn.Module],  # Hydra will pass the instantiated experts here
        lr: float
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList(experts)  # Store experts as ModuleList
        self.lr = lr

    def forward(self, x):
        # Use your experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        # ... rest of your routing logic ... 
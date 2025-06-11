import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class GradientNormalizer:
    """Normalizes gradients across multiple loss components for balanced training"""
    
    def __init__(self, alpha: float = 0.99, eps: float = 1e-8):
        self.alpha = alpha
        self.eps = eps
        self.gradient_norms = defaultdict(lambda: 1.0)
        self.gradient_history = defaultdict(list)
        
    def compute_gradient_norms(self, losses: Dict[str, torch.Tensor], 
                              model: nn.Module) -> Dict[str, float]:
        """Compute gradient norms for each loss component"""
        grad_norms = {}
        
        for name, loss in losses.items():
            if loss.requires_grad and loss.grad_fn is not None:
                # Compute gradients for this loss only
                grads = torch.autograd.grad(
                    loss, 
                    model.parameters(), 
                    retain_graph=True,
                    allow_unused=True
                )
                
                # Calculate gradient norm
                total_norm = 0.0
                for grad in grads:
                    if grad is not None:
                        total_norm += grad.norm(2).item() ** 2
                grad_norm = np.sqrt(total_norm)
                
                # Update running average
                self.gradient_norms[name] = (
                    self.alpha * self.gradient_norms[name] + 
                    (1 - self.alpha) * grad_norm
                )
                grad_norms[name] = self.gradient_norms[name]
                
                # Store history for analysis
                self.gradient_history[name].append(grad_norm)
                if len(self.gradient_history[name]) > 1000:
                    self.gradient_history[name].pop(0)
        
        return grad_norms
    
    def get_balanced_weights(self, grad_norms: Dict[str, float], 
                           base_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate balanced weights based on gradient norms"""
        if not grad_norms:
            return base_weights
            
        # Use the first loss as reference (typically the main loss)
        ref_norm = list(grad_norms.values())[0]
        
        balanced_weights = {}
        for name, base_weight in base_weights.items():
            if name in grad_norms and grad_norms[name] > self.eps:
                # Scale weight inversely proportional to gradient norm
                scale = ref_norm / (grad_norms[name] + self.eps)
                balanced_weights[name] = base_weight * scale
            else:
                balanced_weights[name] = base_weight
                
        return balanced_weights


class UncertaintyWeightedLoss(nn.Module):
    """Implements uncertainty-based multi-task loss weighting"""
    
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute uncertainty-weighted total loss
        Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
        """
        assert len(losses) == self.num_tasks
        
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
            
        return total_loss
    
    def get_weights(self) -> List[float]:
        """Get current task weights"""
        return [torch.exp(-log_var).item() for log_var in self.log_vars]


class DynamicWeightAverage:
    """
    Dynamic Weight Average for multi-task learning
    Based on "End-to-End Multi-Task Learning with Attention"
    """
    
    def __init__(self, num_tasks: int, temperature: float = 2.0):
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.prev_losses = None
        self.weights = torch.ones(num_tasks) / num_tasks
        
    def update_weights(self, current_losses: List[float]) -> List[float]:
        """Update task weights based on loss decrease rates"""
        if self.prev_losses is None:
            self.prev_losses = current_losses
            return self.weights.tolist()
            
        # Calculate relative loss decrease
        loss_ratios = []
        for i in range(self.num_tasks):
            if self.prev_losses[i] > 0:
                ratio = current_losses[i] / self.prev_losses[i]
                loss_ratios.append(ratio)
            else:
                loss_ratios.append(1.0)
                
        loss_ratios = torch.tensor(loss_ratios)
        
        # Apply temperature and softmax
        weights = F.softmax(loss_ratios / self.temperature, dim=0)
        self.weights = weights * self.num_tasks  # Scale to maintain total weight
        
        self.prev_losses = current_losses
        return self.weights.tolist()


class GradientSurgery:
    """
    Gradient surgery for conflicting gradients
    Based on "Gradient Surgery for Multi-Task Learning"
    """
    
    @staticmethod
    def project_conflicting_gradients(gradients: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """Project conflicting gradients to non-conflicting directions"""
        task_names = list(gradients.keys())
        num_tasks = len(task_names)
        
        if num_tasks < 2:
            return gradients
            
        # Flatten gradients for each task
        flat_grads = {}
        shapes = {}
        for name, grads in gradients.items():
            flat_grad_list = []
            shape_list = []
            for g in grads:
                if g is not None:
                    flat_grad_list.append(g.flatten())
                    shape_list.append(g.shape)
            if flat_grad_list:
                flat_grads[name] = torch.cat(flat_grad_list)
                shapes[name] = shape_list
                
        # Check for conflicts and project
        projected_grads = {}
        for i, name_i in enumerate(task_names):
            if name_i not in flat_grads:
                continue
                
            grad_i = flat_grads[name_i].clone()
            
            for j, name_j in enumerate(task_names):
                if i != j and name_j in flat_grads:
                    grad_j = flat_grads[name_j]
                    
                    # Check if gradients conflict (negative cosine similarity)
                    dot_product = torch.dot(grad_i, grad_j)
                    if dot_product < 0:
                        # Project grad_i to be orthogonal to grad_j
                        proj = dot_product / (torch.norm(grad_j) ** 2 + 1e-8)
                        grad_i = grad_i - proj * grad_j
                        
            projected_grads[name_i] = grad_i
            
        # Reshape back to original shapes
        result = {}
        for name, flat_grad in projected_grads.items():
            grads_list = []
            start_idx = 0
            for shape in shapes[name]:
                num_elements = np.prod(shape)
                grad = flat_grad[start_idx:start_idx + num_elements].reshape(shape)
                grads_list.append(grad)
                start_idx += num_elements
            result[name] = grads_list
            
        return result


class AdaptiveLossBalancer:
    """Combines multiple loss balancing strategies"""
    
    def __init__(self, 
                 method: str = "grad_norm",
                 alpha: float = 0.99,
                 temperature: float = 2.0,
                 min_weight: float = 0.01,
                 max_weight: float = 10.0):
        self.method = method
        self.alpha = alpha
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize specific balancers
        self.gradient_normalizer = GradientNormalizer(alpha=alpha)
        self.dynamic_average = None
        self.loss_history = defaultdict(list)
        
    def balance_losses(self, 
                      losses: Dict[str, torch.Tensor],
                      base_weights: Dict[str, float],
                      model: Optional[nn.Module] = None) -> Dict[str, float]:
        """Balance losses using the specified method"""
        
        # Update loss history
        for name, loss in losses.items():
            self.loss_history[name].append(loss.item())
            if len(self.loss_history[name]) > 100:
                self.loss_history[name].pop(0)
                
        if self.method == "grad_norm" and model is not None:
            # Gradient norm balancing
            grad_norms = self.gradient_normalizer.compute_gradient_norms(losses, model)
            weights = self.gradient_normalizer.get_balanced_weights(grad_norms, base_weights)
            
        elif self.method == "magnitude":
            # Magnitude-based balancing
            weights = {}
            loss_values = {name: loss.item() for name, loss in losses.items()}
            
            # Use exponential moving average of losses
            avg_losses = {}
            for name in base_weights:
                if name in self.loss_history and self.loss_history[name]:
                    avg_losses[name] = np.mean(self.loss_history[name][-20:])
                else:
                    avg_losses[name] = loss_values.get(name, 1.0)
                    
            # Calculate inverse scaling
            ref_loss = list(avg_losses.values())[0]
            for name, base_weight in base_weights.items():
                if name in avg_losses and avg_losses[name] > 0:
                    scale = ref_loss / avg_losses[name]
                    weights[name] = base_weight * scale
                else:
                    weights[name] = base_weight
                    
        elif self.method == "dynamic":
            # Dynamic weight averaging
            if self.dynamic_average is None:
                self.dynamic_average = DynamicWeightAverage(len(losses), self.temperature)
                
            loss_list = [losses[name].item() for name in sorted(losses.keys())]
            dynamic_weights = self.dynamic_average.update_weights(loss_list)
            
            weights = {}
            for i, name in enumerate(sorted(base_weights.keys())):
                weights[name] = base_weights[name] * dynamic_weights[i]
                
        else:
            # Default: use base weights
            weights = base_weights.copy()
            
        # Clip weights to reasonable range
        for name in weights:
            weights[name] = np.clip(weights[name], self.min_weight, self.max_weight)
            
        return weights
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get balancing statistics for monitoring"""
        stats = {}
        
        # Loss statistics
        for name, history in self.loss_history.items():
            if history:
                stats[f"{name}_loss"] = {
                    "mean": np.mean(history),
                    "std": np.std(history),
                    "min": np.min(history),
                    "max": np.max(history),
                    "recent": history[-1]
                }
                
        # Gradient statistics
        if hasattr(self, 'gradient_normalizer'):
            for name, norm in self.gradient_normalizer.gradient_norms.items():
                stats[f"{name}_grad_norm"] = {"value": norm}
                
        return stats


def compute_gradient_similarity(grads1: List[torch.Tensor], 
                              grads2: List[torch.Tensor]) -> float:
    """Compute cosine similarity between two sets of gradients"""
    # Flatten and concatenate gradients
    flat_grads1 = []
    flat_grads2 = []
    
    for g1, g2 in zip(grads1, grads2):
        if g1 is not None and g2 is not None:
            flat_grads1.append(g1.flatten())
            flat_grads2.append(g2.flatten())
            
    if not flat_grads1:
        return 0.0
        
    flat_grads1 = torch.cat(flat_grads1)
    flat_grads2 = torch.cat(flat_grads2)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(flat_grads1, flat_grads2, dim=0)
    return similarity.item()


def analyze_gradient_conflicts(gradients: Dict[str, List[torch.Tensor]]) -> Dict[str, float]:
    """Analyze conflicts between different task gradients"""
    task_names = list(gradients.keys())
    conflicts = {}
    
    for i, name1 in enumerate(task_names):
        for j, name2 in enumerate(task_names[i+1:], i+1):
            similarity = compute_gradient_similarity(
                gradients[name1], 
                gradients[name2]
            )
            conflicts[f"{name1}_vs_{name2}"] = similarity
            
    return conflicts
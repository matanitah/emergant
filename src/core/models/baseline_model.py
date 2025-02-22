import numpy as np
import torch
import torch.nn as nn
from core.policies.policy import Policy, MAX_TURN_MAGNITUDE

class BaselineAntPolicy(nn.Module):
    def __init__(self, hidden_sizes):
        """
        Initialize a feedforward neural network with variable hidden layers.
        
        Args:
            hidden_sizes (list): List of integers representing the size of each hidden layer.
        """
        super(BaselineAntPolicy, self).__init__()
        
        input_size = 8
        output_size = 4  # 2 for direction logits, 1 for magnitude mean, 1 for magnitude std
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Build layers dynamically
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Don't add activation after last layer
                layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 8)
        Returns:
            torch.Tensor: Network outputs of shape (batch_size, 4):
                         [:, :2] - direction logits
                         [:, 2] - magnitude mean (in radians)
                         [:, 3] - magnitude std (in radians)
        """
        output = self.network(x)
        
        # Split the output into components
        direction_logits = output[:2]  # Remove second dimension indexing since input may be 1D
        
        # Scale magnitude mean to be between 0 and MAX_TURN_MAGNITUDE (in radians)
        magnitude_mean = torch.sigmoid(output[:, 2]) * MAX_TURN_MAGNITUDE
        
        # Ensure standard deviation is positive but reasonable
        # Max std of ~0.17 radians (~10 degrees)
        magnitude_std = torch.sigmoid(output[:, 3]) * (torch.pi / 18.0)
        
        # Combine outputs
        return torch.cat([
            direction_logits,
            magnitude_mean.unsqueeze(1),
            magnitude_std.unsqueeze(1)
        ], dim=1)
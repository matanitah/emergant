import torch
import torch.distributions as dist
import jax
import jax.numpy as jnp
import flax

# Define maximum turn magnitude in radians (π/4 ≈ 45 degrees)
MAX_TURN_MAGNITUDE = torch.pi / 4

class Policy:
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def get_action(self, observation):
        with torch.no_grad():
            # Convert observation to JAX array and add batch dimension
            obs_array = jnp.array(observation, dtype=jnp.float32)[None, :]
            
            # Get model outputs - now outputs [direction_logits, magnitude_mean, magnitude_std]
            model_output = self.model.apply(obs_array)
            
            # Split outputs into direction and magnitude parameters
            direction_logits = model_output[:, :2]  # First 2 values for left/right
            magnitude_mean = model_output[:, 2]     # Mean for the magnitude
            magnitude_std = torch.abs(model_output[:, 3])  # Standard deviation (must be positive)
            
            # Get direction probabilities and sample
            direction_probs = torch.softmax(direction_logits, dim=1)
            direction_idx = torch.multinomial(direction_probs, 1)
            direction = 'left' if direction_idx.item() == 0 else 'right'
            
            # Create normal distribution for magnitude and sample
            magnitude_dist = dist.Normal(magnitude_mean, magnitude_std)
            magnitude = magnitude_dist.sample()
            
            # Clamp magnitude between 0 and MAX_TURN_MAGNITUDE
            magnitude = torch.clamp(magnitude, 0.0, MAX_TURN_MAGNITUDE).item()

        return (direction, magnitude)

    def learn(self, observation, reward):
        # Convert observation to tensor and add batch dimension
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        
        # Scale reward for numerical stability
        scaled_reward = torch.tensor([[reward * 0.01]], dtype=torch.float32)
        
        # Forward pass to get all outputs
        model_output = self.model(obs_tensor)
        direction_logits = model_output[:, :2]
        magnitude_mean = model_output[:, 2]
        magnitude_std = torch.abs(model_output[:, 3])
        
        # Compute direction probabilities
        direction_probs = torch.softmax(direction_logits, dim=1)
        log_probs = torch.log(direction_probs)
        
        # Create magnitude distribution
        magnitude_dist = dist.Normal(magnitude_mean, magnitude_std)
        magnitude_log_prob = magnitude_dist.log_prob(model_output[:, 2])
        
        # Combine both direction and magnitude log probabilities
        total_log_prob = log_probs.mean() + magnitude_log_prob.mean()
        
        # Policy gradient loss
        loss = -(total_log_prob * scaled_reward).mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

import random
import numpy as np
from config.settings import WIDTH, HEIGHT, PHEROMONE_INFLUENCE, PHEROMONE_STRENGTH
from core.colony import Colony

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize a simple feedforward neural network with one hidden layer
        # The network architecture is: input_layer -> hidden_layer -> output_layer
        
        # weights1: Connects input layer to hidden layer
        # Initialized with small random values (scaled by 0.1) to prevent extreme initial behaviors
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        
        # weights2: Connects hidden layer to output layer
        # Initialized with small random values (scaled by 0.1) to prevent extreme initial behaviors
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        
    def forward(self, x):
        # Forward pass through the network
        # Uses tanh activation function for both layers to output values between -1 and 1
        
        # Process through hidden layer
        self.hidden = np.tanh(np.dot(x, self.weights1))
        # Process through output layer
        self.output = np.tanh(np.dot(self.hidden, self.weights2))
        return self.output
    
    def update(self, learning_rate, gradient):
        # Simple gradient update for reinforcement learning
        # Only updates the output layer weights (weights2) based on the reward signal
        # learning_rate: Controls how much the weights change in response to errors
        # gradient: Computed based on the reward history and action history
        self.weights2 += learning_rate * gradient
        
class Ant:
    def __init__(self, colony, colony_id):
        self.x = colony.x
        self.y = colony.y
        self.colony = colony
        self.colony_x = colony.x
        self.colony_y = colony.y
        self.direction = random.uniform(0, 2 * np.pi)
        self.carrying_food = False
        self.colony_id = colony_id
        self.is_alive = True
        
        # Neural network parameters
        self.input_size = 7  # State observations
        self.hidden_size = 8
        self.output_size = 3  # [turn_left, turn_right, drop_pheromone]
        self.brain = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        
        # Learning parameters
        self.learning_rate = 0.01
        self.reward_history = []
        self.action_history = []

    def get_state(self, pheromone_grid, food_sources):
        # Create state vector for neural network input
        # 1. Distance to colony
        dist_to_colony = np.hypot(self.x - self.colony_x, self.y - self.colony_y) / np.hypot(WIDTH, HEIGHT)
        
        # 2. Angle to colony (normalized)
        angle_to_colony = (np.arctan2(self.colony_y - self.y, self.colony_x - self.x) - self.direction) / (2 * np.pi)
        
        # 3. Carrying food (binary)
        carrying_food = 1.0 if self.carrying_food else 0.0
        
        # 4-6. Pheromone levels in three directions
        pheromone_levels = np.zeros(3)
        for i, angle in enumerate([0, 0.5, -0.5]):
            check_x = min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction + angle) * 5)))
            check_y = min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction + angle) * 5)))
            pheromone_levels[i] = pheromone_grid[check_x][check_y] / PHEROMONE_STRENGTH
            
        # 7. Distance to nearest food
        min_food_dist = 1.0
        for food in food_sources:
            dist = np.hypot(self.x - food.x, self.y - food.y) / np.hypot(WIDTH, HEIGHT)
            min_food_dist = min(min_food_dist, dist)
            
        return np.array([dist_to_colony, angle_to_colony, carrying_food, 
                        *pheromone_levels, min_food_dist])

    def move(self, pheromone_grid, food_sources):
        # Get current state
        state = self.get_state(pheromone_grid, food_sources)
        
        # Get neural network output
        actions = self.brain.forward(state)
        self.action_history.append(actions)
        
        # Apply actions
        turn_left, turn_right, drop_pheromone = actions
        
        # Update direction based on neural network output
        self.direction += 0.3 * (turn_right - turn_left)
        
        # Update position
        self.x += np.cos(self.direction) * 2
        self.y += np.sin(self.direction) * 2
        
        # Keep ants within bounds
        if self.x <= 0 or self.x >= WIDTH-1:
            self.direction = np.pi - self.direction
        if self.y <= 0 or self.y >= HEIGHT-1:
            self.direction = -self.direction

        self.x = max(0, min(WIDTH - 1, self.x))
        self.y = max(0, min(HEIGHT - 1, self.y))
        
        # Calculate reward
        reward = self.calculate_reward()
        self.reward_history.append(reward)
        
        # Learn from experience periodically
        if len(self.reward_history) >= 50:
            self.learn()

    def calculate_reward(self):
        reward = 0
        
        # Reward for picking up food
        if self.carrying_food:
            reward += 0.1
            
        # Reward for successfully returning food to colony
        if self.carrying_food and np.hypot(self.x - self.colony_x, self.y - self.colony_y) < Colony.size:
            reward += 1.0
            
        # Small penalty for wandering too far from colony
        dist_to_colony = np.hypot(self.x - self.colony_x, self.y - self.colony_y)
        reward -= 0.001 * (dist_to_colony / np.hypot(WIDTH, HEIGHT))
        
        return reward

    def learn(self):
        # Simple policy gradient update
        rewards = np.array(self.reward_history)
        actions = np.array(self.action_history)
        
        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-10)
        
        # Calculate gradients and update network
        gradient = np.mean(actions * rewards[:, np.newaxis], axis=0)
        self.brain.update(self.learning_rate, gradient)
        
        # Clear history
        self.reward_history = []
        self.action_history = []

    def drop_pheromone(self, pheromone_grid):
        # Only drop pheromone if neural network suggests it
        if self.action_history and self.action_history[-1][2] > 0:
            pheromone_grid[int(self.x)][int(self.y)] += PHEROMONE_STRENGTH

    def check_food(self, food_sources):
        for food in food_sources:
            if not self.carrying_food and np.hypot(self.x - food.x, self.y - food.y) < food.size:
                self.carrying_food = True
                food.food_count -= 1
                if food.food_count <= 0:
                    food_sources.remove(food)
                return

    def check_colony(self, colony):
        # If ant has returned to its own colony, drop off food.
        if self.colony_id == colony.id:
            if self.carrying_food and np.hypot(self.x - self.colony_x, self.y - self.colony_y) < Colony.size:
                self.colony.increment_food()
                self.carrying_food = False
        # If ant has reached the enemy colony, steal food.
        else:
            if self.carrying_food == False and np.hypot(self.x - colony.x, self.y - colony.y) < Colony.size:
                colony.decrement_food()
                self.carrying_food = True


    # def move(self, pheromone_grid, food_sources):
    #     if self.carrying_food:
    #         # When carrying food, head back to colony
    #         # Calculate angle between current position and colony using arctangent
    #         angle_to_colony = np.arctan2(self.colony_y - self.y, self.colony_x - self.x)
    #         # Gradually adjust direction towards colony (20% adjustment per step)
    #         self.direction += (angle_to_colony - self.direction) * 0.2
    #     else:
    #         # First check for nearby food
    #         closest_food = None
    #         min_distance = float('inf')
            
    #         for food in food_sources:
    #             distance = np.hypot(self.x - food.x, self.y - food.y)
    #             # Ants can see food within 50 units
    #             if distance < 50 and distance < min_distance:
    #                 closest_food = food
    #                 min_distance = distance
            
    #         if closest_food:
    #             # Move towards the food
    #             angle_to_food = np.arctan2(closest_food.y - self.y, closest_food.x - self.x)
    #             # Gradually adjust direction towards food (30% adjustment per step)
    #             self.direction += (angle_to_food - self.direction) * 0.3
    #         else:
    #             # When not carrying food, check surroundings for pheromones
    #             # Check pheromone levels in three directions: straight ahead, right, and left
    #             pheromone_strength = np.array([
    #                 # Check straight ahead (5 units)
    #                 pheromone_grid[min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction) * 5)))][min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction) * 5)))],
    #                 # Check 0.5 radians to the right
    #                 pheromone_grid[min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction + 0.5) * 5)))][min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction + 0.5) * 5)))],
    #                 # Check 0.5 radians to the left
    #                 pheromone_grid[min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction - 0.5) * 5)))][min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction - 0.5) * 5)))],
    #             ])
                
    #             max_pheromone = np.max(pheromone_strength)
    #             if max_pheromone > 0:
    #                 # If pheromones detected, adjust direction towards strongest pheromone trail
    #                 best_direction = np.argmax(pheromone_strength)
    #                 # Convert array index (0,1,2) to direction adjustment (-0.5, 0, 0.5)
    #                 self.direction += (-0.5 + best_direction * 0.5) * PHEROMONE_INFLUENCE
    #             else:
    #                 # If no pheromones detected, add random wandering
    #                 self.direction += random.uniform(-0.3, 0.3)
        
    #     # Update position based on current direction
    #     # Move 2 units in the calculated direction
    #     self.x += np.cos(self.direction) * 2
    #     self.y += np.sin(self.direction) * 2
        
    #     # Keep ants within bounds
    #     if self.x <= 0 or self.x >= WIDTH-1:
    #         self.direction = np.pi - self.direction
    #     if self.y <= 0 or self.y >= HEIGHT-1:
    #         self.direction = -self.direction

    #     self.x = max(0, min(WIDTH - 1, self.x))
    #     self.y = max(0, min(HEIGHT - 1, self.y))

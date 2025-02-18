import random
import numpy as np
from config.settings import *
from core.colony import Colony

class WorkerAnt:
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
        self.brain = colony.hivemind
        
        # Learning parameters
        self.reward_history = []
        self.action_history = []

    def get_state(self, pheromone_grid, food_sources):
        # Create state vector for neural network input
        # 1. Distance to colony (normalized)
        dist_to_colony = np.hypot(self.x - self.colony_x, self.y - self.colony_y) / np.hypot(WIDTH, HEIGHT)
        
        # 2. Angle to colony (normalized)
        angle_to_colony = (np.arctan2(self.colony_y - self.y, self.colony_x - self.x) - self.direction) / (2 * np.pi)
        
        # 3. Carrying food (binary)
        carrying_food = 1.0 if self.carrying_food else 0.0
        
        # 4-6. Pheromone levels in three directions
        pheromone_levels = np.zeros(3)
        for i, angle in enumerate([0, 0.5, -0.5]): # Checks straight ahead, 0.5 radians left, and 0.5 radians right
            check_x = min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction + angle) * 5)))
            check_y = min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction + angle) * 5)))
            pheromone_levels[i] = pheromone_grid[check_x][check_y] / PHEROMONE_STRENGTH
            
        # 7-8. Distance and angle to nearest food within sight range
        min_food_dist = 1.0  # Default when no food is in sight
        angle_to_food = 0.0  # Default when no food is in sight
        for food in food_sources:
            dist = np.hypot(self.x - food.x, self.y - food.y)
            if dist < ANT_RANGE_OF_SIGHT:
                normalized_dist = dist / ANT_RANGE_OF_SIGHT  # Ranges normalized distance from 1 to 0
                if normalized_dist < min_food_dist:
                    min_food_dist = normalized_dist
                    angle_to_food = (np.arctan2(food.y - self.y, food.x - self.x) - self.direction) / (2 * np.pi) # Range normalized from -0.5 to 0.5
            
        return np.array([dist_to_colony, angle_to_colony, carrying_food, 
                        *pheromone_levels, min_food_dist, angle_to_food])

    def move(self, pheromone_grid, food_sources):
        # Get current state
        state = self.get_state(pheromone_grid, food_sources)
        
        # Get neural network output
        actions = self.brain.forward(state)
        self.action_history.append(actions)
        
        # Add random exploration (epsilon-greedy approach)
        if random.random() < 0.1:  # 10% chance of random movement
            actions = np.random.uniform(-1, 1, 2)
        
        # Apply actions
        turn_left, turn_right = actions

        if self.carrying_food:
            self.drop_pheromones(pheromone_grid)
        
        # Update direction based on neural network output
        self.direction += 0.3 * (turn_right - turn_left)
        
        # Update position
        self.x += np.cos(self.direction) * ANT_SPEED
        self.y += np.sin(self.direction) * ANT_SPEED
        
        # Keep ants within bounds
        if self.x <= 0 or self.x >= WIDTH-1:
            self.direction = np.pi - self.direction
        if self.y <= 0 or self.y >= HEIGHT-1:
            self.direction = -self.direction

        self.x = max(0, min(WIDTH - 1, self.x))
        self.y = max(0, min(HEIGHT - 1, self.y))
        
        # Calculate reward
        reward = self.calculate_reward(food_sources, pheromone_grid)
        self.reward_history.append(reward)
        
        # Learn from experience periodically
        if len(self.reward_history) >= REWARD_RATE:
            self.learn()

    def calculate_reward(self, food_sources, pheromone_grid):
        reward = 0
        
        # Reward for picking up food (only when first picked up)
        if self.carrying_food and len(self.reward_history) > 0 and self.reward_history[-1] <= 0:
            reward += 10.0
            
        # Reward for successfully returning food to colony
        if self.carrying_food and np.hypot(self.x - self.colony_x, self.y - self.colony_y) < Colony.size:
            reward += 500.0
            
        # When carrying food: reward/penalty based on angle to colony
        if self.carrying_food:
            # Get angle to colony relative to current direction
            angle_to_colony = np.arctan2(self.colony_y - self.y, self.colony_x - self.x)
            angle_diff = abs(((angle_to_colony - self.direction + np.pi) % (2 * np.pi)) - np.pi)
            
            # Reward for facing towards colony (angle_diff close to 0)
            # Penalty for facing away (angle_diff close to pi)
            angle_reward = (np.pi - angle_diff) / np.pi  # Ranges from 1 (facing colony) to -1 (facing away)
            reward += 0.5 * angle_reward
            
            # Existing distance-based penalties/rewards
            dist_to_colony = np.hypot(self.x - self.colony_x, self.y - self.colony_y)
            reward -= 0.1 * (dist_to_colony / np.hypot(WIDTH, HEIGHT))
            
            if len(self.reward_history) > 0:
                prev_state = self.get_state(pheromone_grid, food_sources)
                prev_colony_dist = prev_state[0] * np.hypot(WIDTH, HEIGHT)  # Unnormalize the distance
                dist_improvement = prev_colony_dist - dist_to_colony
                reward += 0.2 * dist_improvement  # Reward for getting closer to colony
        
        # When not carrying food: reward/penalty based on angle to nearest visible food
        else:
            min_food_dist = float('inf')
            best_angle_to_food = None
            
            for food in food_sources:
                dist = np.hypot(self.x - food.x, self.y - food.y)
                if dist < ANT_RANGE_OF_SIGHT and dist < min_food_dist:
                    min_food_dist = dist
                    best_angle_to_food = np.arctan2(food.y - self.y, food.x - self.x)
            
            if best_angle_to_food is not None:
                # Calculate angle difference and reward/penalty
                angle_diff = abs(((best_angle_to_food - self.direction + np.pi) % (2 * np.pi)) - np.pi)
                angle_reward = (np.pi - angle_diff) / np.pi
                reward += 0.5 * angle_reward
                
                # Distance improvement reward
                if len(self.reward_history) > 0:
                    prev_state = self.get_state(pheromone_grid, food_sources)
                    prev_food_dist = prev_state[6] * ANT_RANGE_OF_SIGHT
                    dist_improvement = prev_food_dist - min_food_dist
                    reward += 0.2 * dist_improvement
        
        return reward

    def learn(self):
        # Simple policy gradient update
        rewards = np.array(self.reward_history)
        actions = np.array(self.action_history)
        
        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + REWARD_NORMALIZER)
        
        # Calculate gradients and update network
        gradient = np.mean(actions * rewards[:, np.newaxis], axis=0)
        self.brain.update(LEARNING_RATE, gradient)
        
        # Clear history
        self.reward_history = []
        self.action_history = []

    def drop_pheromones(self, pheromone_grid):
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
                if colony.decrement_food():
                    self.carrying_food = True



    # SCRIPTED HEURISTIC BASED MOVEMENT METHOD
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
    #             if distance < ANT_RANGE_OF_SIGHT and distance < min_distance:
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

import random
import numpy as np
from config.settings import *
from core.colony import Colony
from core.utils import *



class WorkerAnt:
    def __init__(self, policy):
        self.direction = random.uniform(0, 2 * np.pi)
        self.carrying_food = False
        self.is_alive = True
        self.policy = policy
        self.location = (random.randint(0, WIDTH), random.randint(0, HEIGHT))

    def observe(self, grid_state):
        # Create state vector for neural network input
        # 1. Distance to colony (normalized)
        dist_to_colony = distance_to_colony(self.location, COLONY_POSITION)
        # 2. Angle to colony (normalized)
        angle_to_colony = compute_angle_to_colony(self.location, COLONY_POSITION)
        
        # 3. Carrying food (binary)
        carrying_food = 1.0 if self.carrying_food else 0.0
        
        # 4-6. Pheromone levels in three directions
        pheromone_levels = np.zeros(3)
        for i, angle in enumerate([0, 0.5, -0.5]): # Checks straight ahead, 0.5 radians left, and 0.5 radians right
            # Create a sector to check for pheromones
            sector_pheromones = []
            for r in range(3, 8, 2):  # Check at distances 3, 5, 7 units away
                for theta in np.linspace(-0.2, 0.2, 3):  # Check 3 angles within ±0.2 radians of main angle
                    check_angle = self.direction + angle + theta
                    check_x = min(WIDTH - 1, max(0, int(self.location[0] + np.cos(check_angle) * r)))
                    check_y = min(HEIGHT - 1, max(0, int(self.location[1] + np.sin(check_angle) * r)))
                    sector_pheromones.append(grid_state[check_x, check_y, PHEREMONE_CHANNEL])
            
            # Take the maximum pheromone value in the sector
            pheromone_levels[i] = max(sector_pheromones) / PHEROMONE_STRENGTH
            
        # 7-8. Distance and angle to nearest food within sight range
        min_food_dist = 1.0  # Default when no food is in sight
        angle_to_food = 0.0  # Default when no food is in sight
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if grid_state[x, y, FOOD_CHANNEL] > 0:
                    dist = np.hypot(self.location[0] - x, self.location[1] - y)
                    if dist < ANT_RANGE_OF_SIGHT:
                        normalized_dist = dist / ANT_RANGE_OF_SIGHT  # Ranges normalized distance from 1 to 0
                        if normalized_dist < min_food_dist:
                            min_food_dist = normalized_dist
                            angle_to_food = (np.arctan2(y - self.location[1], x - self.location[0]) - self.direction) / (2 * np.pi) # Range normalized from -0.5 to 0.5
            
        return np.array([dist_to_colony, angle_to_colony, carrying_food, 
                        *pheromone_levels, min_food_dist, angle_to_food])

    def move(self, grid_state):
        # Get current state
        state = self.observe(grid_state)
        action = None
        # Add random exploration (epsilon-greedy approach)
        if random.random() < 0.1:  # 10% chance of random movement
            random_action = random.choice(['left', 'right'])
            random_magnitude = random.gauss(np.pi, np.pi/4) % (2 * np.pi)
            action = (random_action, random_magnitude)
        else:
            action = self.policy.get_action(state)

        self.drop_pheromones(grid_state)
        
        # Update direction based on neural network output
        self.direction += action[1] if action[0] == 'left' else -action[1]
        
        # Update position
        self.location = (self.location[0] + np.cos(self.direction) * ANT_SPEED, self.location[1] + np.sin(self.direction) * ANT_SPEED)
        
        # Keep ants within bounds
        if self.location[0] <= 0 or self.location[0] >= WIDTH-1:
            self.direction = np.pi - self.direction
        if self.location[1] <= 0 or self.location[1] >= HEIGHT-1:
            self.direction = -self.direction

        self.location = (max(0, min(WIDTH - 1, self.location[0])), max(0, min(HEIGHT - 1, self.location[1])))
        
        if self.location[0] == COLONY_POSITION[0] and self.location[1] == COLONY_POSITION[1]:
            self.carrying_food = True
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Update policy
        # self.policy.learn(state, reward)
        
        return reward

    def calculate_reward(self):
        reward = 0
        
        if self.carrying_food:
            reward += CARRY_REWARD
        else:
            reward -= NOT_CARRY_PENALTY

        reward += (1 - distance_to_colony(self.location, COLONY_POSITION)) * DISTANCE_REWARD_SCALAR
           
        return reward

    def drop_pheromones(self, grid_state):
        if self.carrying_food:
            grid_state[int(self.location[0]),int(self.location[1]),PHEREMONE_CHANNEL] += PHEROMONE_STRENGTH

    def check_food(self, grid_state):
        if grid_state[int(self.location[0]),int(self.location[1]),FOOD_CHANNEL] > 0:
            self.carrying_food = True
            grid_state[int(self.location[0]),int(self.location[1]),FOOD_CHANNEL] -= 1

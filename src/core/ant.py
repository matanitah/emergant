import random
import numpy as np
from config.settings import WIDTH, HEIGHT, PHEROMONE_INFLUENCE, PHEROMONE_STRENGTH
from core.colony import Colony
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

    def move(self, pheromone_grid, food_sources):
        if self.carrying_food:
            # When carrying food, head back to colony
            # Calculate angle between current position and colony using arctangent
            angle_to_colony = np.arctan2(self.colony_y - self.y, self.colony_x - self.x)
            # Gradually adjust direction towards colony (20% adjustment per step)
            self.direction += (angle_to_colony - self.direction) * 0.2
        else:
            # First check for nearby food
            closest_food = None
            min_distance = float('inf')
            
            for food in food_sources:
                distance = np.hypot(self.x - food.x, self.y - food.y)
                # Ants can see food within 50 units
                if distance < 50 and distance < min_distance:
                    closest_food = food
                    min_distance = distance
            
            if closest_food:
                # Move towards the food
                angle_to_food = np.arctan2(closest_food.y - self.y, closest_food.x - self.x)
                # Gradually adjust direction towards food (30% adjustment per step)
                self.direction += (angle_to_food - self.direction) * 0.3
            else:
                # When not carrying food, check surroundings for pheromones
                # Check pheromone levels in three directions: straight ahead, right, and left
                pheromone_strength = np.array([
                    # Check straight ahead (5 units)
                    pheromone_grid[min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction) * 5)))][min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction) * 5)))],
                    # Check 0.5 radians to the right
                    pheromone_grid[min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction + 0.5) * 5)))][min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction + 0.5) * 5)))],
                    # Check 0.5 radians to the left
                    pheromone_grid[min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction - 0.5) * 5)))][min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction - 0.5) * 5)))],
                ])
                
                max_pheromone = np.max(pheromone_strength)
                if max_pheromone > 0:
                    # If pheromones detected, adjust direction towards strongest pheromone trail
                    best_direction = np.argmax(pheromone_strength)
                    # Convert array index (0,1,2) to direction adjustment (-0.5, 0, 0.5)
                    self.direction += (-0.5 + best_direction * 0.5) * PHEROMONE_INFLUENCE
                else:
                    # If no pheromones detected, add random wandering
                    self.direction += random.uniform(-0.3, 0.3)
        
        # Update position based on current direction
        # Move 2 units in the calculated direction
        self.x += np.cos(self.direction) * 2
        self.y += np.sin(self.direction) * 2
        
        # Keep ants within bounds
        if self.x <= 0 or self.x >= WIDTH-1:
            self.direction = np.pi - self.direction
        if self.y <= 0 or self.y >= HEIGHT-1:
            self.direction = -self.direction

        self.x = max(0, min(WIDTH - 1, self.x))
        self.y = max(0, min(HEIGHT - 1, self.y))


    def drop_pheromone(self, pheromone_grid):
        if self.carrying_food:
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

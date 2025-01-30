import random
import numpy as np
from config.settings import WIDTH, HEIGHT, PHEROMONE_INFLUENCE, PHEROMONE_STRENGTH
class Ant:
    def __init__(self, colony):
        self.x = colony.x
        self.y = colony.y
        self.colony = colony
        self.colony_x = colony.x
        self.colony_y = colony.y
        self.direction = random.uniform(0, 2 * np.pi)
        self.carrying_food = False

    def move(self, pheromone_grid):
        if self.carrying_food:
            # Move towards the colony
            angle_to_colony = np.arctan2(self.colony_y - self.y, self.colony_x - self.x)
            self.direction += (angle_to_colony - self.direction) * 0.2
        else:
            # Move influenced by pheromones
            pheromone_strength = np.array([
                pheromone_grid[min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction) * 5)))][min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction) * 5)))],
                pheromone_grid[min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction + 0.5) * 5)))][min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction + 0.5) * 5)))],
                pheromone_grid[min(WIDTH - 1, max(0, int(self.x + np.cos(self.direction - 0.5) * 5)))][min(HEIGHT - 1, max(0, int(self.y + np.sin(self.direction - 0.5) * 5)))],
            ])
            best_direction = np.argmax(pheromone_strength)
            if pheromone_strength[best_direction] > 0:
                self.direction += (-0.5 + best_direction * 0.5) * PHEROMONE_INFLUENCE

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
            if not self.carrying_food and np.hypot(self.x - food.x, self.y - food.y) < 5:
                self.carrying_food = True
                food_sources.remove(food)
                return

    def check_colony(self):
        if self.carrying_food and np.hypot(self.x - self.colony_x, self.y - self.colony_y) < 5:
            self.colony.increment_food()
            self.carrying_food = False
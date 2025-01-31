import os
import pygame
import random
import numpy as np
from config.settings import WIDTH, HEIGHT, ANT_COUNT, FOOD_COUNT, PHEROMONE_DECAY, COLORS
from core.ant import Ant
from core.food import Food
from core.colony import Colony
class Simualator:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.colony = Colony(WIDTH // 2, HEIGHT // 2)

        self.ants = [Ant(self.colony) for _ in range(ANT_COUNT)]
        self.food_sources = [Food() for _ in range(FOOD_COUNT)]
        self.pheromone_grid = np.zeros((WIDTH, HEIGHT))
        self.running = True

    def update(self):
        for ant in self.ants:
            ant.move(self.pheromone_grid)
            ant.check_food(self.food_sources)
            ant.check_colony()
            ant.drop_pheromone(self.pheromone_grid)
        
        self.pheromone_grid *= PHEROMONE_DECAY  # Pheromones decay over time

    def draw(self):
        self.screen.fill(COLORS['BLACK'])
        
        # Draw pheromones
        for x in range(WIDTH):
            for y in range(HEIGHT):
                intensity = min(255, int(self.pheromone_grid[x][y] * 10))
                if intensity > 0:
                    self.screen.set_at((x, y), (intensity, intensity, intensity))
        
        # Draw colony
        pygame.draw.circle(self.screen, COLORS['WHITE'], (self.colony.x, self.colony.y), 10)

        # Draw food
        for food in self.food_sources:
            pygame.draw.circle(self.screen, COLORS['GREEN'], (food.x, food.y), food.size // 5)
        
        # Draw ants
        for ant in self.ants:
            color = COLORS['RED'] if ant.carrying_food else COLORS['BLUE']
            pygame.draw.circle(self.screen, color, (int(ant.x), int(ant.y)), 2)
        
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.update()
            self.draw()
            self.clock.tick(30)

        pygame.quit()

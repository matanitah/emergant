import os
import pygame
import random
import numpy as np
from config.settings import *
from core.worker_ant import WorkerAnt
from core.food import Food
from core.colony import Colony
from core.policies.policy import Policy
from core.models.baseline_model import BaselineAntPolicy
from core.utils import *

class Simulator:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.colony = Colony(COLONY_POSITION)  # Left side

        self.worker_ants = [WorkerAnt(policy=Policy(model=BaselineAntPolicy([10]))) for _ in range(ANT_COUNT_PER_COLONY)]
        
        self.grid_state = self.generate_grid_state()
        self.running = True
        self.step = 0
        
    def generate_grid_state(self):
        grid_state = np.zeros((WIDTH, HEIGHT, 2))
        
        # Initialize a fixed number of random food source locations
        for _ in range(FOOD_COUNT):
            # Generate random coordinates
            x = np.random.randint(50, WIDTH-50)
            y = np.random.randint(50, HEIGHT-50)
            
            # Set food presence (channel 0) to 1 at this location
            grid_state[x, y, FOOD_CHANNEL] = np.random.randint(1, 6)
        
        return grid_state

    def update(self):
        # First check if we should end the game
        if np.all(self.grid_state[:, :, FOOD_CHANNEL] == 0):
            self.handle_end_game()
            return
        
        for ant in self.worker_ants:
            ant.move(self.grid_state)
            
        self.grid_state[:, :, PHEREMONE_CHANNEL] *= PHEROMONE_DECAY  # Pheromones decay over time

    def draw(self):
        self.screen.fill(COLORS['BLACK'])
        
        # Draw pheromones
        for x in range(WIDTH):
            for y in range(HEIGHT):
                intensity = min(255, int(self.grid_state[x, y, PHEREMONE_CHANNEL] * 10))
                if intensity > 0:
                    self.screen.set_at((x, y), (intensity, intensity, intensity))
        
        # Draw both colonies
        pygame.draw.circle(self.screen, COLORS['LIGHT_BLUE'], COLONY_POSITION, Colony.size)

        # Display food counts
        font = pygame.font.Font(None, 36)
        colony_text = font.render(f"Colony: {self.colony.food_count}", True, COLORS['LIGHT_BLUE'])
        self.screen.blit(colony_text, (10, 10))  # Top left
        
        # Draw food
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if self.grid_state[x, y, FOOD_CHANNEL] > 0:
                    pygame.draw.circle(self.screen, COLORS['YELLOW'], (x, y), self.grid_state[x, y, FOOD_CHANNEL])
        
        # Draw ants with colony-specific colors
        for ant in self.worker_ants:
            pygame.draw.circle(self.screen, COLORS['GREEN'], (int(ant.location[0]), int(ant.location[1])), 2)
        
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.save_weights()
                    self.running = False
            start_time = pygame.time.get_ticks()
            self.update()
            end_time = pygame.time.get_ticks()
            print(f"Update took {end_time - start_time}ms")
            self.draw()
            pygame.display.flip()
        pygame.quit()

    def handle_end_game(self):
        # Reset the game
        self.reset_game()
        self.step += 1
        
    def reset_game(self):
        # Reset food sources
        self.grid_state = self.generate_grid_state()
        
        # Reset colony food counts
        self.colony.food_count = 0
        
        # Reset ants
        self.worker_ants = [WorkerAnt(policy=Policy(model=BaselineAntPolicy([10]))) for _ in range(ANT_COUNT_PER_COLONY)]

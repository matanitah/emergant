import os
import pygame
import random
import numpy as np
from config.settings import WIDTH, HEIGHT, ANT_COUNT, FOOD_COUNT, PHEROMONE_DECAY, COLORS, MAX_ANTS_PER_COLONY, REPRODUCTION_COOLDOWN
from core.ant import Ant
from core.food import Food
from core.colony import Colony
class Simulator:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.last_reproduction = {
            1: 0,
            2: 0
        }
        self.colony1 = Colony(WIDTH // 4, HEIGHT // 2, 1)  # Left side
        self.colony2 = Colony(3 * WIDTH // 4, HEIGHT // 2, 2)  # Right side

        # Load weights before creating ants
        self.load_weights()

        self.ants1 = [Ant(self.colony1, colony_id=1) for _ in range(ANT_COUNT // 2)]
        self.ants2 = [Ant(self.colony2, colony_id=2) for _ in range(ANT_COUNT // 2)]
        self.ants = self.ants1 + self.ants2  # Combined list for easier iteration

        self.food_sources = [Food() for _ in range(FOOD_COUNT)]
        self.pheromone_grid = np.zeros((WIDTH, HEIGHT))
        self.running = True

    def update(self):
        # Remove dead ants
        self.ants1 = [ant for ant in self.ants1 if ant.is_alive]
        self.ants2 = [ant for ant in self.ants2 if ant.is_alive]
        self.ants = self.ants1 + self.ants2

        for ant in self.ants:
            ant.move(self.pheromone_grid, self.food_sources)
            ant.check_food(self.food_sources)
            
            # Check for colonies nearby
            ant.check_colony(self.colony1)
            ant.check_colony(self.colony2)
                        
            # Combat: Check for nearby enemy ants
            # self.check_combat(ant)
        self.handle_reproduction()
        self.pheromone_grid *= PHEROMONE_DECAY  # Pheromones decay over time

    def check_combat(self, ant):
        enemy_ants = self.ants2 if ant.colony_id == 1 else self.ants1
        
        for enemy in enemy_ants:
            distance = np.sqrt((ant.x - enemy.x)**2 + (ant.y - enemy.y)**2)
            if distance < 5:  # Combat range
                # Simple combat resolution: random chance to kill
                if random.random() < 0.5:
                    enemy.is_alive = False
                else:
                    ant.is_alive = False
                break

    def handle_reproduction(self):
        current_time = pygame.time.get_ticks()
        
        # Check colony 1 reproduction
        if (len(self.ants1) < MAX_ANTS_PER_COLONY and
            current_time - self.last_reproduction[1] > REPRODUCTION_COOLDOWN and
            self.colony1.food_count >= 5):  # Requires 5 food for reproduction
            
            self.colony1.food_count -= 5
            new_ant = Ant(self.colony1, colony_id=1)
            self.ants1.append(new_ant)
            self.ants = self.ants1 + self.ants2
            self.last_reproduction[1] = current_time
        
        # Check colony 2 reproduction
        if (len(self.ants2) < MAX_ANTS_PER_COLONY and 
            current_time - self.last_reproduction[2] > REPRODUCTION_COOLDOWN and
            self.colony2.food_count >= 5):  # Requires 5 food for reproduction
            
            self.colony2.food_count -= 5
            new_ant = Ant(self.colony2, colony_id=2)
            self.ants2.append(new_ant)
            self.ants = self.ants1 + self.ants2
            self.last_reproduction[2] = current_time

    def draw(self):
        self.screen.fill(COLORS['BLACK'])
        
        # Draw pheromones
        for x in range(WIDTH):
            for y in range(HEIGHT):
                intensity = min(255, int(self.pheromone_grid[x][y] * 10))
                if intensity > 0:
                    self.screen.set_at((x, y), (intensity, intensity, intensity))
        
        # Draw both colonies
        pygame.draw.circle(self.screen, COLORS['LIGHT_BLUE'], (self.colony1.x, self.colony1.y), Colony.size)
        pygame.draw.circle(self.screen, COLORS['RED'], (self.colony2.x, self.colony2.y), Colony.size)

        # Display food counts
        font = pygame.font.Font(None, 36)
        colony1_text = font.render(f"Colony 1: {self.colony1.total_food_collected}", True, COLORS['LIGHT_BLUE'])
        colony2_text = font.render(f"Colony 2: {self.colony2.total_food_collected}", True, COLORS['RED'])
        self.screen.blit(colony1_text, (10, 10))  # Top left
        self.screen.blit(colony2_text, (10, 50))  # Below colony 1 text
        
        # Draw food
        for food in self.food_sources:
            pygame.draw.circle(self.screen, COLORS['YELLOW'], (food.x, food.y), food.size)
        
        # Draw ants with colony-specific colors
        for ant in self.ants:
            if ant.is_alive:
                if ant.colony_id == 1:
                    color = COLORS['GREEN'] if ant.carrying_food else COLORS['BLUE']
                else:
                    color = COLORS['ORANGE'] if ant.carrying_food else COLORS['BROWN']
                pygame.draw.circle(self.screen, color, (int(ant.x), int(ant.y)), 2)
        
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.save_weights()
                    self.running = False
            
            self.update()
            self.draw()

        pygame.quit()

    def save_weights(self):
        """Saves the neural network weights for both colonies to text files."""
        # Create weights directory if it doesn't exist
        os.makedirs('weights', exist_ok=True)
        
        # Save weights for colony 1
        with open(f'weights/colony1_weights.txt', 'w') as f:
            f.write("Weights1:\n" + str(self.colony1.hivemind.weights1.tolist()) + "\n")
            f.write("Weights2:\n" + str(self.colony1.hivemind.weights2.tolist()) + "\n")
            f.write("Weights3:\n" + str(self.colony1.hivemind.weights3.tolist()))
            
        # Save weights for colony 2
        with open(f'weights/colony2_weights.txt', 'w') as f:
            f.write("Weights1:\n" + str(self.colony2.hivemind.weights1.tolist()) + "\n")
            f.write("Weights2:\n" + str(self.colony2.hivemind.weights2.tolist()) + "\n")
            f.write("Weights3:\n" + str(self.colony2.hivemind.weights3.tolist()))

    def load_weights(self):
        """Loads neural network weights for both colonies from text files."""
        try:
            # Load weights for colony 1
            with open('weights/colony1_weights.txt', 'r') as f:
                content = f.read()
                # Extract each weight matrix
                for i, prefix in enumerate(['Weights1:', 'Weights2:', 'Weights3:']):
                    start = content.find(prefix) + len(prefix)
                    end = content.find('Weights', start) if i < 2 else len(content)
                    weight_str = content[start:end].strip()
                    weight_matrix = np.array(eval(weight_str))
                    if i == 0:
                        self.colony1.hivemind.weights1 = weight_matrix
                    elif i == 1:
                        self.colony1.hivemind.weights2 = weight_matrix
                    else:
                        self.colony1.hivemind.weights3 = weight_matrix

            # Load weights for colony 2
            with open('weights/colony2_weights.txt', 'r') as f:
                content = f.read()
                # Extract each weight matrix
                for i, prefix in enumerate(['Weights1:', 'Weights2:', 'Weights3:']):
                    start = content.find(prefix) + len(prefix)
                    end = content.find('Weights', start) if i < 2 else len(content)
                    weight_str = content[start:end].strip()
                    weight_matrix = np.array(eval(weight_str))
                    if i == 0:
                        self.colony2.hivemind.weights1 = weight_matrix
                    elif i == 1:
                        self.colony2.hivemind.weights2 = weight_matrix
                    else:
                        self.colony2.hivemind.weights3 = weight_matrix
                        
        except FileNotFoundError:
            print("No weights files found. Starting with random weights.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Starting with random weights.")

import os
import pygame
import random
import numpy as np
from config.settings import *
from core.worker_ant import WorkerAnt
from core.food import Food
from core.colony import Colony
from core.neural_net import NeuralNetwork

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
        self.colony1 = Colony(COLONY_1_POS[0], COLONY_1_POS[1], 1, [6,6,6])  # Left side
        self.colony2 = Colony(COLONY_2_POS[0],COLONY_2_POS[1], 2)  # Right side

        # Load weights before creating ants
        self.load_weights()

        self.worker_ants_1 = [WorkerAnt(self.colony1, colony_id=1) for _ in range(ANT_COUNT_PER_COLONY)]
        self.worker_ants_2 = [WorkerAnt(self.colony2, colony_id=2) for _ in range(ANT_COUNT_PER_COLONY)]
        self.ants = self.worker_ants_1 + self.worker_ants_2  # Combined list for easier iteration

        self.food_sources = [Food() for _ in range(FOOD_COUNT)]
        self.pheromone_grid = np.zeros((WIDTH, HEIGHT))
        self.running = True
        self.generation = 1

    def update(self):
        # First check if we should end the game
        if len(self.food_sources) == 0:
            self.handle_end_game()
            return
        
        # Remove dead ants
        self.worker_ants_1 = [ant for ant in self.worker_ants_1 if ant.is_alive]
        self.worker_ants_2 = [ant for ant in self.worker_ants_2 if ant.is_alive]
        self.ants = self.worker_ants_1 + self.worker_ants_2

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
        enemy_ants = self.worker_ants_2 if ant.colony_id == 1 else self.worker_ants_1
        
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
        if (len(self.worker_ants_1) < MAX_ANTS_PER_COLONY and
            current_time - self.last_reproduction[1] > REPRODUCTION_COOLDOWN and
            self.colony1.food_count >= 5):  # Requires 5 food for reproduction
            
            self.colony1.food_count -= 5
            new_ant = WorkerAnt(self.colony1, colony_id=1)
            self.worker_ants_1.append(new_ant)
            self.ants = self.worker_ants_1 + self.worker_ants_2
            self.last_reproduction[1] = current_time
        
        # Check colony 2 reproduction
        if (len(self.worker_ants_2) < MAX_ANTS_PER_COLONY and 
            current_time - self.last_reproduction[2] > REPRODUCTION_COOLDOWN and
            self.colony2.food_count >= 5):  # Requires 5 food for reproduction
            
            self.colony2.food_count -= 5
            new_ant = WorkerAnt(self.colony2, colony_id=2)
            self.worker_ants_2.append(new_ant)
            self.ants = self.worker_ants_1 + self.worker_ants_2
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
        """Saves the neural network weights and architecture for both colonies to text files."""
        os.makedirs('weights', exist_ok=True)
        
        # Save weights for colony 1
        with open(f'weights/colony1_weights_{self.generation}.txt', 'w') as f:
            # Save layer sizes
            f.write(f"LayerSizes:{self.colony1.hivemind.layer_sizes}\n")
            # Save weights for each layer
            for i, weights in enumerate(self.colony1.hivemind.weights):
                f.write(f"Weights{i}:\n{weights.tolist()}\n")
        
        # Save weights for colony 2
        with open(f'weights/colony2_weights_{self.generation}.txt', 'w') as f:
            # Save layer sizes
            f.write(f"LayerSizes:{self.colony2.hivemind.layer_sizes}\n")
            # Save weights for each layer
            for i, weights in enumerate(self.colony2.hivemind.weights):
                f.write(f"Weights{i}:\n{weights.tolist()}\n")

    def load_weights(self):
        """Loads neural network weights and architecture for both colonies from text files."""
        try:
            # Find the highest generation number from existing weight files
            generation = 0
            if os.path.exists('weights'):
                weight_files = os.listdir('weights')
                for file in weight_files:
                    if file.startswith('colony1_weights_') and file.endswith('.txt'):
                        gen = int(file.split('_')[-1].split('.')[0])
                        generation = max(generation, gen)
            self.generation = generation
            print(f"Starting at generation {self.generation}")
            # Load weights for colony 1
            with open(f'weights/colony1_weights_{generation}.txt', 'r') as f:
                content = f.read()
                
                # Extract layer sizes
                layer_start = content.find('LayerSizes:') + len('LayerSizes:')
                layer_end = content.find('\n', layer_start)
                layer_sizes = eval(content[layer_start:layer_end])
                
                # Initialize network with loaded architecture
                self.colony1.hivemind = NeuralNetwork(layer_sizes)
                
                # Load weights for each layer
                for i in range(len(layer_sizes) - 1):
                    prefix = f'Weights{i}:'
                    start = content.find(prefix) + len(prefix)
                    end = content.find('Weights', start) if i < len(layer_sizes) - 2 else len(content)
                    weight_str = content[start:end].strip()
                    self.colony1.hivemind.weights[i] = np.array(eval(weight_str))

            # Load weights for colony 2
            with open(f'weights/colony2_weights_{generation}.txt', 'r') as f:
                content = f.read()
                
                # Extract layer sizes
                layer_start = content.find('LayerSizes:') + len('LayerSizes:')
                layer_end = content.find('\n', layer_start)
                layer_sizes = eval(content[layer_start:layer_end])
                
                # Initialize network with loaded architecture
                self.colony2.hivemind = NeuralNetwork(layer_sizes)
                
                # Load weights for each layer
                for i in range(len(layer_sizes) - 1):
                    prefix = f'Weights{i}:'
                    start = content.find(prefix) + len(prefix)
                    end = content.find('Weights', start) if i < len(layer_sizes) - 2 else len(content)
                    weight_str = content[start:end].strip()
                    self.colony2.hivemind.weights[i] = np.array(eval(weight_str))
                    
        except FileNotFoundError:
            print("No weights files found. Starting with random weights.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Starting with random weights.")

    def handle_end_game(self):
        print("GAME OVER!!!")
        # Determine winner
        if self.colony1.total_food_collected > self.colony2.total_food_collected:
            winner, loser = self.colony1, self.colony2
        else:
            winner, loser = self.colony2, self.colony1
        
        print(f"WINNER is COLONY {winner.id} with {winner.hivemind.num_layers - 2} hidden layers of sizes {winner.hivemind.hidden_sizes}")
        
        self.save_weights()

        # Mutate the loser's neural network
        loser.mutate()
        winner.init_hivemind()
        
        # Reset the game
        self.reset_game()
        self.generation += 1
        
    def reset_game(self):
        # Reset food sources
        self.food_sources = [Food() for _ in range(FOOD_COUNT)]
        
        # Reset colony food counts
        self.colony1.food_count = 0
        self.colony2.food_count = 0
        self.colony1.total_food_collected = 0
        self.colony2.total_food_collected = 0
        
        # Reset ants
        self.worker_ants_1 = [WorkerAnt(self.colony1, colony_id=1) 
                             for _ in range(ANT_COUNT_PER_COLONY)]
        self.worker_ants_2 = [WorkerAnt(self.colony2, colony_id=2) 
                             for _ in range(ANT_COUNT_PER_COLONY)]
        self.ants = self.worker_ants_1 + self.worker_ants_2

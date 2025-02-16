import pygame
import numpy as np
import ast
import os

class NeuronVisualizer:
    def __init__(self, width=1200, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Neural Network Visualization")
        self.clock = pygame.time.Clock()
        
        # Add mouse position tracking
        self.mouse_pos = (0, 0)
        
        # Colors
        self.colors = {
            'background': (30, 30, 30),
            'neuron': (200, 200, 200),
            'text': (255, 255, 255),
            'positive_weight': (0, 255, 0),
            'negative_weight': (255, 0, 0)
        }
        
        # Network structure
        self.layer_sizes = [8, 8, 8, 2]  # [input, hidden1, hidden2, output]
        self.weights = self.load_weights()
        self.selected_colony = 1
        self.font = pygame.font.Font(None, 24)
        
    def load_weights(self):
        """Load weights from the saved files."""
        weights = {'colony1': [], 'colony2': []}
        
        for colony in [1, 2]:
            try:
                with open(f'weights/colony{colony}_weights.txt', 'r') as f:
                    content = f.read()
                    # Extract each weight matrix
                    for i in range(1, 4):
                        start = content.find(f"Weights{i}:\n") + len(f"Weights{i}:\n")
                        end = content.find("Weights", start) if i < 3 else len(content)
                        weight_str = content[start:end].strip()
                        weight_matrix = np.array(ast.literal_eval(weight_str))
                        weights[f'colony{colony}'].append(weight_matrix)
            except FileNotFoundError:
                print(f"Warning: No weights file found for colony {colony}")
                weights[f'colony{colony}'] = [
                    np.zeros((8, 8)),
                    np.zeros((8, 8)),
                    np.zeros((8, 3))
                ]
        
        return weights
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate the distance from a point to a line segment."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate the distance from point to line
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = ((y2-y1)**2 + (x2-x1)**2)**0.5
        
        if denominator == 0:
            return ((x0-x1)**2 + (y0-y1)**2)**0.5
            
        distance = numerator/denominator
        
        # Check if the point is beyond the line segment
        dot_product = ((x0-x1)*(x2-x1) + (y0-y1)*(y2-y1)) / ((x2-x1)**2 + (y2-y1)**2)
        if dot_product < 0:
            return ((x0-x1)**2 + (y0-y1)**2)**0.5
        elif dot_product > 1:
            return ((x0-x2)**2 + (y0-y2)**2)**0.5
            
        return distance

    def draw_network(self):
        self.screen.fill(self.colors['background'])
        
        # Calculate positions
        layer_spacing = self.width / (len(self.layer_sizes) + 1)
        max_neurons = max(self.layer_sizes)
        neuron_positions = []
        
        # Draw neurons and store their positions
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            neuron_spacing = self.height / (layer_size + 1)
            layer_positions = []
            
            for neuron_idx in range(layer_size):
                x = (layer_idx + 1) * layer_spacing
                y = (neuron_idx + 1) * neuron_spacing
                
                # Draw neuron
                pygame.draw.circle(self.screen, self.colors['neuron'], (x, y), 10)
                
                # Label neurons
                if layer_idx == 0:
                    label = ["dist_colony", "angle_colony", "carrying_food", 
                            "phero_front", "phero_right", "phero_left",
                            "dist_food", "angle_food"][neuron_idx]
                elif layer_idx == len(self.layer_sizes) - 1:
                    label = ["turn_left", "turn_right", "drop_phero"][neuron_idx]
                else:
                    label = f"H{layer_idx}_{neuron_idx}"
                
                text = self.font.render(label, True, self.colors['text'])
                text_rect = text.get_rect(left=x + 15, centery=y)
                self.screen.blit(text, text_rect)
                
                layer_positions.append((x, y))
            neuron_positions.append(layer_positions)
        
        # Draw weights
        weights = self.weights[f'colony{self.selected_colony}']
        for layer_idx in range(len(weights)):
            for i in range(len(neuron_positions[layer_idx])):
                for j in range(len(neuron_positions[layer_idx + 1])):
                    weight = weights[layer_idx][i][j]
                    if abs(weight) > 0.1:  # Only draw significant weights
                        start = neuron_positions[layer_idx][i]
                        end = neuron_positions[layer_idx + 1][j]
                        color = self.colors['positive_weight'] if weight > 0 else self.colors['negative_weight']
                        alpha = int(min(255, abs(weight) * 255))
                        pygame.draw.line(self.screen, (*color[:3], alpha), start, end, max(1, int(abs(weight) * 3)))
                        
                        # Only show weight value if mouse is near the line
                        if self.point_to_line_distance(self.mouse_pos, start, end) < 10:
                            mid_x = (start[0] + end[0]) / 2
                            mid_y = (start[1] + end[1]) / 2
                            weight_text = self.font.render(f"{weight:.2f}", True, self.colors['text'])
                            text_rect = weight_text.get_rect(center=(mid_x, mid_y))
                            # Add background for better readability
                            padding = 2
                            pygame.draw.rect(self.screen, self.colors['background'], 
                                           (text_rect.x - padding, text_rect.y - padding,
                                            text_rect.width + 2*padding, text_rect.height + 2*padding))
                            self.screen.blit(weight_text, text_rect)
        
        # Draw colony selector
        text = self.font.render(f"Viewing Colony {self.selected_colony} (Press SPACE to switch)", True, self.colors['text'])
        self.screen.blit(text, (10, 10))
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.selected_colony = 3 - self.selected_colony  # Toggle between 1 and 2
                    elif event.key == pygame.K_r:
                        self.weights = self.load_weights()  # Reload weights
            
            # Update mouse position
            self.mouse_pos = pygame.mouse.get_pos()
            
            self.draw_network()
            self.clock.tick(30)
        
        pygame.quit()

if __name__ == "__main__":
    visualizer = NeuronVisualizer()
    visualizer.run()

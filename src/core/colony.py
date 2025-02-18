import numpy as np
from core.neural_net import NeuralNetwork
class Colony:
    size = 10
    def __init__(self, x, y, id, hidden_sizes=[8, 8]):
        self.id = id
        self.x = x
        self.y = y
        self.hidden_sizes = hidden_sizes
        self.food_count = 0
        self.total_food_collected = 0
        # Create layer sizes list: input_layer + hidden_layers + output_layer
        self.init_hivemind()

    def init_hivemind(self):
        self.hivemind = NeuralNetwork(self.hidden_sizes)

    def mutate(self):
        # Randomly choose mutation type
        mutation_type = np.random.choice(['add_layer', 'remove_layer', 'modify_size'])
        
        if mutation_type == 'add_layer' and len(self.hidden_sizes) < 6:
            # Add a new hidden layer with random size between 2-12 neurons
            insert_position = np.random.randint(0, len(self.hidden_sizes) + 1)
            new_layer_size = np.random.randint(2, 17)
            self.hidden_sizes.insert(insert_position, new_layer_size)
            
        elif mutation_type == 'remove_layer' and len(self.hidden_sizes) > 2:
            # Remove a random hidden layer
            remove_position = np.random.randint(0, len(self.hidden_sizes))
            self.hidden_sizes.pop(remove_position)
            
        else:  # modify_size
            # Select random layer to modify
            layer_idx = np.random.randint(0, len(self.hidden_sizes))
            # Randomly increase or decrease size by 1-3 neurons
            delta = np.random.randint(-3, 4)
            # Ensure layer size stays reasonable (minimum 3 neurons)
            self.hidden_sizes[layer_idx] = max(3, self.hidden_sizes[layer_idx] + delta)

    def increment_food(self):
        self.food_count += 1
        self.total_food_collected += 1
        
    def decrement_food(self):
        if self.food_count > 0:
            self.food_count -= 1
            return True
        else:
            return False
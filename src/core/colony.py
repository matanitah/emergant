from core.neural_net import NeuralNetwork
class Colony:
    size = 10
    def __init__(self, x, y, id, hidden_sizes=[8, 8]):
        self.id = id
        self.x = x
        self.y = y
        self.food_count = 0
        self.total_food_collected = 0
        # Create layer sizes list: input_layer + hidden_layers + output_layer
        layer_sizes = [8] + hidden_sizes + [2]  # 8 inputs, 2 outputs
        self.hivemind = NeuralNetwork(layer_sizes)
    
    def increment_food(self):
        self.food_count += 1
        self.total_food_collected += 1
        
    def decrement_food(self):
        if self.food_count > 0:
            self.food_count -= 1
            return True
        else:
            return False
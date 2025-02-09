from core.neural_net import NeuralNetwork
class Colony:
    size = 10
    def __init__(self, x, y, id, hidden_size1=8, hidden_size2=8):
        self.id = id
        self.x = x
        self.y = y
        self.food_count = 0
        self.total_food_collected = 0
        self.hivemind = NeuralNetwork(hidden_size1, hidden_size2)
    
    def increment_food(self):
        self.food_count += 1
        self.total_food_collected += 1
        
    def decrement_food(self):
        self.food_count -= 1
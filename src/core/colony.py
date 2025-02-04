from core.neural_net import NeuralNetwork
class Colony:
    size = 10
    def __init__(self, x, y, id):
        self.id = id
        self.x = x
        self.y = y
        self.food_count = 0
        self.hivemind = NeuralNetwork()
    
    def increment_food(self):
        self.food_count += 1
        
    def decrement_food(self):
        self.food_count -= 1
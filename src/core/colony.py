import numpy as np
from config.settings import COLONY_POSITION

class Colony:
    size = 10
    def __init__(self, position):
        self.position = position
        self.food_count = 0

    def increment_food(self):
        self.food_count += 1
        
    def decrement_food(self):
        if self.food_count > 0:
            self.food_count -= 1
            return True
        else:
            return False
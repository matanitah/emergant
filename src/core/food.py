import random
from config.settings import WIDTH, HEIGHT
from math import sqrt

class Food:
    def __init__(self):
        self.food_count = random.randint(50, 250)
        self.size = self.food_count // 25
        
        # Minimum distance from colonies
        MIN_DISTANCE = 100  # Adjust this value as needed
        
        # Keep generating positions until we find one that's far enough from both colonies
        while True:
            self.x = random.randint(50, WIDTH - 50)
            self.y = random.randint(50, HEIGHT - 50)
            
            # Check distance from both colonies
            dist1 = self._distance_to_point(WIDTH // 4, HEIGHT // 2)  # Colony 1
            dist2 = self._distance_to_point(3 * WIDTH // 4, HEIGHT // 2)  # Colony 2
            
            if dist1 >= MIN_DISTANCE and dist2 >= MIN_DISTANCE:
                break
    
    def _distance_to_point(self, x, y):
        return sqrt((self.x - x)**2 + (self.y - y)**2)
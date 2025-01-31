import random
from config.settings import WIDTH, HEIGHT
class Food:
    def __init__(self):
        self.food_count = random.randint(50, 250)
        self.size = self.food_count // 25
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
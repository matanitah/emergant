import random
from config.settings import WIDTH, HEIGHT
class Food:
    def __init__(self):
        self.size = random.randint(10, 50)
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
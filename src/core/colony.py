class Colony:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.food_gathered = 0
    
    def increment_food(self):
        self.food_gathered += 1
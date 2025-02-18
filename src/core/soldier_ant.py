import numpy as np
import random
from config.settings import *

class SoldierAnt:
    def __init__(self, colony, colony_id):
        self.x = colony.x
        self.y = colony.y
        self.colony = colony
        self.colony_x = colony.x 
        self.colony_y = colony.y
        self.direction = random.uniform(0, 2 * np.pi)
        self.colony_id = colony_id
        self.is_alive = True
        self.attack_range = 5
        self.attack_damage = 2
        self.health = 10
        self.speed = 2

    def attack_enemies(self, enemy_colony, enemy_ants):
        # If close to enemy colony, attack it
        dist_to_enemy = np.hypot(self.x - enemy_colony.x, self.y - enemy_colony.y)
        if dist_to_enemy < self.attack_range:
            enemy_colony.take_damage()
            self.is_alive = False
            return

        # Otherwise move towards enemy colony
        angle_to_enemy = np.arctan2(enemy_colony.y - self.y, enemy_colony.x - self.x)
        
        # Add some randomness to movement
        angle_to_enemy += random.uniform(-0.2, 0.2)
        
        self.x += np.cos(angle_to_enemy) * self.speed
        self.y += np.sin(angle_to_enemy) * self.speed

        # Keep ants within bounds
        self.x = max(0, min(WIDTH - 1, self.x))
        self.y = max(0, min(HEIGHT - 1, self.y))


    def move(self, enemy_colony, enemy_ants):
            self.attack_enemies(enemy_colony, enemy_ants)

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.is_alive = False
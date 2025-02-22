import numpy as np
from config.settings import *

MAX_DISTANCE = np.hypot(WIDTH, HEIGHT)
FOOD_CHANNEL = 0
PHEREMONE_CHANNEL = 1

def distance_to_colony(location, colony_position):
    return np.hypot(location[0] - colony_position[0], location[1] - colony_position[1]) / MAX_DISTANCE

def compute_angle_to_colony(location, colony_position):
    return np.arctan2(colony_position[1] - location[1], colony_position[0] - location[0])

def is_in_sight_range(location, target_location):
    return np.hypot(location[0] - target_location[0], location[1] - target_location[1]) < ANT_RANGE_OF_SIGHT
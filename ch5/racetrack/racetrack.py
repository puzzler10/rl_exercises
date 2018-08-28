#%% Setup
import pandas as pd, numpy as np, os
from math import floor
from collections import namedtuple
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

#%%
path = '/Users/tomroth/Dropbox/Reinforcement Learning/reinforcement_learning_exercises/ch5/racetrack/'

track1 = pd.read_csv(path + 'racetrack1.csv', header=None)
track2 = pd.read_csv(path + 'racetrack2.csv', header=None)

def get_landmarks(track):
    """Extract start, finish and boundary coordinates for a track"""
    Loc = namedtuple('Loc', 'row col')
    start_coords = []
    finish_coords = []
    boundary_coords = []
    for row in range(track.shape[0]):
        for col in range(track.shape[1]):
            if   track.iloc[row,col] == 'S':  start_coords.append(Loc(row,col))
            elif track.iloc[row,col] == 'F':  finish_coords.append(Loc(row,col))
            elif track.iloc[row,col] == 'B':  boundary_coords.append(Loc(row,col))
    Coords = namedtuple('Coords', 'start finish boundary')
    return Coords(start_coords, finish_coords, boundary_coords)

def pick_random_start_location(x):
    """Given a list of start line coordinates, pick one at random
    x: list of tuples"""
    idx = floor(np.random.uniform(0, len(x)))
    return x[idx]

def check_boundary_hit(loc, bnds):
    """return True if loc would hit the boundary, false otherwise
    bnds: boundaries"""
    col_locs = [o.col for o in bnds if o.row == loc.row]
    min_col, max_col =  min(col_locs), max(col_locs)
    # In this case the car has hit the side walls
    if loc.col <= min_col or loc.col >= max_col:     return True
    # Else, we could still hit the boundary on the top row,
    # or on a bit that juts out
    for col in col_locs:
        if loc.col == col:          return True
    return False


def change_velocity(vel, dvel, p=0.1):
    """Velocity must be:
            - nonnegative at all times
            - not (0,0), except at the start line
            - have each component between 0 and 4 inclusive
        With probability p dvel= 0, regardless of what was passed into dvel
    vel: current velocity
    dvel: change in velocity"""
    Velocity = namedtuple('Velocity', 'rowChange colChange')
    if np.random.uniform < p:
        dvel = (0,0)
    new_vel = tuple(np.add((1,2), (3,4)))



#%%
# lm = landmarks
Velocity = namedtuple('Velocity', 'rowChange colChange')
Loc = namedtuple('Loc', 'row col')
actions = list(product([-1,0,1], repeat = 2))
#rewards =
vel = Velocity(0,0)
lm = get_landmarks(track1)


loc = pick_random_start_location(lm.start)





















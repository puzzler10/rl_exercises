#%% Setup
import pandas as pd, numpy as np, os
from math import floor
from collections import namedtuple
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.debugger import set_trace
from math import log, e
#%matplotlib auto

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

def get_bounds_dict(bnds):
    bounds_dict = dict()
    for row in range(track.shape[0]):
        bounds_dict[row] = [o.col for o in bnds if o.row == row]
    return bounds_dict


def get_finish_dict(bnds):
    finish_dict = dict()
    for o in bnds:
        finish_dict[o.row] = o.col
    return finish_dict

def pick_random_start_location(x):
    """Given a list of start line coordinates, pick one at random
    x: list of tuples"""
    idx = floor(np.random.uniform(0, len(x)))
    return x[idx]

def hit_landmark(loc, finish=False):
    """return True if loc would hit the landmark (finish or boundaries),
    false otherwise
    finish: True if we're looking for finish line, false if looking for boundaries"""
    # this is where we hit the top row
    if loc.row <= 0:
        if finish:      return False
        else          : return True
    # this case - we're not up to finish territory yet
    # In this case the car has hit the side walls
    if finish:
        if loc.col >= FINISH_DICT[loc.row]:        return True
        if loc.row > max(FINISH_DICT.keys()):      return False
    if not finish:
        min_col, max_col = min(BOUNDS_DICT[loc.row]), max(BOUNDS_DICT[loc.row])
        # one boundary case: it's om the left
        if min_col == max_col:
            if loc.col <= min_col:          return True
        # multiple boundary case
        else:
            if loc.col <= min_col or loc.col >= max_col:     return True
            # deal with boundary that juts out into racetrack
            for col in BOUNDS_DICT[loc.row]:
                if loc.col == col:                           return True
    return False


def change_velocity(vel, dvel, loc, start_locs, p=0.1):
    """Velocity must be:
            - nonnegative at all times
            - not (0,0), except at the start line
            - have each component between 0 and -4 inclusive
            - row is negative numbers because we're moving up the board
            - column is positive because we're moving right
        With probability p dvel= 0, regardless of what was passed into dvel
    vel: current velocity
    dvel: change in velocity
    loc: current_location"""
    if np.random.uniform() < p:
        dvel = (0,0)
    new_vel = tuple(np.add(vel, dvel))
    if new_vel[0] > 0: new_vel = tuple((0, new_vel[1]))
    if new_vel[1] < 0: new_vel = tuple((new_vel[0], 0))
    if new_vel[0] < -4: new_vel = tuple((-4, new_vel[1]))
    if new_vel[1] > 4: new_vel = tuple((new_vel[0], 4))
    if new_vel == (0,0):
        if not is_velocity_valid(new_vel, loc, start_locs):
            # Invalid move: disregard it.
            new_vel = vel
    return new_vel


def is_velocity_valid(vel, loc, start_locs):
    """We can only have a 0,0 velocity if we are at the start line"""
    if vel == (0,0) and loc not in start_locs:    return False
    else:                                         return True


def initialise_dicts(track,actions):
    """Initialise q, C and the intial policy `pi`"""
    q,C,pi = dict(),dict(),dict()
    for i in range(track.shape[0]):
        for j in range(track.shape[1]):
            if track.iloc[i,j] in ['0', 'S'] :
                for a in range(len(actions)):
                    q[i,j,a] = np.random.uniform(0,1)
                    C[i,j,a] = 0
                max_act = np.argmax([q[o] for o in q if o[0] == i and o[1] == j])
                pi[i,j] = actions[max_act]
    return (q,C,pi)

def generate_episode(b, actions, lm):
    """Generate sequences S,A,R through simulating episode of car driving task"""
    S,A,R = [],[],[0]
    loc = pick_random_start_location(lm.start)
    vel = (0,0)
    # start position
    ep_end = False
    while not ep_end:
        p_actions = [b[loc.row,loc.col, i] for i in range(len(actions))]
        dvel = actions[np.argmax(np.cumsum(p_actions) > np.random.uniform())]
        vel = change_velocity(vel, dvel, loc, lm.start)
        S.append(loc)
        A.append(dvel)
        next_loc = tuple(np.add(vel, loc))
        next_loc = Loc(next_loc[0], next_loc[1])
        #print(loc, vel, next_loc, hit_landmark(next_loc, lm.boundary))
        if hit_landmark(next_loc):
            loc = pick_random_start_location(lm.start)
            R.append(-1)
            vel = (0,0)
        elif hit_landmark(next_loc, finish=True):
            R.append(0)
            ep_end=True
        else:
            R.append(-1)
            loc = next_loc
    return (S,A,R)

def update_policy(q, C, S, A, R, pi, b, gamma):
    G = 0
    W = 1
    T = len(R) - 2
    for t in range(T,-1,-1):
        s,r = S[t],R[t+1]
        a = [i for i,o in enumerate(actions) if o == A[t]][0]
        # discounted previous return + next reward
        G = gamma * G  + r
        # cumulative weight sum for each state
        # W is always above 1
        C[s.row, s.col, a] +=  W
        q[s.row, s.col, a] +=  (W / C[s.row, s.col, a]) * (G - q[s.row, s.col, a])
        action = np.argmax([q[s.row, s.col, i] for i in range(len(actions))])
        pi[s.row, s.col] = action
        if a != action:
            break
        W *= (1 / b[s.row, s.col, action])
    return (C,q,pi)

def update_b(epsilon, b,q, states):
    for s in states:
        action = np.argmax([q[s[0], s[1], i] for i in range(len(actions))])
        for a in range(len(actions)):
            if a != action:
                b[s[0], s[1], a] = epsilon / len(actions)
            else:
                b[s[0], s[1], a] = 1 - ((len(actions)-1) * epsilon / len(actions))
    return b

def off_policy_MC_control(track):
    track = track1
    lm = get_landmarks(track)


def update_b(epsilon, b,q, states):
    for s in states:
        action = np.argmax([q[s[0], s[1], i] for i in range(len(actions))])
        for a in range(len(actions)):
            if a != action:
                b[s[0], s[1], a] = epsilon / len(actions)
            else:
                b[s[0], s[1], a] = 1 - ((len(actions)-1) * epsilon / len(actions))
    return b


#%%
# lm = landmarks
Loc = namedtuple('Loc', 'row col')
#rewards =
vel = (0,0)

track = track2
lm = get_landmarks(track)
BOUNDS_DICT = get_bounds_dict(lm.boundary)
FINISH_DICT = get_finish_dict(lm.finish)

actions = list(product([-1,0,1], repeat = 2))
q,C,pi = initialise_dicts(track, actions)

#### setup
# Init b as equiprobable random policy
b = dict()
for key in q.keys():
    b[key] = 1/len(actions)
states = set([(o[0],o[1]) for o in b.keys()])

gamma = 0.9
for i in range(300):
    print(i)
    S,A,R = generate_episode(b, actions, lm)
    C,q,pi = update_policy(q, C, S, A, R, pi, b, gamma)
    #if i < e:         epsilon = 1
    #else:              epsilon = 1 / log(i,e)
    #epsilon=1
    #b = update_b(epsilon, b, q, states)

v = dict()
for s in states:
    v[s[0],s[1]] = np.max([q[s[0], s[1], i] for i in range(len(actions))])

#%%
rows = [o[0] for o in v.keys()]
cols = [o[1] for o in v.keys()]
min_row, max_row = min(rows), max(rows)
min_col, max_col = min(cols), max(cols)
v_arr = np.zeros((max_row+1,max_col+1))
for row in range(min_row, max_row+1):
    for col in range(min_col, max_col+1):
        try:
            v_arr[row,col] = v[row,col]
        except:
            v_arr[row,col] = np.nan

sns.heatmap(v_arr)




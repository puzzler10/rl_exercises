import pytest
from racetrack import *
import numpy as np

Loc = namedtuple('Loc', 'row col')
bnds = [Loc(4,2), Loc(4,7), Loc(3,2), Loc(3,3), Loc(3,4), Loc(3,8)]
start_locs = [Loc(8,2), Loc(8,3), Loc(8,4)]
finish_locs = [Loc(2,9), Loc(3,9), Loc(4,9)]
track = pd.read_csv(path + 'racetrack1.csv', header=None)
BOUNDS_DICT = get_bounds_dict(bnds)
FINISH_DICT = get_finish_dict(finish_locs)

def test_get_landmarks():
    landmarks = get_landmarks(track1)
    assert (0,2) in landmarks.boundary
    assert (38, 7) in landmarks.start
    assert (4,26) in landmarks.finish
    assert (1,26) in landmarks.finish

def test_pick_random_start_location():
    np.random.seed(1000)
    start_loc_set = set()
    for i in range(1000):
        x = pick_random_start_location(start_locs)
        assert x in start_locs
        start_loc_set.add(x)
    assert len(start_loc_set) == len(start_locs)


def test_landmark_hit():
    assert hit_landmark(Loc(4,1)) == True
    assert hit_landmark(Loc(4,2)) == True
    assert hit_landmark(Loc(4,3)) == False
    assert hit_landmark(Loc(4,7)) == True
    assert hit_landmark(Loc(4,8)) == True
    assert hit_landmark(Loc(3,1)) == True
    assert hit_landmark(Loc(3,2)) == True
    assert hit_landmark(Loc(3,4)) == True
    assert hit_landmark(Loc(3,5)) == False
    assert hit_landmark(Loc(3,7)) == False
    assert hit_landmark(Loc(3,8)) == True
    assert hit_landmark(Loc(3,9)) == True
    assert hit_landmark(Loc(2,9) ,finish=True) == True
    assert hit_landmark(Loc(2,10),finish=True) == True
    assert hit_landmark(Loc(2,8),finish=True) == False


def test_change_velocity():
    loc = Loc(8,4)
    assert change_velocity((0,2), (0,1) ,loc,start_locs, p=0) == (0,3)
    assert change_velocity((0,1), (0,-1),loc,start_locs, p=0) == (0,0)
    assert change_velocity((0,0), (0,-1),loc,start_locs, p=0) == (0,0)
    assert change_velocity((0,0), (1,-1),loc,start_locs, p=0) == (0,0)
    assert change_velocity((0,0), (-1,1),loc,start_locs, p=0) == (-1,1)
    assert change_velocity((-4,4), (-1,1),loc,start_locs, p=0) == (-4,4)
    assert change_velocity((0,1), (0,-1),Loc(8,2),start_locs, p=0) == (0,0)
    assert change_velocity((0,1), (0,-1),Loc(9,2),start_locs, p=0) == (0,1)
    assert change_velocity((-3,2), (-1,1),loc,start_locs, p=1) == (-3,2)


def test_is_velocity_valid():
    assert is_velocity_valid((0,0), Loc(8,2), start_locs) == True
    assert is_velocity_valid((0,0), Loc(7,2), start_locs) == False
    assert is_velocity_valid((-1,0), Loc(8,2), start_locs) == True
    assert is_velocity_valid((-1,0), Loc(7,2), start_locs) == True


def test_initialise_dicts():
    actions = list(product([-1,0,1], repeat = 2))
    q,C,pi = initialise_dicts(track, actions)
    assert len(q)/len(actions) == len(pi)
    assert sum(C.values()) == 0
    max_q_arg = np.argmax([q[o] for o in q if o[0] == 8 and o[1] == 2])
    assert pi[8,2] == actions[max_q_arg]


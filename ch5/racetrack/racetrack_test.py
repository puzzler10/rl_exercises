import pytest
from racetrack import *

def test_get_landmarks():
    landmarks = get_landmarks(track1)
    assert (0,2) in landmarks.boundary
    assert (38, 7) in landmarks.start
    assert (4,26) in landmarks.finish
    assert (1,26) in landmarks.finish


def test_boundary_hit():
    Loc = namedtuple('Loc', 'row col')
    bnds = [Loc(4,2), Loc(4,7), Loc(3,2), Loc(3,3), Loc(3,4), Loc(3,8)]
    assert check_boundary_hit(Loc(4,1), bnds) == True
    assert check_boundary_hit(Loc(4,2), bnds) == True
    assert check_boundary_hit(Loc(4,3), bnds) == False
    assert check_boundary_hit(Loc(4,7), bnds) == True
    assert check_boundary_hit(Loc(4,8), bnds) == True
    assert check_boundary_hit(Loc(3,1), bnds) == True
    assert check_boundary_hit(Loc(3,2), bnds) == True
    assert check_boundary_hit(Loc(3,4), bnds) == True
    assert check_boundary_hit(Loc(3,5), bnds) == False
    assert check_boundary_hit(Loc(3,7), bnds) == False
    assert check_boundary_hit(Loc(3,8), bnds) == True
    assert check_boundary_hit(Loc(3,9), bnds) == True
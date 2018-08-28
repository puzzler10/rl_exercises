import numpy as np
import pytest

from blackjack_functions import *

#%% Testing
def test_generate_deck():
    deck = generate_deck()
    assert len(deck) == 52
    assert len([o for o in deck if o == 10]) == 4*4

def test_deal_cards():
    deck = generate_deck()
    assert len(deal_cards(deck)) == 3
    assert len(deal_cards(deck)[0]) == 2
    assert len(deal_cards(deck)[1]) == 2
    assert type(deal_cards(deck)[0]) == list
    assert type(deal_cards(deck)[1]) == list
    assert type(deal_cards(deck)[2]) == list
    assert len(deal_cards(deck)[2]) == 48
    # Test that if you deal two aces, one of them is 1, other is 11
    for i in range(1000):
        assert deal_cards(deck)[1] != [11,11]

def test_get_card_sum():
    assert get_card_sum([1, 3]) == 4
    assert get_card_sum(np.array([1, 3])) == 4

def test_is_bust():
    assert is_bust(20) is False
    assert is_bust(21) is False
    assert is_bust(22) is True

def test_change_ace_value():
    assert change_ace_value([1,1], 'up') == [11,1]
    assert change_ace_value([11,1], 'up') == [11,11]
    assert change_ace_value([1,2,4,11], 'up') == [11,2,4,11]
    assert change_ace_value([11,1], 'down') == [1,1]
    assert change_ace_value([1,2,4,11], 'down') == [1,2,4,1]
    assert change_ace_value([11,11], 'down') == [1,11]
    with pytest.raises(ValueError) as excinfo:
        change_ace_value([2,1], 'keft')
    assert 'Direction' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        change_ace_value([2,1], 'down')
    assert 'flex down' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        change_ace_value([2,11], 'up')
    assert 'flex up' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        change_ace_value([2,2], 'up')
    assert 'aces' in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        change_ace_value('d', 'down')
    assert 'list' in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        change_ace_value(np.array([2,11]), 'down')
    assert 'list' in str(excinfo.value)

def test_hit():
    deck = generate_deck()
    assert len(hit([1,2],deck)[0]) == 3
    assert len(hit([3,4,1],deck)[0]) == 4
    # Test the ace goes from 11 to 1 if you go over 21
    assert hit([11,10],deck)[0][0] == 1
    assert hit([10,11],deck)[0][1] == 1
    # All aces should be 1 if over 21
    for i in range(400):
        assert get_card_sum(hit([10,11],deck)[0]) <= 21

    # Test you don't have a sum over 21 when you hit without useable aces
    with pytest.raises(ValueError) as excinfo:
        hit([10,10,4],deck)
    # Test card is being taken from deck
    assert len(hit([1,2],deck)[1]) == 51

def test_hit_until():
    for i in range(500):
        # Setup
        np.random.seed(i)
        deck = generate_deck()
        player, dealer, deck = deal_cards(deck)
        cards, new_deck = hit_until(player, deck, 21)

        # Check lengths of retuned player cards
        assert len(hit_until(player, deck, 18)[0]) >= len(player)
        assert len(hit_until(player, deck, 2)[0]) == 2
        # Check that the card sum is higher than the value passed in
        assert sum(hit_until(player, deck, 18)[0]) >= 18
        assert sum(hit_until(player, deck, 21)[0]) >= 21
        # Check length of returned deck is correct
        assert len(new_deck) == len(deck) - (len(cards) - len(player))

        # Only the last card should make you bust
        assert sum(cards[0:len(cards)-1]) <= 21
        # Shouldn't go bust if you have a useable ace
        if is_bust(get_card_sum(cards)) :
            assert 11 not in cards

        # Check returned output types
        assert type(hit_until(player, deck, 18)[0]) == list
        assert type(hit_until(player, deck, 18)[1]) == list

    # input checking
    with pytest.raises(ValueError) as excinfo:
        hit_until(player, deck, 1)
    assert 'between 2 and 21' in str(excinfo)
    with pytest.raises(ValueError) as excinfo:
        hit_until(player, deck, 22)
    assert 'between 2 and 21' in str(excinfo)

def test_get_reward():
    assert get_reward(20, 21) == -1
    assert get_reward(5,6) == -1
    assert get_reward(7,4) == 1
    assert get_reward(20,18) == 1
    assert get_reward(20,22) == 1
    assert get_reward(12,12) == 0
    assert get_reward(22,22) == -1
    assert get_reward(22,20) == -1
    assert get_reward(21,21) == 0

def test_has_useable_ace():
    assert has_useable_ace([11,2]) == True
    assert has_useable_ace([11,10,2]) == False
    assert has_useable_ace([2,3,4,11]) == True
    assert has_useable_ace([2,3,4,5,11]) == False
    assert has_useable_ace([3,4]) == False
    assert has_useable_ace([11,1]) == True
    # Shouldn't have a sum under 11 with an ace
    with warnings.catch_warnings(record=True) as w:
        has_useable_ace([1,10])
    assert len(w) ==1
    # Two aces shouldn't be passed into this function
    with pytest.raises(ValueError) as excinfo:
        has_useable_ace([11,11])

def test_initialise_policy(hit_until=20):
    actions = ['hit', 'stick']
    assert type(initialise_policy(hit_until=20)) == list
    assert np.array(initialise_policy(hit_until=20)).shape == (10,10,2)
    # Check that we only have legal actions in array
    policy_flat = np.array(initialise_policy(hit_until=20)).flatten()
    assert sum([1 for o in policy_flat if o in actions]) == len(policy_flat)
    # Check that all 'hit' are above 20
    assert initialise_policy(hit_until=20)[8][0][1] == 'stick'
    assert initialise_policy(hit_until=20)[6][0][1] == 'hit'
    assert initialise_policy(hit_until=20)[9][8][1] == 'stick'
    assert initialise_policy(hit_until=20)[4][8][0] == 'hit'



def test_generate_idx_mappings():
    map_cardsum_idx, map_faceup_idx, map_ace_idx, map_actions = generate_idx_mappings()
    assert len(map_ace_idx) == 2
    assert len(map_cardsum_idx) == 10
    assert len(map_faceup_idx) == 10
    assert map_cardsum_idx[12] == 0
    assert map_cardsum_idx[21] == 9
    assert map_faceup_idx[2] == 0
    assert map_faceup_idx[11] == 9
    assert map_ace_idx[False] == 0
    assert map_ace_idx[True] == 1
    assert map_actions['stick'] == 0
    assert map_actions['hit'] == 1

def test_simulate_game():
    maps = generate_idx_mappings()
    map_cardsum, map_faceup, map_ace = maps[0],maps[1],maps[2]
    policy = initialise_policy(hit_until=20)
    orig_deck = generate_deck()
    for i in range(2000):
        S,A,R = simulate_game(policy,maps, orig_deck, dealer_limit = 17,
                                epsilon=0.1)
        # Check lengths of arrays are correct
        assert len(S) + 1 == len(R)
        assert len(A) + 1 == len(R)
        # If I go bust, my last action should be hit
        last_s = S[len(S)-1]
        card_sum_idx = last_s[0]
        card_sum = list(map_cardsum.keys())[
                list(map_cardsum.values()).index(card_sum_idx)]
        if is_bust(card_sum):
            A[len(A)-1] == 'hit'
            R[len(A)-1] == -1


def test_eval_episode():
    q = np.zeros((10, 10, 2, 2))
    gamma = 1
    returns_sum = np.zeros((10, 10, 2, 2))
    returns_n = np.zeros((10, 10, 2, 2))
    policy = initialise_policy(hit_until=20)
    maps= generate_idx_mappings()
    map_action = maps[3]
    # Test q and returns are updating correctly
    S = [(1, 8, 1)]
    A = ['hit']
    R = [0, -1]
    q = eval_episode(q, policy,returns_sum, returns_n, gamma,S,A,R,maps)
    i,j,k,a = S[0][0],S[0][1],S[0][2],map_action[A[0]]
    #assert returns_sum[i][j][k][a] == -1
    assert q[i][j][k][a] == -1

    S = [(1, 8, 1), (1, 8, 0)]
    A = ['hit', 'hit']
    R = [0, 0, 0]
    q = eval_episode(q,policy, returns_sum, returns_n, gamma,S,A,R,maps)
    i,j,k,a = S[0][0],S[0][1],S[0][2],map_action[A[0]]
    #assert returns[i][j][k][a] == [-1, 0]
    assert q[i][j][k][a] == -0.5
    i,j,k,a = S[1][0],S[1][1],S[1][2],map_action[A[1]]
    #assert returns[i][j][k][a] == [0]
    assert q[i][j][k][a] == 0

    S = [(1, 8, 1), (2, 8, 0), (9,8,0)]
    A = ['hit', 'hit', 'stick']
    R = [0, 0, 0, 1]
    q = eval_episode(q, policy, returns_sum, returns_n, gamma,S,A,R,maps)
    i,j,k,a = S[0][0],S[0][1],S[0][2],map_action[A[0]]
    #assert returns[i][j][k][a] == [-1, 0,1]
    assert q[i][j][k][a] == 0




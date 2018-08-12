#%% Packages
import numpy as np
import copy
import pytest
from sspipe import p, px
from IPython.core.debugger import set_trace
from itertools import permutations
import warnings

#%% Functions
def generate_deck():
    """Generate a deck of cards to use in blackjack.
    Suits aren't relevant here, only the values of each card.
    Ace is represented by 11."""
    # [10,10,10] represents J, Q, K
    card_values = [i for i in range(2, 12)] + [10, 10, 10]
    deck = np.repeat(card_values, 4).tolist()
    return deck


def deal_cards(deck):
    """Deal cards from a deck, without replacement.
    Returns tuple with players cards, dealers cards, and remaining deck"""
    cards = np.random.choice(deck,4,False)
    # Remove dealt cards from the deck
    # Create a deep copy to avoid removing elements in start_deck
    deck_copy = copy.deepcopy(deck)
    for card in cards:
        deck_copy.remove(card)
    player = cards[0:2].tolist()
    dealer = cards[2:4].tolist()
    if player == [11,11]:   change_ace_value(player, 'down')
    if dealer == [11,11]:   change_ace_value(dealer, 'down')
    return (player, dealer, deck_copy)


def get_card_sum(cards):
    """Returns integer with the sum of cards
    Cards is np array, or list"""
    return np.sum(cards)


def is_bust(card_sum):
    if card_sum > 21:  bust =  True
    else:              bust =  False
    return bust

def has_useable_ace(cards):
    """If the player holds an ace that can be
    flexed down (goes from 11 to 1), it is a useable ace. """
    if 1 not in cards and 11 not in cards:  # don't have any ace at all
        return False
    if 11 in cards and get_card_sum(cards) <= 21:  #ace counted as 11 and not bust
        return True
    if cards == [11,11]:
        raise ValueError("There shouldn't be two aces both counted as 11")
    # ace should be counted as 11 here
    if 1 in cards and get_card_sum(cards) <= 11:
        warnings.warn("Ace is counted as 1 when the card sum \
                      is 11 or under, which shouldn't happen")
    else:
        return False


def change_ace_value(cards, direction):
    """ Changes the value of ace in the given direction.
    If two aces are given, only one is changed.
    Cards is np array, or list
    Direction takes either 'up' or 'down'. Up changes from 1 to 11.
        Down changes from 11 to 1. """
    if direction not in ['up', 'down']:
        raise ValueError('Direction must be either "up" or "down"')
    if type(cards) not in [list]:
        raise TypeError('Variable `cards` must be list type')
    if 1 not in cards and 11 not in cards:
        raise ValueError("No aces in cards")
    if direction == 'up' and 1 not in cards:
        raise ValueError("Ace of value 1 not present, can't flex up")
    if direction == 'down' and 11 not in cards:
        raise ValueError("Ace of value 11 not present, can't flex down")

    if   direction == 'up'  : current = 1;  target = 11
    elif direction == 'down': current = 11; target = 1

    ind = [i for i,o in enumerate(cards) if o == current][0]
    cards[ind] = target
    return cards


def hit(cards, deck):
    """Add a card to a players hand `cards` from the deck.
    Returns tuple: (players cards, remaining deck) """
    if is_bust(get_card_sum(cards)):
        raise ValueError("Cannot hit when card sum over 21")
    # Deal card
    card = np.random.choice(deck,1,False).tolist()
    cards += card
    # remove card from deck
    deck_copy = copy.deepcopy(deck)
    deck_copy.remove(card[0])
    # Adjust for aces
    while is_bust(get_card_sum(cards)) and 11 in cards:
        cards = change_ace_value(cards, 'down')
    return (cards, deck_copy)


def hit_until(cards, deck, value):
    """Keep hitting until you have at least this value.
    Returns (cards, deck, is_bust) """
    #set_trace()
    if value < 2 or value > 21:
        raise ValueError("Value must be between 2 and 21")
    cards_copy = copy.deepcopy(cards)
    while sum(cards_copy) < value:
        cards_copy, deck = hit(cards_copy, deck)
    card_sum = get_card_sum(cards_copy)
    return cards_copy, deck, is_bust(card_sum)


def get_reward(player_sum, dealer_sum):
    """Return 1 if player wins
    Return 0 if there is a tie
    Returns -1 if dealer wins"""
    if is_bust(player_sum):    return  -1
    if is_bust(dealer_sum):    return  +1
    if     player_sum < dealer_sum:   return  -1
    elif   player_sum == dealer_sum:  return   0
    elif   player_sum > dealer_sum:   return  +1


def get_list_of_states():
    """State: dealer up card, player_sum, usable_ace
    nonuseable ace: flexing up puts you bust
    useable ace:  can flex up
    """
    pass

#%% Setup
deck = generate_deck()
player, dealer, deck = deal_cards(deck)
faceup = dealer[0]


#%%
actions = ['hit', 'stick']
gamma = 1

# states: 3d array [usable_ace, dealer_show, player_sum]
# returns <- 3d array, size states
# V <- 3d array, size states
# policy <- 4d array of [size(states) x actions]
    # policy is initally: stick if 20 or 21, else hit
# for state in states:
    # sim game:
        # init S, A, R, empty arrays for holding episode steps
        # R[0] = 0   # this one needs to be offset by 1
        # t = 0
        # deal cards
        # until we stop (based on policy):
            # update S[t] with state
            # take action based on policy, update A[t]
            # Update R[t+1]
                # Rewards within a game are 0, so R[t+1] = 0,
                # unless we're at game end
            # t+=1

    # g = 0
    # s_visited = []
    # T = len(S)
    # for t in range(T-1,1,-1):
        # s = S[t]
        # G = gamma * G + R_t+1
        # s_visited.append(s)
        # if s not in s_visited




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
        cards, new_deck, bust = hit_until(player, deck, 21)

        # Check lengths of retuned player cards
        assert len(hit_until(player, deck, 18)[0]) >= len(player)
        assert len(hit_until(player, deck, 2)[0]) == 2
        # Check that the card sum is higher than the value passed in
        assert sum(hit_until(player, deck, 18)[0]) >= 18
        assert sum(hit_until(player, deck, 21)[0]) >= 21
        # Check length of returned deck is correct
        assert len(new_deck) == len(deck) - (len(cards) - len(player))
        # Bust function works okay
        if sum(cards) > 21:   assert bust == True
        else:                 assert bust == False
        # Only the last card should make you bust
        assert sum(cards[0:len(cards)-1]) <= 21
        # Shouldn't go bust if you have a useable ace
        if bust == True:
            assert 11 not in cards

        # Check returned output types
        assert type(hit_until(player, deck, 18)[0]) == list
        assert type(hit_until(player, deck, 18)[1]) == list
        assert type(hit_until(player, deck, 2)[2]) == bool
        assert type(hit_until(player, deck, 21)[2]) == bool

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
















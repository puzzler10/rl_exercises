import numpy as np
import copy
import os
import pytest
from sspipe import p, px
from math import ceil, floor
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
    card_inds = [int(floor(o)) for o in np.random.uniform(0,len(deck),4)]
    # if we have duplicates, redraw
    while len(set(card_inds)) != len(card_inds):
        card_inds = [int(floor(o)) for o in np.random.uniform(0,len(deck),4)]
    cards = [deck[o] for o in card_inds]
    # Remove dealt cards from the deck
    # Create a deep copy to avoid removing elements in start_deck
    deck_copy = [o for o in deck]
    for card in cards:
        deck_copy.remove(card)
    player = cards[0:2]
    dealer = cards[2:4]
    if player == [11,11]:   change_ace_value(player, 'down')
    if dealer == [11,11]:   dealer = [11,1]  # first card should stay 11 for faceup
    return (player, dealer, deck_copy)


def get_card_sum(cards):
    """Returns integer with the sum of cards
    Cards is np array, or list"""
    return sum(cards)


def is_bust(card_sum):
    if card_sum > 21:  bust =  True
    else:              bust =  False
    return bust

def has_useable_ace(cards):
    """If the player holds an ace that can be
    flexed down (goes from 11 to 1), it is a useable ace.
    Returns a boolean"""
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
    card = [deck[floor(np.random.uniform(0,len(deck)))]]
    cards += card
    # remove card from deck
    deck_copy = [o for o in deck]
    deck_copy.remove(card[0])
    # Adjust for aces
    while is_bust(get_card_sum(cards)) and 11 in cards:
        cards = change_ace_value(cards, 'down')
    return (cards, deck_copy)


def hit_until(cards, deck, value):
    """Keep hitting until you have at least this value.
    Returns (cards, deck) """
    #set_trace()
    if value < 2 or value > 21:
        raise ValueError("Value must be between 2 and 21")
    cards_copy = [o for o in cards]
    while sum(cards_copy) < value:
        cards_copy, deck = hit(cards_copy, deck)
    return cards_copy, deck


def get_reward(player_sum, dealer_sum):
    """Return 1 if player wins
    Return 0 if there is a tie
    Returns -1 if dealer wins"""
    if is_bust(player_sum):    return  -1
    if is_bust(dealer_sum):    return  +1
    if     player_sum < dealer_sum:   return  -1
    elif   player_sum == dealer_sum:  return   0
    elif   player_sum > dealer_sum:   return  +1


def generate_idx_mappings():
    """Create mappings between state values and indexes to make
    array storage and retrieval easier."""
    map_cardsum_idx = dict(zip(list(range(12,22)),list(range(10))))
    map_faceup_idx = dict(zip(set(generate_deck()),list(range(10))))
    map_ace_idx = dict(zip([False,True],list(range(2))))
    map_action_idx = dict(zip(['stick','hit'],list(range(2))))
    return(map_cardsum_idx, map_faceup_idx, map_ace_idx, map_action_idx)

def initialise_policy(hit_until = 20):
    """ Returns the policy: a list of shape (10, 10, 2)
    Values are 'hit' for player sum less than hit_until, and 'stick' otherwise"""
    policy = np.empty(shape=(10, 10, 2), dtype=object).tolist()
    map_cardsum, map_faceup, map_ace, _ = generate_idx_mappings()
    #Initial policy is to stick on 20, 21 and hit otherwise
    for csum in range(12,22):
        for faceup in range(2,12):
            for ace in [False,True]:
                i,j,k = map_cardsum[csum],map_faceup[faceup],map_ace[ace]
                if csum >= hit_until: policy[i][j][k] = 'stick'
                else:                 policy[i][j][k] = 'hit'
    return policy

def simulate_game(policy, maps, orig_deck, dealer_limit = 17, epsilon = 0.1):
    """Simulate one game of blackjack.
    dealer_limit: dealer hits until this number
    Returns tuple (S, A, R), with list of states, actions and rewards
    maps: output from generate_idx_mappings"""
    ## Play game
    # Initiate arrays
    S, A, R = [],[],[]
    map_cardsum, map_faceup, map_ace, map_action = maps[0],maps[1],maps[2],maps[3]
    R.append(0)  # Rewards are offset by 1 with time steps
    ## Deal cards
    deck = [o for o in orig_deck]
    player, dealer, deck = deal_cards(deck)
    faceup = dealer[0]
    ## Play game
    game_end = False
    while not game_end:
        ## Player move
        # Update state, action
        csum = get_card_sum(player)
        # If player sum is under 11, we always hit
        if csum <= 11:
            player, deck = hit(player, deck)
        else:
            ace = has_useable_ace(player)
            i,j,k = map_cardsum[csum],map_faceup[faceup],map_ace[ace]
            S.append((i,j,k))
            # Choose randomly epsilon percent of the time, else greedy
            if epsilon > np.random.uniform():
                action = ['stick', 'hit'][floor(np.random.uniform(0,2))]
            else:
                action = policy[i][j][k]
            A.append(action)
            # Evaluate action
            if action == 'hit':
                player, deck = hit(player, deck)
                if is_bust(get_card_sum(player)):
                    game_end = True
                    R.append(-1)
                else:
                    R.append(0)
            elif action == 'stick':
                # Dealer gets cards
                dealer, deck = hit_until(dealer, deck, dealer_limit)
                R.append(get_reward(get_card_sum(player), get_card_sum(dealer)))
                game_end = True
    return(S,A,R)


def eval_episode(q,policy,returns_sum, returns_n,gamma,S,A,R, maps):
    """Updates the action-value function and return function.
    q: preexisting action-value function
    gamma: the reward decay parameter
    S: list of states in episode
    R: list of actions in episode
    maps: output from generate_idx_mappings"""
    map_action = maps[3]
    G = 0
    T = len(R) - 1
    for t in range(T-1, -1,-1):
        s = S[t]
        G = gamma * G + R[t+1]
        if s not in S[0:t]:
            i,j,k,a = s[0],s[1],s[2],map_action[A[t]]
            returns_sum[i][j][k][a] += G
            returns_n[i][j][k][a] += 1
            q[i][j][k][a] = returns_sum[i][j][k][a] / returns_n[i][j][k][a]
            # update policy
            x = q[i][j][k]
            idx = [i for i,o in enumerate(x) if o == max(x)][0]
            policy[i][j][k] = ['stick', 'hit'][idx]
    return q, policy


def policy_iteration(policy, gamma=1, n_sims = 2000, dealer_limit = 17,
                      epsilon=0.1):
    """Find action-value function for a given policy"""
    q = np.zeros((10, 10, 2, 2))
    returns_sum = np.zeros((10, 10, 2, 2))
    returns_n = np.zeros((10, 10, 2, 2))

    maps = generate_idx_mappings()
    orig_deck = generate_deck()
    for i in range(n_sims):
        if (i % 25000 == 0 ): print(i)
        S, A, R = simulate_game(policy, maps, orig_deck, dealer_limit = dealer_limit,
                                epsilon=epsilon)
        q, policy = eval_episode(q, policy, returns_sum, returns_n,
                                  gamma, S, A, R, maps)
    return q,policy

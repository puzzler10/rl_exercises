#%% Packages
import os
os.chdir('/Users/tomroth/Dropbox/Reinforcement Learning/reinforcement_learning_exercises/ch5')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from blackjack_functions import *



#%% Setup

# options
# on-policy, epsilon-greedy


#%%

# Setup stuff
# First axis: sum of player cards. 10 options for 12-21. Under 12 we always hit
# Second axis: dealer's faceup card. 10 options for the value
# Third axis: If player has a useable ace or not. 2 options here.
# Fourth axis: the action, stick or hit


def get_v_from_q(q):
    """Return the value of q corresponding to the policy value...
    which is going to be the maximum value of q for each state"""

    return np.max(q,axis=3)

#def plot_value_function(q):



#%%
gamma = 1
n_sims = 3000000
epsilon = 0.1


initial_hit_until = 20
policy = initialise_policy(hit_until=20)
q, policy = policy_iteration(policy, gamma=gamma, n_sims=n_sims, epsilon=epsilon)

#%%
map_cardsum, map_faceup, map_ace, map_action = generate_idx_mappings()
v = get_v_from_q(q)

inv_map_cardsum = {v: k for k, v in map_cardsum.items()}
inv_map_faceup = {v: k for k, v in map_faceup.items()}
inv_map_faceup[len(inv_map_faceup)-1] = 'A'

yticklabels = [inv_map_cardsum[i] for i in range(v.shape[0])]
xticklabels = [inv_map_faceup[i] for i in range(v.shape[1])]


f, (ax1, ax2) = plt.subplots(1,2)
sns.heatmap(v[:,:,0], vmin = -1.57, vmax=1.29, ax=ax1,
            xticklabels=xticklabels, yticklabels=yticklabels)
sns.heatmap(v[:,:,1], vmin = -1.57, vmax=1.29, ax=ax2,
            xticklabels=xticklabels, yticklabels=yticklabels)

ax1.set_title('No Useable Ace')
ax2.set_title('Useable Ace')
ax1.set_ylabel('Player Sum')
ax2.set_ylabel('Player Sum')
ax1.set_xlabel('Dealer Faceup')
ax2.set_xlabel('Dealer Faceup')

#%%
f, (ax1, ax2) = plt.subplots(1,2)

fun = lambda x : 1 if x == 'hit' else 0

p_noace = np.array(list(map(fun, np.array(policy)[:,:,0].flatten()))).reshape(10,10)
p_ace = np.array(list(map(fun, np.array(policy)[:,:,1].flatten()))).reshape(10,10)


sns.heatmap(p_noace,ax=ax1,
            xticklabels=xticklabels, yticklabels=yticklabels)

sns.heatmap(p_ace,ax=ax2,
            xticklabels=xticklabels, yticklabels=yticklabels)

ax1.set_title('No Useable Ace')
ax2.set_title('Useable Ace')
ax1.set_ylabel('Player Sum')
ax2.set_ylabel('Player Sum')
ax1.set_xlabel('Dealer Faceup')
ax2.set_xlabel('Dealer Faceup')
#%%

_, axes = plt.subplots(2, 2, figsize=(40, 30))
plt.subplots_adjust(wspace=0.1, hspace=0.2)
axes = axes.flatten()

for state, title, axis in zip(v, titles, axes):
    fig = sns.heatmap(np.flipud(v), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                      yticklabels=list(reversed(range(12, 22))))
    fig.set_ylabel('player sum', fontsize=30)
    fig.set_xlabel('dealer showing', fontsize=30)
    fig.set_title(title, fontsize=30)




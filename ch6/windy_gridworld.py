kings_moves = True
allow_zero_move = False
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(14)
nx = 10
ny = 7
start_loc = (0, 3)
goal_loc = (7, 3)


# Wind strength of each column
# negative numbers blow up, positive numbers down
wind = [0, 0, 0, -1, -1, -1, -2, -2, -1, 0]

if kings_moves:
    # going around the unit circle, starting from going right 1 move
    actions = [(1, 0),(1,-1), (0, -1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1)]
    if allow_zero_move:
        actions = actions + [(0,0)]
else:
    # right, left, up, down
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

#### SETUP ---------------------------------
q = np.random.uniform(size=(nx, ny, len(actions)))

#### Functions ----------------------------------
def get_e_greedy_policy(q, epsilon, shape):
    """
    Return an epsilon-greedy policy for use in sarsa.
    :param q: action-value function
    :param epsilon: chance to choose randomly
    :param shape: shape of gridworld, tuple
    :return: policy of shape = (shape, actions)
    """
    x,y = shape
    pi = np.zeros((x, y, len(actions)))
    max_action_idx = np.argmax(q, axis=2)
    for i in range(x):
        for j in range(y):
            for a in range(len(actions)):
                if a == max_action_idx[i, j]:
                    pi[i, j, a] = 1 - ((len(actions) - 1) * (epsilon / len(actions)))
                else:
                    pi[i, j, a] = epsilon / len(actions)
    return pi

def adjust_loc(loc, nx,ny):
    """
    We can't go off the grid with a move. If we try, we should just stay
    where we are now
    :param loc: player's location
    :param nx: number of x locations in grid (columns)
    :param ny: number of y locations in grid (rows)
    :return: updated player location, 2-tuple
    """
    x,y = loc[0],loc[1]
    if x < 0:           x = 0
    elif x > (nx-1):    x = (nx-1)
    if y < 0:           y = 0
    elif y > (ny-1):    y = (ny-1)
    return(x,y)

def add_wind(loc, wind):
    """
    Add wind effect to the player's location
    :param loc: the location on the grid
    :param wind: [nx x 1] list giving wind strengths
    :return: updated location, 2-tuple
    """
    x = loc[0]
    y = loc[1] + wind[x]
    return(x,y)

alpha = 0.8
gamma = 1
t = 1
pi = get_e_greedy_policy(q,1/t,(nx,ny))
min_finish_steps =  100000
finish_t = []
while t < 20000:
    # Generate episode
    loc = start_loc
    a = np.argmax(np.random.uniform() < np.cumsum(pi[loc[0], loc[1]]))
    # move
    ep_steps = 0
    while (loc != goal_loc):
        ep_steps += 1
       # print(t, loc, a)
        s = loc
        action = actions[a]
        loc = tuple(np.add(loc,action))
        loc = adjust_loc(loc, nx,ny)
        loc = add_wind(loc,wind)
        loc = adjust_loc(loc, nx,ny)
        if loc != goal_loc:   r = -1
        elif loc == goal_loc:
            r = 0
            min_finish_steps = min(min_finish_steps, ep_steps)
            finish_t.append(t)
        sprime = loc
        aprime = np.argmax(np.random.uniform() < np.cumsum(pi[sprime[0], sprime[1]]))
        q[s[0], s[1], a] = q[s[0], s[1], a] + alpha *(r + gamma * q[loc[0], loc[1], aprime] - q[s[0], s[1], a])
        a = aprime
        t+=1
        epsilon = 1/t
        pi = get_e_greedy_policy(q, epsilon, (nx, ny))


# plot the finish times
plt.plot(finish_t)
# plt.show(block=True)
# plt.interactive(False)

# plot the best value function in each state
best_q = np.max(q, axis = 2).transpose()
sns.heatmap(best_q)
plt.show(block=True)

# plot the best action in each state
best_actions = np.argmax(q, axis = 2).transpose()
annots =np.array([[str(actions[o][0]) + str(actions[o][1])  for o in a] for a in best_actions])

sns.heatmap(best_actions, cmap = "RdBu_r")
plt.show(block=True)

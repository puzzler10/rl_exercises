import numpy as np
states = ['A', 'B', 'C', 'D', 'E']
actions = ['LEFT', 'RIGHT']

n_episodes = 1000
start_pos = 2

V = np.zeros((5,)).tolist()
n_V = np.zeros((5,)).tolist()

def generate_episode(start_pos):
    S, A, R = [], [], [0]
    pos = start_pos
    while pos >= 0 and pos <= len(states):
        S.append(pos)
        if np.random.uniform() <= 0.5:  pos -= 1;  A.append('LEFT')
        else:                           pos += 1;  A.append('RIGHT')
        if pos > 4:   R.append(1)
        else:         R.append(0)
    return (S, A, R)

for t in range(n_episodes):
    for i in range(len(states)):
        (S, A, R) = generate_episode(i)
        V
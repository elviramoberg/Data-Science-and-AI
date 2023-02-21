from math import inf
import numpy as np


def find_directions(x, y, size):
    (a, b) = size
    # creating a list of possible directions from (x, y)
    poss_dir = []
    # checking where in the matrix (x, y) is to add directions to list
    if x > 0:
        poss_dir.append((x-1, y, 'W'))
    if x < b-1:
        poss_dir.append((x+1, y, 'E'))
    if y > 0:
        poss_dir.append((x, y-1, 'S'))
    if y < a-1:
        poss_dir.append((x, y+1, 'N'))
    return poss_dir


def get_best_V(x, y, reward_matrix, last_V, gamma=1.0, p_move=0.8):
    policy = ''
    best_v = 0
    poss_dir = find_directions(x, y, reward_matrix.shape)
    for (x1, y1, dir) in poss_dir:
        new_v = p_move * (reward_matrix[x1, y1]+gamma*last_V[x1, y1]) + (1-p_move) * (reward_matrix[x, y] + gamma*last_V[x,y])
        if new_v > best_v:
            best_v = new_v
            policy = dir
    return policy, best_v


def state_iteration(V_0, reward_matrix, eps=0.001, gamma=0.9):
    (a,b) = reward_matrix.shape
    last_V = V_0
    policy = np.zeros((a, b), dtype=str)
    best_v = np.zeros((a, b))

    while True:
        for x in range(a):
            for y in range(b):
                policy[x, y], best_v[x, y] = get_best_V(x, y, last_V, reward_matrix, gamma=gamma)
        if (np.abs(best_v-last_V) < eps).all():   # kolla om vi kan skriva nåt annat
            return policy, best_v
        last_V = best_v.copy()


reward_matrix = np.array([(0, 0, 0), (0, 10, 0), (0, 0, 0)])
V_0 = np.zeros(reward_matrix.shape)

policy, best_v = state_iteration(V_0, reward_matrix, eps=0.001)

print(np.flipud(best_v))
print(np.flipud(policy))
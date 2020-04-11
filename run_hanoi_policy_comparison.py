import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

from policy_iteration import policy_iteration
from value_iteration import value_iteration


from towers_of_hanoi import TohEnv
from constants import HANOI_3D
from hanoi_plots import visualize_toh

def count_different_entries(a, b):
    assert a.size == b.size, 'Arrays need to be the same size'
    return a.size - np.sum(np.isclose(a, b))

if __name__ == '__main__':
    gamma = 0.9
    theta = 0.0001
    ENV_NAME = HANOI_3D
    pi_env = TohEnv()
    pi_env = pi_env.unwrapped
    print('policy iteration begin')
    pi_policy, pi_V, pi_iter, pi_time = policy_iteration(pi_env, discount_factor=gamma, theta=theta)

    pi_env.reset()

    pi_env.apply
    

    print('policy iteration end')

    vi_env = TohEnv()
    vi_env = vi_env.unwrapped
    print('value iteration begin')
    vi_policy, vi_V, vi_iter, vi_time = value_iteration(vi_env, discount_factor=gamma, theta=theta)
    print('value iteration end')

    

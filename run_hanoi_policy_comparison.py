import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

from policy_iteration import policy_iteration
from value_iteration import value_iteration


from towers_of_hanoi import TohEnv
from constants import HANOI_6D
from hanoi_plots import visualize_toh

def count_different_entries(a, b):
    assert a.size == b.size, 'Arrays need to be the same size'
    return a.size - np.sum(np.isclose(a, b))

if __name__ == '__main__':
    ENV_NAME = HANOI_6D

    pi_gamma = 0.85
    pi_theta = 0.00001
    pi_env = TohEnv(env_name = ENV_NAME)
    pi_env = pi_env.unwrapped
    pi_policy, pi_V, pi_iter, pi_time = policy_iteration(pi_env, discount_factor=pi_gamma, theta=pi_theta)


    vi_gamma = 0.9
    vi_theta = 0.001    
    vi_env = TohEnv(env_name = ENV_NAME)
    vi_env = vi_env.unwrapped
    vi_policy, vi_V, vi_iter, vi_time = value_iteration(vi_env, discount_factor=vi_gamma, theta=vi_theta)


    print("Results %s Problem" % ENV_NAME)
    print("Policy Iteration: Iterations: %s, Total Time %s"% (pi_iter, pi_time))
    print("Value  Iteration: Iterations: %s, Total Time %s"% (vi_iter, vi_time))
    

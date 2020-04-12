import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

from policy_iteration import policy_iteration
from value_iteration import value_iteration

from constants import FL4x4, FL8x8, FL20x20, TERM_STATE_MAP, GOAL_STATE_MAP
from lake_plots import visualize_policy, visualize_value, visualize_env
from frozen_lake import FrozenLakeEnv

ENV_NAMES = [FL4x4, FL8x8, FL20x20]

def count_different_entries(a, b):
    assert a.size == b.size, 'Arrays need to be the same size'
    return a.size - np.sum(np.isclose(a, b))

if __name__ == '__main__':
    
    for ENV_NAME in ENV_NAMES:
        gamma = 0.9
        theta = 0.0001        
        env_kwargs = {
            'map_name': ENV_NAME,
            'slip_rate': .2,
            'rewards': (-0.1, -1, 1)
        }
        print(ENV_NAME)
        pi_env = FrozenLakeEnv(**env_kwargs)
        pi_env = pi_env.unwrapped
        print('policy iteration begin')
        pi_policy, pi_V, pi_iter, pi_time = policy_iteration(pi_env, discount_factor=gamma, theta=theta)
        print('policy iteration end')
        visualize_policy(pi_policy, ENV_NAME, pi_env.desc.shape,'pi', 'Policy Iteration - Optimal Policy {} Iterations'.format(pi_iter))
        visualize_value(pi_V, ENV_NAME, pi_env.desc.shape,'pi', 'Policy Iteration - Estimated Value of each State')


    for ENV_NAME in ENV_NAMES:
        gamma = 0.85
        theta = 0.001        
        env_kwargs = {
            'map_name': ENV_NAME,
            'slip_rate': .2,
            'rewards': (-0.1, -1, 1)
        }
        vi_env = FrozenLakeEnv(**env_kwargs)
        vi_env = vi_env.unwrapped
        print('value iteration begin')
        vi_policy, vi_V, vi_iter, vi_time = value_iteration(vi_env, discount_factor=gamma, theta=theta)
        print('value iteration end')
        visualize_env(vi_env, ENV_NAME, 'Frozen Lake')
        visualize_policy(vi_policy, ENV_NAME, vi_env.desc.shape,'vi', 'Value Iteration - Optimal Policy {} Iterations'.format(vi_iter))
        visualize_value(vi_V, ENV_NAME, vi_env.desc.shape,'vi', 'Value Iteration - Estimated Value of each State')
    

    #assert np.all(np.isclose(pi_policy, vi_policy, atol=0.05)), "Policies don't match"
    #assert np.all(np.isclose(pi_V, vi_V, atol=0.05)), "Values don't match"




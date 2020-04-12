from pprint import pprint

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

from constants import HANOI_3D, HANOI_4D, HANOI_5D, HANOI_6D
from lake_plots import visualize_policy, visualize_value

from q_learning import q_learning
from towers_of_hanoi import TohEnv
import os

FIGURES_DIRECTORY = './output/hanoi/figures'

if not os.path.exists(FIGURES_DIRECTORY):
    os.makedirs(FIGURES_DIRECTORY)

def get_state_action_value(final_policy):
    return np.max(final_policy, axis=1)

def plot_epsilon_decay(ENV_NAME, epsilon, decay, n_episodes, stats):
    e_prime = epsilon * np.ones(n_episodes)
    for ix in range(n_episodes):
        e_prime[ix] *= decay ** ix
    smoothing_window=10
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed, label='Episode reward')
    plt.plot(e_prime, label='Decayed epsilon value', linestyle='--')
    plt.title("Epsilon-greedy with decay (epsilon=%.1f, decay=%.3f)" % (epsilon, decay))
    plt.xlabel('Episode')
    plt.legend(loc='best')
    file_name = '{}/{}/{}_epsilondecay.png'.format(FIGURES_DIRECTORY, ENV_NAME, 'ql')
    plt.savefig(file_name, format='png', dpi=150) 
    plt.close()

if __name__ == '__main__':


    ENV_NAMES = [HANOI_3D, HANOI_4D, HANOI_5D, HANOI_6D]

    for ENV_NAME in ENV_NAMES:
        if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,ENV_NAME)):
            os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,ENV_NAME))       
        env = TohEnv(env_name=ENV_NAME)
        env = env.unwrapped
        # Tunables
        method='greedy'
        n_episodes = 100000
        gamma = 0.95
        alpha = 0.6
        epsilon = 1.0
        decay = 0.999
        Ne = 10
        q, stats, Nsa, policy = q_learning(
            env=env,
            method=method,
            num_episodes=n_episodes,
            discount_factor=gamma,
            alpha=alpha,
            epsilon=epsilon,
            decay=decay,
            Ne=Ne
        )
        pprint(q)
        pprint(Nsa)
        value = get_state_action_value(policy)
        print(value)
        plot_epsilon_decay(ENV_NAME, epsilon, decay, n_episodes, stats)

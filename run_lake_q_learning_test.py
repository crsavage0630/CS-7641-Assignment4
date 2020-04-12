from pprint import pprint

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

from constants import FL4x4, FL8x8, FL20x20, TERM_STATE_MAP, GOAL_STATE_MAP
from lake_plots import visualize_policy, visualize_value

from q_learning import q_learning
from frozen_lake import FrozenLakeEnv
import os

FIGURES_DIRECTORY = './output/lake/figures'

if not os.path.exists(FIGURES_DIRECTORY):
    os.makedirs(FIGURES_DIRECTORY)
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,FL4x4)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,FL4x4))
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,FL8x8)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,FL8x8))
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,FL20x20)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,FL20x20))   

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


    ENV_NAMES = [FL20x20]

    for ENV_NAME in ENV_NAMES:
        env = FrozenLakeEnv(
            map_name=ENV_NAME,
            rewards=(-0.01, -1, 1), # living, hole, goal
            slip_rate=0.2
        )
        env = env.unwrapped
        # Tunables
        method='greedy'
        n_episodes = 10000
        gamma = 0.90
        alpha = 0.75
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
        visualize_value(value, ENV_NAME, env.desc.shape, 'qlearner','Q Learner - Estimated Value of each State')
        visualize_policy(policy, ENV_NAME, env.desc.shape,'qlearner', 'Q Learner - Optimal Policy')
        plot_epsilon_decay(ENV_NAME, epsilon, decay, n_episodes, stats)



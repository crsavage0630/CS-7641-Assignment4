import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import os

from constants import FL4x4, FL8x8, FL20x20, TERM_STATE_MAP, GOAL_STATE_MAP

from policy_iteration import policy_iteration
from value_iteration import value_iteration
from frozen_lake import FrozenLakeEnv

FIGURES_DIRECTORY = './output/lake/figures'

if not os.path.exists(FIGURES_DIRECTORY):
    os.makedirs(FIGURES_DIRECTORY)
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,FL4x4)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,FL4x4))
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,FL8x8)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,FL8x8))
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,FL20x20)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,FL20x20))    

n_trials = 3
thetas = np.logspace(-5, -1, 5)
gammas = np.linspace(.5, 1, 5)
gammas[-1] = 0.98


def get_environment(ENV_NAME):
    env_kwargs = {
        'map_name': ENV_NAME,
        'slip_rate': .2,
        'rewards': (-0.01, -1, 1)
    }
    env = FrozenLakeEnv(**env_kwargs)
    env = env.unwrapped
    return env


def policy_iteration_test(ENV_NAME):
    env = get_environment(ENV_NAME)
    n_iters = {k: [] for k in thetas}
    runtimes = {k: [] for k in thetas}

    for theta in thetas:
        print('theta=%s' % theta)
        for gamma in gammas:
            temp_n_iters = []
            temp_runtimes = []
            for t in range(n_trials):
                print('theta=%s gamma=%s trial=%s' % (theta,gamma,t))
                _, _, n_iter, runtime = policy_iteration(env, discount_factor=gamma, theta=theta, max_iter=100)
                temp_n_iters.append(n_iter)
                temp_runtimes.append(runtime)
            n_iters[theta].append(np.mean(temp_n_iters))
            runtimes[theta].append(np.mean(temp_runtimes))

    for key, iterlist in n_iters.items():
        plt.plot(gammas, iterlist, label=('theta=%s' % key))
    plt.title('PI - Iterations until Convergence %s Problem' % ENV_NAME)
    plt.legend(loc='upper left')
    plt.xlabel('Gamma')
    plt.ylabel('Iterations')
    file_name = '{}/{}/{}_iterations.png'.format(FIGURES_DIRECTORY, ENV_NAME, 'pi')
    plt.savefig(file_name, format='png', dpi=150) 
    plt.close()

    for key, rt in runtimes.items():
        plt.plot(gammas, [t * 1000 for t in rt], label=('theta=%s' % key))
    plt.title('PI - Time until Convergence %s Problem' % ENV_NAME)
    plt.legend(loc='upper left')
    plt.xlabel('Gamma')
    plt.ylabel('Total Milliseconds')
    file_name = '{}/{}/{}_time.png'.format(FIGURES_DIRECTORY, ENV_NAME, 'pi')
    plt.savefig(file_name, format='png', dpi=150)   
    plt.close()          

def value_iteration_test(ENV_NAME):
    env = get_environment(ENV_NAME)
    n_iters = {k: [] for k in thetas}
    runtimes = {k: [] for k in thetas}

    for theta in thetas:
        print('theta=%s' % theta)
        for gamma in gammas:
            temp_n_iters = []
            temp_runtimes = []
            for t in range(n_trials):
                print('theta=%s gamma=%s trial=%s' % (theta,gamma,t))
                _, _, n_iter, runtime = value_iteration(env, discount_factor=gamma, theta=theta)
                temp_n_iters.append(n_iter)
                temp_runtimes.append(runtime)
            n_iters[theta].append(np.mean(temp_n_iters))
            runtimes[theta].append(np.mean(temp_runtimes))

    for key, iterlist in n_iters.items():
        plt.plot(gammas, iterlist, label=('theta=%s' % key))
    plt.title('VI - Iterations until Convergence %s Problem' % ENV_NAME)
    plt.legend(loc='upper left')
    plt.xlabel('Gamma')
    plt.ylabel('Iterations')
    file_name = '{}/{}/{}_iterations.png'.format(FIGURES_DIRECTORY, ENV_NAME, 'vi')
    plt.savefig(file_name, format='png', dpi=150)  
    plt.close()         

    for key, rt in runtimes.items():
        plt.plot(gammas, [t * 1000 for t in rt], label=('theta=%s' % key))
    plt.title('VI - Time until Convergence %s Problem' % ENV_NAME)
    plt.legend(loc='upper left')
    plt.xlabel('Gamma')
    plt.ylabel('Total Milliseconds)')
    file_name = '{}/{}/{}_time.png'.format(FIGURES_DIRECTORY, ENV_NAME, 'vi')
    plt.savefig(file_name, format='png', dpi=150)     
    plt.close()    



if __name__ == '__main__':

    ENV_NAMES = [FL4x4, FL8x8, FL20x20]

    for ENV_NAME in ENV_NAMES:
        print(ENV_NAME)
        policy_iteration_test(ENV_NAME)
    
    # for ENV_NAME in ENV_NAMES:
    #     print(ENV_NAME)
    #     value_iteration_test(ENV_NAME)
    

          

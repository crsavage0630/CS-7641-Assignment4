import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

from constants import FL4x4, FL_V0_ABBR, FL8x8, FL8x8_V0_ABBR, TERM_STATE_MAP, GOAL_STATE_MAP

from policy_iteration import policy_iteration
from value_iteration import value_iteration
from my_frozen_lake import FrozenLakeEnv

ENV_NAME = FL4x4
ENV_ABBR = FL_V0_ABBR


if __name__ == '__main__':
    ALGO = 'pi'

    env_kwargs = {
        'map_name': ENV_NAME,
        'slip_rate': .2,
        'rewards': (-0.04, -1, 1)
    }
    env = FrozenLakeEnv(**env_kwargs)
    env = env.unwrapped


    if ALGO == 'pi':
        n_trials = 10
        gammas = np.linspace(.05, 1, 20)
        gammas[-1] = 0.98
        thetas = np.logspace(-3, -1, 3)
        print(thetas)
        n_iters = {k: [] for k in thetas}
        runtimes = {k: [] for k in thetas}

        for theta in thetas:
            print('theta=%s' % theta)
            for gamma in gammas:
                print('gamma=%s' % gamma)
                temp_n_iters = []
                temp_runtimes = []
                for t in range(n_trials):
                    print('trial %i' % t)
                    _, _, n_iter, runtime = policy_iteration(env, discount_factor=gamma, theta=theta)
                    temp_n_iters.append(n_iter)
                    temp_runtimes.append(runtime)
                n_iters[theta].append(np.mean(temp_n_iters))
                runtimes[theta].append(np.mean(temp_runtimes))

        for key, iterlist in n_iters.items():
            plt.plot(gammas, iterlist, label=('c=%s' % key))
        plt.title('PI - Iterations until Convergence')
        plt.legend(loc='upper left')
        plt.xlabel('Gamma')
        plt.ylabel('Iterations')
        plt.savefig('figures/%s_%s_iter' % (ENV_ABBR, ALGO))
        plt.show()

        for key, rt in runtimes.items():
            plt.plot(gammas, [t * 1000 for t in rt], label=('c=%s' % key))
        plt.title('PI - Time until Convergence')
        plt.legend(loc='upper left')
        plt.xlabel('Gamma')
        plt.ylabel('Wall clock time (msec)')
        plt.savefig('figures/%s_%s_runtime' % (ENV_ABBR, ALGO))
        plt.show()

    elif ALGO == 'vi':
        error_bound_constants = np.logspace(-4, -1, 4)
        gammas = np.linspace(.05, 1, 20)
        gammas[-1] = 0.98
        n_trials = 10

        n_iters = {k: [] for k in error_bound_constants}
        runtimes = {k: [] for k in error_bound_constants}

        for c in error_bound_constants:
            print('c=%s' % c)
            for gamma in gammas:
                print(gamma)
                temp_n_iters = []
                temp_runtimes = []
                for t in range(n_trials):
                    print('trial %i' % t)
                    _, _, n_iter, runtime = value_iteration(env, discount_factor=gamma, theta=c)
                    temp_n_iters.append(n_iter)
                    temp_runtimes.append(runtime)
                n_iters[c].append(np.mean(temp_n_iters))
                runtimes[c].append(np.mean(temp_runtimes))

        for key, iterlist in n_iters.items():
            plt.plot(gammas, iterlist, label=('c=%.4f' % key).rstrip('0'))
        plt.title('VI - Iterations until Convergence')
        plt.legend(loc='upper left')
        plt.xlabel('Gamma')
        plt.ylabel('Iterations')
        plt.savefig('figures/%s_%s_iter' % (ENV_ABBR, ALGO))
        plt.show()
        for key, rt in runtimes.items():
            plt.plot(gammas, [t * 1000 for t in rt], label=('c=%.4f' % key).rstrip('0'))
        plt.title('VI - Time until Convergence')
        plt.legend(loc='upper left')
        plt.xlabel('Gamma')
        plt.ylabel('Wall clock time (msec)')
        plt.savefig('figures/%s_%s_runtime' % (ENV_ABBR, ALGO))
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns; sns.set()

from constants import TERM_STATE_MAP, GOAL_STATE_MAP, FL4x4, FL8x8, FL20x20

FIGURES_DIRECTORY = './output/lake/figures'

if not os.path.exists(FIGURES_DIRECTORY):
    os.makedirs(FIGURES_DIRECTORY)
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,FL4x4)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,FL4x4))
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,FL8x8)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,FL8x8))
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,FL20x20)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,FL20x20))            

def standard_figure(arr,M,N):
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(arr, cmap='cool')
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(M))
    ax.set_yticklabels(np.arange(N))
    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)    
    return (fig, ax, im)


def visualize_env(env, env_name, title=None):
    shape = env.desc.shape
    M = shape[0]
    N = shape[1]
    arr = np.zeros(shape)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                arr[i, j] = 0.25
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                arr[i, j] = 1.0

    fig, ax, im = standard_figure(arr,M,N)

    for i in range(M):
        for j in range(N):
            if (i, j) == (0, 0):
                ax.text(j, i, 'S', ha='center', va='center', color='k', size=18)
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                ax.text(j, i, 'x', ha='center', va='center', color='k', size=18)
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                ax.text(j, i, '$', ha='center', va='center', color='k', size=18)
            else:
                pass
    fig.tight_layout()
    if title:
        ax.set_title(title)
    file_name = '{}/{}/env.png'.format(FIGURES_DIRECTORY, env_name)
    plt.savefig(file_name, format='png', dpi=150)
    plt.close()


def visualize_policy(pi, env_name, shape, mdp, title=None):
    M = shape[0]
    N = shape[1]
    actions = np.argmax(pi, axis=1).reshape(shape)
    mapping = {
        0: '<',
        1: 'v',
        2: '>',
        3: '^'
    }

    font_size = 14
    if env_name == FL20x20:
        font_size = 9

    arr = np.zeros(shape)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                arr[i, j] = 0.25
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                arr[i, j] = 1.0
    
    fig, ax, im = standard_figure(arr,M,N)

    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                ax.text(j, i, 'x', ha='center', va='center', color='k', size=18)
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                ax.text(j, i, '$', ha='center', va='center', color='k', size=18)
            else:
                ax.text(j, i, mapping[actions[i, j]], ha='center', va='center', color='k', size=font_size)
    # fig.tight_layout()
    if title:
        ax.set_title(title)
    file_name = '{}/{}/{}_policy.png'.format(FIGURES_DIRECTORY, env_name, mdp)
    plt.savefig(file_name, format='png', dpi=150)
    plt.close()

def render_policy(pi, env_name, shape):
    actions = np.argmax(pi, axis=1)
    for index in TERM_STATE_MAP[env_name]:
        actions[index] = 999
    for index in GOAL_STATE_MAP[env_name]:
        actions[index] = 1000

    pi = np.reshape(actions, shape)

    mapping = {
        0: ' < ',
        1: ' v ',
        2: ' > ',
        3: ' ^ ',
        999: ' . ',
        1000: ' $ '
    }
    mapper = np.vectorize(lambda k: mapping[k])
    np.apply_along_axis(lambda row: print(' '.join(row)), axis=1, arr=mapper(pi))


def visualize_value(V, env_name, shape, mdp, title=None):
    font_size = 14
    if env_name == FL20x20:
        font_size = 6
    M = shape[0]
    N = shape[1]
    arr = V.reshape(shape)
    fig, ax, im = standard_figure(arr,M,N)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[env_name]:
                ax.text(j, i, 'x', ha='center', va='center', color='k', size=font_size)
            elif (N * i + j) in GOAL_STATE_MAP[env_name]:
                ax.text(j, i, '$', ha='center', va='center', color='k', size=font_size)
            else:
                ax.text(j, i, '%.2f' % (arr[i, j]), ha='center', va='center', color='k', size=font_size)
    # fig.tight_layout()
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('State-value estimate', rotation=-90, va="bottom")
    if title:
        ax.set_title(title)
    file_name = '{}/{}/{}_value.png'.format(FIGURES_DIRECTORY, env_name, mdp)
    plt.savefig(file_name, format='png', dpi=150)
    plt.close()

def better_desc(desc):
    mapping = {
        b'S': b' S ',
        b'F': b' * ',
        b'H': b' O ',
        b'G': b' $ '
    }
    mapper = np.vectorize(lambda k: mapping[k])
    return mapper(desc)
import cv2
import numpy as np
from os import path
import os
import errno

from constants import HANOI_3D

##Credit abharadwaj42
##https://github.gatech.edu/gist/abharadwaj42/77b9c9384f2c134572f591d20a6564ea

FIGURES_DIRECTORY = './output/hanoi/figures'

if not os.path.exists(FIGURES_DIRECTORY):
    os.makedirs(FIGURES_DIRECTORY)
if not os.path.exists('{}/{}'.format(FIGURES_DIRECTORY,HANOI_3D)):
    os.makedirs('{}/{}'.format(FIGURES_DIRECTORY,HANOI_3D))


def discs():
    return {
        3: [(120, 300), (280, 300), (0, 0, 255)],
        2: [(150, 260), (250, 260), (0, 255, 0)],
        1: [(170, 230), (230, 230), (255, 0, 0)],
        0: [(180, 170), (220, 170), (0, 255, 255)]
    }

def output_folder(env_name):
    return '{}/{}'.format(FIGURES_DIRECTORY,env_name)
    
def visualize_toh(states, env_name, file_prefix):
    step = 0
    for index, state in enumerate(states):
        if index > 20:
            break
        draw_toh(state, output_folder(env_name), file_prefix, step)
        step += 1
        
        
def draw_toh(state, output_folder, file_prefix, step=0):
    img = np.zeros((300, 800, 3), dtype="uint8")
    draw_towers(img)
    for i in range(len(state)):
        peg = state[i]
        draw_discs(img, num_of_discs=len(peg), order=peg, peg=i)
    try:
        os.makedirs(output_folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    file_name = "TOH_" + file_prefix + "_" + str(step) + ".png"
    cv2.imwrite(path.join(output_folder, file_name), img)


def draw_towers(img):
    cv2.line(img, (0, 300), (800, 300), (255, 255, 255), 5)
    cv2.line(img, (200, 10), (200, 300), (255, 255, 255), 5)
    cv2.line(img, (400, 10), (400, 300), (255, 255, 255), 5)
    cv2.line(img, (600, 10), (600, 300), (255, 255, 255), 5)


def draw_discs(img, num_of_discs=4, order=(3, 2, 1, 0), peg=0):
    for i in range(num_of_discs):
        disc_id = order[i]
        details = discs()[disc_id]
        offset = peg * 200
        start = (details[0][0] + offset, details[0][1])
        end = (details[1][0] + offset, details[1][1])
        cv2.line(img, start, end, details[2], 30)
 
import sys
import os

import numpy as np


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} DIR')

    path = sys.argv[1]
    scores = []
    for file in os.listdir(path):
        file_path = f'{path}/{file}'

        with open(file_path, 'r') as f:
            lines = f.readlines()

        score_lines = [float(x.split(':')[1]) for x in lines if x.startswith('Accuracy')]
        scores.append(score_lines[0])

    print('Mean:', np.mean(scores))


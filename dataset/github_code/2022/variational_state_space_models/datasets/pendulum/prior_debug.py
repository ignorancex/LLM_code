import os
import sys
if __name__=="__main__":
    sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
    sys.path.pop(0)

from datasets.pendulum.pendulum import PendulumTrajectoryConfig, PendulumTrajectoryDataset

config = PendulumTrajectoryDataset.BUILDER_CONFIGS[int(sys.argv[1])]
print(config.name)
print(config.Prior)
A = config.A + config.B @ config.K
N = A @ config.Prior @ A.T + config.B @ config.B.T * config.u_Sigma + config.Sigma
print(N)

import tensorflow_datasets as tfds

train = tfds.load('pendulum_trajectory/64x64_lv', split="train")

points = []

for s in tfds.as_numpy(train):
    points.append(s['states'])

import numpy as np
points = np.stack(points)
#points = points.reshape((-1, 2))
print('1')
print(points[:,1].mean(0))
print(np.cov(points[:,1].T))
print('2')
print(points[:,2].mean(0))
print(np.cov(points[:,2].T))
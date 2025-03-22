import os
import sys
if __name__=="__main__":
    sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
    sys.path.pop(0)

import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

from datasets.airsim.airsim import AirsimTrajectoryDataset

config = AirsimTrajectoryDataset.BUILDER_CONFIGS[int(sys.argv[1])]
print(config.Prior)
A = config.A + config.B @ config.K
print(A)
N = A @ config.Prior @ A.T + config.B @ config.B.T * config.u_Sigma + config.Sigma
print(N)
import tensorflow_datasets as tfds
#
train = tfds.load(f'airsim_trajectory/{config.name}', split="train")

points = []

for s in tfds.as_numpy(train):
    points.append(s['states'])

import numpy as np
points = np.stack(points)
#points = points.reshape((-1, 4))
print('0')
print(points[:,0].mean(0))
print(np.cov(points[:,0].T))
print('0_prop')
print(A @ np.cov(points[:,0].T) @ A.T + config.B @ config.B.T * config.u_Sigma + config.Sigma)
print('1')
print(points[:,1].mean(0))
print(np.cov(points[:,1].T))
print('2')
print(points[:,2].mean(0))
print(np.cov(points[:,2].T))

import pandas as pd
import numpy as np
progress = pd.read_csv('/home/francisco/primitives-her/log/pickthrowMAXQ0.5EPS_2prim_greedy-ddpg-FetchPickAndThrow-v1-pickmaxq/progress.csv')["TimeCost(sec)"]
#progress2 =pd.read_csv('/home/francisco/primitives-her/log/multi_step-ddpg-FetchPickObstacle-v1-pickmaxq/progress.csv')["TimeCost(sec)"]
progress3 =pd.read_csv('/home/francisco/C-HGG/log/pickthrow-ddpg-FetchPickAndThrow-v1-hgg-graph-stop-curriculum/progress.csv')["TimeCost(sec)"]

print("MaxQ: {}".format(np.mean(progress.to_numpy())))
#print("Multi-Step: {}".format(np.mean(progress2.to_numpy())))
print("G-HGG: {}".format(np.mean(progress3.to_numpy())))

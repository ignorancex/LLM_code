import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import scipy.stats as stats
import random
import numpy as np
import json
import sys

path=sys.argv[1]
title=sys.argv[2]
dir_path = path.split('/')[0]
plt_name = '_'.join(path.split('.')[1:3])+'.png'

with open(path) as f:
    line = f.read().strip()
    scores = json.loads(line)
#scores = random.sample(scores, 1000)

mean = np.mean(scores)
std = np.std(scores)
print (plt_name, mean, std)

plt.hist(scores, weights=np.ones(len(scores)) / len(scores), bins=30, range=(0.35, 0.95))
#plt.hist(scores, weights=np.ones(len(scores)) / len(scores), bins=30)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.text(0.5, .05, r'$\mu=%.3f,\ \sigma=%.3f$' % (mean, std))
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.title(title)
#plt.show()
plt.savefig(dir_path+'/'+plt_name)

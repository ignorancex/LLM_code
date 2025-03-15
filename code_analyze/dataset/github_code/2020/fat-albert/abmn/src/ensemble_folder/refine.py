import numpy as np

predictions = np.loadtxt('preds.txt')
for i in predictions:
    print(int(i))

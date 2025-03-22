import numpy as np

predictions = np.loadtxt('preds.txt')
f = open("plot_results.txt", "a")
counter = 0
for i in predictions:
    f.write("test:" + str(counter) + " " + str(int(i)) + "\n")
    counter = counter + 1

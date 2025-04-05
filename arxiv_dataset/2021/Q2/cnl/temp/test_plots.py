import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
x = np.arange(300)



linestyles = ["-", "--", "-.", ":"]
markers = ["*", ">", "+", "o"]
markers = ["", "", "", "", "", ""]
#plt.style.use("seaborn-paper")

for style in plt.style.available:

    plt.figure()
    plt.style.use(style)

    for n in range(len(linestyles)):
        plt.plot( x, sp.log(1+x+10*n), linestyle=linestyles[n], marker=markers[n], \
        markevery=10, markersize =8,color="black", label="l={}, m={}".format(linestyles[n], markers[n]) )

    plt.legend()

    plt.savefig("style_{}.png".format(style))

    plt.clf()
    plt.close()

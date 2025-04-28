import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

##Adjust plotting defaults
rcParams["axes.linewidth"] = 4

rcParams["ytick.right"] = True
rcParams["ytick.direction"] = "in"
rcParams["ytick.minor.visible"] = True
rcParams["ytick.major.left"] = True
rcParams["ytick.major.right"] = True
rcParams["ytick.minor.left"] = True
rcParams["ytick.minor.right"] = True
rcParams["ytick.major.size"] = 20
rcParams["ytick.minor.size"] = 10
rcParams["ytick.major.width"] = 2
rcParams["ytick.minor.width"] = 2


rcParams["xtick.top"] = True
rcParams["xtick.direction"] = "in"
rcParams["xtick.minor.visible"] = True
rcParams["xtick.major.top"] = True
rcParams["xtick.major.bottom"] = True
rcParams["xtick.minor.top"] = True
rcParams["xtick.minor.bottom"] = True
rcParams["xtick.major.size"] = 20
rcParams["xtick.minor.size"] = 10
rcParams["xtick.major.width"] = 2
rcParams["xtick.minor.width"] = 2


axesLabelSize = 36
tickLabelSize = 34
textSize = 34

def radiativeTransfer(tau, gamma):
    return (1 + 1/gamma + (1 - 1/gamma)*np.exp(-gamma*tau))**(0.25)



tau = np.logspace(-3,2,10000)

temperatures = []

gamma = [0.25,0.5,1,2.5,3.5]

for i in gamma:
    temperatures.append(radiativeTransfer(tau, i))

fig, ax = plt.subplots(1,1, figsize = (12,8))

logTau = np.log10(tau)
for temp in temperatures:
    ax.plot(temp, logTau, color = "black", lw = 4)


ax.set_xlabel("Temperature $T/c_3$", fontsize = axesLabelSize)
ax.set_ylabel("Log Infared Optical Depth $\\tau$", fontsize = axesLabelSize)

yTicks = [-3,-2,-1,0,1,2]
xTicks = [0.9,1,1.1,1.2,1.3,1.4,1.5]
ax.set_xticks(xTicks)
ax.set_yticks(yTicks)

ax.set_ylim(2,-3)
ax.set_xlim(0.96,1.54)

ax.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)


ax.text(1,-0.5, "$\gamma = 3.5$", fontsize = textSize)
ax.text(1.11,0.25, "$2.5$", fontsize = textSize)
ax.text(1.2,0.6, "$1$", fontsize = textSize)
ax.text(1.325,1, "$0.5$", fontsize = textSize)
ax.text(1.42,1.5, "$0.25$", fontsize = textSize)
plt.tight_layout()

plt.savefig("./Ttau_const_gamma.pdf")
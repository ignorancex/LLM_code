import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
##Standard Imports

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

def radiativeMap(tau, gamma, p):
    newTau = p[0]*np.exp(-p[1]/((1 + 1/gamma + (1 - 1/gamma)*np.exp(-gamma*tau)))**(1/4))
    return newTau

fig2, ax2 = plt.subplots(1,1, figsize = (12,8)) ##Create figure
gammas = np.linspace(0.5,5.5,500)## For orbit diagrams we want a finer granularity
p1 = 1.5
p2 = 30

maxTau = 0
for gamma in gammas:
    d = p1*np.exp(p2*2**(-0.25))
    p = [d,p2] ##Prepare parameters
    tauArr = [0]
    for i in range(300): ##Iterate the map 1000 times
        tauArr.append(radiativeMap(tauArr[i],gamma,p)) 
        if tauArr[i] > maxTau:
            maxTau = tauArr[i]
    xCoor = np.ones(len(tauArr))*gamma ##Plot all points the map visited against the p1 coordinate
    gammaTauArr= []
    for i in tauArr:
        gammaTauArr.append(i*gamma)
    ax2.scatter(xCoor[100:],gammaTauArr[100:], s = 1, color = "black", alpha = 0.2)

ax2.set_xlabel("Visible-to-IR Opacity $\gamma $", fontsize = axesLabelSize)
ax2.set_ylabel("Visible Optical Depth $\gamma\\tau$", fontsize = axesLabelSize)
xMax = np.ceil(maxTau*10)/10

ax2.text(4.4,8, "$p_1 = {:.1f}$".format(p1), fontsize = textSize)
ax2.text(4.4,7, "$p_2 = {:.0f}$".format(p2), fontsize = textSize)


ax2.set_xlim(0,6)
ax2.set_ylim(-0.5,10)
xTicks = [0,1,2,3,4,5,6]
yTicks = np.arange(0,10.1,2.5)
ax2.set_xticks(xTicks)
ax2.set_yticks(yTicks)

ax2.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)

plt.tight_layout()
plt.savefig("../orbitDiagrams/constGammaOrbitDiagram.jpg")
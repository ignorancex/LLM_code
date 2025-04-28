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


axesLabelSize = 40
tickLabelSize = 34
textSize = 34



def gamma(tau, p3, p4):
    return 10**(p3*np.tanh(np.log10(tau)/p4))

fig2, axs = plt.subplots(2,1, figsize = (12,16), sharex = True)
fig2.subplots_adjust(hspace=0)

x = np.logspace(-3,3, 500)

y = gamma(x, 0.5, 0.5)

logX = np.log10(x)
logY = np.log10(y)

gammaTau = np.multiply(x,y)
logX1 = np.log10(gammaTau)

axs[0].plot(logX1, logY, color = "black", lw = 4)
axs[0].plot(logX,logY, ls = "--", color = "xkcd:neon red", lw = 4)
#axs[0].set_xscale("log")
#axs[0].set_yscale("log")
axs[0].set_xlim(-3,3)
axs[0].set_ylim(-0.6,0.6)

yTicks = [-0.5,-0.25,0,0.25,0.5]
xTicks = [-3,-2,-1,0,1,2,3]
axs[0].set_yticks(yTicks)
axs[0].set_xticks(xTicks)
axs[0].set_xticklabels(xTicks, color = "xkcd:neon red")
axs[0].set_xlabel("$\log \\tau$", color = "xkcd:neon red", fontsize = axesLabelSize, labelpad=14)
axs[0].xaxis.set_label_position("top")
#axs[0].xaxis.set_label_coords(0.5,1.18)
#axs[0].yaxis.set_label_coords(-0.1,0.5)


def radiativeTransfer(tau):
    gm = gamma(tau, 0.5, 0.5)
    return (1 + 1/gm + (1 - 1/gm)*np.exp(-gm*tau))**(0.25)

def guillot(tau):
    gm = gamma(tau, 0.5, 0.5)
    return (1 + 1/gm + (gm - 1/gm)*np.exp(-gm*tau))**(0.25) 

y1 = radiativeTransfer(x)
y2 = guillot(x)

axs[1].plot(logX1, y1, color = "black", lw = 4)
axs[1].plot(logX1,y2,color = "navy", ls = "-.", lw = 4)
#axs[1].set_xscale("log")
axs[1].set_ylim(1,1.25)

yTicks = [1.05,1.1,1.15,1.2]
axs[1].set_yticks(yTicks)
axs[1].set_ylim(1,1.25)

axs[1].set_xlim(-3,3)
axs[1].set_xticks(xTicks)
axs[1].set_xticklabels(xTicks)

axs[1].text( -2.7,1.165, "Map B: $T$", fontsize = textSize)
axs[1].text( -2.7,1.05, "Map C: $T'$", fontsize = textSize, color = "navy")

axs[0].set_ylabel("$\log\gamma$", fontsize = axesLabelSize)
axs[1].set_ylabel("$T/c_3$ or $T'/c_3$", fontsize = axesLabelSize)
axs[1].set_xlabel("$\log\gamma \\tau $", fontsize = axesLabelSize)


axs[0].tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
axs[0].tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
axs[0].tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10, labeltop = True, labelbottom = False)
axs[0].tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
axs[1].tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
axs[1].tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
axs[1].tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
axs[1].tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)

plt.tight_layout(pad = 0)

plt.savefig("../Ttau_var_gamma_b.pdf")
plt.show()

fig2,ax2 = plt.subplots(1,1, figsize = (12,8))
ax2.plot(logX, logY, color = "black", lw = 4)
ax2.set_xlim(-3,3)
ax2.set_ylim(-0.6,0.6)

yTicks = [-0.5,-0.25,0,0.25,0.5]
xTicks = [-3,-2,-1,0,1,2,3]
ax2.set_yticks(yTicks)
ax2.set_xticks(xTicks)
ax2.set_ylabel("$\log\gamma$", fontsize = axesLabelSize)
ax2.set_xlabel("$\log\\tau$", fontsize = axesLabelSize)

ax2.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)


plt.tight_layout()
plt.savefig("./gamma_v_tau.pdf")
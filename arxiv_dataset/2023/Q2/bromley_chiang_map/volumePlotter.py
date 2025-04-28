import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

chaoticP1, chaoticP2, chaoticP3, lyapunovExponent = np.loadtxt(".\data\chaoticPointsMC.txt", delimiter = ',', unpack = True)
#data\chaoticPointsGuillot2.txt
rcParams["axes.linewidth"] = 3
rcParams["axes.labelsize"] = 19

rcParams["ytick.right"] = True
rcParams["ytick.direction"] = "in"
rcParams["ytick.minor.visible"] = True
rcParams["ytick.major.left"] = True
rcParams["ytick.major.right"] = True
rcParams["ytick.minor.left"] = True
rcParams["ytick.minor.right"] = True 
rcParams["ytick.major.size"] = 12
rcParams["ytick.minor.size"] = 6



rcParams["xtick.top"] = True
rcParams["xtick.direction"] = "in"
rcParams["xtick.minor.visible"] = True
rcParams["xtick.major.top"] = True
rcParams["xtick.major.bottom"] = True
rcParams["xtick.minor.top"] = True
rcParams["xtick.minor.bottom"] = True
rcParams["xtick.major.size"] = 12
rcParams["xtick.minor.size"] = 6

tickLabelSize = 28
axesLabelSize = 34
textSize = 28

    
def viewAdjustment(theta, phi, chaoticP1, chaoticP2, chaoticP3):
    alpha = []
    viewX = 1.5*np.cos(theta*np.pi/180)*np.cos(phi*np.pi/180)
    viewY = 1.5*np.cos(theta*np.pi/180)*np.sin(phi*np.pi/180)
    viewZ = 1.5*np.sin(theta*np.pi/180)
    for i in range(len(chaoticP1)):
                x = chaoticP1[i]
                y = chaoticP2[i]
                z = chaoticP3[i]
                x = x - 0.5
                y = (y-20)/20 - 0.5
                z = z/2 - 0.5
                dist = np.sqrt((viewX-x)**2+(viewY-y)**2+(viewZ-z)**2)
                alpha.append(dist)

    maxAlpha = max(alpha)
    minAlpha = min(alpha)
    for i in range(len(alpha)):
        alpha[i] = 10**(-(maxAlpha - alpha[i])/maxAlpha)
    return alpha


##Plot 1
theta = 15
phi = -60
alpha = viewAdjustment(theta, phi, chaoticP1,chaoticP2,chaoticP3)
fig = plt.figure(figsize = (24,20))
ax1 = fig.add_subplot(221,projection = '3d')
logChaoticP1 = np.log10(chaoticP1)
scatt = ax1.scatter3D(chaoticP1, chaoticP2, chaoticP3, c = lyapunovExponent, cmap = "viridis_r", alpha = alpha, marker = '.')
#ax.scatter(logChaoticC1, chaoticC2, chaoticC3)

ax1.set_xlabel("$p_1$", fontsize = axesLabelSize, labelpad = 25)
ax1.set_ylabel("$p_2$", fontsize = axesLabelSize, labelpad = 25)
ax1.set_zlabel("$p_3$", fontsize = axesLabelSize, labelpad = 25)

#[0.2,0.4,0.6,0.8,1.0]
#[0.1,0.2,0.3]
xTicks = [0.2,0.4,0.6,0.8,1.0]
yTicks = [20,25,30,35,40]
zTicks = [0.6,1.0,1.4,1.8]
ax1.set_xticks(xTicks)
ax1.set_yticks(yTicks)
ax1.set_zticks(zTicks)
plt.gca().invert_xaxis()

ax1.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 5)
ax1.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 5)
ax1.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 5)
ax1.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 5)
ax1.tick_params(axis = 'z', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax1.tick_params(axis = 'z', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)

ax1.text(0.8,30,0.5,"valley", fontsize = textSize, weight = "bold")
ax1.text(1.0,19,1.4,"wall",fontsize = textSize, weight = "bold")
ax1.text(0.25,40,1.4,"wall", fontsize = textSize, weight = "bold")


ax1.view_init(theta,phi)

##Plot 2
theta = -15
phi = 120
alpha = viewAdjustment(theta, phi, chaoticP1,chaoticP2,chaoticP3)

ax2 = fig.add_subplot(222,projection = '3d')
scatt = ax2.scatter3D(chaoticP1, chaoticP2, chaoticP3, c = lyapunovExponent, cmap = "viridis_r", alpha = alpha, marker = '.')
#ax.scatter(logChaoticC1, chaoticC2, chaoticC3)
ax2.set_xlabel("$p_1$", fontsize = axesLabelSize, labelpad = 20)
ax2.set_ylabel("$p_2$", fontsize = axesLabelSize, labelpad = 20)
ax2.set_zlabel("$p_3$", fontsize = axesLabelSize, labelpad = 23)

xTicks = [0.2,0.4,0.6,0.8,1.0]
yTicks = np.arange(20,40.01,5)
zTicks = np.arange(0.6,2.01,0.4)
ax2.set_xticks(xTicks)
ax2.set_yticks(yTicks)
ax2.set_zticks(zTicks)

ax2.text(0.4,25,0.9,"valley", fontsize = textSize, weight = "bold")
ax2.text(0.1,33,1.4,"wall", fontsize = textSize, weight = "bold")
ax2.text(1.1,40,1.4,"wall", fontsize = textSize, weight = "bold")

ax2.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 15)
ax2.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 15)
ax2.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 15)
ax2.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 15)
ax2.tick_params(axis = 'z', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'z', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)


ax2.view_init(theta,phi)

##Plot 3
theta = -10
phi = 80
alpha = viewAdjustment(theta, phi, chaoticP1,chaoticP2,chaoticP3)


ax3 = fig.add_subplot(223,projection = '3d')
scatt = ax3.scatter3D(chaoticP1, chaoticP2, chaoticP3, c = lyapunovExponent, cmap = "viridis_r", alpha = alpha, marker = '.')
#ax.scatter(logChaoticC1, chaoticC2, chaoticC3)
ax3.set_xlabel("$p_1$", fontsize = axesLabelSize, labelpad = 20)
ax3.set_ylabel("$p_2$", fontsize = axesLabelSize, labelpad = 23)
ax3.set_zlabel("$p_3$", fontsize = axesLabelSize, labelpad = 33)

xTicks = [0.2,0.4,0.6,0.8,1.0]
yTicks = [20,30,40]
zTicks = np.arange(0.6,2.01,0.4)
ax3.set_xticks(xTicks)
ax3.set_yticks(yTicks)
ax3.set_zticks(zTicks)

ax3.text(0.55,20,0.6,"valley", fontsize = textSize,weight = "bold")
ax3.text(0.25,40,1,"low $\mathbf{p_1}$ \n wall", fontsize = textSize,weight = "bold")
ax3.text(1.05,20,1.4,"high $\mathbf{p_1}$ \n wall", fontsize = textSize, weight = "bold")

ax3.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 15)
ax3.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 15)
ax3.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 15)
ax3.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 15)
ax3.tick_params(axis = 'z', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 15)
ax3.tick_params(axis = 'z', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 15)


ax3.view_init(theta,phi)

##Plot 4
theta = 0
phi = 0
alpha = viewAdjustment(theta, phi, chaoticP1,chaoticP2,chaoticP3)


ax4 = fig.add_subplot(224,projection = '3d')
scatt = ax4.scatter3D(chaoticP1, chaoticP2, chaoticP3, c = lyapunovExponent, cmap = "viridis_r", alpha = alpha, marker = '.')
#ax.scatter(logChaoticC1, chaoticC2, chaoticC3)
#ax4.set_xlabel("$p_1$", fontsize = axesLabelSize, labelpad = 20)
ax4.set_ylabel("$p_2$", fontsize = axesLabelSize, labelpad = 35)
ax4.set_zlabel("$p_3$", fontsize = axesLabelSize, labelpad = 25)

xTicks = np.arange(0.1,0.31,0.05)
yTicks = np.arange(20,40.01,5)
zTicks = np.arange(0.6,2.01,0.4)
ax4.set_xticks([])
ax4.set_yticks(yTicks)
ax4.set_zticks(zTicks)

ax4.text(0.6,23,0.6,"valley", fontsize = textSize, weight = "bold")

ax4.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 15)
ax4.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 15)
ax4.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax4.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax4.tick_params(axis = 'z', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax4.tick_params(axis = 'z', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)


ax4.view_init(theta,phi)

cbaxes = fig.add_axes([1,0,0.03,1])
cbaxes.tick_params(axis = 'y', pad = 10)

cbar = fig.colorbar(scatt, ax = [ax1,ax2,ax3,ax4], orientation = "vertical", cax = cbaxes, label = "Lyapunov Exponent (iteration$^{-1}$)")
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(28)
cbar.set_label(label = "Lyapunov exponent (iteration$^{-1}$)", size = 36)
cbaxes.yaxis.set_label_coords(2.5,0.5)

plt.tight_layout()
plt.savefig("./3dPlots/chaoticLocus.png")
plt.show()
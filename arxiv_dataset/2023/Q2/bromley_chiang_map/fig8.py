import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import rcParams

##Adjust plotting defaults
rcParams["axes.linewidth"] = 3.5

rcParams["ytick.right"] = True
rcParams["ytick.direction"] = "in"
rcParams["ytick.minor.visible"] = True
rcParams["ytick.major.left"] = True
rcParams["ytick.major.right"] = True
rcParams["ytick.minor.left"] = True
rcParams["ytick.minor.right"] = True
rcParams["ytick.major.size"] = 16
rcParams["ytick.minor.size"] = 8



rcParams["xtick.top"] = True
rcParams["xtick.direction"] = "in"
rcParams["xtick.minor.visible"] = True
rcParams["xtick.major.top"] = True
rcParams["xtick.major.bottom"] = True
rcParams["xtick.minor.top"] = True
rcParams["xtick.minor.bottom"] = True
rcParams["xtick.major.size"] = 16
rcParams["xtick.minor.size"] = 8

titleSize = 28
axesLabelSize = 34
tickLabelSize = 26
textSize = 26

def pierrehumbert(tau, p):
    '''
    Map using the two stream solution derived in Pierrehumbert (20xx)
    $$\gamma = 10^{p3\tanh(\log(\frac{\tau}{p4}))}$$
    $$\tau_{i+1} = p1e^{\frac{-p2}{(1 + \frac{1}{\gamma} + (1-\frac{1}{gamma})e^{-\gamma\tau})^{-\frac{1}{4}}}}
    p is an array of 4 values [d,p2,p3,4]
    '''
    gamma = 10**(p[2]*np.tanh(np.log10(tau)/p[3]))
    newTau = p[0]*np.exp(-p[1]/((1 + 1/gamma + (1 - 1/gamma)*np.exp(-gamma*tau)))**(1/4))
    return newTau


def nDeriv(f, x, args):
    '''
    Numerical derivative over a delta of 0.00001 to the right
    '''
    delta = f(x+0.00001, args) - f(x, args)
    return delta/0.00001

def lyapunovExp(f, x0, args):
    '''
    Calculates the lyapunov exponent using the method in Strogatz Ch 10

    '''
    lyExp = 0
    x = x0
    for i in range(300):
        x = f(x,args)
    for i in range(10000):
        lyExp += np.log(np.abs(nDeriv(f,x,args))) 
        x = f(x,args)
    lyExp /= 10000
    return lyExp


##Define Parameters of Interest
p1 = 0.8 ##[0,1] for pierrehumbert, [1,10] for Guillot
p2 = 38 ##[20,40]

p3 = 0.6  ##[0,2]
p4 = 0.5
d = p1*np.exp(p2*(2)**(-0.25))  
p = [d,p2,p3,p4]

tauArr = [0]
perturbedTauArr = []
difference = []
xi = [0]
yi = [0]

for i in range(1000):
    tauArr.append(pierrehumbert(tauArr[i],p))
    if i == 300:
        perturbedTauArr.append(tauArr[-1]+1e-10)
        difference.append(np.abs(perturbedTauArr[-1]-tauArr[-1]))
    elif i > 300:
        perturbedTauArr.append(pierrehumbert(perturbedTauArr[-1],p))
        difference.append(np.abs(perturbedTauArr[-1]-tauArr[-1]))
    xi.append(xi[-1])
    yi.append(pierrehumbert(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])


maxVal = max(tauArr)
xMax = math.ceil(maxVal*10)/10 + 0.1 
##Axes max set to have the plot more centered. Can change to end on major tick

x = np.linspace(0,xMax-0.1,100)
y = pierrehumbert(x,p)

fig,ax = plt.subplots(2,2,figsize = (19.2,12.8))

ax[0][0].plot(range(100,200),tauArr[100:200], marker = 'o', color = "black")
ax[0][0].set_xlabel("Iteration i", fontsize = axesLabelSize)
ax[0][0].set_ylabel("Optical Depth $\\tau$", fontsize = axesLabelSize)
##ax[0][0].set_title("Time Series", fontsize = titleSize)



ax[0][0].set_xlim(95,205)
ax[0][0].set_ylim(-0.1,xMax)
xTicks = [100,120,140,160,180,200]
yTicks = np.arange(0,xMax+0.1,0.5)
ax[0][0].set_xticks(xTicks)
ax[0][0].set_yticks(yTicks)

lyExp = lyapunovExp(pierrehumbert, 0, p)
y1 = 1e-10 * np.exp(lyExp*range(150))

ax[1][0].plot(difference[:150], color = "black")
ax[1][0].plot(range(150), y1, color = "black", ls = "--")
ax[1][0].set_xlabel("Iteration i", fontsize = axesLabelSize)
ax[1][0].set_ylabel("$\delta \\tau$", fontsize = axesLabelSize)
ax[1][0].set_yscale("log")
##ax[1][0].set_title("Correlation Plot", fontsize = titleSize)


ax[1][0].text(105,0.25e-5, "$p_1 = {0}$".format(p1), fontsize = textSize)
ax[1][0].text(105,0.25e-6, "$p_2 = {0}$".format(p2), fontsize = textSize)
ax[1][0].text(105,0.25e-7, "$p_3 = {0}$".format(p3), fontsize = textSize)
ax[1][0].text(105,0.25e-8, "$p_4 = {0}$".format(p4), fontsize = textSize)
ax[1][0].text(105,0.25e-9, "$\lambda = ${0:.4f}".format(lyExp), fontsize = textSize)
ax[1][0].text(30,5e-9*np.exp(30*lyExp), "$\delta \\tau = \delta \\tau_0 e^{\lambda i}$", fontsize = textSize, rotation = 40)

ax[1][0].set_xlim(-5,155)
ax[1][0].set_ylim(1e-11,1e1)
xTicks = np.arange(0,141,20)
yTicks = [1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1]
ax[1][0].set_xticks(xTicks)
ax[1][0].set_yticks(yTicks)


ax[0][1].plot(x,x, ls = "-.", color = "tab:green", lw = 3)
ax[0][1].plot(x,y, color = "tab:orange", lw = 3)
ax[0][1].plot(xi[100:300], yi[100:300], color = "black", lw = 1, alpha = 0.5)
ax[0][1].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize)
ax[0][1].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize)


##ax[0][1].set_title("Cobweb Plot", fontsize = titleSize)

xSpacing = 0.5
if xMax < 2.5:
    xSpacing = 0.2

ax[0][1].set_xlim(-0.1,2.1)
ax[0][1].set_ylim(-0.1,2.1)
xTicks = np.arange(0,2.1,0.5)
yTicks = np.arange(0,2.1,0.5)
ax[0][1].set_xticks(xTicks)
ax[0][1].set_yticks(yTicks)

gammas = 10**(p3*np.tanh(np.log10(tauArr)/p4))
gammaTau = np.multiply(tauArr, gammas)


bins,_,__ = ax[1][1].hist(gammaTau[100:], 50, color = "black")
ax[1][1].set_xlabel("$\gamma\\tau$", fontsize = axesLabelSize)
ax[1][1].set_ylabel("Count", fontsize = axesLabelSize)
##ax[1][1].set_title("Histogram", fontsize = titleSize)

maxBins = max(bins/50)*50
ax[1][1].set_xlim(-0.1,4.1)
ax[1][1].set_ylim(0,300)
xTicks = np.arange(0,4.1,0.5)
yTicks = np.arange(0,301,50)
ax[1][1].set_xticks(xTicks)
ax[1][1].set_yticks(yTicks)

for axes in ax:
    for axis in axes:
        axis.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
        axis.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
        axis.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
        axis.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)

##fig.suptitle("Plots for p1 = " + str(p1) + ",p2 = " + str(p2) + ",p3 = " + str(p3) + ",p4 = " + str(p4), fontsize = 20)
plt.tight_layout()
plt.savefig("../timeSeriesPlots/timeSeries"+str(p1)+str(p2)+str(p3)+".pdf")

print(lyExp)
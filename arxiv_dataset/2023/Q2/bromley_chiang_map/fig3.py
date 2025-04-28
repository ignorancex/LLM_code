import numpy as np
import math
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


axesLabelSize = 40
tickLabelSize = 34
textSize = 34

def constantGamma(tau, p):
    newTau = p[0]*np.exp(-p[1]/((1 + 1/p[2] + (1 - 1/p[2])*np.exp(-p[2]*tau)))**(1/4))
    return newTau

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

def guillot(tau,p):
    '''
    Map using the two stream solution derived in Guillot (20xx)
    $$\gamma = 10^{p3\tanh(\log(\frac{\tau}{p4}))}$$
    $$\tau_{i+1} = p1e^{\frac{-p2}{(1 + \frac{1}{\gamma} + (\gamma-\frac{1}{gamma})e^{-\gamma\tau})^{-\frac{1}{4}}}}
    p is an array of 4 values [d,p2,p3,4]
    '''
    gamma = 10**(p[2]*np.tanh(np.log10(tau)/p[3]))
    newTau = p[0]*np.exp(-p[1]/((1 + 1/gamma + (gamma - 1/gamma)*np.exp(-gamma*tau)))**(1/4))
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
p1 = 1.5
p2 = 30
gamma = 0.5
d = p1*np.exp(p2*2**(-0.25))
p = [d, p2, gamma]

fig,ax = plt.subplots(3,2,figsize = (24,24))
fig.subplots_adjust(wspace=0.0, hspace = 0.25)

x = np.linspace(-5,30,100)
y = constantGamma(x,p)

tauArr = [0]
xi = [0]
yi = [-5]
for i in range(50):
    tauArr.append(constantGamma(tauArr[-1],p))
    xi.append(xi[-1])
    yi.append(constantGamma(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])

ax[0][0].plot(x,y,color = "tab:blue", ls = "--", lw = 4, zorder= 3)
ax[0][0].plot(x,x,color = "tab:green", ls = "-.", lw = 4,zorder= 1)
ax[0][0].plot(xi,yi,color = "black", linewidth = 2, alpha = 0.8, zorder= 2)

ax[0][1].plot(tauArr, marker = 'o', color = "black")

tauArr = [23.5]
xi = [23.5]
yi = [-5]
for i in range(50):
    tauArr.append(constantGamma(tauArr[-1],p))
    xi.append(xi[-1])
    yi.append(constantGamma(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])

ax[0][0].plot(xi,yi,color = "black", linewidth = 2, alpha = 0.8, zorder= 2)

ax[0][1].plot(tauArr, marker = 'o', color = "black")

ax[0][0].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize)
ax[0][0].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize)
ax[0][1].set_xlabel("Iteration $i$", fontsize = axesLabelSize)
ax[0][1].yaxis.set_label_position("right")
ax[0][1].set_ylabel("$\\tau$",fontsize = axesLabelSize)

ax[0][1].yaxis.tick_right()

xTicks = [0,10,20,30,40,50]
yTicks = [0,5,10,15,20,25]

ax[0][0].set_xlim(-3,28)
ax[0][0].set_ylim(-3,28)
ax[0][0].set_xticks(yTicks)
ax[0][0].set_yticks(yTicks)
ax[0][1].set_xlim(-6,56)
ax[0][1].set_ylim(-3,28)
ax[0][1].set_xticks(xTicks)
ax[0][1].set_yticks(yTicks)

ax[0][1].text(36,11,"$ p_1 = 1.5$", fontsize = textSize)
ax[0][1].text(36,7,"$ p_2 = 30$", fontsize = textSize)
ax[0][1].text(36,3,"$ \gamma = 0.5$", fontsize = textSize)

gamma = 2.5
p = [d, p2, gamma]

x = np.linspace(-1,3,100)
y = constantGamma(x,p)


tauArr = [0]
xi = [0]
yi = [-1]
for i in range(50):
    tauArr.append(constantGamma(tauArr[-1],p))
    xi.append(xi[-1])
    yi.append(constantGamma(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])

ax[1][0].plot(x,y,color = "tab:blue", ls = "--",lw = 4, zorder= 3)
ax[1][0].plot(x,x,color = "tab:green", ls = "-.", lw = 4, zorder= 1)
ax[1][0].plot(xi,yi,color = "black", linewidth = 2, alpha = 0.8, zorder= 2)

ax[1][1].plot(tauArr, marker = 'o', color = "black")

ax[1][0].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize)
ax[1][0].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize)
ax[1][1].set_xlabel("Iteration $i$", fontsize = axesLabelSize)
ax[1][1].yaxis.set_label_position("right")
ax[1][1].set_ylabel("$\\tau$",fontsize = axesLabelSize)

ax[1][1].yaxis.tick_right()

yTicks = [0,0.5,1.0,1.5]

ax[1][0].set_xlim(-0.25,1.75)
ax[1][0].set_ylim(-0.25,1.75)
ax[1][0].set_xticks(yTicks)
ax[1][0].set_yticks(yTicks)
ax[1][1].set_xlim(-6,56)
ax[1][1].set_ylim(-0.25,1.75)
ax[1][1].set_xticks(xTicks)
ax[1][1].set_yticks(yTicks)

ax[1][1].text(36,1.21,"$ p_1 = 1.5$", fontsize = textSize)
ax[1][1].text(36,0.95,"$ p_2 = 30$", fontsize = textSize)
ax[1][1].text(36,0.69,"$ \gamma = 2.5$", fontsize = textSize)

gamma = 3.5
p = [d, p2, gamma]

x = np.linspace(-1,3,100)
y = constantGamma(x,p)


tauArr = [0]
xi = [0]
yi = [-1]
for i in range(50):
    tauArr.append(constantGamma(tauArr[-1],p))
    xi.append(xi[-1])
    yi.append(constantGamma(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])

ax[2][0].plot(x,y,color = "tab:blue", ls = "--", lw = 4,zorder= 3)
ax[2][0].plot(x,x,color = "tab:green", ls = "-.", lw = 4, zorder= 1)
ax[2][0].plot(xi,yi,color = "black", linewidth = 2, alpha = 0.8, zorder= 2)

ax[2][1].plot(tauArr, marker = 'o', color = "black")

ax[2][0].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize)
ax[2][0].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize)
ax[2][1].set_xlabel("Iteration $i$", fontsize = axesLabelSize)
ax[2][1].yaxis.set_label_position("right")
ax[2][1].set_ylabel("$\\tau$",fontsize = axesLabelSize)

ax[2][1].yaxis.tick_right()

ax[2][0].set_xlim(-0.25,1.75)
ax[2][0].set_ylim(-0.25,1.75)
ax[2][0].set_xticks(yTicks)
ax[2][0].set_yticks(yTicks)
ax[2][1].set_xlim(-6,56)
ax[2][1].set_ylim(-0.25,1.75)
ax[2][1].set_xticks(xTicks)
ax[2][1].set_yticks(yTicks)

ax[2][1].text(36,1.40,"$ p_1 = 1.5$", fontsize = textSize)
ax[2][1].text(36,1.15,"$ p_2 = 30$", fontsize = textSize)
ax[2][1].text(36,0.90,"$ \gamma = 3.5$", fontsize = textSize)

for axes in ax:
    for axis in axes:
        axis.tick_params(axis = 'x', bottom = True, top = True, right = True, left = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
        axis.tick_params(axis = 'x', bottom = True, top = True, right = True, left = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
        axis.tick_params(axis = 'y', bottom = True, top = True, right = True, left = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
        axis.tick_params(axis = 'y', bottom = True, top = True, right = True, left = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)






plt.savefig("../timeSeriesPlots/const_gamma_time_series.pdf")
plt.show()
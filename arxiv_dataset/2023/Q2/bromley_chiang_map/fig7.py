import numpy as np
import matplotlib.pyplot as plt
import math
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
p1 = 0.4 ##[0,1] for pierrehumbert, [1,10] for Guillot
p2 = 38 ##[20,40]

p3 = 0.6  ##[0,2]
p4 = 0.5
d = p1*np.exp(p2*2**(-0.25))
p = [d,p2,p3,p4]

tauArr = [0]
xi = [0]
yi = [0]

for i in range(1000):
    tauArr.append(pierrehumbert(tauArr[i],p))
    xi.append(xi[-1])
    yi.append(pierrehumbert(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])


maxVal = max(tauArr)
xMax = math.ceil(maxVal*10)/10 + 0.1 
##Axes max set to have the plot more centered. Can change to end on major tick

x = np.linspace(0,1.4,100)
y = pierrehumbert(x,p)

fig,ax = plt.subplots(2,2,figsize = (19.2,12.8), sharex=True, sharey = True)
fig.subplots_adjust(hspace=0.0,wspace=0.0)


ax[0][0].plot(x,x, ls = "-.", color = "tab:green", lw = 3)
ax[0][0].plot(x,y, color = "tab:blue", lw = 3)
ax[0][0].plot(xi[100:300], yi[100:300], color = "black", alpha = 0.8)
ax[0][0].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize, labelpad = 14)
ax[0][0].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize)
ax[0][0].text(-0.05,1.4,"$p_1 = 0.4$", fontsize = textSize)

p1 = 0.5 ##[0,1] for pierrehumbert, [1,10] for Guillot
p2 = 38 ##[20,40]

p3 = 0.6  ##[0,2]
p4 = 0.5
d = p1*np.exp(p2*2**(-0.25))
p = [d,p2,p3,p4]

tauArr = [0]
xi = [0]
yi = [0]



for i in range(1000):
    tauArr.append(pierrehumbert(tauArr[i],p))
    xi.append(xi[-1])
    yi.append(pierrehumbert(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])

maxVal = max(tauArr)
xMax = math.ceil(maxVal*10)/10 + 0.1 

x = np.linspace(0,1.4,100)
y = pierrehumbert(x,p)


ax[0][1].plot(x,x, ls = "-.", color = "tab:green", lw = 3)
ax[0][1].plot(x,y, color = "tab:blue", lw = 3)
ax[0][1].plot(xi[100:300], yi[100:300], color = "black", alpha = 0.8)
ax[0][1].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize, labelpad = 14)
ax[0][1].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize, labelpad = 14)
ax[0][1].text(-0.05,1.4,"$p_1 = 0.5$", fontsize = textSize)

p1 = 0.525 ##[0,1] for pierrehumbert, [1,10] for Guillot
p2 = 38 ##[20,40]

p3 = 0.6  ##[0,2]
p4 = 0.5
d = p1*np.exp(p2*2**(-0.25))
p = [d,p2,p3,p4]

tauArr = [0]
xi = [0]
yi = [0]

for i in range(1000):
    tauArr.append(pierrehumbert(tauArr[i],p))
    xi.append(xi[-1])
    yi.append(pierrehumbert(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])

maxVal = max(tauArr)
xMax = math.ceil(maxVal*10)/10 + 0.1 

x = np.linspace(0,1.4,100)
y = pierrehumbert(x,p)

ax[1][0].plot(x,x, ls = "-.", color = "tab:green", lw = 3)
ax[1][0].plot(x,y, color = "tab:blue", lw = 3)
ax[1][0].plot(xi[100:300], yi[100:300], color = "black", alpha = 0.8)
ax[1][0].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize)
ax[1][0].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize)
ax[1][0].text(-0.05,1.45,"$p_1 = 0.525$", fontsize = textSize)

p1 = 0.5702
p2 = 38 ##[20,40]

p3 = 0.6  ##[0,2]
p4 = 0.5
d = p1*np.exp(p2*2**(-0.25))
p = [d,p2,p3,p4]

tauArr = [0]
xi = [0]
yi = [0]

for i in range(1000):
    tauArr.append(pierrehumbert(tauArr[i],p))
    xi.append(xi[-1])
    yi.append(pierrehumbert(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])

maxVal = max(tauArr)
xMax = math.ceil(maxVal*10)/10 + 0.1 

x = np.linspace(0,1.6,100)
y = pierrehumbert(x,p)

ax[1][1].plot(x,x, ls = "-.", color = "tab:green", lw = 4)
ax[1][1].plot(x,y, color = "tab:blue", lw = 4)
ax[1][1].plot(xi[100:300], yi[100:300], color = "black", alpha = 0.8)
ax[1][1].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize)
ax[1][1].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize, labelpad = 14)
ax[1][1].text(-0.05,1.45,"$p_1 = 0.5702$", fontsize = textSize)

ticks = np.arange(0,1.7,0.4)

##ax[0][1].set_title("Cobweb Plot", fontsize = titleSize)




ax[0][0].tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10, labeltop = True, labelbottom = False)
ax[0][0].tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax[0][0].tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10, labelleft = True, labelright = False)
ax[0][0].tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax[0][0].xaxis.set_label_position("top")

ax[0][1].tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10, labeltop = True, labelbottom = False)
ax[0][1].tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax[0][1].tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10, labelleft = False, labelright = True)
ax[0][1].tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax[0][1].xaxis.set_label_position("top")
ax[0][1].yaxis.set_label_position("right")

ax[1][0].tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10, labeltop = False, labelbottom = True)
ax[1][0].tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax[1][0].tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10, labelleft = True, labelright = False)
ax[1][0].tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)

ax[1][1].tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10, labeltop = False, labelbottom = True)
ax[1][1].tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax[1][1].tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10, labelleft = False, labelright = True)
ax[1][1].tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax[1][1].yaxis.set_label_position("right")
##fig.suptitle("Plots for p1 = " + str(p1) + ",p2 = " + str(p2) + ",p3 = " + str(p3) + ",p4 = " + str(p4), fontsize = 20)
#plt.tight_layout()

for axes in ax:
    for axis in axes:
        axis.set_xlim(-0.2,1.8)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)


ax[0][0].set_ylim(-0.2,1.8)
ax[0][1].set_ylim(-0.2,1.8)
ax[1][0].set_ylim(-0.2,1.8)
ax[1][1].set_ylim(-0.2,1.8)
##Why does this set one axis size for all 4 panels. I want the lower ones to be taller than the upper ones


plt.savefig("./cobwebs.pdf")
plt.show()
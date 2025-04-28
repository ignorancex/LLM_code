import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle

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
    for i in range(1000):
        lyExp += np.log(np.abs(nDeriv(f,x,args))) 
        x = f(x,args)
    lyExp /= 1000
    return lyExp

p1s = np.linspace(0,1.25) ## [0,1] for pierrehumbert, [1,10] for guillot
p2 = 38 ## [20,40]
p3 = 0.6 ## [0,2] but preferred to be lower
p4 = 0.5 ##Fixed at 0.5 for some reason



fig1, ax1 = plt.subplots(1,1, figsize = (12,8)) ##Create figure
tempX = np.linspace(0,5,1000)
d= p1s[-1]*np.exp(p2*(2)**(-0.25))
topCurve = pierrehumbert(tempX,[d,p2,p3,p4])
maxVal = max(topCurve)
yMax = np.ceil(maxVal*20)/20


x = np.linspace(0,2.5,100) ##Create the space to show the maps over
ax1.plot(x,x,color = "tab:green", alpha = 1, ls = '-.', zorder = -1)##Plot the 1:1 line

chaoticLabel = False
for i in range(len(p1s)):
    p1 = p1s[i]
    d = p1*np.exp(p2*(2)**(-0.25)) ##Prepare parameters
    p = [d,p2,p3,p4]
    chaotic = False
    if lyapunovExp(pierrehumbert,1e-10,p) > 0: ##We define the map as chaotic if its lypunov exponent is positive
        chaotic = True
    #print(p1, lyapunovExp(pierrehumbert,1e-10,p))
    color = "tab:blue"
    lineStyle = '--'
    if chaotic: ##Color the chaotic maps orange
        color = "tab:orange"
        lineStyle = '-'
    y = pierrehumbert(x,p)
    if i == 0:
        ax1.plot(x,y, c = color, ls = lineStyle, label = "Regular", alpha = 0.8)
    elif chaotic == True and chaoticLabel == False:
        ax1.plot(x,y, c = color, ls = lineStyle, label = "Chaotic")
        chaoticLabel = True
    else:
        ax1.plot(x,y, c = color, ls = lineStyle)


labelIndex = np.where(topCurve == maxVal)
ax1.text(0.65,1.8, "$p_1 = 1.25$", fontsize = textSize, rotation = -65)
ax1.text(0.6,-0.3, "$p_1 = 0.0$", fontsize = textSize, fontweight = "bold")
ax1.text(2,1.3, "$p_2 = {:.0f}$".format(p2), fontsize = 28)
ax1.text(2,0.9, "$p_3 = {:.1f}$".format(p3), fontsize = 28)
ax1.text(2,0.5, "$p_4 = {:.1f}$".format(p4), fontsize = 28)
ax1.text(1.1,1.25, "$\\tau(i+1) = \\tau(i)$", fontsize = textSize, rotation = 24)

ax1.set_xlabel("Optical Depth $\\tau(i)$", fontsize = axesLabelSize)
ax1.set_ylabel("Optical Depth $\\tau(i+1)$", fontsize = axesLabelSize)

ax1.add_patch(Rectangle((1.6,2.45), 0.9,0.83, ec = "black", fill = False, lw = 1.5))
ax1.add_patch(Rectangle((1.9,0.25), 0.6,1.4, ec = "black", fill = False, lw = 1.5))

ax1.legend(loc = (0.65,0.735), frameon = False, fontsize = 28)


ax1.set_xlim(-0.1, 2.6)
ax1.set_ylim(-0.5,3.5)
xTicks = np.arange(0, 2.51, 0.5)
yTicks = np.arange(0, 3.1, 0.5)
ax1.set_xticks(xTicks)
ax1.set_yticks(yTicks)

ax1.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax1.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax1.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax1.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)

plt.tight_layout()

fig1.savefig("./mapFamily380.6.png")


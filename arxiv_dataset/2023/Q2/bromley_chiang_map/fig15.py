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
    for i in range(1000):
        lyExp += np.log(np.abs(nDeriv(f,x,args))) 
        x = f(x,args)
    lyExp /= 1000
    return lyExp

fig2, ax2 = plt.subplots(1,1, figsize = (12,8)) ##Create figure
p1s = np.linspace(0,0.3,1000)## For orbit diagrams we want a finer granularity
p2 = 35 ## [20,40]
p3 = 0.6 ## [0,2] but preferred to be lower
p4 = 0.5 ##Fixed at 0.5 for some reason


maxTau = 0

for p1 in p1s:
    for j in np.linspace(0,1,5):
        d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))
        p = [d,p2,p3,p4] ##Prepare parameters
        tauArr = [j*p1*40]
        for i in range(200): ##Iterate the map 1000 times
            tauArr.append(guillot(tauArr[i],p)) 
            if tauArr[i] > maxTau:
                maxTau = tauArr[i]
        xCoor = np.ones(len(tauArr))*p1 ##Plot all points the map visited against the p1 coordinate
        ax2.scatter(xCoor[100:],tauArr[100:], s = 1, color = "black", alpha = 0.1)

ax2.set_xlabel("$p_1$", fontsize = axesLabelSize)
ax2.set_ylabel("Infrared Optical Depth $\\tau$", fontsize = axesLabelSize)
xMax = np.ceil(maxTau*10)/10


ax2.text(0.02,10, "$p_2 = {:.0f}$".format(p2), fontsize = textSize)
ax2.text(0.02,8.5, "$p_3 = {:.1f}$".format(p3), fontsize = textSize)
ax2.text(0.02,7, "$p_4 = {:.1f}$".format(p4), fontsize = textSize)


ax2.set_xlim(-0.01,0.31)
ax2.set_ylim(-0.5,12.5)
xTicks = np.arange(0,0.31,0.05)
yTicks = np.arange(0,12.1,2)
ax2.set_xticks(xTicks)
ax2.set_yticks(yTicks)

ax2.tick_params(axis = 'x', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'x', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'y', bottom = True, top = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
ax2.tick_params(axis = 'y', bottom = True, top = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)

plt.tight_layout()

fig2.savefig("../orbitDiagrams/guillotOrbitDiagram"+str(p2)+str(p3)+".jpg")

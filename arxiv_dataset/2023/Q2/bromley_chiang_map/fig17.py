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

def guillot(tau,p):
    '''
    Map using the two stream solution derived in Guillot (20xx)
    $$\gamma = 10^{p_3\tanh(\log(\frac{\tau}{p_4}))}$$
    $$\tau_{i+1} = p_1e^{\frac{-p_2}{(1 + \frac{1}{\gamma} + (\gamma-\frac{1}{gamma})e^{-\gamma\tau})^{-\frac{1}{4}}}}$$
    p is an array of 4 values [d,p2,p3,4]
    '''
    gamma = 10**(p[2]*np.tanh(np.log10(tau)/p[3]))
    newTau = p[0]*np.exp(-p[1]/((1 + 1/gamma + (gamma - 1/gamma)*np.exp(-gamma*tau)))**(1/4))
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

fig, ax = plt.subplots(3,2, figsize = (19.2,19.2))

x = np.logspace(-2,2,500)

p1 = 0.023
p2 = 35
p3 = 0.6
p4 = 0.5
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]

y1 = guillot(x,p)

ax[0][0].plot(np.log10(x),np.log10(y1), color = "tab:blue", lw = 3)

p1 = 0.03
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]

y2 = guillot(x,p)

ax[0][0].plot(np.log10(x),np.log10(y2), color = "tab:blue", lw = 3)

p1 = 0.05
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]

y3 = guillot(x,p)
ax[0][0].plot(np.log10(x),np.log10(y3), color = "tab:blue", lw = 3)

p1 = 0.25
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]

y4 = guillot(x,p)
ax[0][0].plot(np.log10(x),np.log10(y4), color = "tab:blue", lw = 3)

p1 = 0.5
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]

y5 = guillot(x,p)
ax[0][0].plot(np.log10(x),np.log10(y5), color = "tab:blue", lw = 3)

ax[0][0].plot(np.log10(x),np.log10(x), ls = "-.", color = "tab:green", lw = 3)

xi = [0.4]
yi = [0.001]
xi2 = [0.6]
yi2 = [0.001]
xi3 = [1]
yi3 = [0.001]
p1 = 0.023
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]
y2 = guillot(x,p)

for i in range(200):
    xi.append(xi[-1])
    yi.append(guillot(xi[-1],p)) 
    xi.append(yi[-1])
    yi.append(yi[-1])
    xi2.append(xi2[-1])
    yi2.append(guillot(xi2[-1],p)) 
    xi2.append(yi2[-1])
    yi2.append(yi2[-1])
    xi3.append(xi3[-1])
    yi3.append(guillot(xi3[-1],p)) 
    xi3.append(yi3[-1])
    yi3.append(yi3[-1])



ax[0][1].plot(np.log10(x),np.log10(y2), color = "tab:blue", ls = "--", lw = 3, zorder = 3)
ax[0][1].plot(np.log10(xi),np.log10(yi),color = "black", linewidth = 2, alpha = 0.7, zorder= 2)
ax[0][1].plot(np.log10(xi2),np.log10(yi2),color = "xkcd:neon red", linewidth = 2, alpha = 0.7, zorder= 2)
ax[0][1].plot(np.log10(xi3),np.log10(yi3),color = "navy", linewidth = 2, alpha = 0.7, zorder= 2)
ax[0][1].plot(np.log10(x),np.log10(x),color = "tab:green", ls = "-.", lw = 3, zorder= 1)

x = np.logspace(-2,1.5,500)
xi = [0.35]
yi = [0.001]
xi2 = [0.6]
yi2 = [0.001]
xi3 = [1.2]
yi3 = [0.001]
p1 = 0.03
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]
y3 = guillot(x,p)

for i in range(200):
    xi.append(xi[-1])
    yi.append(guillot(xi[-1],p)) 
    xi.append(yi[-1])
    yi.append(yi[-1])
    xi2.append(xi2[-1])
    yi2.append(guillot(xi2[-1],p)) 
    xi2.append(yi2[-1])
    yi2.append(yi2[-1])
    xi3.append(xi3[-1])
    yi3.append(guillot(xi3[-1],p)) 
    xi3.append(yi3[-1])
    yi3.append(yi3[-1])



ax[1][0].plot(np.log10(x),np.log10(y3), color = "tab:blue", ls = "--", lw = 3, zorder = 3)
ax[1][0].plot(np.log10(xi),np.log10(yi),color = "black", linewidth = 2, alpha = 0.7, zorder= 2)
ax[1][0].plot(np.log10(xi2),np.log10(yi2),color = "tab:red", linewidth = 1, alpha = 0.7, zorder= 2)
ax[1][0].plot(np.log10(xi3),np.log10(yi3),color = "navy", linewidth = 1, alpha = 0.7, zorder= 2)
ax[1][0].plot(np.log10(xi2[0:3]),np.log10(yi2[0:3]),color = "xkcd:neon red", linewidth = 2, alpha = 0.7, zorder= 2)
ax[1][0].plot(np.log10(xi3[0:3]),np.log10(yi3[0:3]),color = "navy", linewidth = 2, alpha = 0.7, zorder= 2)
ax[1][0].plot(np.log10(x),np.log10(x),color = "tab:green", ls = "-.", lw = 3, zorder= 1)

x = np.logspace(-2,1.5,500)
xi = [0.05]
yi = [0.001]
xi2 = [0.2]
yi2 = [0.001]
xi3 = [1.2]
yi3 = [0.001]
p1 = 0.05
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]
y4 = guillot(x,p)

for i in range(200):
    xi.append(xi[-1])
    yi.append(guillot(xi[-1],p)) 
    xi.append(yi[-1])
    yi.append(yi[-1])
    xi2.append(xi2[-1])
    yi2.append(guillot(xi2[-1],p)) 
    xi2.append(yi2[-1])
    yi2.append(yi2[-1])
    xi3.append(xi3[-1])
    yi3.append(guillot(xi3[-1],p)) 
    xi3.append(yi3[-1])
    yi3.append(yi3[-1])



ax[1][1].plot(np.log10(x),np.log10(y4), color = "tab:blue", ls = "--", lw = 3, zorder = 3)
ax[1][1].plot(np.log10(xi),np.log10(yi),color = "black", linewidth = 2, alpha = 0.7, zorder= 2)
ax[1][1].plot(np.log10(xi2),np.log10(yi2),color = "xkcd:neon red", linewidth = 2, alpha = 0.7, zorder= 2)
ax[1][1].plot(np.log10(xi3),np.log10(yi3),color = "navy", linewidth = 2, alpha = 0.7, zorder= 2)
ax[1][1].plot(np.log10(x),np.log10(x),color = "tab:green", ls = "-.", lw = 3, zorder= 1)

x = np.linspace(0,12,500)
xi = [1.2]
yi = [0.001]
p1 = 0.25
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]
y5 = guillot(x,p)

for i in range(200):
    xi.append(xi[-1])
    yi.append(guillot(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])

ax[2][0].plot(x,y5, color = "tab:blue", ls = "--", lw = 3, zorder = 3)
ax[2][0].plot(xi,yi,color = "black", linewidth = 2, alpha = 0.7, zorder= 2)
ax[2][0].plot(x,x,color = "tab:green", ls = "-.", lw = 3, zorder= 1)

x = np.linspace(0,15,500)
xi = [1.2]
yi = [0.001]
p1 = 0.5
d = p1*np.exp(p2*(1+10**(-p3))**(-0.25))  
p = [d,p2,p3,p4]
y6 = guillot(x,p)

for i in range(200):
    xi.append(xi[-1])
    yi.append(guillot(xi[-1],p))
    xi.append(yi[-1])
    yi.append(yi[-1])

ax[2][1].plot(x,y6, color = "tab:blue", ls = "--", lw = 3, zorder = 3)
ax[2][1].plot(xi,yi,color = "black", linewidth = 2, alpha = 0.7, zorder= 2)
ax[2][1].plot(x,x,color = "tab:green", ls = "-.", lw = 3, zorder= 1)


xTicks = np.arange(-2,2.1,1)
yTicks = np.arange(-2,2.1,1)
ax[0][0].set_xlim(-2,2)
ax[0][0].set_ylim(-2,2)
ax[0][0].set_xticks(xTicks)
ax[0][0].set_yticks(yTicks)


ax[0][0].text(-1.7,1.4,"$p_2 = 35$", fontsize = textSize)
ax[0][0].text(-1.7,1,"$p_3 = 0.6$", fontsize = textSize)
ax[0][0].text(-1.7,0.6,"$p_4 = 0.5$", fontsize = textSize)

textSize = 16
ax[0][0].text(1.3,-1.85,"$0.023$", fontsize = textSize)
ax[0][0].text(1.3,-1.485,"$0.03$", fontsize = textSize)
ax[0][0].text(1.3,-1.25,"$0.05$", fontsize = textSize)
ax[0][0].text(1.3,-0.55,"$0.25$", fontsize = textSize)
ax[0][0].text(1.0,-0.15,"$p_1 = 0.5$", fontsize = textSize)

ax[0][0].set_xlabel("$\log\\tau(i)$", fontsize = axesLabelSize)
ax[0][0].set_ylabel("$\log\\tau(i+1)$", fontsize = axesLabelSize)

textSize = 34
xTicks = np.arange(-2,0.6,0.5)
yTicks = np.arange(-2,0.6,0.5)
ax[0][1].set_xlim(-2,0.5)
ax[0][1].set_ylim(-2,0.5)
ax[0][1].set_xticks(xTicks)
ax[0][1].set_yticks(yTicks)

ax[0][1].text(-1.6,0, "$p_1 = 0.023$", fontsize = textSize)
ax[0][1].set_xlabel("$\log\\tau(i)$", fontsize = axesLabelSize)
ax[0][1].set_ylabel("$\log\\tau(i+1)$", fontsize = axesLabelSize)

xTicks = np.arange(-2,0.6,0.5)
yTicks = np.arange(-2,0.6,0.5)
ax[1][0].set_xlim(-2,0.5)
ax[1][0].set_ylim(-2,0.5)
ax[1][0].set_xticks(xTicks)
ax[1][0].set_yticks(yTicks)

ax[1][0].text(-1.75,0,"$p_1 = 0.03$", fontsize = textSize)
ax[1][0].set_xlabel("$\log\\tau(i)$", fontsize = axesLabelSize)
ax[1][0].set_ylabel("$\log\\tau(i+1)$", fontsize = axesLabelSize)


xTicks = np.arange(-1.5,0.6,0.5)
yTicks = np.arange(-1.5,0.6,0.5)
ax[1][1].set_xlim(-1.5,0.5)
ax[1][1].set_ylim(-1.5,0.5)
ax[1][1].set_xticks(xTicks)
ax[1][1].set_yticks(yTicks)

ax[1][1].text(-1.2,0.13,"$p_1 = 0.05$", fontsize = textSize)
ax[1][1].set_xlabel("$\log\\tau(i)$", fontsize = axesLabelSize)
ax[1][1].set_ylabel("$\log\\tau(i+1)$", fontsize = axesLabelSize)


xTicks = np.arange(0,12.1,2)
yTicks = np.arange(0,12.1,2)
ax[2][0].set_xlim(-0.5,12.5)
ax[2][0].set_ylim(-0.5,12.5)
ax[2][0].set_xticks(xTicks)
ax[2][0].set_yticks(yTicks)

ax[2][0].text(0.8,10.7,"$p_1 = 0.25$", fontsize = textSize)
ax[2][0].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize)
ax[2][0].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize)

xTicks = np.arange(0,15.1,2.5)
yTicks = np.arange(0,20.1,5)
ax[2][1].set_xlim(-0.5,15.5)
ax[2][1].set_ylim(-1,21)
ax[2][1].set_xticks(xTicks)
ax[2][1].set_yticks(yTicks)

ax[2][1].text(2,17.5,"$p_1 = 0.5$", fontsize = textSize)
ax[2][1].set_xlabel("$\\tau(i)$", fontsize = axesLabelSize)
ax[2][1].set_ylabel("$\\tau(i+1)$", fontsize = axesLabelSize)


for axes in ax:
    for axis in axes:
        axis.tick_params(axis = 'x', bottom = True, top = True, right = True, left = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
        axis.tick_params(axis = 'x', bottom = True, top = True, right = True, left = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)
        axis.tick_params(axis = 'y', bottom = True, top = True, right = True, left = True, which = "major", direction = "in", labelsize = tickLabelSize, pad = 10)
        axis.tick_params(axis = 'y', bottom = True, top = True, right = True, left = True, which = "minor", direction = "in", labelsize = tickLabelSize, pad = 10)

plt.tight_layout()
fig.savefig("./MapCTimeSeriesA.pdf")
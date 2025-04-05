import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate
import pylab 

import matplotlib
'''
font = {'family': 'serif',
                    'serif':  'Computer Modern Roman',
                    'weight': 200,
                    'size':   30}

matplotlib.rc('font', **font)
plt.rc('text', usetex=True)
'''
x,y,z = np.loadtxt("2DLLHValues.txt", unpack = True)



x = np.log10(x)
y = np.log10(y)

yticks  = [-2,-1,0,1,2]
ylabels = [0.01,0.1,1,10,100]
xticks  = [-3,-2.0,-1.0,0]
xlabels = [0.001,0.01,0.1,1]
limits  = [-3.0, 0, -2,2]
#xlabel = r"sin$^{2}$(2$\theta_{24}$)" 
xlabel = "s2(2theta24)" 
#ylabel = r"$\Delta$m$^{2}_{41}$ [eV$^{2}$] "
ylabel = 'Deltam2'
fig = pylab.figure(figsize=[6,7])
ax = fig.add_subplot(111)

nInterp = 1000

mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) 
x = x[mask]
y = y[mask]
z = z[mask]


xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(),y.max(), nInterp)
xi, yi = np.meshgrid(xi, yi)


zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')

plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
    extent=[x.min(), x.max(), y.min(), y.max()], aspect="auto", cmap = plt.cm.get_cmap(plt.cm.Spectral,100),alpha=1)

cbar = plt.colorbar()
cbar.set_label('DLLH2', rotation=90, fontsize = 13)
cbar.ax.tick_params(labelsize=20) 
plt.clim(vmin=0, vmax=10) 

plt.xlabel(xlabel, fontsize=13)        
plt.ylabel(ylabel, fontsize=13)
        
levels = [4.61,5.99,9.21] 
levels_label = [r'90$\%$ C.L.',r'95$\%$ C.L.',r'99$\%$ C.L.']
CS1 = plt.contour(xi, yi, zi, levels = [4.61,5.99,9.21], colors=('k','k','k'), alpha=1, linestyles=('-','--',':'),  linewidths=(1,1,1))
                
pylab.xticks(xticks,xlabels)
pylab.yticks(yticks,ylabels)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

pylab.axis(limits)
plt.tight_layout()
plt.show() 



x,y,z = np.loadtxt("BayesFactorValues.txt", unpack = True)
z = np.asarray(z)
x = np.log10(x)
y = np.log10(y)
mask       = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) #& np.isfinite(dllh2)
x          = x[mask]
y          = y[mask]
z          = z[mask]
min_index       = np.where(z == np.min(z))[0][0]
bf_x = x[min_index]
bf_y = y[min_index]

clow = -2.5
chigh = 2.5
color = 'RdBu'

#label = r'Log$_{10}$(Bayes Factor)'
label = 'Log10(Bayes Factor)'

figsize = [6,7]


yticks  = [-2,-1,0,1,2]
ylabels = [0.01,0.1,1,10,100]

xticks  = [-3.0,-2.0,-1.0,0]
xlabels = [0.001,0.01,0.1,1]

limits  = [-3.0, 0, -2,2]
#xlabel  = r"sin$^{2}$(2$\theta_{24}$)"  
#ylabel  = r"$\Delta$m$^{2}_{41}$ [eV$^{2}$] "
xlabel = "s2(2theta24)" 
ylabel = 'Deltam2'


fig= plt.figure(figsize=(figsize[0],figsize[1]))
ax = fig.add_subplot(111)

plt.axis(limits)
plt.xticks(xticks,xlabels, fontsize=20)
plt.xlabel(xlabel, fontsize=20)  
plt.yticks(yticks,ylabels, fontsize=20)
plt.ylabel(ylabel, fontsize=20)
nInterp = 1000

xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(),y.max(), nInterp)
xi, yi = np.meshgrid(xi, yi)

zi = sp.interpolate.griddata((x, y), z, (xi, yi), method='linear')
im = plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
       extent=[x.min(), x.max(), y.min(), y.max()], aspect="auto", cmap = plt.cm.get_cmap(color,100),alpha=1)#plt.cm.Spectral  

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

cbar = colorbar(im)
cbar.set_label(label, rotation=90, fontsize = 20)
cbar.ax.tick_params(labelsize=20) 

plt.clim(vmin=clow, vmax=chigh) 
    
L1 = plt.contour(xi, yi, zi, levels = [-0.5], colors= 'r', alpha=1, linestyles=('-'),  linewidths=1,label=r'Substantial')

plt.plot(bf_x, bf_y,  marker='*',color='k', markersize=20)
plt.plot(bf_x, bf_y,  marker='*',color = 'w',markersize=10, linewidth=0)
plt.tight_layout()

plt.show()



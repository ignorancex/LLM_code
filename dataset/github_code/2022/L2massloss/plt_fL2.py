import numpy as np
import pylab as pl
from mplchange import *
import pickle
from math import pi, sqrt, sin, cos, tan, log, log10, floor, ceil
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap

fdir = './disk_sltns/'

M2_in_Msun = 1.4
q = .5     # mass ratio M2/M1
case = 1    # 1:Hrich, 2:Hpoor, 3:Hrich-lowZ

savename = 'grid_M%.1f_q%.1f_case%d' % (M2_in_Msun, q, case)

fL2outer_only = True  # only including the contribution from the outer disk
T_contours = True    # whether to add temperature contours
fL2_cmap = 'hot'   # cmap for fL2 contours (only used when comp_Hrich=True)


with open(fdir + savename + '.pkl', 'rb') as f:
    data_all = pickle.load(f)
    

info = data_all[0]
print(info)
logM1dot = data_all[1]
loga = data_all[2]
fL2outer = 10**data_all[3]
fL2inner = 10**data_all[4]
logT = data_all[6]
logQratio = data_all[8]
    

# total fL2
fL2arr = fL2outer + fL2inner

x_arr = logM1dot
y_arr = loga
xlabel = r'$\log|\dot{M}_1|/(M_\odot\rm \,yr^{-1})$'
ylabel = r'$\log a/R_\odot$'

if fL2outer_only:
    plt_image = fL2outer
    zlabel = r'$f_{\rm L2}$ (linear)'
    dividing_point = 0.1
    # plt_image = np.log10((1 - fL2outer))
    # zlabel = r'$\log (1-f_{\rm L2}^{\rm outer})$'
else:
    plt_image = np.log10(fL2arr)
    zlabel = r'$\logf_{\rm L2}^{\rm tot}$'
    dividing_point = -1
    # plt_image = np.log10(fL2arr/(1 - fL2arr))
    # zlabel = r'log$\,f_{\rm L2}/(1-f_{\rm L2})$'
    
plt_contour = logT
text = r'$%g\,M_\odot$' % M2_in_Msun

fig = pl.figure(figsize=(13, 10))
ax = fig.add_axes([0.105, 0.11, 0.88, 0.85])
min_val, max_val = np.amin(plt_image), np.amax(plt_image)
print('max, min=', max_val, min_val)

# create my own colormap
cres = 2056
# top = cm.get_cmap('YlOrBr_r', cres)  # this one looks better
top = cm.get_cmap('Oranges_r', cres)
bottom = cm.get_cmap('Blues', cres)
newcolors = np.vstack((top(np.linspace(0, 1, int(cres*(dividing_point-min_val)/(max_val-min_val)))),
                       bottom(np.linspace(0, 1, int(cres*(max_val-dividing_point)/(max_val-min_val))))
                       ))
newcmp = ListedColormap(newcolors, name='OrangeBlue')

if fL2outer_only:
    step = 0.2
    levels = np.arange(ceil(min_val*10)/10, floor(max_val*10)/10 + step, step)
    if M2_in_Msun > 5:
        fname = 'fL2outer_BH' + str(case)
    else:
        fname = 'fL2outer_NS' + str(case)
else:
    levels = np.arange(ceil(min_val*5)/5+0.3, floor(max_val*10)/10+1)
    if M2_in_Msun > 5:
        fname = 'fL2tot_BH' + str(case)
    else:
        fname = 'fL2tot_NS' + str(case)
lgM1dotmin, lgamin = min(logM1dot), min(loga)
lgM1dotmax, lgamax = max(logM1dot), max(loga)
if fL2outer_only:
    x = lgM1dotmin + 0.11*(lgM1dotmax-lgM1dotmin)
    y = lgamin + 0.08*(lgamax-lgamin)
else:
    x = lgM1dotmin + 0.77*(lgM1dotmax-lgM1dotmin)
    y = lgamin + 0.2*(lgamax-lgamin)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(x, y, text, fontsize=40, color='k', bbox=props)
levels = list(OrderedDict.fromkeys(levels))
# print(levels)
strs = [('%.1f' % num) for num in levels]
# print(levels, strs)
ax.set_xlabel(xlabel, labelpad=-2)
ax.set_ylabel(ylabel)
im = ax.imshow(plt_image.transpose(),
               interpolation='bicubic', origin='lower',
               cmap=newcmp, aspect='auto', alpha=0.7,
               extent=(min(x_arr), max(x_arr),
                       min(y_arr), max(y_arr)))
im.set_clim(vmin=min_val, vmax=max_val)

CB = pl.colorbar(im, ax=ax, ticks=levels)
CB.ax.set_yticklabels(strs)
CB.ax.set_ylabel(zlabel, labelpad=3)
CB.ax.minorticks_off()

# fL2 contours
color1 = 'k'
X, Y = np.meshgrid(x_arr, y_arr)

CS = ax.contour(X, Y, plt_image.transpose(),
                levels, linestyles='solid', colors=color1, linewidths=2, alpha=0.5)
    
fmt = {}
for l, s in zip(CS.levels, strs):
    fmt[l] = s
if fL2outer_only:
    fs = 33
else:
    fs = 30    # only for fL2_tot
pl.clabel(CS, CS.levels, inline=True, fmt=fmt,
          fontsize=fs, colors=None)

if T_contours:
    # temperature contours
    color2 = 'darkblue'
    # Nlevels = 7
    min_val, max_val = np.amin(plt_contour), np.amax(plt_contour)
    # T_levels = np.linspace(ceil(min_val*4)/4, floor(max_val*4)/4,
    #                      Nlevels, endpoint=True)
    T_levels = np.arange(3.3, 6.8, 0.5)
    T_levels = T_levels[1:]
    while T_levels[0] < min_val:
        T_levels = T_levels[1:]
    while T_levels[-1] > max_val:
        T_levels = T_levels[:-1]
    CS = ax.contour(X, Y, plt_contour.transpose(),
                    T_levels, colors=color2, linewidths=3, alpha=0.4)
    T_strs = [('%.1f' % num) for num in T_levels]
    T_strs[-2] = r'log$\, T/\mathrm{K}=$' + T_strs[-2]
    fmt = {}
    for l, s in zip(CS.levels, T_strs):
        fmt[l] = s
    pl.clabel(CS, CS.levels, inline=True, fmt=fmt,
              fontsize=25, colors=color2)


c = 'cyan'
# CS2 = ax.contour(X, Y, logQratio.transpose(), levels=[-2],
#                  colors=c, linewidths=4, alpha=0.6,
#                  linestyles='dashed')
CS2 = ax.contour(X, Y, logQratio.transpose(), levels=[-1],
                 colors=c, linewidths=4, alpha=0.6,
                 linestyles='dashed')
CS3 = ax.contour(X, Y, logQratio.transpose(), levels=[0],
                 colors=c, linewidths=4, alpha=0.6,
                 linestyles='dotted')

pl.savefig(fdir + savename + '.png')
print(savename + ' saved')

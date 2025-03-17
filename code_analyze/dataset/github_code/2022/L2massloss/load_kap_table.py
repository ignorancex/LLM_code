import numpy as np
import pylab as pl
from mplchange import *
from scipy.interpolate import RectBivariateSpline
import scipy.optimize as optimization
from matplotlib import cm
from matplotlib.colors import ListedColormap
from math import pi, sqrt, sin, cos, tan, log, log10, exp, floor, ceil

fdir = './kap_data/'

case = 2  # 1 -- Hrich, solarZ; 2 -- Hpoor, solarZ; 3 -- Hrich, lowZ
plt_kap = False  # turn this on to make opacity plots
extrap = False   # extrapolation beyond grid may not be accurate [no need to]

if case == 1:
    fig_label = r'$X=0.7, Z=0.02$'
    fig_savename = 'Hrich_solar'
    fname1 = 'gn93_z0.02_x0.7.data'   # Hrich
    fname2 = 'lowT_fa05_gn93_z0.02_x0.7.data'
elif case == 2:
    fig_label = r'$X=0, Z=0.02$'
    fig_savename = 'Hpoor_solar'
    fname1 = 'gn93_z0.02_x0.0.data'   # Hpoor
    fname2 = 'lowT_fa05_gn93_z0.02_x0.0.data'
else:
    fig_label = r'$X=0.7, Z=0.001$'
    fig_savename = 'Hrich_lowZ'
    fname1 = 'gn93_z0.001_x0.7.data'   # Hrich - lowZ
    fname2 = 'lowT_fa05_gn93_z0.001_x0.7.data'


def parse(fname):
    lgRarr = np.loadtxt(fname, max_rows=1, skiprows=5, unpack=True, dtype=float)
    data = np.loadtxt(fname, skiprows=7, unpack=True, dtype=float)
    lgTarr = data[0]
    NR = len(lgRarr)
    NT = len(lgTarr)
    lgkap = np.zeros((NR, NT), dtype=float)
    for i in range(NR):
        for j in range(NT):
            lgkap[i, j] = data[i+1][j]
    return lgRarr, lgTarr, lgkap


def sigm(x, x0, dx):   # sigmoid function between 0 and 1
    arg = (x-x0)/dx
    if arg < -50:
        return 0.
    if arg > 50:
        return 1.
    else:
        return exp(arg)/(exp(arg) + 1)


# ----- highT opacity table 1
lgR1, lgT1, lgkap1 = parse(fdir+fname1)
lgRmin, lgRmax = min(lgR1), max(lgR1)
lgTmin1, lgTmax1 = min(lgT1), max(lgT1)
intp_lgkap1 = RectBivariateSpline(lgR1, lgT1, lgkap1, kx=3, ky=3, s=0)

# ----- lowT opacity table 2 (which has the same lgRmin and lgRmax)
lgR2, lgT2, lgkap2 = parse(fdir+fname2)
lgTmin2, lgTmax2 = min(lgT2), max(lgT2)
intp_lgkap2 = RectBivariateSpline(lgR2, lgT2, lgkap2, kx=3, ky=3, s=0)
lgT_blend_center = (lgTmax2 + lgTmin1)/2
dlgT_blend = (lgTmax2-lgTmin1)/10   # width for the sigmoid function

# print('lgTmin=', lgTmin2)

# full grid to interpolate upon
NlgR_old, NlgT_old = 200, 200
lgRgrid_old = np.linspace(lgRmin, lgRmax, NlgR_old, endpoint=True)
lgTgrid_old = np.linspace(min(lgTmin1, lgTmin2), max(lgTmax1, lgTmax2),
                          NlgT_old, endpoint=True)
lgkapgrid_old = np.zeros((NlgR_old, NlgT_old), dtype=float)

for i in range(NlgR_old):
    x = lgRgrid_old[i]
    for j in range(NlgT_old):
        y = lgTgrid_old[j]
        if y < lgTmin1:  # use linear extrapolation
            yleft = lgT1[0]
            yright = lgT1[1]
            zleft = intp_lgkap1(x, yleft)[0][0]
            zright = intp_lgkap1(x, yright)[0][0]
            slope = (zright - zleft)/(yright - yleft)
            z1 = zleft + slope * (y - yleft)
        else:
            z1 = intp_lgkap1(x, y)[0][0]
        if y > lgTmax2:  # use linear extrapolation
            yleft = lgT2[-1]
            yright = lgT2[-2]
            zleft = intp_lgkap2(x, yleft)[0][0]
            zright = intp_lgkap2(x, yright)[0][0]
            slope = (zright - zleft)/(yright - yleft)
            z2 = zleft + slope * (y - yleft)
        else:
            z2 = intp_lgkap2(x, y)[0][0]
        weight = sigm(y, lgT_blend_center, dlgT_blend)
        # smoothly connect these two functions
        # such that z ~ z2 if y < lgT_blend_center
        #           z ~ z1 if y > lgT_blend_center
        z = weight * z1 + (1 - weight) * z2
        lgkapgrid_old[i, j] = z

lgRgrid = lgRgrid_old
lgTgrid = lgTgrid_old
lgkapgrid = lgkapgrid_old
intp_lgkapgrid = RectBivariateSpline(lgRgrid, lgTgrid, lgkapgrid,
                                         kx=3, ky=3, s=0)
# lgkap at any lgR and lgT within the old grid is:  intp_lgkapgrid(lgR, lgT)[0][0]


def func_kap_R(x, A, B, alpha):  # note: x = R/Rgrid[0], where Rgrid[0]=1e-8
    return A * np.power(x, alpha) + B


if extrap:
    lgR1 = lgRgrid_old
    lgT1 = lgTgrid_old
    lgkap1 = lgkapgrid_old
    lgRmin1, lgRmax1 = lgR1[0], lgR1[-1]
    lgTmin, lgTmax = lgT1[0], lgT1[-1]
    intp_lgkap1 = RectBivariateSpline(lgR1, lgT1, lgkap1, kx=3, ky=3, s=0)

    # first we extrapolate to a grid at smaller lgR
    Delta_lgR = 0.4   # the width of the overlapping region
    lgR_extension = 1  # extension in log space
    NlgR2, NlgT2 = 50, 50
    lgR2 = np.linspace(lgR1[0]-lgR_extension, lgR1[0] + Delta_lgR, NlgR2, endpoint=True)
    lgRmin2, lgRmax2 = lgR2[0], lgR2[-1]
    lgT2 = np.linspace(lgT1[0], lgT1[-1], NlgT2, endpoint=True)
    lgkap2 = np.zeros((NlgR2, NlgT2), dtype=float)

    dx = 0.02   # spacing between data points
    guess = [1, 1, 1]   # only initialization
    for j in range(NlgT2):
        lgT = lgT2[j]
        xarr = np.arange(0, Delta_lgR, dx)   # data points
        yarr = np.array([intp_lgkap1(x, lgT)[0][0] for x in (xarr + lgR1[0])])
        bounds = ([0, 0, 0], [5, 1.3, 3])
        if j == 0:
            popt, pcov = optimization.curve_fit(func_kap_R, 10**xarr, 10**yarr,
                                                bounds=bounds)
        else:
            popt, pcov = optimization.curve_fit(func_kap_R, 10**xarr, 10**yarr,
                                                guess, bounds=bounds)
        # popt, pcov = optimization.curve_fit(func_kap_R, 10**xarr, 10**yarr,
        #                                     bounds=bounds)
        guess = popt  # use this as the guess for the next j
        # print(guess)
        for i in range(NlgR2):
            lgkap2[i, j] = log10(func_kap_R(10**(lgR2[i]-lgR2[0]), *popt))
    intp_lgkap2 = RectBivariateSpline(lgR2, lgT2, lgkap2, kx=3, ky=3, s=0)
    # print(np.shape(lgkap2))

    # then we blend these two grids together
    lgR_blend_center = (lgRmax2 + lgRmin1)/2
    dlgR_blend = (lgRmax2-lgRmin1)/10   # width for the sigmoid function

    # full grid to interpolate upon
    NlgR, NlgT = 200, 200
    lgRgrid = np.linspace(min(lgRmin1, lgRmin2), max(lgRmax1, lgRmax2),
                          NlgR, endpoint=True)
    lgTgrid = np.linspace(lgTmin, lgTmax, NlgT, endpoint=True)
    lgkapgrid = np.zeros((NlgR, NlgT), dtype=float)

    for j in range(NlgT):
        y = lgTgrid[j]
        for i in range(NlgR):
            x = lgRgrid[i]
            if x > lgRmin1:  # within the first grid
                z1 = intp_lgkap1(x, y)[0][0]
            else:  # below the first grid, use linear extrapolation
                xleft = lgR1[0]
                xright = lgR1[1]
                zleft = intp_lgkap1(xleft, y)[0][0]
                zright = intp_lgkap1(xright, y)[0][0]
                slope = (zright - zleft)/(xright - xleft)
                z1 = zleft + slope * (x - xleft)
            if y < lgRmax2:  # within the second grid
                z2 = intp_lgkap2(x, y)[0][0]
            else:  # above the second grid, use linear extrapolation
                xleft = lgR2[-1]
                xright = lgR2[-2]
                zleft = intp_lgkap2(xleft, y)[0][0]
                zright = intp_lgkap2(xright, y)[0][0]
                slope = (zright - zleft)/(xright - xleft)
                z2 = zleft + slope * (x - xleft)
            weight = sigm(x, lgR_blend_center, dlgR_blend)
            # smoothly connect these two functions
            # such that z ~ z2 if x < lgT_blend_center
            #           z ~ z1 if x > lgT_blend_center
            z = weight * z1 + (1 - weight) * z2
            lgkapgrid[i, j] = z

    intp_lgkapgrid = RectBivariateSpline(lgRgrid, lgTgrid, lgkapgrid,
                                             kx=3, ky=3, s=0)
    # -- lgkap at any lgR and lgT within the grid is:  intp_lgkapgrid(lgR, lgT)[0][0]

if plt_kap:   # plot kap
    # plot the results on the new grid
    xlabel = r'$\log R = \log\rho/(\mathrm{g\,cm^{-3}}) - 3\log T/\mathrm{K}} + 18$'
    ylabel = r'$\log T/\mathrm{K}$'
    zlabel = r'$\log\kappa_{\rm R}/(\rm cm^2\,g^{-1})$'
    xarr = lgRgrid
    yarr = lgTgrid
    plt_image = lgkapgrid

    # xarr = lgR2
    # yarr = lgT2
    # plt_image = lgkap2
    print('plotting opacity image')

    fig = pl.figure(figsize=(13, 10))
    ax = fig.add_axes([0.08, 0.11, 0.93, 0.85])
    min_val, max_val = np.amin(plt_image), np.amax(plt_image)
    print('max, min values=', max_val, min_val)

    min_val, max_val = -6, 5
    # create my own colormap
    dividing_point = -0.5
    cres = 2056
    top = cm.get_cmap('Oranges_r', cres)
    bottom = cm.get_cmap('Blues', cres)
    newcolors = np.vstack((top(np.linspace(0, 1, int(cres * (dividing_point - min_val) / (max_val - min_val)))),
                           bottom(np.linspace(0, 1, int(cres * (max_val - dividing_point) / (max_val - min_val))))
                           ))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    levels = np.arange(ceil(min_val), floor(max_val)+1, 1)
    # levels = np.arange(ceil(min_val*2)/2, floor(max_val*2)/2+1, 1)
    strs = [('%.1f' % num).replace('.0', '') for num in levels]
    # print(levels, strs)
    ax.set_xlabel(xlabel, labelpad=-2)
    ax.set_ylabel(ylabel)
    im = ax.imshow(plt_image.transpose(),
                   interpolation='bicubic', origin='lower',
                   cmap=newcmp, aspect='auto', alpha=0.7,
                   extent=(min(xarr), max(xarr),
                           min(yarr), max(yarr)))
    im.set_clim(vmin=min_val, vmax=max_val)

    CB = pl.colorbar(im, ax=ax, ticks=levels)
    CB.ax.set_yticklabels(strs)
    CB.ax.set_ylabel(zlabel, labelpad=3)
    CB.ax.minorticks_off()

    # # ----contours
    color1 = 'k'
    X, Y = np.meshgrid(xarr, yarr)
    CS = ax.contour(X, Y, plt_image.transpose(),
                    levels, colors=color1, linewidths=3, alpha=0.5)
    fmt = {}
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    fs = 31
    pl.clabel(CS, CS.levels, inline=True, fmt=fmt,
              fontsize=fs, colors=color1)
    ax.text(-7.5, 8.3, fig_label, ha='left', va='center', )

    pl.savefig(fdir + 'kap_blend_' + fig_savename + '.png')

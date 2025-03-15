# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py as h5
import sys, os
import matplotlib.ticker as ticker

#from scipy.stats import mstats
#from scipy.optimize import curve_fit
#from visualize import imshow

#from configSetup import Configuration
#from combine_files import get_file_list
#from combine_files import combine_tiles
#from scipy.ndimage.filters import gaussian_filter

import argparse

# visualize matrix
def imshow(ax,
           grid, xmin, xmax, ymin, ymax,
           cmap='plasma',
           vmin = 0.0,
           vmax = 1.0,
           clip = -1.0,
           cap = None,
           aspect = 'auto',
          ):

    ax.clear()
    ax.minorticks_on()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    #ax.set_xlim(-3.0, 3.0)
    #ax.set_ylim(-3.0, 3.0)

    extent = [ xmin, xmax, ymin, ymax ]

    if clip == None:
        mgrid = grid
    elif type(clip) == tuple:
        cmin, cmax = clip
        print(cmin, cmax)
        mgrid = np.ma.masked_where( np.logical_and(cmin <= grid, grid <= cmax), grid)
    else:
        mgrid = np.ma.masked_where(grid <= clip, grid)

    if cap != None:
        mgrid = np.clip(mgrid, cap )

    mgrid = mgrid.T
    im = ax.imshow(mgrid,
              extent=extent,
              origin='lower',
              interpolation='nearest',
              cmap = cmap,
              vmin = vmin,
              vmax = vmax,
              aspect=aspect,
              #vmax = Nrank,
              #alpha=0.5
              )
    return im

default_values = {
    'cmap':"viridis",
    'vmin': None,
    'vmax': None,
    'clip': None,
    'aspect':1,
    'vsymmetric':None,
    'title':'',
    'derived':False,
}

default_turbulence_values = {
        'rho': {'title': r"$\rho$",
                'vmin': 0.0,
                'vmax': 0.8,
                },
        'jz': {'title': r"$J_z$",
               'cmap': "RdBu",
               'vsymmetric':True,
               'vmin': -0.0800,
               'vmax':  0.0800,
                },
        'bz': {'title': r"$B_z$",
               'cmap': "RdBu",
               'vsymmetric':True,
                },
        'je': {'title': r"$\mathbf{J} \cdot \mathbf{E}$",
               'cmap': "BrBG",
               'vmin': -0.0150,
               'vmax':  0.0150,
               'vsymmetric':True,
               'derived':True,
                },
}



def read_var(f5F, var):
    try:
        val = f5F[var][:,:,0]
    except:
        nx = f5F['Nx'][()]
        ny = f5F['Ny'][()]
        nz = f5F['Nz'][()]
        #print("reshaping 1D array into multiD with {} {} {}".format(nx,ny,nz))

        val = f5F[var][:]
        val = np.reshape(val, (nx, ny, nz))
        val = val[:,:,0]

    return val



def plot2dturb_single(
        ax,
        var,
        info,
        title= None,
        vmin = None,
        vmax = None,
        cmap = None,
        clip = None,
        ):

    #--------------------------------------------------
    # unpack incoming arguments that modify defaults
    args = {}

    #general defaults
    for key in default_values:
        args[key] = default_values[key]

    #overwrite with turbulence defaults
    try:
        for key in default_turbulence_values[var]:
            args[key] = default_turbulence_values[var][key]
    except:
        pass

    #finally, overwrite with user given values
    for key in args:
        try:
            user_val = eval(key)
            if not(user_val == None):
                args[key] = user_val
                print("overwriting {} key with {}".format(key, user_val))
        except:
            pass

    print('--------------------------------------------------')
    print("reading {}".format(info['fields_file']))

    f5F = h5.File(info['fields_file'],'r')

    # normal singular variables
    if not(args['derived']):
        val = read_var(f5F, var)

    # composite quantities
    else:
        print("building composite variable")

        if var == "je":
            jx = read_var(f5F, "jx")
            jy = read_var(f5F, "jy")
            jz = read_var(f5F, "jz")

            ex = read_var(f5F, "ex")
            ey = read_var(f5F, "ey")
            ez = read_var(f5F, "ez")

            val = jx*ex + jy*ey + jz*ez


    #--------------------------------------------------
    # get shape
    nx, ny = np.shape(val)
    #print("nx={} ny={}".format(nx, ny))

    xmin = 0.0
    ymin = 0.0
    xmax = nx/info['skindepth']
    ymax = ny/info['skindepth']


    # else set vmin and vmax using normal min/max
    if args['vmin'] == None:
        args['vmin'] = np.min(val)
    if args['vmax'] == None:
        args['vmax'] = np.max(val)

    # make color limits symmetric
    if args['vsymmetric']:
        vminmax = np.maximum( np.abs(args['vmin']), np.abs(args['vmax']) )
        args['vmin'] = -vminmax
        args['vmax'] =  vminmax


    # finally, re-check that user did not give vmin/vmax
    args['vmin'] = vmin if not(vmin == None) else args['vmin']
    args['vmax'] = vmax if not(vmax == None) else args['vmax']

    #nor the project default
    args['vmin'] = default_turbulence_values[var]['vmin'] if not(vmin == None) else args['vmin']
    args['vmax'] = default_turbulence_values[var]['vmax'] if not(vmax == None) else args['vmax']

    #--------------------------------------------------
    # print all options
    print("--- {} ---".format(var))
    for key in args:
        if not(key == None):
            print(" setting {}: {}".format(key, args[key]))


    #--------------------------------------------------

    im = imshow(ax, val, xmin, xmax, ymin, ymax,
           cmap = args['cmap'],
           vmin = args['vmin'],
           vmax = args['vmax'],
           clip = args['clip'],
           aspect=args['aspect'],
           )

    #--------------------------------------------------
    # zoom-in/limits
    ax.set_xlim((600.0, 900.0))
    ax.set_ylim((300.0, 600.0))


    #--------------------------------------------------
    # colorbar

    ax.set_xlabel(r"$x$ $(c/\omega_p)$")
    ax.set_ylabel(r"$y$ $(c/\omega_p)$")
    plt.subplots_adjust(left=0.15, bottom=0.10, right=0.87, top=0.97)

    wskip = 0.2
    pad = 0.01
    pos = ax.get_position()

    axleft   = pos.x0
    axbottom = pos.y0
    axright  = pos.x0 + pos.width
    axtop    = pos.y0 + pos.height

    cax = plt.fig.add_axes([axright+pad, axbottom+wskip, 0.01, axtop-axbottom-2*wskip])

    cb = plt.fig.colorbar(im, cax=cax,
            orientation='vertical',
            ticklocation='right')

    cax.text(1.0, 1.03, args['title'], transform=cax.transAxes)

    slap = str(info['lap']).rjust(4, '0')
    fname = var+'_{}.pdf'.format(slap)
    plt.savefig(fname)
    cb.remove()



#--------------------------------------------------

# Default simulation values
class Conf:
    c_omp = 5.0
    stride = 5


if __name__ == "__main__":

    #setup argparser
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--var',
            dest='var',
            default=None,
            type=str,
            help='Variable to analyze')

    args = parser.parse_args()

    conf = Conf()



    plt.fig = plt.figure(1, figsize=(4,3.5), dpi=200)
    plt.rc('font',  family='sans-serif')
    #plt.rc('text',  usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes',  labelsize=8)


    gs = plt.GridSpec(1, 1)
    gs.update(hspace = 0.0)
    gs.update(wspace = 0.0)

    axs = []
    axs.append( plt.subplot(gs[0,0]) )

    axs[0].set_xlabel(r"$x$ $(c/\omega_p)$")
    axs[0].set_ylabel(r"$y$ $(c/\omega_p)$")

    #--------------------------------------------------
    # read file

    fdir = '../data/'
    print("plotting {}".format(fdir))

    fname_F = "flds"

    lap = 6600

    #general file info
    info = {}
    info['lap'] = lap
    info['fields_file']   = fdir + 'flds_'+str(lap)+'.h5'
    info['skindepth'] = conf.c_omp/conf.stride

    print(info['fields_file'])

    #plot
    plot2dturb_single(axs[0], args.var, info)

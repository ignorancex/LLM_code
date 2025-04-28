# poreutils.py
# Date-Modified: 2021-12-29
# Author: Adrian Del Maestro
# Helper functions for analyzing 4He inside nanopores

import mpmath as mp
from dgutils import colors as colortools
import numpy as np
import os
from cmaptools import readcpt
import matplotlib.colors as mpl_colors
from copy import copy

π = np.pi

red = '#fe7979'
orange = '#febc56'
green = '#aef6a8'
lblue = '#41c8ff' 
blue = '#0028e8'
dblue = '#0f06b4'
purple = '#090076'

pore_colors = {'μ = -7.0 K':lblue,'μ = -19.0 K':green, 'μ = -27.0 K':orange, 'μ = -47.0 K':red}
pore_labels = {'μ = -7.0 K':'full','μ = -19.0 K':'3 layers', 'μ = -27.0 K':'2 layers', 'μ = -47.0 K':'1 layer'}

# setup a possible custom font path
font_path,bold_font_path = '.','.'
if 'LOCAL_FONT_PATH' in os.environ:
    font_path = os.environ['LOCAL_FONT_PATH'] + os.path.sep + 'HelveticaNeue/HelveticaNeue-Light-08.ttf'
    bold_font_path = os.environ['LOCAL_FONT_PATH'] + os.path.sep + 'HelveticaNeue/HelveticaNeue-Bold-02.ttf'


# ------------------------------------------------------------------------------\
def get_masked_palette(cpt_file, over_color='w', under_color='k'):

    cmap = readcpt(cpt_file,hinge=None)
    #colors = colortools.get_linear_colors(cmap,100)
    
    # setup a masked color pallette
    palette = copy(cmap)
    palette.set_over(over_color, 1.0)
    palette.set_under(under_color, 1.0)
    
    return palette

# -------------------------------------------------------------------------------
def get_props(c, ls='None'):
    return {'mec':c, 'ls':ls, 'ecolor':c, 'marker':'o', 'mfc':colortools.get_alpha_hex(c,0.5,real=True), 
            'color':c}

# -------------------------------------------------------------------------------
def g2_LL_T0Linf(z,ρₒ,K,A):
    return 1.0 - K/(2*π**2*ρₒ**2*z**2) + (A/(ρₒ**2*z**(2.0*K)))*np.cos(2*π*ρₒ*z)

# -------------------------------------------------------------------------------
def g2_LL_T0Linf_max(z,ρₒ,K,A,sign):
    return 1.0 - K/(2*π**2*ρₒ**2*z**2) + sign*(A/(ρₒ**2*z**(2.0*K)))

# -------------------------------------------------------------------------------
def g2_LL_T0L(z,ρₒ,L,K,A):
    d = (L/π)*np.abs(np.sin(π*z/L))
    return 1.0 - K/(2*π**2*ρₒ**2*d**2) + (A/(ρₒ**2*d**(2.0*K)))*np.cos(2*π*ρₒ*z)

# -------------------------------------------------------------------------------
def g2_LL_T0L_max(z,ρₒ,L,K,A,sign):
    d = (L/π)*np.abs(np.sin(π*z/L))
    return 1.0 - K/(2*π**2*ρₒ**2*d**2) + (A/(ρₒ**2*d**(2.0*K)))*sign

# -------------------------------------------------------------------------------
def g2_LL_TL(z,ρₒ,L,T,K,v,A):
    
    z̄ = π*z/L
    q = np.exp(-π*v/(L*T))
    
    θ1 = np.frompyfunc(lambda n,z,q,p: mp.jtheta(n,z,q,derivative=p), 4,1)
    
    t1 = (π/L)**2 * ( θ1(1,z̄,q,2)/θ1(1,z̄,q,0) - θ1(1,z̄,q,1)/θ1(1,z̄,q,0)**2 )
    t2 = float(2*mp.eta(1j*v/(L*T)).real)*np.exp(-π*v/(6.0*L*T)) / θ1(1,z̄,q,0)

    return np.array(1.0 + K/(2*π**2*ρₒ**2)*t1 +(A/ρₒ**2)*t2**(2*K)*np.cos(2*π*ρₒ*z), dtype=float)

# -------------------------------------------------------------------------------
def g2_LL_TL_max(z,ρₒ,L,T,K,v,A,sign):
        
    z̄ = π*z/L
    q = np.exp(-π*v/(L*T))
    
    θ1 = np.frompyfunc(lambda n,z,q,p: mp.jtheta(n,z,q,derivative=p), 4,1)
    t1 = (π/L)**2 * ( θ1(1,z̄,q,2)/θ1(1,z̄,q,0) - θ1(1,z̄,q,1)/θ1(1,z̄,q,0)**2 )
    t2 = float(2*mp.eta(1j*v/(L*T)).real)*np.exp(-π*v/(6.0*L*T)) / θ1(1,z̄,q,0)
    
    res = np.array(1.0 + K/(2*π**2*ρₒ**2)*t1 +(A/ρₒ**2)*t2**(2*K)*sign,dtype=float)
    
    return res

# --------------------------------------------------------------------------------------
def ω_LL(p,q,KL,kF):
    '''The hard-rod prediction for the scattering maxima'''
    return (4.0/KL)*np.abs(q/(2*kF) + p*(q/(2*kF))**2)

# --------------------------------------------------------------------------------------
def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
        print(target)
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex:
        for ax in axs.flat[:-1]:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
            ax.xaxis.set_visible(False)

    # Turn off y tick labels and offset text for all but the left most column
    if sharey:
        for ax in axs.flat[:-1]:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


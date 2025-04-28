import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.visualization import ImageNormalize, AsinhStretch
import cmasher as cmr
from astropy.coordinates import spherical_to_cartesian
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from astropy.coordinates import cartesian_to_spherical
import sys

font = {"family":"serif",
        "size":15}
plt.rc('font', **font)
fontsize = 15



###### Plot Cumilative Extinction from -30 to 30 #################################
def plot_ext_Galplane(ext_med_cube, l_bounds_pred):

    image = ext_med_cube 

    fig = plt.figure(figsize=(25, 10)) #width, height
    ax = fig.add_subplot(1, 1, 1)
    colmap = cmr.get_sub_cmap("cmr.chroma", 0.05, 1)
    normalize = ImageNormalize(image, vmin=0, vmax=4, stretch=AsinhStretch(a=0.4)) #vmax=5 #vmax=np.nanmax(image) #stretch=AsinhStretch(a=0.8)
    img = ax.imshow(image[::-1,::-1,-1].T, origin="upper", aspect="auto", cmap=colmap,
                        extent =(np.max(l_bounds_pred), np.min(l_bounds_pred), -30, 30), #np.min(b_bounds_pred), np.max(b_bounds_pred)
                        #vmin=np.min(image), vmax=np.max(image)
                        norm=normalize 
                        )
    cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
    cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
    ax.set_xlabel("$l [^{\\circ}]$", fontsize=fontsize, family="cursive")
    ax.set_ylabel("$b [^{\\circ}]$", fontsize=fontsize, family="cursive")


    plt.tight_layout() #pad=4, h_pad=0.5, w_pad=0.5
    plt.savefig("Ext_Cumulative.png") 
    #plt.show()
    plt.close()





########  Plot Cumilative Extinction in Hammer projection with Gal Center in the Center of the Plot ################

def plot_fullMW_Hammer(ext_med_cube, l_bounds_pred, b_bounds_pred):

    if np.max(l_bounds_pred) > 180:

            image = np.concatenate([ext_med_cube [-np.sum(l_bounds_pred > 180):, :], ext_med_cube [:-np.sum(l_bounds_pred > 180), :]], axis=0)
            l_bounds_plot = np.concatenate([l_bounds_pred[-np.sum(l_bounds_pred > 180):] - 360, l_bounds_pred[:-np.sum(l_bounds_pred > 180)]], axis=0)
            print(l_bounds_plot)

    else:
            image = ext_med_cube 
            l_bounds_plot = l_bounds_pred 


    fig = plt.figure(figsize=(20, 11)) #width, height

    ax = fig.add_subplot(1, 1, 1, projection="hammer")
    colmap = cmr.get_sub_cmap("cmr.chroma", 0.05, 1)
    normalize = ImageNormalize(image, vmin=0, vmax=4, stretch=AsinhStretch(a=0.5)) #vmax=np.nanmax(image), stretch=AsinhStretch(a=0.8)
    img = ax.pcolormesh(np.deg2rad(l_bounds_plot)[::-1], np.deg2rad(b_bounds_pred[::-1]), image[:,::-1,-1].T,
                        cmap=colmap,norm=normalize 
                        )
    cbar = plt.colorbar(img, orientation="horizontal", shrink=0.7, pad=0.025, aspect=50)
    cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
    #ax.set_xlabel("$l [^{\\circ}]$", fontsize=fontsize, family="cursive")
    #ax.set_ylabel("$b [^{\\circ}]$", fontsize=fontsize, family="cursive")
    #ax.invert_xaxis()
    # ax.set_xlim(180, -180)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])


    plt.tight_layout() #pad=2, h_pad=0.5, w_pad=0.1
    plt.savefig("Ext_Cumulative_Hammer.png") 
    #plt.show()
    plt.close()



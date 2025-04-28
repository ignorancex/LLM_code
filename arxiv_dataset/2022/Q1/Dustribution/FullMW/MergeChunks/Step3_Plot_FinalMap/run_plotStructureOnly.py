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

from plot_fullMW_lbd import plot_ext_Galplane, plot_fullMW_Hammer
from plot_xyExt_slicesAlongZ import plot_Ext_xy_SlicesAlongZ
from plot_xyDense_slicesAlongZ import plot_Dense_xy_slicesAlongZ, plot_Dense_xy_slicesAlongZ_300pcZoom, plot_Dense_xy_slicesAlongZ_withContours
from plot_xyIntegExt_slicesAlongZ import plot_IntegExt_xy_IntegAlongZ, plot_IntegExt_xy_IntegAlongZ_300pcZoom, plot_IntegExt_xy_IntegAlongZ_withContours, plot_IntegExt_xy_IntegAlongZ_within_bounds

from plot_xzDense_slicesAlongY import plot_Dense_xz_slicesAlongY, plot_Dense_xz_slicesAlongY_300pcZoom
from plot_xzIntegExt_slicesAlongY import plot_IntegExt_xz_IntegAlongY, plot_IntegExt_xz_IntegAlongY_300pcZoom

from plot_yzDense_slicesAlongX import plot_Dense_yz_slicesAlongX, plot_Dense_yz_slicesAlongX_300pcZoom
from plot_yzIntegExt_slicesAlongX import plot_IntegExt_yz_IntegAlongX, plot_IntegExt_yz_IntegAlongX_300pcZoom


font = {"family":"serif",
        "size":15}
plt.rc('font', **font)
fontsize = 15



if __name__=="__main__":


    l_bounds_pred = np.load("../Stitched_FullMap/FullMW_merged_l_bounds.pkl.npy", allow_pickle=True)
    b_bounds_pred = np.load("../Stitched_FullMap/FullMW_merged_b_bounds.pkl.npy", allow_pickle=True)
    d_bounds_pred = np.load("../Stitched_FullMap/FullMW_merged_d_bounds.pkl.npy", allow_pickle=True)
    ext_med_cube = np.load("../Stitched_FullMap/FullMW_CumExt_Weighted_Median.pkl.npy", allow_pickle=True)
    dense_med_cube = np.load("../Stitched_FullMap/FullMW_Dens_Weighted_Median.pkl.npy", allow_pickle=True)

    #Reinterpolate images or not - we only need to reinterpolate image if the final stiched image changes, if not we just load the saved interpolated images from the plotting routine
    re_interp = True


#     #### XY plots along Z #####
#     plot_ext_Galplane(ext_med_cube, l_bounds_pred) #Plot Cumilative Extinction from -30 to 30
#     plot_fullMW_Hammer(ext_med_cube, l_bounds_pred, b_bounds_pred) #Plot Cumilative Extinction in Hammer projection with Gal Center in the Center of the Plot

#     plot_Dense_xy_slicesAlongZ(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp) #Plot xy  in density mag/pc plots along z slices - plot units mag/pc
#     plot_Dense_xy_slicesAlongZ_300pcZoom(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)#Plot xy  for 300x300 pc cube in density mag/pc plots along z slices - plot units mag/pc
#     plot_Dense_xy_slicesAlongZ_withContours(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)

#     plot_Ext_xy_SlicesAlongZ(ext_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp) #Plot xy plots of extinction sliced along z - plots units in mag

#     plot_IntegExt_xy_IntegAlongZ(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)#Plot xy  integrated extinction along z (integrated density cube along z) - plot units mag
#     plot_IntegExt_xy_IntegAlongZ_300pcZoom(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)#Plot 300x300 pc cube of xy  integrated extinction along z (integrated density cube along z) - plot units mag
#     plot_IntegExt_xy_IntegAlongZ_withContours(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp) #Plot xy  integrated extinction along z (integrated density cube along z) with Contour along extinction levels- plot units mag

#     #### XZ plots along Y #####
#     plot_Dense_xz_slicesAlongY(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp) 
#     plot_Dense_xz_slicesAlongY_300pcZoom(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)

#     plot_IntegExt_xz_IntegAlongY(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)
#     plot_IntegExt_xz_IntegAlongY_300pcZoom(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)


#     #### YZ plots along Z #####
#     plot_Dense_yz_slicesAlongX(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp) 
#     plot_Dense_yz_slicesAlongX_300pcZoom(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)

#     plot_IntegExt_yz_IntegAlongX(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)
#     plot_IntegExt_yz_IntegAlongX_300pcZoom(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp)

    #### Plot IntegXY plots but only with given z boundaries ######
    plot_IntegExt_xy_IntegAlongZ_within_bounds(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp, z_bounds = [-300, 300])
    plot_IntegExt_xy_IntegAlongZ_within_bounds(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp, z_bounds = [-500, 500])
    plot_IntegExt_xy_IntegAlongZ_within_bounds(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp, z_bounds = [-750, 750])
    plot_IntegExt_xy_IntegAlongZ_within_bounds(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp, z_bounds = [-1000, 1000])










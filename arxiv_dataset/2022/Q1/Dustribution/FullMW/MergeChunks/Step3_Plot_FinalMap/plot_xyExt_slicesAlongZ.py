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


### Calc b Coords func ###
def coords_to_index(coordinates, midpoint_array):
    #first filter out coordiantes which are outside the limits of the array
    #coords_used = np.where( np.logical_and((coordinates > np.min(midpoint_array)), (coordinates < np.max(midpoint_array))) )[0]
    #Instead of filtering out values which are outside the range of interest, we will make sure that the indices
    #for values which are outside the range end up set to -1 by pre-filling the array with -1 and then overwriting it where we can
    #as a result, map_coordinates will insert 0.0 in all entries with an index of -1, ensuring that we get sensible
    #output and that all pixels in our image are filled.
    indices = np.full_like(coordinates, -1)
    print(len(coordinates))
    for i, m in enumerate(midpoint_array[:-1]):
        inds = np.argwhere(np.logical_and((m < coordinates),
                                          (coordinates < midpoint_array[i+1])
                                          )
                           )
        r = midpoint_array[i+1] - m #range bin covers in midpoint array
        indices[inds] = (coordinates[inds] - m)/r + i
    return indices



############ Plot xy plots where lbd extinction cube is converted xyz and sliced along z - plots units in mag ####################
def plot_Ext_xy_SlicesAlongZ(ext_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp):

    colmap = cmr.get_sub_cmap("cmr.chroma", 0.05, 1)

    #Interpolate data for new stitched image
    if re_interp: 

        ext_med_cube  = np.where(np.isfinite(ext_med_cube), ext_med_cube, 0)

        l_mid = (l_bounds_pred[1:] + l_bounds_pred[:-1])/2
        l_mid[l_mid > 180] = l_mid[l_mid>180] -360
        b_mid = (b_bounds_pred[1:] + b_bounds_pred[:-1])/2
        d_mid = (d_bounds_pred[1:] + d_bounds_pred[:-1])/2

        #Meshgrid ensures that all possible combinations of lbd coords are output as xyz coords so that we have all possible xyz coords
        print("Creating lbd mesh and xyz mid")
        lbd_mesh = np.meshgrid(l_mid, b_mid, d_bounds_pred)
        coords_lbd = np.array([lbd_mesh[0].flatten(), lbd_mesh[1].flatten(), lbd_mesh[2].flatten()]).T

        print("calc xyz mid")
        x_mid, y_mid, z_mid = spherical_to_cartesian(lbd_mesh[2], np.deg2rad(lbd_mesh[1]), np.deg2rad(lbd_mesh[0])) 

        #Now make the desired coordinates
        print("xyz min max vals calculated")
        x_min = np.min(x_mid.value)
        y_min = np.min(y_mid.value)
        z_min = np.min(z_mid.value)

        x_max = np.max(x_mid.value)
        y_max = np.max(y_mid.value)
        z_max = np.max(z_mid.value)

        npix = 300 #Cell size in xyz for the plot (assuming a 3 kpc dist radius this is 20 pc resolution for the plot)

        x_points = np.linspace(x_min, x_max, npix)
        y_points = np.linspace(y_min, y_max, npix)
        z_points = [0, 100, 300, 400] #we don't want npix number of z slices - z can go from -500 to 500 pc
        np.save("Ext_xy_slicesAlongZ_Xpoints.pkl",x_points, allow_pickle=True)
        np.save("Ext_xy_slicesAlongZ_Ypoints.pkl",y_points, allow_pickle=True)
        np.save("Ext_xy_slicesAlongZ_Zpoints.pkl",z_points, allow_pickle=True)

        x_mesh, y_mesh = np.meshgrid(x_points, y_points)

        print("Starting z slices")
        for z in z_points:
            z_mesh = np.full_like(x_mesh, z)


            plotting_coords = np.array([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()])
            
            #get spherical coods of plotting_coords - required
            plotting_coords_spherical_d, plotting_coords_spherical_b, plotting_coords_spherical_l = cartesian_to_spherical(plotting_coords[0], plotting_coords[1], plotting_coords[2])


            plotting_coords_spherical_l = np.rad2deg(plotting_coords_spherical_l.value)
            plotting_coords_spherical_l[plotting_coords_spherical_l > 180] = plotting_coords_spherical_l[plotting_coords_spherical_l > 180] - 360.
            plotting_coords_spherical_b = np.rad2deg(plotting_coords_spherical_b.value)
            plotting_coords_spherical_d = plotting_coords_spherical_d.value

            #Get the pixel indeces to be used in interpolation
            plotting_indices_l = coords_to_index(plotting_coords_spherical_l, l_mid)
            plotting_indices_b = coords_to_index(plotting_coords_spherical_b, b_mid)
            plotting_indices_d = coords_to_index(plotting_coords_spherical_d, d_mid)

            interp_indices = np.hstack((plotting_indices_l[..., np.newaxis], plotting_indices_b[..., np.newaxis], plotting_indices_d[..., np.newaxis])).T #we need to transpose because we want an array with shape(3, npixels), but otherwise this would result in shape(npixels, 3). We need to use [..., np.newaxis] for each array as otherwise they would get concatenated along their existing axis, which would result in shape(3*npixels)

            print("Creating linear interpolate function")
            interp_points = map_coordinates(ext_med_cube , interp_indices)
            # print("interp_points ==", interp_points)
            # print("interp_points shape ==", interp_points.shape)

            #now reshape it for plotting
            interp_image = np.reshape(interp_points, (npix, npix))
            np.save("Ext_xy_slicesAlongZ_Z"+str(z)+"pc_Image.pkl",interp_image, allow_pickle=True)
            # print("interp_image ==", interp_image)
            # print("interp_image shape ==", interp_image.shape)

            print("Beginning plotting")
            fig = plt.figure(figsize=(15, 11)) #width, height
            ax = fig.add_subplot(1, 1, 1)
            normalize = ImageNormalize(interp_image, vmin=0, vmax=np.nanmax(interp_image), stretch=AsinhStretch(a=0.8))
            img = ax.imshow(interp_image[::-1,:], origin="upper", aspect="auto", cmap=colmap,                 #interp_image[::-1,::-1,-1].T
                                extent =(np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)),
                                norm=normalize 
                                )
            cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
            cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
            ax.set_xlabel("x [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
            ax.set_ylabel("y [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")


            plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
            plt.savefig("Ext_xy_slicesAlongZ_Z"+str(z)+"pc.png") 
            #plt.show()
            plt.close()

        
    #Load from saved interpolated data for existing stiched image
    else: 
        x_points = np.load("Ext_xy_slicesAlongZ_Xpoints.pkl.npy", allow_pickle=True)
        y_points = np.load("Ext_xy_slicesAlongZ_Ypoints.pkl.npy", allow_pickle=True)
        z_points = np.load("Ext_xy_slicesAlongZ_Zpoints.pkl.npy", allow_pickle=True)

        for z in z_points:
        
            interp_image = np.load("Ext_xy_slicesAlongZ_Z"+str(z)+"pc_Image.pkl.npy", allow_pickle=True)

            print("Beginning plotting")
            fig = plt.figure(figsize=(15, 11)) #width, height
            ax = fig.add_subplot(1, 1, 1)
            normalize = ImageNormalize(interp_image, vmin=0, vmax=np.nanmax(interp_image), stretch=AsinhStretch(a=0.8))
            img = ax.imshow(interp_image[::-1,:], origin="upper", aspect="auto", cmap=colmap,                 #interp_image[::-1,::-1,-1].T
                                extent =(np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)),
                                norm=normalize 
                                )
            cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
            cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
            ax.set_xlabel("x [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
            ax.set_ylabel("y [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")


            plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
            plt.savefig("Ext_xy_slicesAlongZ_Z"+str(z)+"pc.png") 
            #plt.show()
            plt.close()


















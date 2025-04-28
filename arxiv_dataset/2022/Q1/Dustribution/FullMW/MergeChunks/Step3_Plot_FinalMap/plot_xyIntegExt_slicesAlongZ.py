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





########### Plot xy  integrated extinction along z (integrated density cube along z) - plot units mag ##############
def plot_IntegExt_xy_IntegAlongZ(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp):

    colmap = cmr.get_sub_cmap("cmr.chroma", 0.05, 1)

    #Interpolate data for new stitched image
    if re_interp: 
    
        gpy_dens_cube = dense_med_cube
        gpy_dens_cube = np.where(np.isfinite(gpy_dens_cube), gpy_dens_cube, 0)

        l_mid = (l_bounds_pred[1:] + l_bounds_pred[:-1])/2
        l_mid[l_mid > 180] = l_mid[l_mid>180] -360
        b_mid = (b_bounds_pred[1:] + b_bounds_pred[:-1])/2
        d_mid = (d_bounds_pred[1:] + d_bounds_pred[:-1])/2

        #Meshgrid ensures that all possible combinations of lbd coords are output as xyz coords so that we have all possible xyz coords
        print("Creating lbd mesh and xyz mid")
        lbd_mesh = np.meshgrid(l_mid, b_mid, d_mid)
        coords_lbd = np.array([lbd_mesh[0].flatten(), lbd_mesh[1].flatten(), lbd_mesh[2].flatten()]).T

        print("calc xyz mid")
        x_mid, y_mid, z_mid = spherical_to_cartesian(lbd_mesh[2], np.deg2rad(lbd_mesh[1]), np.deg2rad(lbd_mesh[0])) 

        #Now make the desired coordinates
        # print("xyz min max vals calculated")
        x_min = np.min(x_mid.value)
        y_min = np.min(y_mid.value)
        z_min = np.min(z_mid.value)

        x_max = np.max(x_mid.value)
        y_max = np.max(y_mid.value)
        z_max = np.max(z_mid.value)

        npix = 300 #Cell size in xyz for the plot (assuming a 3 kpc dist radius this is 20 pc resolution for the plot)

        x_points = np.linspace(x_min, x_max, npix)
        y_points = np.linspace(y_min, y_max, npix)
        z_points = np.linspace(z_min, z_max, npix)
        np.save("IntegExt_xy_IntegAlongZ_Xpoints.pkl",x_points, allow_pickle=True)
        np.save("IntegExt_xy_IntegAlongZ_Ypoints.pkl",y_points, allow_pickle=True)
        np.save("IntegExt_xy_IntegAlongZ_Zpoints.pkl",z_points, allow_pickle=True)

        x_mesh, y_mesh, z_mesh = np.meshgrid(x_points, y_points, z_points)
        interp_coords = np.array([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()])

        #get spherical coods of plotting_coords - required
        interp_coords_spherical_d, interp_coords_spherical_b, interp_coords_spherical_l = cartesian_to_spherical(interp_coords[0], interp_coords[1], interp_coords[2])

        interp_coords_spherical_l = np.rad2deg(interp_coords_spherical_l.value)
        interp_coords_spherical_l[interp_coords_spherical_l > 180] = interp_coords_spherical_l[interp_coords_spherical_l > 180] - 360.
        interp_coords_spherical_b = np.rad2deg(interp_coords_spherical_b.value)
        interp_coords_spherical_d = interp_coords_spherical_d.value

        #Get the pixel indeces to be used in interpolation
        interp_indices_l = coords_to_index(interp_coords_spherical_l, l_mid)
        interp_indices_b = coords_to_index(interp_coords_spherical_b, b_mid)
        interp_indices_d = coords_to_index(interp_coords_spherical_d, d_mid)

        interp_indices = np.hstack((interp_indices_l[..., np.newaxis], interp_indices_b[..., np.newaxis], interp_indices_d[..., np.newaxis])).T #we need to transpose because we want an array with shape(3, npixels), but otherwise this would result in shape(npixels, 3). We need to use [..., np.newaxis] for each array as otherwise they would get concatenated along their existing axis, which would result in shape(3*npixels)


        print("Creating linear interpolate function")
        interp_points = map_coordinates(gpy_dens_cube, interp_indices)

        #now reshape it for integration
        interp_cube = np.reshape(interp_points, (npix, npix, npix)) #xyz axis order

        print("Begin integration along z axis for xy integ extinction plot")
        interp_image = np.trapz(interp_cube, z_points, axis=2) #integrated density (to get full integ extinction) along z axis
        np.save("IntegExt_xy_IntegAlongZ_Image.pkl", interp_image, allow_pickle=True)

        print("Beginning plotting")
        fig = plt.figure(figsize=(15, 11)) #width, height
        ax = fig.add_subplot(1, 1, 1)
        normalize = ImageNormalize(interp_image, vmin=0, vmax=1, stretch=AsinhStretch(a=0.8)) #vmin=0, vmax=np.nanmax(interp_image)#vmin=0.0001, vmax=0.085. #, stretch=AsinhStretch(a=0.01). #vmin=0, vmax=0.01
        img = ax.imshow(interp_image[::-1,:], origin="upper", aspect="auto", cmap=colmap,
                            extent =(np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)),
                            norm=normalize 
                            )
        cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
        cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
        ax.set_xlabel("x [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
        ax.set_ylabel("y [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")


        plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
        plt.savefig("IntegExt_xy_IntegAlongZ.png") 
        #plt.show()
        plt.close()


    #Load from saved interpolated data for existing stiched image
    else: 

        x_points = np.load("IntegExt_xy_IntegAlongZ_Xpoints.pkl.npy", allow_pickle=True)
        y_points = np.load("IntegExt_xy_IntegAlongZ_Ypoints.pkl.npy", allow_pickle=True)

        interp_image = np.load("IntegExt_xy_IntegAlongZ_Image.pkl.npy", allow_pickle=True)

        print("Beginning plotting")
        fig = plt.figure(figsize=(15, 11)) #width, height
        ax = fig.add_subplot(1, 1, 1)
        normalize = ImageNormalize(interp_image, vmin=0, vmax=1, stretch=AsinhStretch(a=0.8)) #vmin=0, vmax=np.nanmax(interp_image)#vmin=0.0001, vmax=0.085. #, stretch=AsinhStretch(a=0.01). #vmin=0, vmax=0.01
        img = ax.imshow(interp_image[::-1,:], origin="upper", aspect="auto", cmap=colmap,
                            extent =(np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)),
                            norm=normalize 
                            )
        cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
        cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
        ax.set_xlabel("x [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
        ax.set_ylabel("y [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")


        plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
        plt.savefig("IntegExt_xy_IntegAlongZ.png") 
        #plt.show()
        plt.close()






########### Plot 300x300 pc cube of xy  integrated extinction along z (integrated density cube along z) - plot units mag ##############
def plot_IntegExt_xy_IntegAlongZ_300pcZoom(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp):  

    colmap = cmr.get_sub_cmap("cmr.chroma", 0.05, 1)  

    #Interpolate data for new stitched image
    if re_interp: 

        gpy_dens_cube = dense_med_cube
        gpy_dens_cube = np.where(np.isfinite(gpy_dens_cube), gpy_dens_cube, 0)

        l_mid = (l_bounds_pred[1:] + l_bounds_pred[:-1])/2
        l_mid[l_mid > 180] = l_mid[l_mid>180] -360
        b_mid = (b_bounds_pred[1:] + b_bounds_pred[:-1])/2
        d_mid = (d_bounds_pred[1:] + d_bounds_pred[:-1])/2

        #Meshgrid ensures that all possible combinations of lbd coords are output as xyz coords so that we have all possible xyz coords
        print("Creating lbd mesh and xyz mid")
        lbd_mesh = np.meshgrid(l_mid, b_mid, d_mid)
        coords_lbd = np.array([lbd_mesh[0].flatten(), lbd_mesh[1].flatten(), lbd_mesh[2].flatten()]).T

        print("calc xyz mid")
        x_mid, y_mid, z_mid = spherical_to_cartesian(lbd_mesh[2], np.deg2rad(lbd_mesh[1]), np.deg2rad(lbd_mesh[0])) 

        #Now make the desired coordinates
        # print("xyz min max vals calculated")
        x_min = np.min(x_mid.value)
        y_min = np.min(y_mid.value)
        z_min = np.min(z_mid.value)

        x_max = np.max(x_mid.value)
        y_max = np.max(y_mid.value)
        z_max = np.max(z_mid.value)

        npix = 500 #Cell size in xyz for the plot (assuming a 3 kpc dist radius this is 20 pc resolution for the plot)

        x_points = np.linspace(x_min, x_max, npix)
        y_points = np.linspace(y_min, y_max, npix)
        z_points = np.linspace(z_min, z_max, npix)

        x_mesh, y_mesh, z_mesh = np.meshgrid(x_points, y_points, z_points)
        interp_coords = np.array([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()])

        #get spherical coods of plotting_coords - required
        interp_coords_spherical_d, interp_coords_spherical_b, interp_coords_spherical_l = cartesian_to_spherical(interp_coords[0], interp_coords[1], interp_coords[2])

        interp_coords_spherical_l = np.rad2deg(interp_coords_spherical_l.value)
        interp_coords_spherical_l[interp_coords_spherical_l > 180] = interp_coords_spherical_l[interp_coords_spherical_l > 180] - 360.
        interp_coords_spherical_b = np.rad2deg(interp_coords_spherical_b.value)
        interp_coords_spherical_d = interp_coords_spherical_d.value

        #Get the pixel indeces to be used in interpolation
        interp_indices_l = coords_to_index(interp_coords_spherical_l, l_mid)
        interp_indices_b = coords_to_index(interp_coords_spherical_b, b_mid)
        interp_indices_d = coords_to_index(interp_coords_spherical_d, d_mid)

        interp_indices = np.hstack((interp_indices_l[..., np.newaxis], interp_indices_b[..., np.newaxis], interp_indices_d[..., np.newaxis])).T #we need to transpose because we want an array with shape(3, npixels), but otherwise this would result in shape(npixels, 3). We need to use [..., np.newaxis] for each array as otherwise they would get concatenated along their existing axis, which would result in shape(3*npixels)


        print("Creating linear interpolate function")
        interp_points = map_coordinates(gpy_dens_cube, interp_indices)

        #now reshape it for integration
        interp_cube = np.reshape(interp_points, (npix, npix, npix)) #xyz axis order

        print("Begin integration along z axis for xy integ extinction plot")
        interp_image = np.trapz(interp_cube, z_points, axis=2) #integrated density (to get full integ extinction) along z axis
        np.save("IntegExt_xy_IntegAlongZ_Image_300pcZoom.pkl", interp_image, allow_pickle=True)

        print("Beginning plotting")
        fig = plt.figure(figsize=(15, 11)) #width, height
        ax = fig.add_subplot(1, 1, 1)
        normalize = ImageNormalize(interp_image, vmin=0, vmax=1, stretch=AsinhStretch(a=0.8)) #vmin=0, vmax=np.nanmax(interp_image)#vmin=0.0001, vmax=0.085. #, stretch=AsinhStretch(a=0.01). #vmin=0, vmax=0.01
        img = ax.imshow(interp_image[::-1,:], origin="upper", aspect="auto", cmap=colmap,
                            extent =(np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)),
                            norm=normalize 
                            )
        cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
        cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
        ax.set_xlabel("x [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
        ax.set_ylabel("y [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
        plt.xlim(-300, 300)
        plt.ylim(-300, 300)


        plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
        plt.savefig("IntegExt_xy_IntegAlongZ_ZoomInner300pc.png") 
        #plt.show()
        plt.close()


    #Load from saved interpolated data for existing stiched image
    else: 

        x_points = np.load("IntegExt_xy_IntegAlongZ_Xpoints.pkl.npy", allow_pickle=True)
        y_points = np.load("IntegExt_xy_IntegAlongZ_Ypoints.pkl.npy", allow_pickle=True)

        interp_image = np.load("IntegExt_xy_IntegAlongZ_Image_300pcZoom.pkl.npy", allow_pickle=True)

        print("Beginning plotting")
        fig = plt.figure(figsize=(15, 11)) #width, height
        ax = fig.add_subplot(1, 1, 1)
        normalize = ImageNormalize(interp_image, vmin=0, vmax=1, stretch=AsinhStretch(a=0.8)) #vmin=0, vmax=np.nanmax(interp_image)#vmin=0.0001, vmax=0.085. #, stretch=AsinhStretch(a=0.01). #vmin=0, vmax=0.01
        img = ax.imshow(interp_image[::-1,:], origin="upper", aspect="auto", cmap=colmap,
                            extent =(np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)),
                            norm=normalize 
                            )
        cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
        cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
        ax.set_xlabel("x [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
        ax.set_ylabel("y [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
        plt.xlim(-300, 300)
        plt.ylim(-300, 300)


        plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
        plt.savefig("IntegExt_xy_IntegAlongZ_ZoomInner300pc.png") 
        #plt.show()
        plt.close()






########### Plot xy  integrated extinction along z (integrated density cube along z) with Contour along extinction levels- plot units mag ##############
def plot_IntegExt_xy_IntegAlongZ_withContours(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp):

    x_points = np.load("IntegExt_xy_IntegAlongZ_Xpoints.pkl.npy", allow_pickle=True)
    y_points = np.load("IntegExt_xy_IntegAlongZ_Ypoints.pkl.npy", allow_pickle=True)

    interp_image = np.load("IntegExt_xy_IntegAlongZ_Image.pkl.npy", allow_pickle=True)

    colmap = cmr.get_sub_cmap("cmr.chroma", 0.05, 1)  

    print("Beginning plotting")
    fig = plt.figure(figsize=(15, 11)) #width, height
    ax = fig.add_subplot(1, 1, 1)
    normalize = ImageNormalize(interp_image, vmin=0, vmax=1, stretch=AsinhStretch(a=0.8)) #vmin=0, vmax=np.nanmax(interp_image)#vmin=0.0001, vmax=0.085. #, stretch=AsinhStretch(a=0.01). #vmin=0, vmax=0.01
    img = ax.imshow(interp_image[::-1,:], origin="upper", aspect="auto", cmap=colmap,
                        extent =(np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)),
                        norm=normalize 
                        )
    cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
    cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)

    contours = ax.contour(interp_image, levels=[0.3, 0.6, 1], origin="upper",  colors="black", 
                extent = (np.min(x_points), np.max(x_points), np.max(y_points), np.min(y_points))) #Contours matching our GP predictions
    ax.clabel(contours, inline=True, fontsize=20, fmt="%2.1f")

    ax.set_xlabel("x [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
    ax.set_ylabel("y [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")


    plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
    plt.savefig("IntegExt_xy_IntegAlongZ_withContours.png") 
    #plt.show()
    plt.close()





########### Plot xy  integrated extinction along z (integrated density cube along z) - plot units mag ##############
def plot_IntegExt_xy_IntegAlongZ_within_bounds(dense_med_cube, l_bounds_pred, b_bounds_pred, d_bounds_pred, re_interp, z_bounds = None):

    colmap = cmr.get_sub_cmap("cmr.chroma", 0.05, 1)
    
    if z_bounds is None:
        z_bounds = [-500, 500] #pc, default bounds

    #Interpolate data for new stitched image
    if re_interp: 
    
        gpy_dens_cube = dense_med_cube
        gpy_dens_cube = np.where(np.isfinite(gpy_dens_cube), gpy_dens_cube, 0)

        l_mid = (l_bounds_pred[1:] + l_bounds_pred[:-1])/2
        l_mid[l_mid > 180] = l_mid[l_mid>180] -360
        b_mid = (b_bounds_pred[1:] + b_bounds_pred[:-1])/2
        d_mid = (d_bounds_pred[1:] + d_bounds_pred[:-1])/2

        #Meshgrid ensures that all possible combinations of lbd coords are output as xyz coords so that we have all possible xyz coords
        print("Creating lbd mesh and xyz mid")
        lbd_mesh = np.meshgrid(l_mid, b_mid, d_mid)
        coords_lbd = np.array([lbd_mesh[0].flatten(), lbd_mesh[1].flatten(), lbd_mesh[2].flatten()]).T

        print("calc xyz mid")
        x_mid, y_mid, z_mid = spherical_to_cartesian(lbd_mesh[2], np.deg2rad(lbd_mesh[1]), np.deg2rad(lbd_mesh[0])) 

        #Now make the desired coordinates
        # print("xyz min max vals calculated")
        x_min = np.min(x_mid.value)
        y_min = np.min(y_mid.value)
        z_min = np.min(z_mid.value)

        x_max = np.max(x_mid.value)
        y_max = np.max(y_mid.value)
        z_max = np.max(z_mid.value)

        npix = 300 #Cell size in xyz for the plot (assuming a 3 kpc dist radius this is 20 pc resolution for the plot)

        x_points = np.linspace(x_min, x_max, npix)
        y_points = np.linspace(y_min, y_max, npix)
        z_points = np.linspace(z_min, z_max, npix)
        np.save("IntegExt_xy_IntegAlongZ_Xpoints.pkl",x_points, allow_pickle=True)
        np.save("IntegExt_xy_IntegAlongZ_Ypoints.pkl",y_points, allow_pickle=True)
        np.save("IntegExt_xy_IntegAlongZ_Zpoints.pkl",z_points, allow_pickle=True)

        x_mesh, y_mesh, z_mesh = np.meshgrid(x_points, y_points, z_points)
        interp_coords = np.array([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()])

        #get spherical coods of plotting_coords - required
        interp_coords_spherical_d, interp_coords_spherical_b, interp_coords_spherical_l = cartesian_to_spherical(interp_coords[0], interp_coords[1], interp_coords[2])

        interp_coords_spherical_l = np.rad2deg(interp_coords_spherical_l.value)
        interp_coords_spherical_l[interp_coords_spherical_l > 180] = interp_coords_spherical_l[interp_coords_spherical_l > 180] - 360.
        interp_coords_spherical_b = np.rad2deg(interp_coords_spherical_b.value)
        interp_coords_spherical_d = interp_coords_spherical_d.value

        #Get the pixel indeces to be used in interpolation
        interp_indices_l = coords_to_index(interp_coords_spherical_l, l_mid)
        interp_indices_b = coords_to_index(interp_coords_spherical_b, b_mid)
        interp_indices_d = coords_to_index(interp_coords_spherical_d, d_mid)

        interp_indices = np.hstack((interp_indices_l[..., np.newaxis], interp_indices_b[..., np.newaxis], interp_indices_d[..., np.newaxis])).T #we need to transpose because we want an array with shape(3, npixels), but otherwise this would result in shape(npixels, 3). We need to use [..., np.newaxis] for each array as otherwise they would get concatenated along their existing axis, which would result in shape(3*npixels)


        print("Creating linear interpolate function")
        interp_points = map_coordinates(gpy_dens_cube, interp_indices)

        #now reshape it for integration
        interp_cube = np.reshape(interp_points, (npix, npix, npix)) #xyz axis order
        np.save("IntegExt_xy_IntegAlongZ_Cube.pkl", interp_cube, allow_pickle=True)

        # Find the indices of the z points which are closest to the z bounds
        i_z_min = np.argmin(np.abs(z_points - z_bounds[0]))
        i_z_max = np.argmin(np.abs(z_points - z_bounds[1]))

        # now we use those indices to define the limits of the integration
        print(f"Begin integration along z axis (bounds: {z_bounds[0]}, {z_bounds[1]}) for xy integ extinction plot")
        interp_image = np.trapz(interp_cube[:,:,i_z_min:i_z_max], z_points[i_z_min:i_z_max], axis=2) #integrated density (to get full integ extinction) along z axis
        np.save("IntegExt_xy_IntegAlongZ_bounded_Image.pkl", interp_image, allow_pickle=True)

        print("Beginning plotting")
        fig = plt.figure(figsize=(15, 11)) #width, height
        ax = fig.add_subplot(1, 1, 1)
        normalize = ImageNormalize(interp_image, vmin=0, vmax=1, stretch=AsinhStretch(a=0.8)) #vmin=0, vmax=np.nanmax(interp_image)#vmin=0.0001, vmax=0.085. #, stretch=AsinhStretch(a=0.01). #vmin=0, vmax=0.01
        img = ax.imshow(interp_image[::-1,:], origin="upper", aspect="auto", cmap=colmap,
                            extent =(np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)),
                            norm=normalize 
                            )
        cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
        cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
        ax.set_xlabel("x [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
        ax.set_ylabel("y [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")


        plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
        plt.savefig(f"IntegExt_xy_IntegAlongZ_bounded_{z_bounds[0]}_{z_bounds[1]}.png") 
        #plt.show()
        plt.close()


    #Load from saved interpolated data for existing stiched image
    else: 

        x_points = np.load("IntegExt_xy_IntegAlongZ_Xpoints.pkl.npy", allow_pickle=True)
        y_points = np.load("IntegExt_xy_IntegAlongZ_Ypoints.pkl.npy", allow_pickle=True)
        z_points = np.load("IntegExt_xy_IntegAlongZ_Zpoints.pkl.npy", allow_pickle=True)

        #interp_image = np.load("IntegExt_xy_IntegAlongZ_Image.pkl.npy", allow_pickle=True)
        interp_cube = np.load("IntegExt_xy_IntegAlongZ_Cube.pkl.npy", allow_pickle=True)

        # Find the indices of the z points which are closest to the z bounds
        i_z_min = np.argmin(np.abs(z_points - z_bounds[0]))
        i_z_max = np.argmin(np.abs(z_points - z_bounds[1]))

        # now we use those indices to define the limits of the integration
        print(f"Begin integration along z axis (bounds: {z_bounds[0]}, {z_bounds[1]}) for xy integ extinction plot")
        interp_image = np.trapz(interp_cube[:,:,i_z_min:i_z_max], z_points[i_z_min:i_z_max], axis=2) #integrated density (to get full integ extinction) along z axis
        np.save("IntegExt_xy_IntegAlongZ_bounded_Image.pkl", interp_image, allow_pickle=True)

        print("Beginning plotting")
        fig = plt.figure(figsize=(15, 11)) #width, height
        ax = fig.add_subplot(1, 1, 1)
        normalize = ImageNormalize(interp_image, vmin=0, vmax=1, stretch=AsinhStretch(a=0.8)) #vmin=0, vmax=np.nanmax(interp_image)#vmin=0.0001, vmax=0.085. #, stretch=AsinhStretch(a=0.01). #vmin=0, vmax=0.01
        img = ax.imshow(interp_image[::-1,:], origin="upper", aspect="auto", cmap=colmap,
                            extent =(np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)),
                            norm=normalize 
                            )
        cbar = plt.colorbar(img, orientation="vertical", shrink=0.7, pad=0.025)
        cbar.set_label("Extinction [mag]", fontsize=fontsize, labelpad=20)
        ax.set_xlabel("x [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")
        ax.set_ylabel("y [pc]", fontsize=fontsize, family="cursive", style="oblique", weight="bold")


        plt.tight_layout(pad=4, h_pad=0.5, w_pad=0.5)
        plt.savefig(f"IntegExt_xy_IntegAlongZ_bounded_{z_bounds[0]}_{z_bounds[1]}.png") 
        #plt.show()
        plt.close()








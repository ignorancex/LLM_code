import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from glob import glob
import os
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, \
                    AsymmetricPercentileInterval, SqrtStretch, ManualInterval
import h5py 
from astropy.nddata import CCDData
import astropy.constants as const
from astropy.wcs import FITSFixedWarning
import warnings
warnings.simplefilter("ignore", category=FITSFixedWarning)
from PIL import Image
from datetime import datetime, timedelta
from ccdproc import ImageFileCollection
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FixedLocator, \
                 FixedFormatter, LogLocator, StrMethodFormatter
from matplotlib import patches
from matplotlib.markers import MarkerStyle
import cmcrameri.cm as cmcm
import cmasher as cmr
from scipy import ndimage
from scipy.io import readsav
import copy
from juanfit import SpectrumFitSingle, SpectrumFitRow, gaussian
import argparse
import sys
import inflect
from num2tex import num2tex

def fit_and_plot(line,order):
    def func_img_xpixel_to_xarcsec(x):
        return (x - img_center[0])/img_pixel_to_arcsec

    def func_img_xarcsec_to_xpixel(x):
        return x*img_pixel_to_arcsec + img_center[0]

    def func_img_ypixel_to_yarcsec(x):
        return (x - img_center[1])/img_pixel_to_arcsec

    def func_img_yarcsec_to_ypixel(x):
        return x*img_pixel_to_arcsec + img_center[1]

    def plot_colorbar(im, ax, width="3%", height="100%",loc="lower left",fontsize=14,
                    bbox_to_anchor=(1.02, 0., 1, 1)):
        clb_ax = inset_axes(ax,width=width,height=height,loc=loc,
                    bbox_to_anchor=bbox_to_anchor,
                    bbox_transform=ax.transAxes,
                    borderpad=0)
        clb = plt.colorbar(im,pad = 0.05,orientation='vertical',ax=ax,cax=clb_ax)
        clb_ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        clb_ax.yaxis.get_offset_text().set_fontsize(fontsize)
        clb_ax.tick_params(labelsize=fontsize)
        return clb, clb_ax

    def create_rec_eqs():
        return patches.Rectangle((eis_eqs_xstart, eis_eqs_ystart),
                                eis_eqs_fovx, eis_eqs_fovy,linewidth=2,edgecolor="red",
                                facecolor="none",alpha=0.6)

    def create_limb_circle(rsun):
        return patches.Circle((0,0),rsun,linewidth=2,edgecolor="grey",
                                facecolor="none",alpha=0.6,ls="--")
    
    def veff2teff(veff):
        return 55.85*const.u.cgs.value*(veff*1e5)**2/2/const.k_B.cgs.value/1e6

    def teff2veff(teff):
        return np.sqrt(2*const.k_B.cgs.value*teff*1e6/(55.85*const.u.cgs.value))/1e5

    with h5py.File("../../sav/Eclipse/LimbTrack/sun_pos_linear_fit.h5", 'r') as hf:
        sun_x_fitparam = hf['sun_x_fitparam'][:]
        sun_y_fitparam = hf['sun_y_fitparam'][:]

        sun_x_fitpoly = np.poly1d(sun_x_fitparam)
        sun_y_fitpoly = np.poly1d(sun_y_fitparam)

    eis_idl_sav = readsav("../../sav/EIS/EQSPY/EQSPY_lvl1_offset_170821_tilt_cor.sav",verbose=False)
    eis_eqs_xcen, eis_eqs_ycen = np.array((eis_idl_sav["xcen"], eis_idl_sav["ycen_195"])) + eis_idl_sav["xy_correct_aia"]
    eis_eqs_fovx, eis_eqs_fovy = np.array((eis_idl_sav["fovx"], eis_idl_sav["fovy"]))
    eis_eqs_xstart = eis_eqs_xcen - eis_eqs_fovx/2.
    eis_eqs_xend = eis_eqs_xcen + eis_eqs_fovx/2.
    eis_eqs_ystart = eis_eqs_ycen - eis_eqs_fovy/2.
    eis_eqs_yend = eis_eqs_ycen + eis_eqs_fovy/2.

    if line == "FeXIV":
        green_path = "../../src/EclipseSpectra2017/MikesData_l1/Green/"
        totality_green_im_collection = ImageFileCollection(green_path,
                            glob_include="TotalitySequence*.fit")
        totality_green_df = totality_green_im_collection.summary.to_pandas()
        totality_green_df["date-obs"] = pd.to_datetime(totality_green_df["date-obs"])

        totality_green_df_cut = totality_green_df.loc[(totality_green_df['date-obs'] >= datetime(2017,8,21,17,46,38)) & 
                                                (totality_green_df['date-obs'] < datetime(2017,8,21,17,47,0))]
        totality_green_df_cut.sort_values(by="date-obs")
        totality_green_df_cut = totality_green_df_cut.reset_index(drop=True)

        totality_green_df_ext = totality_green_df.loc[(totality_green_df['date-obs'] >= datetime(2017,8,21,17,45,36)) & 
                                                (totality_green_df['date-obs'] < datetime(2017,8,21,17,47,8))]
        totality_green_df_ext.sort_values(by="date-obs")                                        
        totality_green_df_ext = totality_green_df_ext.reset_index(drop=True)


        FeXIV_line_cont_frame = CCDData.read("../../src/EclipseSpectra2017/MitchellData/MitchellFeXIVLine_ContRatio.fits",unit="adu")
        FeXIV_line_cont_image = FeXIV_line_cont_frame.data
        sun_center_FeXIV = (np.float64(FeXIV_line_cont_frame.header["SUNX"]),np.float64(FeXIV_line_cont_frame.header["SUNY"]))
        FeXIV_line_cont_xslice = slice(372-300,372+301)
        FeXIV_line_cont_yslice = slice(383-220,383+221)
        FeXIV_line_cont_cutout = FeXIV_line_cont_image[FeXIV_line_cont_yslice, FeXIV_line_cont_xslice]
        FeXIV_rotate_center = (sun_center_FeXIV[0] - FeXIV_line_cont_xslice.start, sun_center_FeXIV[1] - FeXIV_line_cont_yslice.start)
        FeXIV_line_cont_image_rot_scipy = ndimage.rotate(FeXIV_line_cont_cutout, angle=360 - np.float64(FeXIV_line_cont_frame.header["SUNROT"]),reshape=False,order=1)

        # green_frame_example = CCDData.read(os.path.join(green_path,totality_green_df_cut.iloc[0]["file"]),hdu=0,unit="adu")
        green_frame_wavelength = CCDData.read(os.path.join(green_path,totality_green_df_cut.iloc[0]["file"]),hdu=1,unit="adu").data

        img_center = np.array([300,220])
        slit_pos = 209.4
        rsun_arcsec = 950.0
        rsun_context_pixel = 71.4
        pixel_ratio = rsun_context_pixel/np.float64(FeXIV_line_cont_frame.header["MOONR"])
        img_pixel_to_arcsec = np.float64(FeXIV_line_cont_frame.header["SUNR"])/rsun_arcsec
        pixel_ratio_to_arcsec = rsun_context_pixel/np.float64(FeXIV_line_cont_frame.header["MOONR"])*img_pixel_to_arcsec
        rotate_angle_context = -27.5


        if order == 63:
            with h5py.File("../../sav/Eclipse/FlatField/skyflat_green_1d_FeXIV_63rd.h5", 'r') as hf:
                flatfield_1d = hf['flatfield_1d'][:]
            FeXIV_xslice = slice(345,395)

            with h5py.File("../../sav/Eclipse/FitResults/FeXIV_63.h5",'r') as hf:
                green_fit_matrix_ext = hf['green_fit_matrix_ext'][:]
                green_fit_matrix_ext_err = hf['green_fit_matrix_ext_err'][:]
                green_fit_matrix_bin_ext = hf['green_fit_matrix_bin_ext'][:]
                green_fit_matrix_bin_ext_err = hf['green_fit_matrix_bin_ext_err'][:]
                green_fit_filename_index = hf['green_fit_filename_index'][:]

        if order == 62:
            with h5py.File("../../sav/Eclipse/FlatField/skyflat_green_1d_FeXIV_62nd.h5", 'r') as hf:
                flatfield_1d = hf['flatfield_1d'][:]
            FeXIV_xslice = slice(680,730)

            with h5py.File("../../sav/Eclipse/FitResults/FeXIV_62.h5",'r') as hf:
                green_fit_matrix_ext = hf['green_fit_matrix_ext'][:]
                green_fit_matrix_ext_err = hf['green_fit_matrix_ext_err'][:]
                green_fit_matrix_bin_ext = hf['green_fit_matrix_bin_ext'][:]
                green_fit_matrix_bin_ext_err = hf['green_fit_matrix_bin_ext_err'][:]
                green_fit_filename_index = hf['green_fit_filename_index'][:]        

        if order == 61:
            with h5py.File("../../sav/Eclipse/FlatField/skyflat_green_1d_FeXIV_61st.h5", 'r') as hf:
                flatfield_1d = hf['flatfield_1d'][:]
            FeXIV_xslice = slice(1010,1060)

            with h5py.File("../../sav/Eclipse/FitResults/FeXIV_61.h5",'r') as hf:
                green_fit_matrix_ext = hf['green_fit_matrix_ext'][:]
                green_fit_matrix_ext_err = hf['green_fit_matrix_ext_err'][:]
                green_fit_matrix_bin_ext = hf['green_fit_matrix_bin_ext'][:]
                green_fit_matrix_bin_ext_err = hf['green_fit_matrix_bin_ext_err'][:]
                green_fit_filename_index = hf['green_fit_filename_index'][:]      

        # starttime_green_ext = datetime(2017,8,21,17,45,36)

        green_limb_loc = np.array([396.,625.,])

        x_1d_grid_green_ext = np.arange(-63,125,1,dtype=np.float64)
        y_1d_grid_green_ext = np.arange(np.mean(green_limb_loc) - 699.,  np.mean(green_limb_loc) - 349., 1, dtype=np.float64)

        y_1d_grid_green_arcsec_ext = y_1d_grid_green_ext/(np.diff(green_limb_loc)/2.)*rsun_arcsec * \
                np.float64(FeXIV_line_cont_frame.header["MOONR"])/np.float64(FeXIV_line_cont_frame.header["SUNR"])
        x_1d_grid_green_arcsec_ext = x_1d_grid_green_ext * (sun_x_fitpoly(10) - sun_x_fitpoly(9.5))/pixel_ratio_to_arcsec
        y_1d_grid_green_arcsec_bin_ext = np.average(y_1d_grid_green_arcsec_ext.reshape(-1,5),axis=1)

        x_2d_grid_green_arcsec_ext, y_2d_grid_green_arcsec_ext = np.meshgrid(x_1d_grid_green_arcsec_ext, y_1d_grid_green_arcsec_ext)
        x_2d_grid_green_arcsec_bin_ext, y_2d_grid_green_arcsec_bin_ext = np.meshgrid(x_1d_grid_green_arcsec_ext, y_1d_grid_green_arcsec_bin_ext)

        y_green_step_correction_ext = (sun_y_fitpoly(np.linspace(0,93.5,188)) - sun_y_fitpoly(62))/rsun_context_pixel*rsun_arcsec * \
                np.float64(FeXIV_line_cont_frame.header["MOONR"])/np.float64(FeXIV_line_cont_frame.header["SUNR"])
        y_green_step_correction_ext = np.flip(y_green_step_correction_ext)

        y_2d_grid_green_arcsec_correct_ext = y_2d_grid_green_arcsec_ext + y_green_step_correction_ext[np.newaxis,:]
        y_2d_grid_green_arcsec_bin_correct_ext = y_2d_grid_green_arcsec_bin_ext + y_green_step_correction_ext[np.newaxis,:]

        x_2d_grid_green_arcsec_rot_ext = np.cos(np.deg2rad(np.abs(rotate_angle_context)))*x_2d_grid_green_arcsec_ext + \
                                    np.sin(np.deg2rad(np.abs(rotate_angle_context)))*y_2d_grid_green_arcsec_correct_ext

        y_2d_grid_green_arcsec_rot_ext = - np.sin(np.deg2rad(np.abs(rotate_angle_context)))*x_2d_grid_green_arcsec_ext + \
                                    np.cos(np.deg2rad(np.abs(rotate_angle_context)))*y_2d_grid_green_arcsec_correct_ext

        x_2d_grid_green_arcsec_bin_rot_ext = np.cos(np.deg2rad(np.abs(rotate_angle_context)))*x_2d_grid_green_arcsec_bin_ext + \
                                    np.sin(np.deg2rad(np.abs(rotate_angle_context)))*y_2d_grid_green_arcsec_bin_correct_ext

        y_2d_grid_green_arcsec_bin_rot_ext = - np.sin(np.deg2rad(np.abs(rotate_angle_context)))*x_2d_grid_green_arcsec_bin_ext + \
                                    np.cos(np.deg2rad(np.abs(rotate_angle_context)))*y_2d_grid_green_arcsec_bin_correct_ext

        pixel_size_green = np.abs(np.mean(np.diff((green_frame_wavelength/order/10.)[FeXIV_xslice])))
        inst_width_pix_green = 1.86
        inst_width_nm_green = pixel_size_green*inst_width_pix_green
        inst_width_pix_err_green = 0.4
        inst_width_nm_err_green = pixel_size_green*inst_width_pix_err_green

        img_xpixel_array = np.arange(FeXIV_line_cont_image_rot_scipy.shape[1])
        img_ypixel_array = np.arange(FeXIV_line_cont_image_rot_scipy.shape[0])

        img_xarcsec_array = func_img_xpixel_to_xarcsec(img_xpixel_array)
        img_yarcsec_array = func_img_ypixel_to_yarcsec(img_ypixel_array)

        slit_xshift_green = sun_x_fitpoly(62) - slit_pos
        slit_center_x_green =  - slit_xshift_green/pixel_ratio_to_arcsec*np.cos(np.deg2rad(np.abs(rotate_angle_context)))
        slit_center_y_green =  slit_xshift_green/pixel_ratio_to_arcsec*np.sin(np.deg2rad(np.abs(rotate_angle_context)))

        mask_bad_fit_matrix_ext = np.where(np.any(np.abs(green_fit_matrix_ext_err/green_fit_matrix_ext)[:3,:,:] > 0.5,axis=0) | \
                np.any(np.isnan(green_fit_matrix_ext_err),axis=0))
        
        mask_bad_fit_matrix_bin_ext = np.where(np.any(np.abs(green_fit_matrix_bin_ext_err/green_fit_matrix_bin_ext)[:3,:,:] > 0.2,axis=0) | \
                        np.any(np.isnan(green_fit_matrix_bin_ext_err),axis=0))

        # kernel = Gaussian2DKernel(x_stddev=1,y_stddev=2)
        # kernel_bin = Gaussian2DKernel(x_stddev=1,y_stddev=0.2)

        green_where_disk_ext = np.where((x_2d_grid_green_arcsec_rot_ext + slit_center_x_green)**2 + \
                                            (y_2d_grid_green_arcsec_rot_ext + slit_center_y_green)**2 < 970**2)

        green_line_int_masked = np.copy(green_fit_matrix_ext[1,:,:])
        # green_line_int_masked[mask_bad_fit_matrix_ext] = np.nan
        # green_line_int_masked = interpolate_replace_nans(green_line_int_masked,kernel)
        green_line_int_masked[green_where_disk_ext] = np.nan

        green_line_int_masked_err = np.copy(green_fit_matrix_ext_err[1,:,:])
        green_line_int_masked_err[green_where_disk_ext] = np.nan

        green_cont_masked = np.copy(green_fit_matrix_ext[3,:,:]+green_fit_matrix_ext[4,:,:])
        # green_cont_masked[mask_bad_fit_matrix_ext] = np.nan
        # green_cont_masked = interpolate_replace_nans(green_cont_masked, kernel)
        green_cont_masked[green_where_disk_ext] = np.nan

        green_line_cont_ratio_masked = green_fit_matrix_ext[1,:,:]/(green_fit_matrix_ext[3,:,:]+green_fit_matrix_ext[4,:,:])
        # green_line_cont_ratio_masked[mask_bad_fit_matrix_ext] = np.nan
        # green_line_cont_ratio_masked = interpolate_replace_nans(green_line_cont_ratio_masked,kernel)
        green_line_cont_ratio_masked[green_where_disk_ext] = np.nan

        green_where_disk_bin_ext = np.where((x_2d_grid_green_arcsec_bin_rot_ext + slit_center_x_green)**2 + \
                                            (y_2d_grid_green_arcsec_bin_rot_ext + slit_center_y_green)**2 < 940**2)

        green_vlos_masked = -(np.copy(green_fit_matrix_bin_ext[0,:,:]) - 530.29)/530.29*const.c.cgs.value*1e-5
        green_vlos_masked[np.where(green_fit_matrix_bin_ext[1,:,:] < 15)] = np.nan
        green_vlos_masked[mask_bad_fit_matrix_bin_ext] = np.nan
        green_vlos_masked[green_where_disk_bin_ext] = np.nan
        green_vlos_masked = green_vlos_masked - np.nanmedian(green_vlos_masked)
        # green_vlos_masked = interpolate_replace_nans(green_vlos_masked,kernel_bin)
        green_vlos_masked[green_where_disk_bin_ext] = np.nan

        green_vlos_masked_err = green_fit_matrix_bin_ext_err[0,:,:]/530.29*const.c.cgs.value*1e-5
        green_vlos_masked_err[np.where(green_fit_matrix_bin_ext[1,:,:] < 15)] = np.nan
        green_vlos_masked_err[green_where_disk_bin_ext] = np.nan

        green_fwhm_masked = np.copy(green_fit_matrix_bin_ext[2,:,:])
        green_fwhm_masked[np.where(green_fit_matrix_bin_ext[1,:,:] < 15)] = np.nan
        green_fwhm_masked[mask_bad_fit_matrix_bin_ext] = np.nan
        # green_fwhm_masked_err[green_where_disk_bin_ext] = np.nan        
        green_veff_masked = np.sqrt(green_fwhm_masked**2 - inst_width_nm_green**2)/530.29*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2))
        # green_veff_masked = interpolate_replace_nans(green_veff_masked,kernel_bin)
        green_veff_masked[green_where_disk_bin_ext] = np.nan

        green_fwhm_masked_err = np.copy(green_fit_matrix_bin_ext_err[2,:,:])
        green_fwhm_masked_err[np.where(green_fit_matrix_bin_ext[1,:,:] < 15)] = np.nan
        green_fwhm_masked_err[green_where_disk_bin_ext] = np.nan     
        green_veff_masked_err = np.sqrt(green_fwhm_masked**2/(green_fwhm_masked**2 - inst_width_nm_green**2)* \
            green_fwhm_masked_err**2 + inst_width_nm_green**2/(green_fwhm_masked**2 - inst_width_nm_green**2)* \
            inst_width_nm_err_green**2)/530.29*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2))


        colorbar_width = "10%"

        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(11.5,7.6),constrained_layout=True)
        fig.get_layout_engine().set(rect=(0.02,0.02,0.87,0.95))

        im1 = ax1.pcolormesh(img_xarcsec_array,img_yarcsec_array,FeXIV_line_cont_image_rot_scipy,vmin=0.2,vmax=1.2,
                            cmap=cmr.jungle_r,shading="auto",rasterized=True)

        plot_colorbar(im1, ax1,width=colorbar_width)

        # for ax_ in (ax2,ax3,ax4,ax5,ax6):
        #     ax_.pcolormesh(img_xarcsec_array,img_yarcsec_array,FeXIV_line_cont_image_rot_scipy,vmin=0.2,vmax=1.2,
        #                     cmap=cmr.jungle_r,shading="auto",rasterized=True,alpha=0.5)

        # green_where_disk_ext = np.where((x_2d_grid_green_arcsec_rot_ext + slit_center_x_green)**2 + \
        #                                         (y_2d_grid_green_arcsec_rot_ext + slit_center_y_green)**2 < 940**2)

        # green_line_int_masked = np.copy(green_fit_matrix_ext[1,:,:])
        # green_line_int_masked[green_where_disk_ext] = np.nan

        norm_green_line_int = ImageNormalize(green_line_int_masked,interval=ManualInterval(0,850),
                                                stretch=SqrtStretch())

        im2 = ax2.pcolormesh(x_2d_grid_green_arcsec_rot_ext + slit_center_x_green,
                        y_2d_grid_green_arcsec_rot_ext + slit_center_y_green,
                        green_line_int_masked,cmap=cmr.jungle_r,rasterized=True,norm=norm_green_line_int)


        plot_colorbar(im2, ax2,width=colorbar_width)

        # green_cont_masked = np.copy(green_fit_matrix_ext[3,:,:]+green_fit_matrix_ext[4,:,:])
        # # green_cont_masked[mask_bad_fit_matrix_ext] = np.nan
        # # green_cont_masked = interpolate_replace_nans(green_cont_masked, kernel)
        # green_cont_masked[green_where_disk_ext] = np.nan

        # green_line_cont_ratio_masked = green_fit_matrix_ext[1,:,:]/(green_fit_matrix_ext[3,:,:]+green_fit_matrix_ext[4,:,:])
        # # green_line_cont_ratio_masked[mask_bad_fit_matrix_ext] = np.nan
        # # green_line_cont_ratio_masked = interpolate_replace_nans(green_line_cont_ratio_masked,kernel)
        # green_line_cont_ratio_masked[green_where_disk_ext] = np.nan
        
        im3 = ax3.pcolormesh(x_2d_grid_green_arcsec_rot_ext + slit_center_x_green,
                        y_2d_grid_green_arcsec_rot_ext + slit_center_y_green,
                        green_cont_masked,cmap=cmr.jungle_r,rasterized=True,vmin=1000,vmax=10000)

        im3_clb, im3_clbax = plot_colorbar(im3, ax3,width=colorbar_width)
        im3_clbax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        im3_clbax.yaxis.get_offset_text().set_fontsize(14)
        im3_clbax.yaxis.get_offset_text().set(va="bottom",ha="left")
        im3_clbax.yaxis.get_offset_text().set_position((1,1.1))

        # green_line_cont_ratio_masked = green_fit_matrix_ext[1,:,:]/green_fit_matrix_ext[3,:,:]
        # green_line_cont_ratio_masked[green_where_disk_ext] = np.nan

        im4 = ax4.pcolormesh(x_2d_grid_green_arcsec_rot_ext + slit_center_x_green,
                        y_2d_grid_green_arcsec_rot_ext + slit_center_y_green,
                        green_line_cont_ratio_masked,cmap=cmr.jungle_r,rasterized=True,vmin=0,vmax=0.2)

        plot_colorbar(im4, ax4,width=colorbar_width)

        # green_where_disk_bin_ext = np.where((x_2d_grid_green_arcsec_bin_rot_ext + slit_center_x_green)**2 + \
        #                                         (y_2d_grid_green_arcsec_bin_rot_ext + slit_center_y_green)**2 < 940**2)

        # green_vlos_masked = -(np.copy(green_fit_matrix_bin_ext[0,:,:]) - 530.29)/530.29*const.c.cgs.value*1e-5
        # green_vlos_masked[np.where(green_fit_matrix_bin_ext[1,:,:] < 15)] = np.nan
        # green_vlos_masked[green_where_disk_bin_ext] = np.nan
        # green_vlos_masked = green_vlos_masked - np.nanmedian(green_vlos_masked)
        # green_vlos_masked_err = green_fit_matrix_bin_ext_err[0,:,:]/530.29*const.c.cgs.value*1e-5
        # green_vlos_masked_err[np.where(green_fit_matrix_bin_ext[1,:,:] < 15)] = np.nan
        # green_vlos_masked_err[green_where_disk_bin_ext] = np.nan

        im5 = ax5.pcolormesh(x_2d_grid_green_arcsec_bin_rot_ext + slit_center_x_green,
                        y_2d_grid_green_arcsec_bin_rot_ext + slit_center_y_green,
                        green_vlos_masked,cmap=cmcm.vik_r,rasterized=True,vmin=-10,vmax=10)

        plot_colorbar(im5, ax5,width=colorbar_width)

        # fwhm_masked = np.copy(green_fit_matrix_bin_ext[2,:,:])
        # fwhm_masked[np.where(green_fit_matrix_bin_ext[1,:,:] < 15)] = np.nan
        # fwhm_masked[green_where_disk_bin_ext] = np.nan
        # fwhm_masked_err = np.copy(green_fit_matrix_bin_ext_err[2,:,:])
        # fwhm_masked_err[np.where(green_fit_matrix_bin_ext[1,:,:] < 15)] = np.nan
        # fwhm_masked_err[green_where_disk_bin_ext] = np.nan        
        # veff_masked = np.sqrt(fwhm_masked**2 - inst_width_nm_green**2)/530.29*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2))
        # veff_masked_err = fwhm_masked/np.sqrt(fwhm_masked**2 - inst_width_nm_green**2)* \
        #     fwhm_masked_err/530.29*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2))

        # im6 = ax6.pcolormesh(x_2d_grid_green_arcsec_bin_rot_ext + slit_center_x_green,
        #                 y_2d_grid_green_arcsec_bin_rot_ext + slit_center_y_green,veff_masked,
        #                 cmap=cmcm.batlowK,rasterized=True,vmin=0.07/530.29*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2)),
        #                 vmax=0.11/530.29*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2)))

        # plot_colorbar(im6, ax6,width=colorbar_width)

        im6 = ax6.pcolormesh(x_2d_grid_green_arcsec_bin_rot_ext + slit_center_x_green,
                        y_2d_grid_green_arcsec_bin_rot_ext + slit_center_y_green,
                        green_veff_masked,cmap=cmcm.batlowK,rasterized=True,
                        vmin=0.07/530.29*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2))
                        ,vmax=0.12/530.29*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2)))


        clb6, clb_ax6 = plot_colorbar(im6, ax6,width=colorbar_width,bbox_to_anchor=(1.3, 0., 1, 1),fontsize=16)
        clb_ax6.yaxis.set_ticks_position("left")
        clb_ax6.set_ylabel(r"$\boldsymbol{v_{\rm eff}\ [\mathrm{km\,s^{-1}}]}$",fontsize=16)
        clb_ax6.yaxis.set_label_position("left")
        clb_ax6_teff = clb_ax6.secondary_yaxis("right",functions=(veff2teff, teff2veff))
        clb_ax6_teff.tick_params(labelsize=16)
        clb_ax6_teff.set_ylabel(r"$\boldsymbol{T_{\rm eff}\ [\mathrm{MK}]}$",fontsize=16)

        ax1.set_ylabel("Solar-Y [arcsec]",fontsize=14)
        ax4.set_xlabel("Solar-X [arcsec]",fontsize=14)
        ax4.set_ylabel("Solar-Y [arcsec]",fontsize=14)
        ax5.set_xlabel("Solar-X [arcsec]",fontsize=14)
        ax6.set_xlabel("Solar-X [arcsec]",fontsize=14)

        ax1.set_title(r"Fe \textsc{xiv} 530.3\,nm",fontsize=16)
        ax2.set_title(r"Fe \textsc{xiv} 530.3\,nm $I_{\rm tot}$ [arb.u.]",fontsize=16)
        ax3.set_title(r"Fe \textsc{xiv} 530.3\,nm $I_{\rm cont}$ [arb.u.]",fontsize=16)
        ax4.set_title(r"Fe \textsc{xiv} 530.3\,nm $I_{\rm tot}/I_{\rm cont}$",fontsize=16)
        ax5.set_title(r"Fe \textsc{xiv} 530.3\,nm $v_{\rm LOS}\ \left[\mathrm{km\,s^{-1}}\right]$",fontsize=16)
        ax6.set_title(r"Fe \textsc{xiv} 530.3\,nm $v_{\rm eff}\ \left[\mathrm{km\,s^{-1}}\right]$",fontsize=16)

        xlim_zoomin = [-1500,0]
        ylim_zoomin = [-1000,1000]

        for ax_ in (ax2,ax3,ax5,ax6):
            ax_.tick_params(labelleft=False)

        for ax_ in (ax1,ax2,ax3):
            ax_.tick_params(labelbottom=False)

        for ax_ in (ax1,ax2,ax3,ax4,ax5,ax6):
            ax_.contour(img_xarcsec_array,img_yarcsec_array,FeXIV_line_cont_image_rot_scipy,levels=[0.4,0.65,0.9],alpha=0.6,
                    colors=['grey'])
            ax_.add_patch(create_rec_eqs())
            ax_.add_patch(create_limb_circle(rsun_arcsec))
            ax_.set_aspect(1)
            ax_.tick_params(labelsize=14)
            ax_.set_xlim(xlim_zoomin)
            ax_.set_ylim(ylim_zoomin)

        fig.canvas.draw()
        GetFitProfile(fig, (ax1,ax2,ax3,ax4,ax5,ax6),x_2d_grid_green_arcsec_rot_ext + slit_center_x_green,
                        y_2d_grid_green_arcsec_rot_ext + slit_center_y_green,
                        x_2d_grid_green_arcsec_bin_rot_ext + slit_center_x_green,
                        y_2d_grid_green_arcsec_bin_rot_ext + slit_center_y_green,
                        green_fit_filename_index, green_path,
                        totality_green_df_ext,FeXIV_xslice,flatfield_1d,order, green_fit_matrix_ext,
                        green_fit_matrix_ext_err, green_fit_matrix_bin_ext, green_fit_matrix_bin_ext_err,
                        green_vlos_masked, green_vlos_masked_err, green_veff_masked, green_veff_masked_err)
        
        plt.show()

    elif line == "FeX":
        red_path = "../../src/EclipseSpectra2017/MikesData_l1/Red/"

        totality_red_im_collection = ImageFileCollection(red_path,
                                glob_include="TotalitySequence*.fit")
        totality_red_df = totality_red_im_collection.summary.to_pandas()
        totality_red_df["date-obs"] = pd.to_datetime(totality_red_df["date-obs"])

        totality_red_df_cut = totality_red_df.loc[(totality_red_df['date-obs'] >= datetime(2017,8,21,17,46,40)) & 
                                                (totality_red_df['date-obs'] < datetime(2017,8,21,17,47,0))]
        totality_red_df_cut.sort_values(by="date-obs")
        totality_red_df_cut = totality_red_df_cut.reset_index(drop=True)

        totality_red_df_ext = totality_red_df.loc[(totality_red_df['date-obs'] >= datetime(2017,8,21,17,45,36)) & 
                                                (totality_red_df['date-obs'] < datetime(2017,8,21,17,47,2))]
        totality_red_df_ext.sort_values(by="date-obs")
        totality_red_df_ext = totality_red_df_ext.reset_index(drop=True)

        FeXI_line_cont_frame = CCDData.read("../../src/EclipseSpectra2017/MitchellData/MitchellFeXILine_ContRatio.fits",unit="adu")

        FeXI_line_cont_image = FeXI_line_cont_frame.data
        sun_center_FeXI = (np.float64(FeXI_line_cont_frame.header["SUNX"]),np.float64(FeXI_line_cont_frame.header["SUNY"]))
        FeXI_line_cont_xslice = slice(372-300,372+301)
        FeXI_line_cont_yslice = slice(383-220,383+221)
        FeXI_line_cont_cutout = FeXI_line_cont_image[FeXI_line_cont_yslice, FeXI_line_cont_xslice]
        FeXI_rotate_center = (sun_center_FeXI[0] - FeXI_line_cont_xslice.start, sun_center_FeXI[1] - FeXI_line_cont_yslice.start)
        FeXI_line_cont_image_rot_scipy = ndimage.rotate(FeXI_line_cont_cutout, angle=360 - np.float64(FeXI_line_cont_frame.header["SUNROT"]),reshape=False,order=1)

        red_frame_wavelength = CCDData.read(os.path.join(red_path,totality_red_df_cut.iloc[0]["file"]),hdu=1,unit="adu").data

        img_center = np.array([300,220])
        slit_pos = 209.4
        rsun_arcsec = 950.0
        rsun_context_pixel = 71.4
        pixel_ratio = rsun_context_pixel/np.float64(FeXI_line_cont_frame.header["MOONR"])
        img_pixel_to_arcsec = np.float64(FeXI_line_cont_frame.header["SUNR"])/rsun_arcsec
        pixel_ratio_to_arcsec = rsun_context_pixel/np.float64(FeXI_line_cont_frame.header["MOONR"])*img_pixel_to_arcsec
        rotate_angle_context = -27.5

        if order == 53:
            with h5py.File("../../sav/Eclipse/FlatField/skyflat_red_1d_FeX_53rd.h5", 'r') as hf:
                flatfield_1d = hf['flatfield_1d'][:]
            FeX_xslice = slice(1025,1075)

            with h5py.File("../../sav/Eclipse/FitResults/FeX_53.h5",'r') as hf:
                red_fit_matrix_ext = hf['red_fit_matrix_ext'][:]
                red_fit_matrix_ext_err = hf['red_fit_matrix_ext_err'][:]
                red_fit_matrix_bin_ext = hf['red_fit_matrix_bin_ext'][:]
                red_fit_matrix_bin_ext_err = hf['red_fit_matrix_bin_ext_err'][:]
                red_fit_filename_index = hf['red_fit_filename_index'][:]

        if order == 52:
            with h5py.File("../../sav/Eclipse/FlatField/skyflat_red_1d_FeX_52nd.h5", 'r') as hf:
                flatfield_1d = hf['flatfield_1d'][:]
            FeX_xslice = slice(602,652)

            with h5py.File("../../sav/Eclipse/FitResults/FeX_52.h5",'r') as hf:
                red_fit_matrix_ext = hf['red_fit_matrix_ext'][:]
                red_fit_matrix_ext_err = hf['red_fit_matrix_ext_err'][:]
                red_fit_matrix_bin_ext = hf['red_fit_matrix_bin_ext'][:]
                red_fit_matrix_bin_ext_err = hf['red_fit_matrix_bin_ext_err'][:]
                red_fit_filename_index = hf['red_fit_filename_index'][:]

        if order == 51:
            with h5py.File("../../sav/Eclipse/FlatField/skyflat_red_1d_FeX_51st.h5", 'r') as hf:
                flatfield_1d = hf['flatfield_1d'][:]
            FeX_xslice = slice(205,255)

            with h5py.File("../../sav/Eclipse/FitResults/FeX_51.h5",'r') as hf:
                red_fit_matrix_ext = hf['red_fit_matrix_ext'][:]
                red_fit_matrix_ext_err = hf['red_fit_matrix_ext_err'][:]
                red_fit_matrix_bin_ext = hf['red_fit_matrix_bin_ext'][:]
                red_fit_matrix_bin_ext_err = hf['red_fit_matrix_bin_ext_err'][:]
                red_fit_filename_index = hf['red_fit_filename_index'][:]

        starttime_red_ext = datetime(2017,8,21,17,45,36)

        red_limb_loc = np.array([366.,592.,])
        x_1d_grid_red_ext = np.arange(-51,125,1,dtype=np.float64) + 8
        y_1d_grid_red_ext = np.arange(np.mean(red_limb_loc) - 699.,  np.mean(red_limb_loc) - 349., 1, dtype=np.float64)

        y_1d_grid_red_arcsec_ext = y_1d_grid_red_ext/(np.diff(red_limb_loc)/2.)*rsun_arcsec * \
                np.float64(FeXI_line_cont_frame.header["MOONR"])/np.float64(FeXI_line_cont_frame.header["SUNR"])
        x_1d_grid_red_arcsec_ext = x_1d_grid_red_ext * (sun_x_fitpoly(10) - sun_x_fitpoly(9.5))/pixel_ratio_to_arcsec
        y_1d_grid_red_arcsec_bin_ext = np.average(y_1d_grid_red_arcsec_ext.reshape(-1,5),axis=1)

        x_2d_grid_red_arcsec_ext, y_2d_grid_red_arcsec_ext = np.meshgrid(x_1d_grid_red_arcsec_ext, y_1d_grid_red_arcsec_ext)
        x_2d_grid_red_arcsec_bin_ext, y_2d_grid_red_arcsec_bin_ext = np.meshgrid(x_1d_grid_red_arcsec_ext, y_1d_grid_red_arcsec_bin_ext)

        y_red_step_correction_ext = (sun_y_fitpoly(np.linspace(0,87.5,176) - 4) - sun_y_fitpoly(66))/rsun_context_pixel*rsun_arcsec * \
                np.float64(FeXI_line_cont_frame.header["MOONR"])/np.float64(FeXI_line_cont_frame.header["SUNR"])
        y_red_step_correction_ext = np.flip(y_red_step_correction_ext)

        y_2d_grid_red_arcsec_correct_ext = y_2d_grid_red_arcsec_ext + y_red_step_correction_ext[np.newaxis,:]
        y_2d_grid_red_arcsec_bin_correct_ext = y_2d_grid_red_arcsec_bin_ext + y_red_step_correction_ext[np.newaxis,:]

        x_2d_grid_red_arcsec_rot_ext = np.cos(np.deg2rad(np.abs(rotate_angle_context)))*x_2d_grid_red_arcsec_ext + \
                                    np.sin(np.deg2rad(np.abs(rotate_angle_context)))*y_2d_grid_red_arcsec_correct_ext

        y_2d_grid_red_arcsec_rot_ext = - np.sin(np.deg2rad(np.abs(rotate_angle_context)))*x_2d_grid_red_arcsec_ext + \
                                    np.cos(np.deg2rad(np.abs(rotate_angle_context)))*y_2d_grid_red_arcsec_correct_ext

        x_2d_grid_red_arcsec_bin_rot_ext = np.cos(np.deg2rad(np.abs(rotate_angle_context)))*x_2d_grid_red_arcsec_bin_ext + \
                                    np.sin(np.deg2rad(np.abs(rotate_angle_context)))*y_2d_grid_red_arcsec_bin_correct_ext

        y_2d_grid_red_arcsec_bin_rot_ext = - np.sin(np.deg2rad(np.abs(rotate_angle_context)))*x_2d_grid_red_arcsec_bin_ext + \
                                    np.cos(np.deg2rad(np.abs(rotate_angle_context)))*y_2d_grid_red_arcsec_bin_correct_ext

        
        pixel_size_red = np.abs(np.mean(np.diff((red_frame_wavelength/order/10.)[FeX_xslice])))
        inst_width_pix_red = 2.12
        inst_width_nm_red = pixel_size_red*inst_width_pix_red

        img_center = np.array([300,220])
        img_xpixel_array = np.arange(FeXI_line_cont_image_rot_scipy.shape[1])
        img_ypixel_array = np.arange(FeXI_line_cont_image_rot_scipy.shape[0])

        img_xarcsec_array = func_img_xpixel_to_xarcsec(img_xpixel_array)
        img_yarcsec_array = func_img_ypixel_to_yarcsec(img_ypixel_array)

        colorbar_width = "10%"
        slit_xshift_red = sun_x_fitpoly(62) - slit_pos

        slit_center_x_red =  - slit_xshift_red/pixel_ratio_to_arcsec*np.cos(np.deg2rad(np.abs(rotate_angle_context)))
        slit_center_y_red =  slit_xshift_red/pixel_ratio_to_arcsec*np.sin(np.deg2rad(np.abs(rotate_angle_context)))

        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        im1 = ax1.pcolormesh(img_xarcsec_array,img_yarcsec_array,FeXI_line_cont_image_rot_scipy,vmin=1,vmax=4,
                            cmap=cmcm.lajolla,shading="auto",rasterized=True)

        plot_colorbar(im1, ax1,width=colorbar_width)

        red_where_disk_ext = np.where((x_2d_grid_red_arcsec_rot_ext + slit_center_x_red)**2 + \
                                                (y_2d_grid_red_arcsec_rot_ext + slit_center_y_red)**2 < 940**2)

        for ax_ in (ax2,ax3,ax4,ax5,ax6):
            ax_.pcolormesh(img_xarcsec_array,img_yarcsec_array,FeXI_line_cont_image_rot_scipy,vmin=1,vmax=4,
                                cmap=cmcm.lajolla,shading="auto",rasterized=True,alpha=0.6)

        red_line_int_masked = np.copy(red_fit_matrix_ext[1,:,:])
        red_line_int_masked[red_where_disk_ext] = np.nan
        norm_red_line_int = ImageNormalize(red_line_int_masked,interval=ManualInterval(0,350),
                                stretch=SqrtStretch())

        im2 = ax2.pcolormesh(x_2d_grid_red_arcsec_rot_ext + slit_center_x_red,
                        y_2d_grid_red_arcsec_rot_ext + slit_center_y_red,
                        red_line_int_masked,cmap=cmcm.bamako_r,rasterized=True,norm=norm_red_line_int)


        plot_colorbar(im2, ax2,width=colorbar_width)

        # red_cont_masked = np.copy(red_fit_matrix_ext[3,:,:])
        # red_cont_masked[red_where_disk_ext] = np.nan
        red_cont_masked = np.copy(red_fit_matrix_ext[3,:,:]+red_fit_matrix_ext[4,:,:])
        # red_cont_masked[red_mask_bad_fit_matrix_ext] = np.nan
        # red_cont_masked = interpolate_replace_nans(red_cont_masked, kernel)
        red_cont_masked[red_where_disk_ext] = np.nan

        # red_line_cont_ratio_masked = red_fit_matrix_ext[1,:,:]/red_fit_matrix_ext[3,:,:]
        # red_line_cont_ratio_masked[red_mask_bad_fit_matrix_ext] = np.nan
        # red_line_cont_ratio_masked = interpolate_replace_nans(red_line_cont_ratio_masked,kernel)
        # red_line_cont_ratio_masked[red_where_disk_ext] = np.nan
        red_line_cont_ratio_masked = red_line_int_masked/red_cont_masked

        im3 = ax3.pcolormesh(x_2d_grid_red_arcsec_rot_ext + slit_center_x_red,
                        y_2d_grid_red_arcsec_rot_ext + slit_center_y_red,
                        red_cont_masked,cmap=cmcm.lajolla,rasterized=True,vmin=1000,vmax=10000)

        im3_clb, im3_clbax = plot_colorbar(im3, ax3,width=colorbar_width)
        im3_clbax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        im3_clbax.yaxis.get_offset_text().set_fontsize(14)
        im3_clbax.yaxis.get_offset_text().set(va="bottom",ha="left")
        im3_clbax.yaxis.get_offset_text().set_position((0,1.01))

        # red_line_cont_ratio_masked = red_fit_matrix_ext[1,:,:]/red_fit_matrix_ext[3,:,:]
        # red_line_cont_ratio_masked[red_where_disk_ext] = np.nan

        im4 = ax4.pcolormesh(x_2d_grid_red_arcsec_rot_ext + slit_center_x_red,
                        y_2d_grid_red_arcsec_rot_ext + slit_center_y_red,
                        red_line_cont_ratio_masked,cmap=cmcm.bamako_r,rasterized=True,vmin=0,vmax=0.1)

        plot_colorbar(im4, ax4,width=colorbar_width)

        red_where_disk_bin_ext = np.where((x_2d_grid_red_arcsec_bin_rot_ext + slit_center_x_red)**2 + \
                                                (y_2d_grid_red_arcsec_bin_rot_ext + slit_center_y_red)**2 < 940**2)

        red_vlos_masked = -(np.copy(red_fit_matrix_bin_ext[0,:,:]) - 637.451)/637.451*const.c.cgs.value*1e-5
        red_vlos_masked[np.where(red_fit_matrix_bin_ext[1,:,:] < 10)] = np.nan
        red_vlos_masked[red_where_disk_bin_ext] = np.nan
        red_vlos_masked = red_vlos_masked - np.nanmedian(red_vlos_masked)
        red_vlos_masked_err = red_fit_matrix_bin_ext_err[0,:,:]/637.451*const.c.cgs.value*1e-5
        red_vlos_masked_err[np.where(red_fit_matrix_bin_ext[1,:,:] < 10)] = np.nan
        red_vlos_masked_err[red_where_disk_bin_ext] = np.nan

        im5 = ax5.pcolormesh(x_2d_grid_red_arcsec_bin_rot_ext + slit_center_x_red,
                        y_2d_grid_red_arcsec_bin_rot_ext + slit_center_y_red,
                        red_vlos_masked,cmap=cmcm.vik_r,rasterized=True,vmin=-10,vmax=10)

        plot_colorbar(im5, ax5,width=colorbar_width)

        fwhm_masked = np.copy(red_fit_matrix_bin_ext[2,:,:])
        fwhm_masked[np.where(red_fit_matrix_bin_ext[1,:,:] < 10)] = np.nan
        fwhm_masked[red_where_disk_bin_ext] = np.nan
        fwhm_masked_err = np.copy(red_fit_matrix_bin_ext_err[2,:,:])
        fwhm_masked_err[np.where(red_fit_matrix_bin_ext[1,:,:] < 10)] = np.nan
        fwhm_masked_err[red_where_disk_bin_ext] = np.nan
        veff_masked = np.sqrt(fwhm_masked**2 - inst_width_nm_red**2)/637.451*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2))
        veff_masked_err = fwhm_masked/np.sqrt(fwhm_masked**2 - inst_width_nm_red**2)* \
            fwhm_masked_err/637.451*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2))
        im6 = ax6.pcolormesh(x_2d_grid_red_arcsec_bin_rot_ext + slit_center_x_red,
                        y_2d_grid_red_arcsec_bin_rot_ext + slit_center_y_red,
                        veff_masked,cmap=cmcm.batlowK,rasterized=True,vmin=0.06/637.451*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2)),
                        vmax=0.18/637.451*const.c.cgs.value*1e-5/np.sqrt(4*np.log(2)))

        plot_colorbar(im6, ax6,width=colorbar_width)

        ax1.set_ylabel("Solar-Y [arcsec]",fontsize=14)
        ax4.set_xlabel("Solar-X [arcsec]",fontsize=14)
        ax4.set_ylabel("Solar-Y [arcsec]",fontsize=14)
        ax5.set_xlabel("Solar-X [arcsec]",fontsize=14)
        ax6.set_xlabel("Solar-X [arcsec]",fontsize=14)

        ax1.set_title(r"Fe \textsc{xi} 789.2\,nm",fontsize=16)
        ax2.set_title(r"Fe \textsc{x} 637.4\,nm $I_{\rm tot}$ [arb.u.]",fontsize=16)
        ax3.set_title(r"Fe \textsc{x} 637.4\,nm $I_{\rm cont}$ [arb.u.]",fontsize=16)
        ax4.set_title(r"Fe \textsc{x} 637.4\,nm $I_{\rm tot}/I_{\rm cont}$",fontsize=16)
        ax5.set_title(r"Fe \textsc{x} 637.4\,nm $v_{\rm LOS}\ \left[\mathrm{km\,s^{-1}}\right]$",fontsize=16)
        ax6.set_title(r"Fe \textsc{x} 637.4\,nm $v_{\rm eff}\ \left[\mathrm{km\,s^{-1}}\right]$",fontsize=16)

        xlim_zoomin = [-1400,600]
        ylim_zoomin = [-1200,1200]

        for ax_ in (ax2,ax3,ax5,ax6):
            ax_.tick_params(labelleft=False)

        for ax_ in (ax1,ax2,ax3):
            ax_.tick_params(labelbottom=False)

        for ax_ in (ax1,ax2,ax3,ax4,ax5,ax6):
            ax_.contour(img_xarcsec_array,img_yarcsec_array,FeXI_line_cont_image_rot_scipy,levels=[3],alpha=0.6,
                    colors=['#FFC408'])
            ax_.add_patch(create_rec_eqs())
            ax_.add_patch(create_limb_circle(rsun_arcsec))
            ax_.set_aspect(1)
            ax_.tick_params(labelsize=14)
            ax_.set_xlim(xlim_zoomin)
            ax_.set_ylim(ylim_zoomin)

        GetFitProfile(fig, (ax1,ax2,ax3,ax4,ax5,ax6),x_2d_grid_red_arcsec_rot_ext + slit_center_x_red,
                        y_2d_grid_red_arcsec_rot_ext + slit_center_y_red,
                        x_2d_grid_red_arcsec_bin_rot_ext + slit_center_x_red,
                        y_2d_grid_red_arcsec_bin_rot_ext + slit_center_y_red,
                        red_fit_filename_index, red_path,
                        totality_red_df_ext,FeX_xslice,flatfield_1d,order, red_fit_matrix_ext,
                        red_fit_matrix_ext_err, red_fit_matrix_bin_ext, red_fit_matrix_bin_ext_err,
                        red_vlos_masked, red_vlos_masked_err, veff_masked, veff_masked_err)
        
        plt.show()


def give_ax_index(target_ax, axes):
    for ii, ax_ in enumerate(axes):
        if target_ax is ax_:
            return ii

def find_nearest_pixel(x,y,x_grid,y_grid):
    distance = np.sqrt((x - x_grid)**2 + (y - y_grid)**2)
    index = np.unravel_index(np.argmin(distance),distance.shape)
    return index

class GetFitProfile:
    def __init__(self, fig, axes, x_grid, y_grid, x_grid_bin, y_grid_bin,
                file_grid, path, file_df,wvl_slice,flatfield_1d,order,
                fit_matrix, fit_matrix_err, fit_bin_matrix, fit_bin_matrix_err,
                vlos_masked,vlos_masked_err,veff_masked,veff_masked_err,
                cont_slice_1 = slice(0,10), cont_slice_2 = slice(40,50),
                params_prec = {"int":2,"wvl":1,"fwhm":1}):
        self.fig = fig
        self.axes = axes
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.x_grid_bin = x_grid_bin
        self.y_grid_bin = y_grid_bin
        self.file_grid = file_grid
        self.path = path
        self.file_df = file_df
        self.wvl_slice = wvl_slice
        self.flatfield_1d = flatfield_1d
        self.order = order
        self.fit_matrix = fit_matrix
        self.fit_matrix_err = fit_matrix_err
        self.fit_bin_matrix = fit_bin_matrix
        self.fit_bin_matrix_err = fit_bin_matrix_err
        self.vlos_masked = vlos_masked 
        self.vlos_masked_err = vlos_masked_err
        self.veff_masked = veff_masked
        self.veff_masked_err = veff_masked_err
        self.cont_slice_1 = cont_slice_1
        self.cont_slice_2 = cont_slice_2
        self.params_prec = params_prec

        self.cid = fig.canvas.mpl_connect('button_press_event',self)
    
    def __call__(self, event):
        x_select_loc, y_select_loc = event.xdata,event.ydata
        ax_index = give_ax_index(event.inaxes,self.axes)

        print("Axis: {:d} Solar-X: {:.2f} Solar-Y: {:.2f} Heliocentric Distance {:.2f}".format(ax_index,
                         x_select_loc, y_select_loc, np.sqrt(x_select_loc**2 + y_select_loc**2)/950.))

        if ax_index in [0,1,2,3]:
            y_select_pixel,x_select_pixel = find_nearest_pixel(x_select_loc, y_select_loc, 
                        self.x_grid,self.y_grid)
            solarx_select, solary_select = self.x_grid[y_select_pixel,x_select_pixel], self.y_grid[y_select_pixel,x_select_pixel]
            file_index = self.file_grid[x_select_pixel]
            frame = CCDData.read(os.path.join(self.path,self.file_df.iloc[file_index,0]),hdu=0,unit="adu")
            frame_wavelength = CCDData.read(os.path.join(self.path,self.file_df.iloc[file_index,0]),hdu=2,unit="adu").data
            frame_wavelength = frame_wavelength/self.order/10.
            profile_to_fit = frame.data/self.flatfield_1d[:,np.newaxis]/frame.header["exptime"]
            profile_to_fit = np.flip(profile_to_fit,axis=0)
            profile_to_fit = profile_to_fit[y_select_pixel,self.wvl_slice]
            frame_wavelength_sliced = frame_wavelength[self.wvl_slice]

            cont_wvl = frame_wavelength_sliced[np.r_[self.cont_slice_1, self.cont_slice_2]]
            cont_int = profile_to_fit[np.r_[self.cont_slice_1, self.cont_slice_2]]
            cont_fit_param = np.polyfit(cont_wvl, cont_int, deg = 1)
            cont_fit_poly = np.poly1d(cont_fit_param)

            profile_res = profile_to_fit - cont_fit_poly(frame_wavelength_sliced)
            profile_fit = gaussian(frame_wavelength_sliced,*self.fit_matrix[:3,y_select_pixel,x_select_pixel]) + \
                self.fit_matrix[3,y_select_pixel,x_select_pixel]#+self.fit_matrix[4,y_select_pixel,x_select_pixel] - np.mean(cont_fit_poly(frame_wavelength_sliced))
            wvl_to_plot = np.linspace(frame_wavelength_sliced[0],frame_wavelength_sliced[-1],301)
            profile_fit_to_plot = gaussian(wvl_to_plot,*self.fit_matrix[:3,y_select_pixel,x_select_pixel]) + \
                self.fit_matrix[3,y_select_pixel,x_select_pixel]#+self.fit_matrix[4,y_select_pixel,x_select_pixel] - np.mean(cont_fit_poly(frame_wavelength_sliced))
            fit_res = profile_res - profile_fit

            plot_xlim = self.fit_matrix[0,y_select_pixel,x_select_pixel] + \
                    3*np.array([-self.fit_matrix[2,y_select_pixel,x_select_pixel],
                        self.fit_matrix[2,y_select_pixel,x_select_pixel]])

            line_wvl_plot,int_total_plot,fwhm_plot = self.fit_matrix[:3,y_select_pixel,x_select_pixel]
            line_wvl_err_plot,int_total_err_plot,fwhm_err_plot = self.fit_matrix_err[:3,y_select_pixel,x_select_pixel]

            int_total_text_fmt = r'$I_0 = {:#.{int_data_prec}g}\pm{:#.{int_err_prec}g}$'
            line_wvl_text_fmt = r'$\lambda_0 = {:#.{wvl_data_prec}g}\pm{:#.{wvl_err_prec}g}$'
            fwhm_text_fmt = r'$\Delta \lambda = {:#.{fwhm_data_prec}g}\pm{:#.{fwhm_err_prec}g}$'

        elif ax_index in [4,5]:
            y_select_pixel,x_select_pixel = find_nearest_pixel(x_select_loc, y_select_loc, 
                        self.x_grid_bin,self.y_grid_bin)
            solarx_select, solary_select = self.x_grid_bin[y_select_pixel,x_select_pixel], self.y_grid_bin[y_select_pixel,x_select_pixel]
            file_index = self.file_grid[x_select_pixel]
            frame = CCDData.read(os.path.join(self.path,self.file_df.iloc[file_index,0]),hdu=0,unit="adu")
            frame_wavelength = CCDData.read(os.path.join(self.path,self.file_df.iloc[file_index,0]),hdu=2,unit="adu").data
            frame_wavelength = frame_wavelength/self.order/10.
            profile_to_fit = frame.data/self.flatfield_1d[:,np.newaxis]/frame.header["exptime"]
            profile_to_fit = np.flip(profile_to_fit,axis=0)
            profile_to_fit = np.mean(profile_to_fit[y_select_pixel*5:(y_select_pixel+1)*5,self.wvl_slice],axis=0)
            frame_wavelength_sliced = frame_wavelength[self.wvl_slice]

            cont_wvl = frame_wavelength_sliced[np.r_[self.cont_slice_1, self.cont_slice_2]]
            cont_int = profile_to_fit[np.r_[self.cont_slice_1, self.cont_slice_2]]
            cont_fit_param = np.polyfit(cont_wvl, cont_int, deg = 1)
            cont_fit_poly = np.poly1d(cont_fit_param)

            profile_res = profile_to_fit - cont_fit_poly(frame_wavelength_sliced)
            profile_fit = gaussian(frame_wavelength_sliced,*self.fit_bin_matrix[:3,y_select_pixel,x_select_pixel]) + \
                self.fit_bin_matrix[3,y_select_pixel,x_select_pixel]# - np.mean(cont_fit_poly(frame_wavelength_sliced))

            wvl_to_plot = np.linspace(frame_wavelength_sliced[0],frame_wavelength_sliced[-1],301)
            profile_fit_to_plot = gaussian(wvl_to_plot,*self.fit_bin_matrix[:3,y_select_pixel,x_select_pixel]) + \
                self.fit_bin_matrix[3,y_select_pixel,x_select_pixel]# - np.mean(cont_fit_poly(frame_wavelength_sliced))
            fit_res = profile_res - profile_fit

            plot_xlim = self.fit_bin_matrix[0,y_select_pixel,x_select_pixel] + \
                    3*np.array([-self.fit_bin_matrix[2,y_select_pixel,x_select_pixel],
                        self.fit_bin_matrix[2,y_select_pixel,x_select_pixel]])

            line_wvl_plot,int_total_plot,fwhm_plot = self.fit_bin_matrix[:3,y_select_pixel,x_select_pixel]
            line_wvl_err_plot,int_total_err_plot,fwhm_err_plot = self.fit_bin_matrix_err[:3,y_select_pixel,x_select_pixel]
            vlos_plot, vlos_err_plot = (self.vlos_masked[y_select_pixel,x_select_pixel], 
                                        self.vlos_masked_err[y_select_pixel,x_select_pixel])
            veff_plot, veff_err_plot = (self.veff_masked[y_select_pixel,x_select_pixel], 
                                        self.veff_masked_err[y_select_pixel,x_select_pixel])

            int_total_text_fmt = r'$I_0 = {:#.{int_data_prec}g}\pm{:#.{int_err_prec}g}$'
            line_wvl_text_fmt = r'$\lambda_0 = {:#.{wvl_data_prec}g}\pm{:#.{wvl_err_prec}g}$'
            fwhm_text_fmt = r'$\Delta \lambda = {:#.{fwhm_data_prec}g}\pm{:#.{fwhm_err_prec}g}$'
            vlos_text_fmt = r'$v_{{\rm LOS}} = {:#.{vlos_data_prec}g}\pm{:#.{vlos_err_prec}g}$'
            veff_text_fmt = r'$v_{{\rm eff}} = {:#.{veff_data_prec}g}\pm{:#.{veff_err_prec}g}$'
                    

            
        print("X index: {:d} Y index: {:d}".format(x_select_pixel, y_select_pixel))
        print("File index: {:d} Filename: {}".format(file_index,self.file_df.iloc[file_index,0]))
        # print(vlos_plot,vlos_err_plot,veff_plot)


        # fig = plt.figure(figsize=(8,6),constrained_layout=True)
        # gs_fig = fig.add_gridspec(1, 2,figure=fig,width_ratios=[3.,1])
        # gs_plot = gs_fig[0].subgridspec(2, 1,height_ratios=[5,2])
        # ax = fig.add_subplot(gs_plot[0])
        # ax_res = fig.add_subplot(gs_plot[1])

        fig, (ax,ax_res) = plt.subplots(2,1,figsize=(7,7),gridspec_kw={"height_ratios":[5,2]},constrained_layout=True)
        ax.tick_params(labelbottom=False)

        ax.step(frame_wavelength_sliced,profile_res,where="mid",
                    color="#E87A90",label = r"$I_{\rm obs}$",lw=2,zorder=15)
        ax.fill_between(frame_wavelength_sliced,
        np.ones_like(frame_wavelength_sliced)*np.min(profile_res),profile_res,
                    step='mid',color="#FEDFE1",alpha=0.6)

        ax.plot(wvl_to_plot,profile_fit_to_plot,color="black",ls="-",label = r"$I_{\rm fit}$",lw=2,
                            zorder=16,alpha=0.7)
        
        ax_res.scatter(frame_wavelength_sliced,fit_res,marker="o",s=15,color="#E9002D")
        ax_res.axhline(0,ls="--",lw=2,color="#91989F",alpha=0.7) 

        ax.get_shared_x_axes().join(ax, ax_res)
        ax.set_xlim(plot_xlim)

        inflect_eng = inflect.engine()
        if ax_index in [0,1,2,3]:
            title_ext = ""
        elif ax_index in [4,5]:
            title_ext = r"\textbf{ 5-pixel aver.}"
        if "Green" in self.file_df.iloc[file_index,0]:
            title = r"\textbf{{Fe \textsc{{xiv}} 530.3 nm {} order}}".format(inflect_eng.ordinal(self.order)) + title_ext + \
                "\n" + r"\textbf{{Solar-X: {:.2f} Solar-Y: {:.2f} Distance: {:.2f}}} $\boldsymbol{{R_\odot}}$".format(
                         solarx_select, solary_select, np.sqrt(solarx_select**2 + solary_select**2)/950.)
        elif "Red" in self.file_df.iloc[file_index,0]:
            title = r"\textbf{{Fe \textsc{{x}} 637.4 nm {} order}}".format(inflect_eng.ordinal(self.order)) + title_ext +\
                "\n" + r"\textbf{{Solar-X: {:.2f} Solar-Y: {:.2f} Distance: {:.2f}}} $\boldsymbol{{R_\odot}}$".format(
                         solarx_select, solary_select, np.sqrt(solarx_select**2 + solary_select**2)/950.)
        ax.set_title(title,fontsize=16)

        ax.set_ylabel("Intensity [arb.u]",fontsize=16)
        ax_res.set_ylabel("Res. [arb.u]",fontsize=16)
        ax_res.set_xlabel(r"$\textrm{Wavelength}\ \lambda\ [\mathrm{nm}]$",fontsize=16)
        for ax_ in (ax, ax_res):
            ax_.tick_params(labelsize=16,direction="in",right=True,top=True,which="both")

        # gs_text = gs_fig[1].subgridspec(1,1)
        # ax_text = fig.add_subplot(gs_text[0])
        # ax_text.axis("off")
        
        
        wvl_data_prec = np.ceil(np.log10(np.abs(line_wvl_plot))).astype("int") - \
            np.ceil(np.log10(line_wvl_err_plot)).astype("int") + self.params_prec["wvl"]

        ax.text(0.03,0.92,line_wvl_text_fmt.format(num2tex(line_wvl_plot),
        num2tex(line_wvl_err_plot),wvl_data_prec = wvl_data_prec,wvl_err_prec = self.params_prec["wvl"]),ha = 'left',va = 'center', 
        color = 'black',fontsize = 16,linespacing=1.5,transform=ax.transAxes)

        int_data_prec = np.ceil(np.log10(np.abs(int_total_plot))).astype("int") - \
        np.ceil(np.log10(int_total_err_plot)).astype("int") + self.params_prec["int"]

        ax.text(0.03,0.82,int_total_text_fmt.format(num2tex(int_total_plot),
        num2tex(int_total_err_plot),int_data_prec = int_data_prec,int_err_prec = self.params_prec["int"]),ha = 'left',va = 'center', 
        color = 'black',fontsize = 16,linespacing=1.5,transform=ax.transAxes)

        fwhm_data_prec = np.ceil(np.log10(np.abs(fwhm_plot))).astype("int") - \
            np.ceil(np.log10(fwhm_err_plot)).astype("int") + self.params_prec["fwhm"]

        ax.text(0.03,0.72,fwhm_text_fmt.format(num2tex(fwhm_plot),
        num2tex(fwhm_err_plot),fwhm_data_prec = fwhm_data_prec,fwhm_err_prec = self.params_prec["fwhm"]),ha = 'left',va = 'center', 
        color = 'black',fontsize = 16,linespacing=1.5,transform=ax.transAxes)

        if ax_index in [4,5]:
            vlos_data_prec = np.ceil(np.log10(np.abs(vlos_plot))).astype("int") - \
                np.ceil(np.log10(vlos_err_plot)).astype("int") + self.params_prec["wvl"]

            ax.text(0.03,0.62,vlos_text_fmt.format(num2tex(vlos_plot),
            num2tex(vlos_err_plot),vlos_data_prec = vlos_data_prec,vlos_err_prec = self.params_prec["wvl"]),ha = 'left',va = 'center', 
            color = 'black',fontsize = 16,linespacing=1.5,transform=ax.transAxes)

            veff_data_prec = np.ceil(np.log10(np.abs(veff_plot))).astype("int") - \
                np.ceil(np.log10(veff_err_plot)).astype("int") + self.params_prec["fwhm"]

            ax.text(0.03,0.52,veff_text_fmt.format(num2tex(veff_plot),
            num2tex(veff_err_plot),veff_data_prec = veff_data_prec,veff_err_prec = self.params_prec["fwhm"]),ha = 'left',va = 'center', 
            color = 'black',fontsize = 16,linespacing=1.5,transform=ax.transAxes)

        # fit_model = SpectrumFitSingle(data=profile_res,wvl=green_frame_wavelength_sliced,line_number=1,
        #             line_wvl_init=green_frame_wavelength_sliced[np.argmax(profile_res)],int_max_init=profile_res.max(),fwhm_init=0.1)
        # fit_model.plot(plot_fit=False)
        plt.show()





def fit_spectra(image, wvl, wavelength_slice, ypix_slice, cont_slice_1, cont_slice_2,nbin=5,plot_fit=False):
    image_sliced = image[ypix_slice, wavelength_slice]
    if (nbin == 1) or (nbin is None):
        pass
    else:
        image_sliced = np.average(image_sliced.reshape(-1,nbin,image_sliced.shape[1]),axis=1)
    wvl_sliced = wvl[wavelength_slice]
    fit_params = np.zeros((4,image_sliced.shape[0]))
    fit_errs = np.zeros((4,image_sliced.shape[0]))

    for ii in range(image_sliced.shape[0]):
        fit_params[:,ii], fit_errs[:,ii] = fit_spectra_single(wvl_sliced[np.r_[cont_slice_1, cont_slice_2]], 
                                image_sliced[ii, np.r_[cont_slice_1, cont_slice_2]],wvl_sliced, image_sliced[ii,:],
                                plot_fit=plot_fit)

    return fit_params, fit_errs

def fit_spectra_single(cont_wvl, cont_int, wvl, int,plot_fit=False):
    cont_fit_param = np.polyfit(cont_wvl, cont_int, deg = 1)
    cont_fit_poly = np.poly1d(cont_fit_param)

    int_res = int - cont_fit_poly(wvl)

    fit_model = SpectrumFitSingle(data=int_res,wvl=wvl,line_number=1,
                        line_wvl_init=wvl[np.argmax(int_res)],int_max_init=int_res.max(),fwhm_init=0.1)

    try:
        fit_model.run_lse()
    except RuntimeError:
        pass
    if plot_fit:
        fit_model.plot(plot_params=False)
        print(fit_model.fwhm_fit)

    return np.array([fit_model.line_wvl_fit[0], fit_model.int_total_fit[0], fit_model.fwhm_fit[0],
                     cont_fit_poly(fit_model.line_wvl_fit[0]) + fit_model.int_cont_fit]), \
            np.array([fit_model.line_wvl_err[0], fit_model.int_total_err[0], fit_model.fwhm_err[0],
                    fit_model.int_cont_err])



                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--line", help="the spectral line to plot, FeXIV or FeX")
    parser.add_argument("-o","--order", help="order of the spectral line to plot. FeXIV: 61, 62, 63 (default); FeX: 51, 52 (default), 53",
                        type=int)
    args = parser.parse_args()
    if args.line == "FeXIV":
        if args.order not in [61,62,63]:
            args.order = 63
            print("No valid order for Fe XIV 530.3, use default order = 63.")
        fit_and_plot("FeXIV",args.order)
            
    elif args.line == "FeX":
        if args.order not in [51,52,53]:
            args.order = 52
            print("No valid order for Fe X 637.4, use default order = 52.")
        fit_and_plot("FeX",args.order)
    else:
        sys.exit("Please give the spectral line to plot with -l or --line, FeXIV or FeX")




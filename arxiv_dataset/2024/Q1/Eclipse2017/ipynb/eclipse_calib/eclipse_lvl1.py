import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from glob import glob
import os
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, AsymmetricPercentileInterval
import h5py 
from astropy.nddata import CCDData
from PIL import Image
from datetime import datetime, timedelta
from ccdproc import ImageFileCollection
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FixedLocator, FixedFormatter, LogLocator
import cmcrameri.cm as cmcm
import cmasher as cmr
from scipy import ndimage

#Eclipse Images
green_path = "../../src/EclipseSpectra2017/MikesData/VaderEclipseDayGreen2017aug21/"
green_path_save = "../../src/EclipseSpectra2017/MikesData_l1/Green/"
red_path = "../../src/EclipseSpectra2017/MikesData/VaderEclipseDayRed2017aug21/"
red_path_save = "../../src/EclipseSpectra2017/MikesData_l1/Red/"

totality_green_im_collection = ImageFileCollection(green_path,
                        glob_include="TotalitySequence*.fit")
totality_green_df = totality_green_im_collection.summary.to_pandas()
totality_green_df["date-obs"] = pd.to_datetime(totality_green_df["date-obs"])

totality_red_im_collection = ImageFileCollection(red_path,
                        glob_include="TotalitySequence*.fit")
totality_red_df = totality_red_im_collection.summary.to_pandas()
totality_red_df["date-obs"] = pd.to_datetime(totality_red_df["date-obs"])

totality_green_df_cut = totality_green_df.loc[(totality_green_df['date-obs'] >= datetime(2017,8,21,17,45,21)) & 
                                        (totality_green_df['date-obs'] <= datetime(2017,8,21,17,48,8))]
totality_green_df_cut = totality_green_df_cut.reset_index(drop=True)


totality_red_df_cut = totality_red_df.loc[(totality_red_df['date-obs'] >= datetime(2017,8,21,17,45,21)) & 
                                        (totality_red_df['date-obs'] <= datetime(2017,8,21,17,48,8))]
totality_red_df_cut = totality_red_df_cut.reset_index(drop=True)


#Read Bias Images
with h5py.File("../../sav/Eclipse/Bias/master_bias_dc_red_1s_proto.h5", 'r') as hf:
    bias_dc_red_1s = hf['image'][:]

with h5py.File("../../sav/Eclipse/Bias/master_bias_dc_red_3s_proto.h5", 'r') as hf:
    bias_dc_red_3s = hf['image'][:]

with h5py.File("../../sav/Eclipse/Bias/master_bias_dc_green_1s_proto.h5", 'r') as hf:
    bias_dc_green_1s = hf['image'][:]

with h5py.File("../../sav/Eclipse/Bias/master_bias_dc_green_3s_proto.h5", 'r') as hf:
    bias_dc_green_3s = hf['image'][:]

#Read curvature corrections
with h5py.File("../../sav/Eclipse/Curvature/master_curvature_green.h5", 'r') as hf:
    xpos_map_coordinate_green = hf['xpos_map_coordinate'][:]
    xstart_pixel_green = hf['xpos_map_coordinate'].attrs['xstart_pixel']
    xend_pixel_green = hf['xpos_map_coordinate'].attrs['xend_pixel']

    ypos_map_coordinate_green = hf['ypos_map_coordinate'][:]
    ystart_pixel_green = hf['ypos_map_coordinate'].attrs['ystart_pixel']
    yend_pixel_green = hf['ypos_map_coordinate'].attrs['yend_pixel']

testx_slice_mapcoor_green = slice(xstart_pixel_green,xend_pixel_green)
testy_slice_mapcoor_green = slice(ystart_pixel_green,yend_pixel_green)


with h5py.File("../../sav/Eclipse/Curvature/master_curvature_red.h5", 'r') as hf:
    xpos_map_coordinate_red = hf['xpos_map_coordinate'][:]
    xstart_pixel_red = hf['xpos_map_coordinate'].attrs['xstart_pixel']
    xend_pixel_red = hf['xpos_map_coordinate'].attrs['xend_pixel']

    ypos_map_coordinate_red = hf['ypos_map_coordinate'][:]
    ystart_pixel_red = hf['ypos_map_coordinate'].attrs['ystart_pixel']
    yend_pixel_red = hf['ypos_map_coordinate'].attrs['yend_pixel']

testx_slice_mapcoor_red = slice(xstart_pixel_red,xend_pixel_red)
testy_slice_mapcoor_red = slice(ystart_pixel_red,yend_pixel_red)

#Read Wavelength files 
with h5py.File("../../sav/Eclipse/Wavelength/master_wavelength_curvature_green.h5", 'r') as hf:
    wavelength_times_order_green = hf['wavelength_times_order'][:]
    wavelength_times_order_shift_green = hf['wavelength_times_order_shift'][:]

with h5py.File("../../sav/Eclipse/Wavelength/master_wavelength_curvature_red.h5", 'r') as hf:
    wavelength_times_order_red = hf['wavelength_times_order'][:]
    wavelength_times_order_shift_red = hf['wavelength_times_order_shift'][:]

for ii, row_ in totality_green_df_cut.iterrows():
    green_filename = row_["file"]
    green_img, green_hdr = fits.getdata(os.path.join(green_path,green_filename),header=True)
    exptime = np.float64(green_hdr["EXPTIME"])

    if exptime <= 1.5:
        green_img_unbias = green_img - bias_dc_green_1s
        green_readout_noise = 12.2
    elif 2.5 < exptime < 3.5:
        green_img_unbias = green_img - bias_dc_green_3s
        green_readout_noise = 13.1
    
    green_img_curv_corr = ndimage.map_coordinates(green_img_unbias[testy_slice_mapcoor_green, testx_slice_mapcoor_green],
                                                    (ypos_map_coordinate_green, xpos_map_coordinate_green),order=1)

    green_hdr["comments"] = "Green detector. Bias removed, curvature corrected."
    green_hdr["XWS"] = (xstart_pixel_green, "X window start after the curvature correction")
    green_hdr["YWS"] = (ystart_pixel_green, "X window start after the curvature correction")
    green_hdr["RONOISE"] = (green_readout_noise, "Readout noise")
    wavelength_green_hdr = fits.Header()
    wavelength_green_hdr["comments"] = "Wavelength calibrated from laboratory hydrogen and helium frames."
    wavelength_shift_green_hdr = fits.Header()
    wavelength_shift_green_hdr["comments"] = "Wavelength absolutedly calibrated from chromospheric lines."
    
    primary_hdu = fits.PrimaryHDU(data=green_img_curv_corr,header=green_hdr)
    wavelength_hdu = fits.ImageHDU(data=wavelength_times_order_green,header= wavelength_green_hdr)
    wavelength_shift_hdu = fits.ImageHDU(data=wavelength_times_order_shift_green,header=wavelength_shift_green_hdr)
    hdul_green = fits.HDUList([primary_hdu, wavelength_hdu, wavelength_shift_hdu])

    green_filename_save = green_filename[:-4] + "_l1" + green_filename[-4:]
    hdul_green.writeto(os.path.join(green_path_save,green_filename_save),overwrite=True)

for ii, row_ in totality_red_df_cut.iterrows():
    red_filename = row_["file"]
    red_img, red_hdr = fits.getdata(os.path.join(red_path,red_filename),header=True)
    exptime = np.float64(red_hdr["EXPTIME"])

    if exptime <= 1.5:
        red_img_unbias = red_img - bias_dc_red_1s
        red_readout_noise = 9.92
    elif 2.5 < exptime < 3.5:
        red_img_unbias = red_img - bias_dc_red_3s
        red_readout_noise = 10.2
    
    red_img_curv_corr = ndimage.map_coordinates(red_img_unbias[testy_slice_mapcoor_red, testx_slice_mapcoor_red],
                                                    (ypos_map_coordinate_red, xpos_map_coordinate_red),order=1)

    red_hdr["comments"] = "Red detector. Bias removed, curvature corrected."
    red_hdr["XWS"] = (xstart_pixel_red, "X window start after the curvature correction")
    red_hdr["YWS"] = (ystart_pixel_red, "X window start after the curvature correction")
    red_hdr["RONOISE"] = (red_readout_noise, "Readout noise")
    wavelength_red_hdr = fits.Header()
    wavelength_red_hdr["comments"] = "Wavelength calibrated from laboratory hydrogen and helium frames."
    wavelength_shift_red_hdr = fits.Header()
    wavelength_shift_red_hdr["comments"] = "Wavelength absolutedly calibrated from chromospheric lines."
    
    primary_hdu = fits.PrimaryHDU(data=red_img_curv_corr,header=red_hdr)
    wavelength_hdu = fits.ImageHDU(data=wavelength_times_order_red,header= wavelength_red_hdr)
    wavelength_shift_hdu = fits.ImageHDU(data=wavelength_times_order_shift_red,header=wavelength_shift_red_hdr)
    hdul_red = fits.HDUList([primary_hdu, wavelength_hdu, wavelength_shift_hdu])

    red_filename_save = red_filename[:-4] + "_l1" + red_filename[-4:]
    hdul_red.writeto(os.path.join(red_path_save,red_filename_save),overwrite=True)
    




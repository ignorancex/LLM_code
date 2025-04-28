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

video_frame_fnames = sorted(glob("../../sav/Eclipse/Video/frame*.jpg"))
video_frame_fnames = video_frame_fnames[34:200]

video_frame_cube = np.zeros((1080,1920,3,200-34),dtype="uint8")
for ii, video_frame_fname in enumerate(video_frame_fnames):
    with Image.open(video_frame_fname) as im:
        video_frame_cube[:,:,:,ii] = np.asarray(im)

video_vertical_slice =  slice(390,710)
video_horizontal_slice = slice(746,1160)
video_frame_cube = video_frame_cube[video_vertical_slice,video_horizontal_slice,:,:]

video_time_array = np.arange(datetime(2017,8,21,17,45,23),datetime(2017,8,21,17,48,9),timedelta(seconds=1)).astype(datetime)
video_time_array = video_time_array[13:97]

slit_pos = 209.4
rotate_angle_context = -30
sun_center_pixel_plot = np.array([300,220])

with h5py.File("../../sav/Eclipse/LimbTrack/sun_pos_linear_fit.h5", 'r') as hf:
    sun_x_fitparam = hf['sun_x_fitparam'][:]
    sun_y_fitparam = hf['sun_y_fitparam'][:]

sun_x_fitpoly = np.poly1d(sun_x_fitparam)
sun_y_fitpoly = np.poly1d(sun_y_fitparam)

green_path = "../../src/EclipseSpectra2017/MikesData/VaderEclipseDayGreen2017aug21/"
red_path = "../../src/EclipseSpectra2017/MikesData/VaderEclipseDayRed2017aug21/"

totality_green_im_collection = ImageFileCollection(green_path,
                        glob_include="TotalitySequence*.fit")
totality_green_df = totality_green_im_collection.summary.to_pandas()
totality_green_df["date-obs"] = pd.to_datetime(totality_green_df["date-obs"])

totality_red_im_collection = ImageFileCollection(red_path,
                        glob_include="TotalitySequence*.fit")
totality_red_df = totality_red_im_collection.summary.to_pandas()
totality_red_df["date-obs"] = pd.to_datetime(totality_red_df["date-obs"])

FeXI_line_cont_frame = CCDData.read("../../src/EclipseSpectra2017/MitchellData/MitchellFeXILine_ContRatio.fits",unit="adu")
FeXI_line_cont_image = FeXI_line_cont_frame.data
sun_center_FeXI = (np.float64(FeXI_line_cont_frame.header["SUNX"]),np.float64(FeXI_line_cont_frame.header["SUNY"]))
FeXI_line_cont_xslice = slice(372-300,372+301)
FeXI_line_cont_yslice = slice(383-220,383+221)
FeXI_line_cont_cutout = FeXI_line_cont_image[FeXI_line_cont_yslice, FeXI_line_cont_xslice]
FeXI_rotate_center = (sun_center_FeXI[0] - FeXI_line_cont_xslice.start, sun_center_FeXI[1] - FeXI_line_cont_yslice.start)
FeXI_line_cont_image_rot_scipy = ndimage.rotate(FeXI_line_cont_cutout, angle=360 - np.float64(FeXI_line_cont_frame.header["SUNROT"]),reshape=False,order=1)


FeXIV_line_cont_frame = CCDData.read("../../src/EclipseSpectra2017/MitchellData/MitchellFeXIVLine_ContRatio.fits",unit="adu")
FeXIV_line_cont_image = FeXIV_line_cont_frame.data
sun_center_FeXIV = (np.float64(FeXIV_line_cont_frame.header["SUNX"]),np.float64(FeXIV_line_cont_frame.header["SUNY"]))
FeXIV_line_cont_xslice = slice(372-300,372+301)
FeXIV_line_cont_yslice = slice(383-220,383+221)
FeXIV_line_cont_cutout = FeXIV_line_cont_image[FeXIV_line_cont_yslice, FeXIV_line_cont_xslice]
FeXIV_rotate_center = (sun_center_FeXIV[0] - FeXIV_line_cont_xslice.start, sun_center_FeXIV[1] - FeXIV_line_cont_yslice.start)
FeXIV_line_cont_image_rot_scipy = ndimage.rotate(FeXIV_line_cont_cutout, angle=360 - np.float64(FeXIV_line_cont_frame.header["SUNROT"]),reshape=False,order=1)

pixel_ratio = 71.4/np.float64(FeXI_line_cont_frame.header["MOONR"])
gs_kw = dict(width_ratios=[1,2.3])


for ii, ii_time in enumerate(video_time_array[:]):
    fig, axes = plt.subplots(2,2,figsize=(16,8.5),gridspec_kw=gs_kw,constrained_layout=True)
    ((ax_img, ax_specg),(ax_imr, ax_specr)) = axes

    ax_img.imshow(FeXIV_line_cont_image_rot_scipy,vmin=0.2,vmax=1.2,cmap=cmr.jungle_r,origin="lower")
    ax_imr.imshow(FeXI_line_cont_image_rot_scipy,vmin=1,vmax=4,cmap=cmcm.lajolla,origin="lower")


    ax_img.set_title("Fe XIV 530.3 nm {}".format(ii_time.strftime("%H:%M:%S")),fontsize=14)
    ax_imr.set_title("Fe XI 789.2 nm {}".format(ii_time.strftime("%H:%M:%S")),fontsize=14)

    ax_img.set_ylabel("SOLAR-Y [Pixel]",fontsize=14)
    ax_imr.set_xlabel("SOLAR-X [Pixel]",fontsize=14)
    ax_imr.set_ylabel("SOLAR-Y [Pixel]",fontsize=14)

    if ii < 5:
        slit_xshift = sun_x_fitpoly(5) - slit_pos
    else:
        slit_xshift = sun_x_fitpoly(ii) - slit_pos

    slit_start_x = sun_center_pixel_plot[0] - slit_xshift/pixel_ratio*np.cos(np.deg2rad(np.abs(rotate_angle_context)))
    slit_start_y = sun_center_pixel_plot[1] + slit_xshift/pixel_ratio*np.sin(np.deg2rad(np.abs(rotate_angle_context)))

    ax_img.axline((slit_start_x,slit_start_y),slope=1/np.tan(np.deg2rad(np.abs(rotate_angle_context))),color="red",lw=2)
    ax_imr.axline((slit_start_x,slit_start_y),slope=1/np.tan(np.deg2rad(np.abs(rotate_angle_context))),color="red",lw=2)


    green_nearest_fname = totality_green_df.loc[(totality_green_df['date-obs'] 
                - video_time_array[ii]).abs().idxmin(),"file"]
    red_nearest_fname = totality_red_df.loc[(totality_red_df['date-obs'] 
                - video_time_array[ii]).abs().idxmin(),"file"]

    green_frame = CCDData.read(os.path.join(green_path,green_nearest_fname),unit="adu")
    red_frame = CCDData.read(os.path.join(red_path,red_nearest_fname),unit="adu")

    green_image = green_frame.data[300:750,:]/green_frame.header["EXPTIME"]
    red_image = red_frame.data[300:750,:]/red_frame.header["EXPTIME"]

    norm_green = ImageNormalize(green_image,stretch=LogStretch())
    norm_red = ImageNormalize(red_image,stretch=LogStretch())

    im_green = ax_specg.pcolormesh(np.arange(green_frame.header["NAXIS1"]),np.arange(450)+300,
                        green_image,cmap=cmcm.lajolla,norm=norm_green,shading='auto',rasterized=True)

    im_red = ax_specr.pcolormesh(np.arange(red_frame.header["NAXIS1"]),np.arange(450)+300,
                        red_image,cmap=cmcm.lajolla,norm=norm_red,shading='auto',rasterized=True)

    ax_specg.set_title("Green Detector {} {}".format(green_frame.header["DATE-OBS"][-8:],green_nearest_fname),fontsize=14)

    ax_specr.set_title("Red Detector {} {}".format(red_frame.header["DATE-OBS"][-8:],red_nearest_fname),fontsize=14)
    
    ax_specg.invert_yaxis()
    ax_specr.invert_yaxis()

    ax_specg.set_ylabel("CCD-Y [Pixel]",fontsize=14)
    ax_specr.set_xlabel("CCD-X [Pixel]",fontsize=14)
    ax_specr.set_ylabel("CCD-Y [Pixel]",fontsize=14)
    

        
    for ax_ in axes.flatten():
        ax_.set_aspect(1)
        ax_.tick_params(labelsize=13,right=True,top=True)
    
    plt.savefig(fname=os.path.join("../../sav/Eclipse/Video_RotSlit/","Video_RotSlit_{:03d}.png".format(ii)),format="png",
                dpi=144,bbox_inches="tight")
    fig.clf()
    for ax_ in axes.flatten():
        ax_.cla()
    plt.close(fig)

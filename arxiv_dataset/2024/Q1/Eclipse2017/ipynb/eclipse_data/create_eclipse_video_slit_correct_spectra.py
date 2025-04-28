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
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FixedLocator, FixedFormatter, LogLocator, StrMethodFormatter
from matplotlib import patches
from matplotlib.markers import MarkerStyle
import cmcrameri.cm as cmcm
import cmasher as cmr
from scipy import ndimage
import copy

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
rotate_angle_context = -28.
sun_center_pixel_plot = np.array([300,220])

with h5py.File("../../sav/Eclipse/LimbTrack/sun_pos_linear_fit.h5", 'r') as hf:
    sun_x_fitparam = hf['sun_x_fitparam'][:]
    sun_y_fitparam = hf['sun_y_fitparam'][:]

sun_x_fitpoly = np.poly1d(sun_x_fitparam)
sun_y_fitpoly = np.poly1d(sun_y_fitparam)

green_path = "../../src/EclipseSpectra2017/MikesData_l1/Green/"
red_path = "../../src/EclipseSpectra2017/MikesData_l1/Red/"

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

rsun_arcsec = 950.0
rsun_context_pixel = 71.4
pixel_ratio = rsun_context_pixel/np.float64(FeXI_line_cont_frame.header["MOONR"])
img_pixel_to_arcsec = np.float64(FeXI_line_cont_frame.header["SUNR"])/rsun_arcsec
pixel_ratio_to_arcsec = rsun_context_pixel/np.float64(FeXI_line_cont_frame.header["MOONR"])*img_pixel_to_arcsec

gs_kw = dict(width_ratios=[1,2.2],hspace=0.05)

eis_eqs_xcen, eis_eqs_ycen = np.array((-895.061, 390.811))
eis_eqs_fovx, eis_eqs_fovy = np.array((119.808, 160.0))
eis_eqs_xstart = eis_eqs_xcen - eis_eqs_fovx/2.
eis_eqs_xend = eis_eqs_xcen + eis_eqs_fovx/2.
eis_eqs_ystart = eis_eqs_ycen - eis_eqs_fovy/2.
eis_eqs_yend = eis_eqs_ycen + eis_eqs_fovy/2.

eis_spch_fovx, eis_spch_fovy = np.array((60.,512.))
eis_spch_xstart = -15.
eis_spch_xend = 45.
eis_spch_ystart = -1358.5
eis_spch_yend = -855.

def create_rec_eqs():
    return patches.Rectangle((eis_eqs_xstart, eis_eqs_ystart),
                            eis_eqs_fovx, eis_eqs_fovy,linewidth=0,edgecolor="grey",
                            facecolor="grey",alpha=0.6)
def create_rec_spch():
    return patches.Rectangle((eis_spch_xstart, eis_spch_ystart),
                            eis_spch_fovx, eis_spch_fovy,linewidth=0,edgecolor="grey",
                            facecolor="grey",alpha=0.6)


img_center = np.array([300,220])
img_xpixel_array = np.arange(FeXIV_line_cont_image_rot_scipy.shape[1])
img_ypixel_array = np.arange(FeXIV_line_cont_image_rot_scipy.shape[0])

def func_img_xpixel_to_xarcsec(x):
    return (x - img_center[0])/img_pixel_to_arcsec

def func_img_xarcsec_to_xpixel(x):
    return x*img_pixel_to_arcsec + img_center[0]

def func_img_ypixel_to_yarcsec(x):
    return (x - img_center[1])/img_pixel_to_arcsec

def func_img_yarcsec_to_ypixel(x):
    return x*img_pixel_to_arcsec + img_center[1]
def func_one_to_one(x):
    return x

img_xarcsec_array = func_img_xpixel_to_xarcsec(img_xpixel_array)
img_yarcsec_array = func_img_ypixel_to_yarcsec(img_ypixel_array)

for ii, ii_time in enumerate(video_time_array[:]):
    fig, axes = plt.subplots(2,2,figsize=(16,9),gridspec_kw=gs_kw,constrained_layout=True)
    ((ax_img, ax_specg),(ax_imr, ax_specr)) = axes

    ax_img.pcolormesh(img_xarcsec_array,img_yarcsec_array,FeXIV_line_cont_image_rot_scipy,vmin=0.2,vmax=1.2,
                    cmap=cmr.jungle_r,shading="auto",rasterized=True)
    ax_imr.pcolormesh(img_xarcsec_array,img_yarcsec_array,FeXI_line_cont_image_rot_scipy,vmin=1,vmax=4,
                    cmap=cmcm.lajolla,shading="auto",rasterized=True)

    
    ax_img.add_patch(create_rec_eqs())
    ax_imr.add_patch(create_rec_eqs())
    ax_img.add_patch(create_rec_spch())
    ax_imr.add_patch(create_rec_spch())

    ax_img.set_title("Fe XIV 530.3 nm {}".format(ii_time.strftime("%H:%M:%S")),fontsize=14)
    ax_imr.set_title("Fe XI 789.2 nm {}".format(ii_time.strftime("%H:%M:%S")),fontsize=14)
    
    ax_img_xpixel = ax_img.secondary_xaxis("top",functions=(func_img_xarcsec_to_xpixel,func_img_xpixel_to_xarcsec))
    ax_img_ypixel = ax_img.secondary_yaxis("right",functions=(func_img_yarcsec_to_ypixel,func_img_ypixel_to_yarcsec))

    ax_imr_xpixel = ax_imr.secondary_xaxis("top",functions=(func_img_xarcsec_to_xpixel,func_img_xpixel_to_xarcsec))
    ax_imr_ypixel = ax_imr.secondary_yaxis("right",functions=(func_img_yarcsec_to_ypixel,func_img_ypixel_to_yarcsec))

    for ax_ in (ax_img_xpixel,ax_img_ypixel,ax_imr_xpixel,ax_imr_ypixel):
        ax_.tick_params(labelsize=12)

    ax_img.set_ylabel("Solar-Y [arcsec]",fontsize=14)
    ax_imr.set_xlabel("Solar-X [arcsec]",fontsize=14)
    ax_imr.set_ylabel("Solar-Y [arcsec]",fontsize=14)



    if ii < 5:
        slit_xshift = sun_x_fitpoly(5) - slit_pos
    else:
        slit_xshift = sun_x_fitpoly(ii) - slit_pos

    slit_center_x =  - slit_xshift/pixel_ratio_to_arcsec*np.cos(np.deg2rad(np.abs(rotate_angle_context)))
    slit_center_y =  slit_xshift/pixel_ratio_to_arcsec*np.sin(np.deg2rad(np.abs(rotate_angle_context)))

    slit_top_x = slit_center_x + rsun_arcsec*np.sin(np.deg2rad(np.abs(rotate_angle_context)))
    slit_top_y = slit_center_y + rsun_arcsec*np.cos(np.deg2rad(np.abs(rotate_angle_context)))

    slit_bottom_x = 2*slit_center_x - slit_top_x
    slit_bottom_y = 2*slit_center_y - slit_top_y

    slit_limb_marker = MarkerStyle("_")
    slit_limb_marker._transform.rotate_deg(-30)

    ax_img.scatter([slit_bottom_x, slit_top_x],[slit_bottom_y, slit_top_y],s=20,marker=slit_limb_marker,color="red",alpha=0.8)
    ax_imr.scatter([slit_bottom_x, slit_top_x],[slit_bottom_y, slit_top_y],s=20,marker=slit_limb_marker,color="red",alpha=0.8)


    ax_img.axline((slit_center_x,slit_center_y),slope=1/np.tan(np.deg2rad(np.abs(rotate_angle_context))),color="red",lw=2,alpha=0.8)
    ax_imr.axline((slit_center_x,slit_center_y),slope=1/np.tan(np.deg2rad(np.abs(rotate_angle_context))),color="red",lw=2,alpha=0.8)


    green_nearest_fname = totality_green_df.loc[(totality_green_df['date-obs'] 
                - video_time_array[ii]).abs().idxmin(),"file"]
    red_nearest_fname = totality_red_df.loc[(totality_red_df['date-obs'] 
                - video_time_array[ii]).abs().idxmin(),"file"]

    green_frame = CCDData.read(os.path.join(green_path,green_nearest_fname),hdu=0,unit="adu")
    green_wavelength = CCDData.read(os.path.join(green_path,green_nearest_fname),hdu=1,unit="angstrom").data
    red_frame = CCDData.read(os.path.join(red_path,red_nearest_fname),hdu=0,unit="adu")
    red_wavelength = CCDData.read(os.path.join(red_path,red_nearest_fname),hdu=1,unit="angstrom").data

    green_image = green_frame.data/green_frame.header["EXPTIME"]
    red_image = red_frame.data/red_frame.header["EXPTIME"]

    norm_green = ImageNormalize(green_image,stretch=LogStretch())
    norm_red = ImageNormalize(red_image,stretch=LogStretch())

    # im_green = ax_specg.pcolormesh(np.arange(green_frame.header["NAXIS1"]),np.arange(green_frame.header["NAXIS2"]),
    #                     green_image,cmap=cmcm.lajolla,norm=norm_green,shading='auto',rasterized=True)

    im_green = ax_specg.pcolormesh(green_wavelength/62./10.,np.arange(green_frame.header["NAXIS2"]) + green_frame.header["YWS"],
                        green_image,cmap=cmcm.lajolla,norm=norm_green,shading='auto',rasterized=True)
    

    # im_red = ax_specr.pcolormesh(np.arange(red_frame.header["NAXIS1"]),np.arange(green_frame.header["NAXIS2"]),
    #                     red_image,cmap=cmcm.lajolla,norm=norm_red,shading='auto',rasterized=True)

    im_red = ax_specr.pcolormesh(red_wavelength/52./10.,np.arange(red_frame.header["NAXIS2"]) + red_frame.header["YWS"],
                        red_image,cmap=cmcm.lajolla,norm=norm_red,shading='auto',rasterized=True)

    ax_specg.set_title("Green Detector {} {}".format(green_frame.header["DATE-OBS"][-8:],green_nearest_fname),fontsize=14)

    ax_specr.set_title("Red Detector {} {}".format(red_frame.header["DATE-OBS"][-8:],red_nearest_fname),fontsize=14)
    
    ax_specg.invert_yaxis()
    ax_specr.invert_yaxis()

    ax_specg.set_xlabel("62nd order Wavelength [nm]",fontsize=14)
    ax_specg.set_ylabel("CCD-Y [Pixel]",fontsize=14)
    
    ax_specr.set_xlabel("52nd order Wavelength [nm]",fontsize=14)
    ax_specr.set_ylabel("CCD-Y [Pixel]",fontsize=14)

    for ax_ in (axes.flatten()):
        ax_.tick_params(labelsize=12)
    
    ax_specg_2 = ax_specg.twiny()
    ax_specg_2.set_xlim((green_frame.header["NAXIS1"]-0.5,green_frame.header["XWS"]-0.5))
    ax_specr_2 = ax_specr.twiny()
    ax_specr_2.set_xlim((red_frame.header["XWS"]-0.5,red_frame.header["NAXIS1"]-0.5))

    for ax_ in (ax_specg_2,ax_specr_2):
        ax_.tick_params(labelsize=12)

    FeXIV_tick_locs = 530.286*np.array([63.,62.,61.])/62.
    FeX_tick_locs = 637.451*np.array([51.,52.,53.])/52.

    green_limb_locs = np.array([396,625])
    red_limb_locs = np.array([366,592])

    # green_limb_locs = np.array([396,625]) + (sun_y_fitpoly(ii) - sun_y_fitpoly(70))/rsun_context_pixel*np.diff(green_limb_locs)
    # red_limb_locs = np.array([366,592]) + (sun_y_fitpoly(ii) - sun_y_fitpoly(70))/rsun_context_pixel*np.diff(red_limb_locs)

    ax_specg_xlim = ax_specg.get_xlim()
    ax_specg_ylim = ax_specg.get_ylim()
    ax_specr_xlim = ax_specr.get_xlim()
    ax_specr_ylim = ax_specr.get_ylim()

    ax_specg.set_xticks(list(ax_specg.get_xticks()) + FeXIV_tick_locs.tolist())
    ax_specr.set_xticks(list(ax_specr.get_xticks()) + FeX_tick_locs.tolist())

    ax_specg.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax_specr.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax_specg.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax_specr.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    #fig.canvas.draw()
    #print(ax_specg.get_xmajorticklabels())

    for FeXIV_xtickline, FeXIV_xticklabel, xtick_loc in zip(filter(lambda x: x.get_marker() == 3, ax_specg.get_xticklines()),
                            ax_specg.get_xticklabels(),ax_specg.xaxis.get_ticklocs()):
        if xtick_loc in FeXIV_tick_locs:
            FeXIV_xticklabel.set_visible(False)
            FeXIV_xtickline.set_markeredgecolor("red")
            FeXIV_xtickline.set_markeredgewidth(3)

    for FeX_xtickline, FeX_xticklabel, xtick_loc in zip(filter(lambda x: x.get_marker() == 3, ax_specr.get_xticklines()),
                            ax_specr.get_xticklabels(),ax_specr.xaxis.get_ticklocs()):
        if xtick_loc in FeX_tick_locs:
            FeX_xticklabel.set_visible(False)
            FeX_xtickline.set_markeredgecolor("red")
            FeX_xtickline.set_markeredgewidth(3)
    
    # print([x_.get_marker() for x_ in ax_specg.get_yticklines()])

    #after the slit is pointed to off-limbs
    if ii >= 66:
        ax_specg.set_yticks(list(ax_specg.get_yticks()) + green_limb_locs.tolist())
        ax_specr.set_yticks(list(ax_specr.get_yticks()) + red_limb_locs.tolist())

        for green_limb_ytickline, green_limb_yticklabel, ytick_loc in zip(filter(lambda x: x.get_marker() == 0, ax_specg.get_yticklines()),
                                ax_specg.get_yticklabels(),ax_specg.yaxis.get_ticklocs()):

            if ytick_loc in green_limb_locs:
                green_limb_yticklabel.set_visible(False)
                green_limb_ytickline.set_markeredgecolor("red")
                green_limb_ytickline.set_markeredgewidth(3)

        for red_limb_ytickline, red_limb_yticklabel, ytick_loc in zip(filter(lambda x: x.get_marker() == 0, ax_specr.get_yticklines()),
                                ax_specr.get_yticklabels(),ax_specr.yaxis.get_ticklocs()):
            if ytick_loc in red_limb_locs:
                red_limb_yticklabel.set_visible(False)
                red_limb_ytickline.set_markeredgecolor("red")
                red_limb_ytickline.set_markeredgewidth(3)

    ax_specg.set_xlim(ax_specg_xlim)
    ax_specr.set_xlim(ax_specr_xlim)
    ax_specg.set_ylim(ax_specg_ylim)
    ax_specr.set_ylim(ax_specr_ylim)

    ax_specg_right = ax_specg.secondary_yaxis("right",functions=(func_one_to_one, func_one_to_one))
    ax_specg_right.tick_params(labelright=False)
    ax_specr_right = ax_specr.secondary_yaxis("right",functions=(func_one_to_one, func_one_to_one))
    ax_specr_right.tick_params(labelright=False)

    for ax_ in (ax_img, ax_imr):
        ax_.set_aspect(1)

    
    # plt.show()
    plt.savefig(fname=os.path.join("../../sav/Eclipse/Video_RotSlit_SpecCorr/","Video_RotSlit_SpecCorr_{:03d}.png".format(ii)),format="png",
                dpi=144,bbox_inches="tight")
    fig.clf()
    for ax_ in axes.flatten():
        ax_.cla()
    plt.close(fig)

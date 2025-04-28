import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u
from glob import glob
import os
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, AsymmetricPercentileInterval
import h5py 
from PIL import Image
from datetime import datetime, timedelta
from ccdproc import ImageFileCollection
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FixedLocator, FixedFormatter, LogLocator
import cmcrameri.cm as cmcm
from juanfit import SpectrumFitSingle, SpectrumFitRow
from specutils.utils.wcs_utils import vac_to_air, air_to_vac
from scipy import ndimage
import cv2

video_frame_fnames = sorted(glob("../../sav/Eclipse/Video/frame*.jpg"))
video_frame_fnames = video_frame_fnames[34+13:97+34]
n_fnames = len(video_frame_fnames)

video_frame_cube = np.zeros((1080,1920,3,n_fnames),dtype="uint8")
for ii, video_frame_fname in enumerate(video_frame_fnames):
    with Image.open(video_frame_fname) as im:
        video_frame_cube[:,:,:,ii] = np.asarray(im)

video_vertical_slice =  slice(390,710)
video_horizontal_slice = slice(746,1200)
slit_pos = 209.4

video_frame_cube_flip = np.flip(video_frame_cube[video_vertical_slice,video_horizontal_slice,:,:], axis=0)
video_time_array = np.arange(datetime(2017,8,21,17,45,23),datetime(2017,8,21,17,48,9),timedelta(seconds=1)).astype(datetime)
video_time_array = video_time_array[13:97]
sun_x = np.zeros(n_fnames)
sun_y = np.zeros(n_fnames)
rsun = np.zeros(n_fnames)
for ii in range(n_fnames):
    totality_context_flip_gray = cv2.cvtColor(video_frame_cube_flip[:,:,:,ii], cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(totality_context_flip_gray, cv2.HOUGH_GRADIENT, 1, 1000,param1=500, param2=0.9, minRadius=69, maxRadius=72)

    fig, ax = plt.subplots(figsize=(10,6),constrained_layout=True)
    ax.imshow(video_frame_cube_flip[:,:,:,ii])
    ax.axvline(slit_pos,color="red",lw=2,alpha=0.7)

    if circles is not None:
        n_circle = circles.shape[1]
        for circle_ in circles[0]:
            circle_x, circle_y, circle_radius = circle_
            circle_plot = plt.Circle((circle_x, circle_y), circle_radius, color='red',fill=False,lw=3,alpha=0.7)
            ax.add_patch(circle_plot)

        if n_circle == 1:
            ax.set_title("i = {:03d} {} SUNX = {:.1f} SUNY = {:.1f}".format(ii, video_time_array[ii],circle_x, circle_y))
            sun_x[ii] = circle_x
            sun_y[ii] = circle_y
            rsun[ii] = circle_radius
        else:
            print("{:03d} failed".format(ii))
    else:
        print("{:03d} failed".format(ii))

    ax.axis("scaled")
    ax.tick_params(labelsize=16,length=4,width=1.2)






    plt.savefig(fname=os.path.join("../../sav/Eclipse/Video_TrackLimb/","Video_TrackLimb_{:03d}.png".format(ii)),format="png",
                dpi=120)
    fig.clf()
    ax.cla()
    plt.close(fig)

with h5py.File("../../sav/Eclipse/LimbTrack/sun_pos.h5", 'w') as hf:
    df_sunx = hf.create_dataset("sun_x",  data=sun_x)
    df_suny = hf.create_dataset("sun_y",  data=sun_y)
    df_rsun = hf.create_dataset("rsun",  data=rsun)
    df_sunx.attrs["start_index"] = 13+34
    df_sunx.attrs["end_index"] = 97+34

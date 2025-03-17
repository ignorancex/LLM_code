import os
import sys
import numpy as np
import imageio as iio
import time
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from utils import get_random_crop

"""
Script for visualizing the various data sources for the Krycklan site
"""


# Global vars
BASE_PATH = 'data'
DO_PLOT = 1
CROP_SIZE = 100

# Read rainfall data at Svartberget
rain_data = scipy.io.loadmat(os.path.join(BASE_PATH,
                             'rain_data.mat'))['rain_data'][0][0]
rain_data = rain_data[1]  # actual measurements
rain_data_norm = rain_data / np.max(rain_data)

# Read waterflow measurements
flow_data = scipy.io.loadmat(os.path.join(BASE_PATH,
                             'flow_data.mat'))['flow_data'][0]
flow_data_coords = np.squeeze(np.asarray([flow_data[i][0][0][4] for i in range(len(flow_data))]))
flow_data = np.squeeze(np.asarray([flow_data[i][0][0][3] for i in range(len(flow_data))])).T
flow_data_norm = flow_data / np.nanmax(flow_data)
flow_data_mean = np.nanmean(flow_data, 1)
flow_data_mean_norm = flow_data_mean / np.max(flow_data_mean)

# Read image plain coordinates of measurement points
ids_and_img_coords = scipy.io.loadmat(os.path.join(BASE_PATH,
                                      'ids_and_img_coords.mat'))['ids_and_img_coords']

# Plot rainfall and waterflow data -- also plot the mean flow in the same plot
# as the rainfall (both normalized to be max equal to 1)
if DO_PLOT:
    plt.figure(figsize=(9,3))
    plt.subplot(131)
    plt.plot(rain_data, 'o', mfc='none')
    plt.title('Rainfall at Svartberget')
    plt.xlabel('Day')
    plt.ylabel('Rainfall')
    plt.grid()
    plt.subplot(132)
    plt.plot(flow_data, 'x')
    plt.title('Water flow at sites')
    plt.xlabel('Day')
    plt.ylabel('Water flow')
    plt.grid()
    plt.subplot(133)
    plt.plot(rain_data_norm, 'o', mfc='none')
    plt.plot(flow_data_mean_norm, 'x', alpha=0.5)
    plt.title('Normalized rainfall + mean water flow')
    plt.xlabel('Day')
    plt.legend(['Rain', 'Flow'])
    plt.grid()
    #plt.draw()

    plt.savefig('data_fig1.png')
    plt.cla()
    plt.clf()
    plt.close('all')

# Read images
img_with_meas_points = iio.imread(os.path.join(BASE_PATH, 'matpunkter_siteid.JPG'))
hyd_conn = iio.imread(os.path.join(BASE_PATH, 'hydraulic_conductivity.png'))
soil_type = iio.imread(os.path.join(BASE_PATH, 'soil_type.png'))
soil_depth = iio.imread(os.path.join(BASE_PATH, 'soil_depth.png'))
soil_moisture = iio.imread(os.path.join(BASE_PATH, 'soil_moisture.png'))
soil_cover = iio.imread(os.path.join(BASE_PATH, 'soil_cover.png'))
all_imgs = [img_with_meas_points, hyd_conn, soil_type, soil_depth, soil_moisture, soil_cover]
H, W = img_with_meas_points.shape[:2]

# Show images
if DO_PLOT:
    plt.figure(figsize=(9,3))
    plt.subplot(231)
    plt.imshow(img_with_meas_points)
    plt.title('Map with measurement points')
    plt.subplot(232)
    plt.imshow(hyd_conn)
    plt.title('Hydraulic conductivity')
    plt.subplot(233)
    plt.imshow(soil_type)
    plt.title('Soil type')
    plt.subplot(234)
    plt.imshow(soil_depth)
    plt.title('Soil depth')
    plt.subplot(235)
    plt.imshow(soil_moisture)
    plt.title('Soil moisture')
    plt.subplot(236)
    plt.imshow(soil_cover)
    plt.title('Soil cover')
    #plt.draw()

    plt.savefig('data_fig2.png')
    plt.cla()
    plt.clf()
    plt.close('all')

    # Show image together with measurement points and points where we actually
    # have measurements
    plt.figure()
    plt.imshow(img_with_meas_points)
    plt.scatter(ids_and_img_coords[:, 1], ids_and_img_coords[:, 0], marker='x', color='r')
    plt.scatter(flow_data_coords[:, 1], flow_data_coords[:, 0], marker='o', facecolor='none', color='b')
    plt.title('Map with measurement points')
    plt.legend(['Sites', 'Sites with data'])
    #plt.draw()

    plt.savefig('data_fig3.png')
    plt.cla()
    plt.clf()
    plt.close('all')

# Normalize images to the [-1, 1]-range
all_imgs = [2 * np.float32(img) / 255 - 1 for img in all_imgs]

# Get (same) random crop of each image
random_crop, coords_inside, _ = get_random_crop(H, W, CROP_SIZE, flow_data_coords)

# Show cropped region on top of original image
if DO_PLOT:
    plt.figure()
    plt.imshow(img_with_meas_points)
    plt.scatter(ids_and_img_coords[:, 1], ids_and_img_coords[:, 0], marker='x', color='r')
    plt.scatter(flow_data_coords[:, 1], flow_data_coords[:, 0], marker='o', facecolor='none', color='b')
    plt.title('Map with measurement points + random crop')
    plt.legend(['Sites', 'Sites with data'])
    rect = patches.Rectangle((random_crop[2], random_crop[0]),
                             random_crop[3] - random_crop[2],
                             random_crop[1] - random_crop[0],
                             linewidth=1, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    #plt.draw()

    plt.savefig('data_fig4.png')
    plt.cla()
    plt.clf()
    plt.close('all')

# Crop actual image + create binary indicator image
img_crop = img_with_meas_points[random_crop[0]:random_crop[1],
                                random_crop[2]:random_crop[3], :]
indicator_crop = np.zeros((H, W), dtype=bool)
coords_inside = np.int32(np.round(coords_inside))
indicator_crop[coords_inside[:, 0], coords_inside[:, 1]] = 1
indicator_crop = indicator_crop[random_crop[0]:random_crop[1],
                                random_crop[2]:random_crop[3]]

# Show cropped image + indicator
if DO_PLOT:
    plt.figure(figsize=(9,3))
    plt.subplot(121)
    plt.imshow(img_crop)
    plt.title('Crop with measurement points')
    plt.subplot(122)
    plt.imshow(indicator_crop)
    plt.title('Indicator crop')
    #plt.show()

    plt.savefig('data_fig5.png')
    plt.cla()
    plt.clf()
    plt.close('all')

print("DONE")

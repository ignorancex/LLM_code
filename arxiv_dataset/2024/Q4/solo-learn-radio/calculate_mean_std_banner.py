import numpy as np
from astropy.io import fits
import json
import os
import pandas as pd
from astropy.visualization import MinMaxInterval
from astropy.visualization import AsinhStretch, LogStretch, SqrtStretch, SquaredStretch
from astropy.visualization import LinearStretch, PowerStretch, SinhStretch

def add_normalization_channels(img):
    ch0 = PowerStretch(2) + MinMaxInterval()
    ch1 = SquaredStretch() + MinMaxInterval()
    ch2 = SinhStretch() + MinMaxInterval()
    ch3 = LinearStretch(slope=1, intercept=0) + MinMaxInterval()
    ch4 = LogStretch(a=1000) + MinMaxInterval()
    ch5 = SqrtStretch() + MinMaxInterval()
    ch6 = AsinhStretch() + MinMaxInterval()

    stacked_image = np.dstack((
        ch0(img), ch1(img), ch2(img),
        ch3(img),
        ch4(img), ch5(img), ch6(img),))
    return stacked_image
    

def load_info(root, datalist):
    info_file = os.path.join(root, datalist)
    df = pd.read_json(info_file, orient="index")
    return df

def removeNaNs(im):
    # obtain the image histogram
    non_nan_idx = ~np.isnan(im)
    min_image = np.min(im[non_nan_idx])
    im[~non_nan_idx] = min_image # set NaN values to min_values
    
    return im
        
root = "../Banner-cutout_factor2.5-full/meerkat"
datalist = "info.json"
df = load_info(root, datalist)

mean    = 0.
std     = 0.
mean_0  = 0.
std_0   = 0.
mean_1  = 0.
std_1   = 0.
mean_2  = 0.
std_2   = 0.
for fits_path in df["target_path"]:
    img = removeNaNs(fits.getdata(os.path.join(root,fits_path)).astype(np.float32))
    stacked_image = add_normalization_channels(img)
    mean    += np.mean(stacked_image, axis=(0, 1))
    std     += np.std(stacked_image, axis=(0, 1))
    mean_0  += np.mean(stacked_image[:,:,0], axis=(0, 1))
    std_0   += np.std(stacked_image[:,:,0], axis=(0, 1))
    mean_0  += np.mean(stacked_image[:,:,1], axis=(0, 1))
    std_0   += np.std(stacked_image[:,:,1], axis=(0, 1))
    mean_0  += np.mean(stacked_image[:,:,2], axis=(0, 1))
    std_0   += np.std(stacked_image[:,:,2], axis=(0, 1))
    mean_1  += np.mean(stacked_image[:,:,3], axis=(0, 1))
    std_1   += np.std(stacked_image[:,:,3], axis=(0, 1))
    mean_2  += np.mean(stacked_image[:,:,4], axis=(0, 1))
    std_2   += np.std(stacked_image[:,:,4], axis=(0, 1))
    mean_2  += np.mean(stacked_image[:,:,5], axis=(0, 1))
    std_2   += np.std(stacked_image[:,:,5], axis=(0, 1))
    mean_2  += np.mean(stacked_image[:,:,6], axis=(0, 1))
    std_2   += np.std(stacked_image[:,:,6], axis=(0, 1))

mean /= len(df)
std  /= len(df)
mean_0  /= 3*len(df)
std_0   /= 3*len(df)
mean_1  /= len(df)
std_1   /= len(df)
mean_2  /= 3*len(df)
std_2   /= 3*len(df)

print(f"Mean: {mean} ; Std: {std}")
print(f"Mean_0: {mean_0} ; Std_0: {std_0}")
print(f"Mean_1: {mean_1} ; Std_1: {std_1}")
print(f"Mean_2: {mean_2} ; Std_2: {std_2}")
from astropy.io import fits
from PIL import Image
import numpy as np

def load_fits_image(image_path):
    return fits.getdata(image_path).astype(np.float32)

def load_npy_image(image_path):
    return np.load(image_path).astype(np.float32)

def load_rgb_image(image_path):
    return np.array(Image.open(image_path)).astype(np.float32)

def load_grey_image(image_path):
    return np.array(Image.open(image_path)).astype(np.float32)

# VLASS
# img = np.load(image_path)
# Mirabest
# np.squeeze(np.load(image_path)) # Mirabest (salvarlo con una dimensione in meno)
# ROBIN
# img = fits.getdata(image_path).astype(np.float32)
# RGZ
# img = fits.getdata(image_path).astype(np.float32)
# FRG
# img = np.array(Image.open(image_path)).astype(np.float32)
# Hulk - Banner
# img = self.removeNaNs(fits.getdata(image_path).astype(np.float32))
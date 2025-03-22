import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
from io import BytesIO
from PIL import Image as PILImage
import cv2


def add_salt_and_pepper(x, var):
    mask1 = torch.rand_like(x[1]) < var/2
    mask2 = torch.rand_like(x[1]) < var/2
    for i in range(3):
        x[i][mask1] = 0
        x[i][mask2] = 1
    return x

def add_gaussian_noise(x, var):
    noise = torch.randn_like(x) * var
    x = x + noise
    return x

def add_gaussian_blur(x, var):
    x_np = x.permute(1, 2, 0).cpu().numpy()  
    channel_images = [x_np[:, :, i] for i in range(x_np.shape[2])]
    blurred_channels = [gaussian_filter(channel, sigma=var, mode='reflect') for channel in channel_images]
    x_blurred = np.stack(blurred_channels, axis=2)
    x_blurred = torch.tensor(x_blurred, dtype=x.dtype).permute(2, 0, 1) 
    x_blurred = torch.clamp(x_blurred, 0, 1)
    return x_blurred

def add_shot_noise(x, var):
    shot_noise = torch.poisson(x) * var
    x += shot_noise
    return x

def add_impulse_noise(x, var):
    noisy_image = random_noise(x / 255.0, mode='s&p', amount=var)
    noisy_image = np.clip(noisy_image, 0, 1) * 255
    #noisy_image = noisy_image.astype(np.uint8)
    return torch.from_numpy(noisy_image)

def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()



def add_frost(x, severity=1):
    xtype = x.dtype
    x = x.permute(1, 2, 0).cpu().numpy()
    severity =  int(severity)

    c = [(0.98, 0.2), (0.96, 0.22), (0.94, 0.24), (0.9, 0.26), (0.88, 0.27), (0.86, 0.28), (0.84, 0.29), (0.82, 0.3), (0.8, 0.31), (0.78, 0.32)][severity - 1]

    idx = np.random.randint(5)
    filename = ['PATH_TO_FROST_PNGS/frost1.png', 
                'PATH_TO_FROST_PNGS/frost2.png', 
                'PATH_TO_FROST_PNGS/frost3.png', 
                'PATH_TO_FROST_PNGS/frost4.jpg', 
                'PATH_TO_FROST_PNGS/frost5.jpg', 
                'PATH_TO_FROST_PNGS/frost6.jpg'][idx]
    frost = cv2.imread(filename)
    frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 32), np.random.randint(0, frost.shape[1] - 32)
    frost = frost[x_start:x_start + 32, y_start:y_start + 32][..., [2, 1, 0]]
    frost = frost / 255.
    x = np.clip(c[0] * np.array(x) + c[1] * frost, 0, 1)
    #print(np.max(x), np.min(x), "\n")
    x = torch.tensor(x, dtype=xtype).permute(2, 0, 1) 
    return x


def add_frost_TIN(x, severity=1):
    xtype = x.dtype
    x = x.permute(1, 2, 0).cpu().numpy()
    severity =  int(severity)

    c = [(0.98, 0.2), (0.96, 0.22), (0.94, 0.24), (0.9, 0.26), (0.88, 0.27), (0.86, 0.28), (0.84, 0.29), (0.82, 0.3), (0.8, 0.31), (0.78, 0.32)][severity - 1]


    idx = np.random.randint(5)
    filename = ['PATH_TO_FROST_PNGS/frost1.png', 
                'PATH_TO_FROST_PNGS/frost2.png', 
                'PATH_TO_FROST_PNGS/frost3.png', 
                'PATH_TO_FROST_PNGS/frost4.jpg', 
                'PATH_TO_FROST_PNGS/frost5.jpg', 
                'PATH_TO_FROST_PNGS/frost6.jpg'][idx]    frost = cv2.imread(filename)
    frost = cv2.resize(frost, (0, 0), fx=0.3, fy=0.3)

    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 64), np.random.randint(0, frost.shape[1] - 64)
    frost = frost[x_start:x_start + 64, y_start:y_start + 64][..., [2, 1, 0]]
    frost = frost / 255.
    x = np.clip(c[0] * np.array(x) + c[1] * frost, 0, 1)
    #print(np.max(x), np.min(x), "\n")
    x = torch.tensor(x, dtype=xtype).permute(2, 0, 1) 
    return x







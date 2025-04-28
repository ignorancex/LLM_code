# coding: utf-8
from __future__ import print_function  # Python 2/3 compatibility
import numpy
import tifffile
from preprocessing import *
from unet2 import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def PIL2array(img):
    """ Convert a PIL/Pillow image to a numpy array """
    return numpy.array(img.getdata(),
                       numpy.uint8).reshape(img.size[1], img.size[0])

def division(a, b):
    a1 = a // b
    a2 = a % b
    if a2>0:
        a1 = a1 + 1
    return a1, a2

def division_padding(a, b, arg=2):
    stride = b // arg
    a1 = a // stride - arg + 1
    a2 = a % stride
    if a2 > 0:
        a1 = a1 + 1
    return a1, a2

def box(x, y, block_size, l, w):
    x1 = (x + 256) if x + block_size + 1 <= l else l
    y1 = (y + 256) if y + block_size + 1 <= w else w
    return x, y, x1, y1

# split into given size
def split(imageAddr, save=False, block_size=256, enhence=False, padding=False, enhence_ind=0, padding_index=2,):
    img = Image.open(imageAddr)
    images = []
    for i, page in enumerate(ImageSequence.Iterator(img)):
        page.load()
        page = page.convert('L')
        if enhence :
            if enhence_ind==0:
                max = np.max(skimage.img_as_float(page))
                page = color_enhence(page, round(1/ max))
            else:
                page = color_enhence(page,enhence_ind)
        images.append(page)
    h = len(images)  # of frames in a tif
    length, width = images[0].size
    if padding:
        l, padding_l = division_padding(length, block_size, padding_index)
        w, padding_w = division_padding(width, block_size, padding_index)
    else:
        l, padding_l = division(length, block_size)
        w, padding_w = division(width, block_size)
    splitImages = []
    for i in range(len(images)):
        for row in range(w):
            for col in range(l):
                if padding:
                    x = col * block_size // padding_index
                    y = row * block_size // padding_index
                else:
                    x = col * block_size
                    y = row * block_size
                splitImage = images[i].crop(box(x, y, block_size, length, width))
                if splitImage.size[0] < 256 or splitImage.size[1] < 256:
                    valid_page = Image.new('L', (block_size, block_size))
                    valid_page.paste(splitImage, (0, 0))
                    splitImage = valid_page
                splitImages.append(splitImage)

    # storing slices of images(optional)
    if save:
        for i in range(len(splitImages)):
            splitImages[i].save(r"data\tmp\gene\%04d.png"%i)
    return length, width, h, splitImages


def integration(images, length, width, h, saveAddre="test.tif",nparray=False, block_size=256):
    frames = []
    l, padding_l = division(length, block_size)
    w, padding_w = division(width, block_size)

    for i in range(h):
        frames.append(Image.new('L', (length, width)))
        for j in range(l * w * i, l * w * (i + 1)):
            x = j % l * block_size
            y = j // l % w * block_size
            l_crop = block_size if x + block_size <= length else length - x
            w_crop = block_size if y + block_size <= width else width - y
            if nparray:
                image = images[j][:,:,0]
                image = Image.fromarray(np.uint8(image),"L")
            else:
                image = images[j]
            # image.show()
            im = image.crop((0, 0, l_crop, w_crop))
            # print(l_crop, w_crop, box(x, y, block_size, length, width))
            frames[i].paste(im, box(x, y, block_size, length, width))
            frames[i].show()
    with tifffile.TiffWriter(saveAddre) as tiff:
        for img in frames:
            tiff.save(PIL2array(img))
    print("Done")

def integrationFromPNG(length, width, h, readAddress, saveAddre="test.tif", block_size=256, padding=False,padding_index=2):
    filelist = sorted(glob.glob(readAddress))
    frames = []
    images = []
    if padding:
        l, padding_l = division_padding(length, block_size, padding_index)
        w, padding_w = division_padding(width, block_size, padding_index)
    else:
        l, padding_l = division(length, block_size)
        w, padding_w = division(width, block_size)
    for fn in filelist:  # For each name in the list
        img = Image.open(fn)
        images.append(img)
    for i in range(h):
        frames.append(Image.new('L', (length, width)))
        for j in range(l * w * i, l * w * (i + 1)):
            if padding:
                x = j % l * block_size // padding_index
                y = j // l % w * block_size // padding_index
            else:
                x = j % l * block_size
                y = j // l % w * block_size
            l_crop = block_size if x + block_size <= length else length - x
            w_crop = block_size if y + block_size <= width else width - y
            image = images[j]
            # image = Image.open(filelist[j])
            im = image.crop((0, 0, l_crop, w_crop))
            frames[i].paste(im, box(x, y, block_size, length, width), im)
            # image.close()
    with tifffile.TiffWriter(saveAddre) as tiff:
        for img in frames:
            # img = (img//127)*255
            tiff.save(PIL2array(img))
    print("Done")


def newGenerator(images,flag_multi_class = False):
    for img in images:
        # print(type(img.size))
        img = np.reshape(img, img.size + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        img = img / 255
        yield img

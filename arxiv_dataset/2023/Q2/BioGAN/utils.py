"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
import scipy.misc
import imageio
from PIL import Image

try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [1024, 768])
    img = img/127.5 - 1
    return img

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    # img_C = imread(image_path[2])

    if not is_testing:
        img_A = scipy.misc.imresize(img_A, [1054, 798])
        img_B = scipy.misc.imresize(img_B, [1054, 798])
        # img_C = scipy.misc.imresize(img_C, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, 30 )))
        w1 = int(np.ceil(np.random.uniform(1e-2, 30)))
        img_A = img_A[h1:h1+1024, w1:w1+768]
        img_B = img_B[h1:h1+1024, w1:w1+768]
        # img_C = img_C[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
            # img_C = np.fliplr(img_C)

    else:
        img_A = scipy.misc.imresize(img_A, [1024, 768])
        img_B = scipy.misc.imresize(img_B, [1024, 768])
        # img_C = scipy.misc.imresize(img_C, [fine_size, fine_size])

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    # img_C = img_C/127.5 - 1.

    img_ABC = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_ABC

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def save_test_images(images, size, image_path):
    temp_image = reverse_image(images)
    temp_image = scipy.misc.imresize(temp_image[0] , [4032, 3024])
    im = temp_image / 255.
    im = np.expand_dims(im , axis=0)
    return imsave( im , size, image_path)


def imread(path, is_grayscale = False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.toimage(merge(images, size), cmin=0.0, cmax=1.0).save(path)
    # return scipy.misc.imsave(path, d)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return ( np.asarray(images)+1.)/2.

def get_img(src, img_size=False):
   img = imageio.imread(src, pilmode='RGB') # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = np.array(Image.fromarray(img).resize(img_size[:2]))
   return img

def reverse_image(image):
    return (np.asarray(image)+1.)* 127.5

def load_style_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [1024, 768])
    img = img/127.5 - 1
    return img
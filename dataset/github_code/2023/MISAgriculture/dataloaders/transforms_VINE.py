'''


Original implementation:  https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py


https://medium.com/mlearning-ai/understanding-torchvision-functionalities-for-pytorch-part-2-transforms-886b60d5c23a

'''


import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms.functional import normalize
from PIL import Image
import numbers
import cv2

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)



#===================================================================================
#
#  Multi Modal Tranformations
#
#===================================================================================


class ToTensor:
    def __init__(self):
        self.tensor = T.ToTensor()

    def __call__(self,rgb,mask):
        
        if not torch.is_tensor(rgb):
            mask = cv2.convertScaleAbs(mask)  #convert numpy.uint16 to uint8 (?)
            rgb = self.tensor(rgb,)
            mask = self.tensor(mask)        #(!!!)
            #mask = torch.from_numpy(mask)
            mask = mask.type(torch.float32)
            rgb  = rgb.type(torch.float32)
            #mask = torch.permute(mask,(-1,0,1))  #comentado
            #mask = self.tensor(mask)

        return(rgb,mask)


        
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self,rgb,mask):
        rgb = normalize(rgb, self.mean,self.std)
        return(rgb,mask)

class SingleCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb,):
        for t in self.transforms:
            
            rgb = t(rgb)
        return rgb


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, mask):
        for t in self.transforms:
            rgb,mask = t(rgb,mask)
        return rgb,mask


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = SingleCompose(transforms)

        return transform

    def __call__(self, img, depth, lbl):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        transf = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transf(img), depth, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomScale():
    def __init__(self, min_scale=0.1,max_sclae=1.2, interpolation=Image.BILINEAR):
        self.min_scale = min_scale
        self.max_scale = max_sclae 
        self.interpolation = interpolation

    def __call__(self, img, depth ,lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        assert img.shape[2:3] == lbl.shape[2:3]
        scale = random.uniform(self.min_scale, self.max_scale)
        target_size = (int(img.shape[1]*scale), int(img.shape[2]*scale) )
        img = F.resize(img, target_size, self.interpolation)
        depth = F.resize(depth, target_size, self.interpolation)
        lbl = F.resize(lbl, target_size, Image.NEAREST)
        return img,depth,lbl

    #def __repr__(self):
    #    interpolate_str = _pil_interpolation_to_str[self.interpolation]
    #    return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, depth, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        img = F.center_crop(img, self.size)
        depth = F.center_crop(depth, self.size)
        lbl = F.center_crop(lbl, self.size)
        return img,depth,lbl 

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class Resize:
    def __init__(self,size):
        self.size = size
        self.resize = T.Resize(self.size)
    def __call__(self, rgb, mask):

        rgb = self.resize(rgb)
        mask = self.resize(mask) 
        return(rgb,mask)
  
class RandomRotate:
    def __init__(self, min_angle, max_angle=None):
        self.min_angle = min_angle
        if max_angle is None:
            max_angle = min_angle
        self.max_angle = max_angle


    def __call__(self, rgb, mask):
        angle = random.randint(self.min_angle, self.max_angle)
        rgb  = F.rotate(rgb,angle)
        mask = F.rotate(mask,angle)
        return(rgb,mask)

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, rgb, mask):
        #if random.random() < self.flip_prob:
        rgb = F.hflip(rgb)
        mask = F.hflip(mask)
        return(rgb, mask)


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomMask():
    def __init__(self,mask_size=[50,50]):
        self.size= mask_size
    
    def __call__(self,rgb,depth,lbl):

        assert rgb.shape[2:3] == depth.shape[2:3], 'Modalities have not the same shape'
        
        c,h,w = rgb.shape
        # Generate random coordinates for the mask
        m_i = random.randint( self.size[0],h-self.size[0]) # mask origin rows
        m_j = random.randint(0+self.size[1],w-self.size[1]) # mask origin colums 
        
        rgb[:,(m_i-self.size[0]):(m_i+self.size[0]),m_j-self.size[1]:m_j+self.size[1]]=0
        depth[:,(m_i-self.size[0]):(m_i+self.size[0]),m_j-self.size[1]:m_j+self.size[1]] = 0
       
        return(rgb,depth,lbl)

class RandomPixelsErease():
    def __init__(self,height=(0,224),width=(0,224),pixel_percent=0.3,super_pixel=3):
        assert len(height) == 2, 'Height dim is wrong'
        assert len(width) == 2, 'Height dim is wrong'
        assert pixel_percent >= 0  and pixel_percent <= 1,'Percentage Pixel value out of range'
        self.pixel_percent = pixel_percent
        self.h = height
        self.w = width
        self.super_pixel = super_pixel
    
    def __call__(self,rgb,depth,lbl):

        assert rgb.shape[2:3] == depth.shape[2:3], 'Modalities have not the same shape'
        
        n_pixel_total = np.prod(rgb.shape).item()
        # Generate random coordinates for the mask
        n_pixel = int(round(n_pixel_total*(self.pixel_percent),0))

        randm_height = np.random.randint(low = self.h[0], high = self.h[1], size=n_pixel)
        randm_with = np.random.randint(low = self.w[0],high=self.w[1], size=n_pixel)
        
        rgb[:,randm_height,randm_with]   = 0
        depth[:,randm_height,randm_with] = 0
       
        return(rgb,depth,lbl)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        c,w, h = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, depth, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.shape[1:] == lbl.shape[1:], 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            depth = F.pad(depth, self.padding)
            lbl = F.pad(lbl, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            depth = F.pad(depth, padding=int((1 + self.size[1] - depth.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            depth = F.pad(depth, padding=int((1 + self.size[0] - depth.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        depth = F.crop(depth, i, j, h, w)
        lbl = F.crop(lbl, i, j, h, w)
        return img, depth, lbl

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, rgb, mask):
        rgb = F.center_crop(rgb, self.size)
        mask = F.center_crop(mask, self.size)
        return  rgb,mask


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class AdjustSaturation:
    def __init__(self,saturation_value=4): # Sat_value = 0 (whiite and black), 1 same image, 2 enhances sat by a factor of 2;
        self.max_sat_value = saturation_value
    def __call__(self,rgb, mask):
        sat = random.random()*self.max_sat_value
        rgb = F.adjust_saturation(rgb,sat)
        return(rgb, mask)

class AdjustBrightness:
    def __init__(self,brightness_factor = 3):
        self.brightness_factor = brightness_factor
    def __call__(self,rgb, mask):
        value = random.random()*self.brightness_factor
        rgb = F.adjust_brightness(rgb,value)
        return(rgb, mask)

class Equalize:
    def __init__(self):
        pass
    def __call__(self,rgb, mask):
        rgb = rgb.to(torch.uint8)
        rgb = F.to_pil_image(rgb)
        rgb = F.equalize(rgb)
        #rgb = F.to_tensor(rgb)
        rgb = F.pil_to_tensor(rgb)

        return(rgb,mask)


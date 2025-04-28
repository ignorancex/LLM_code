from typing import Callable, List, Optional, Sequence, Type, Union
from astropy.visualization import MinMaxInterval
from astropy.visualization import AsinhStretch, LogStretch, SqrtStretch, SquaredStretch
from astropy.visualization import LinearStretch, PowerStretch, SinhStretch
from astropy.stats import sigma_clip
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import random
import functools
import torch

class AddNormChannels:
    def __init__(self, cfg):
        """Add 3 normalized channels as a callable object.

        Args:None
        """
        self.norm_map = {
            "power": PowerStretch(2),
            "squared": SquaredStretch(),
            "sinh": SinhStretch(),
            "minmax": LinearStretch(slope=1, intercept=0),
            "log": LogStretch(a=1000),
            "sqrt": SqrtStretch(),
            "asinh": AsinhStretch(),
            #"sigmaclip": sigmaclip(5, 30),
            #"zscale": zscale(0.25),
            #"zscale": zscale(0.40)
        }
        self.ch0 = cfg.ch0
        self.ch1 = cfg.ch1
        self.ch2 = cfg.ch2

    def __call__(self, img: Image) -> Image:
        """Add normalizations channels to the image. It works for radio continuum images extracted from fits files e.g.

        Args:
            img (Image): a single channel image in npy format

        Returns:
            Image: blurred image.
        """

        ch0_stretch = self.norm_map[random.choices(self.ch0)[0]]
        ch1_stretch = self.norm_map[random.choices(self.ch1)[0]]
        ch2_stretch = self.norm_map[random.choices(self.ch2)[0]]

        transform_ch0 =  ch0_stretch + MinMaxInterval()
        transform_ch1 =  ch1_stretch + MinMaxInterval()
        transform_ch2 =  ch2_stretch + MinMaxInterval() # AsymmetricPercentileInterval(0.4, 1.)

        stacked_image = np.dstack((transform_ch0(img), transform_ch1(img), transform_ch2(img)))
        stacked_image_pil = Image.fromarray(np.uint8(stacked_image*255))
        """
        if any(np.isnan(return_image_pil)):
            print("NAN")
        """
        return stacked_image_pil
    
class Pad2Square:
    def __init__(self):
        """Pad to square

        Args:None
        """

    def __call__(self, img: Image) -> Image:
        """Add normalizations channels to the image. It works for radio continuum images extracted from fits files e.g.

        Args:
            img (Image): a single channel image in npy format

        Returns:
            Image: blurred image.
        """

        image_npy = np.array(img)
        H = image_npy.shape[0]
        W = image_npy.shape[1]
        if H < W:
            dif = W - H
            if dif % 2 == 0:
                up_pad = bot_pad = int(dif / 2)
            else:
                up_pad = int(dif / 2)
                bot_pad = int(dif - up_pad)
            #p2d = (up_pad, bot_pad, 0, 0)
            #image_npy = F.pad(image_npy, p2d, "constant", 0)
            image_npy = np.pad(image_npy, ((up_pad, bot_pad), (0, 0), (0,0)),
                    mode='constant', constant_values=0.) 
            
        if H > W:
            dif = H - W
            if dif % 2 == 0:
                left_pad = right_pad = int(dif / 2)
            else:
                left_pad = int(dif / 2)
                right_pad = int(dif - left_pad)
            #p2d = (0, 0, left_pad, right_pad)
            #image_npy = F.pad(image_npy, p2d, "constant", 0)
            image_npy = np.pad(image_npy, ((0, 0), (left_pad, right_pad), (0,0)), mode='constant', constant_values=0.)
        return Image.fromarray(np.uint8(image_npy))

class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)

class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)

class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"

class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join(str(transform) for transform in self.transforms)

class ZScaleTransformer(object):
	""" Apply zscale transformation to each channel """

	def __init__(self, contrasts=[0.25,0.25,0.25], **kwparams):
		""" Create a data pre-processor object """

		self.contrasts= contrasts
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			print("ERROR: Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check constrast dim vs nchans
		nchans= data.shape[-1]
	
		if len(self.contrasts)<nchans:
			print("ERROR: Invalid constrasts given (constrast list size=%d < nchans=%d)" % (len(self.contrasts), nchans))
			return None
		
		# - Transform each channel
		#data_stretched= np.copy(data)
		data_stretched= data

		for i in range(data.shape[-1]):
			data_ch= data_stretched[:,:,i]
			if self.contrasts[i]<=0:
				data_transf= data_ch
			else:
				transform= ZScaleInterval(contrast=self.contrasts[i]) # able to handle NANs
				data_transf= transform(data_ch)
			data_stretched[:,:,i]= data_transf

		# - Scale data
		data_stretched[~cond]= 0 # Restore 0 and nans set in original data

		return data_stretched

class SigmaClipper(object):
	""" Clip all pixels below zlow=mean-(sigma_low*std) and above zhigh=mean + (sigma_up*std) """

	def __init__(self, sigma_low=10.0, sigma_up=10.0, chid=-1, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.sigma_low= sigma_low
		self.sigma_up= sigma_up
		self.chid= chid # -1=do for all channels, otherwise clip only selected channel

	def __clip(self, data):
		""" Clip channel input """

		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]

		# - Clip all pixels that are below sigma clip
		#logger.debug("Clipping all pixel values <(mean - %f x stddev) and >(mean + %f x stddev) ..." % (self.sigma_low, self.sigma_up))
		res= sigma_clip(data_1d, sigma_lower=self.sigma_low, sigma_upper=self.sigma_up, masked=True, return_bounds=True)
		thr_low= res[1]
		thr_up= res[2]

		#data_clipped= np.copy(data)
		data_clipped= data
		data_clipped[data_clipped<thr_low]= thr_low
		data_clipped[data_clipped>thr_up]= thr_up
		data_clipped[~cond]= 0
		
		return data_clipped
		

	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			print("ERROR: Input data is None!")
			return None

		# - Loop over channels and get bgsub data
		#data_clipped= np.copy(data)
		data_clipped= data

		for i in range(data.shape[-1]):
			if self.chid!=-1 and i!=self.chid:
				continue	
			data_ch_clipped= self.__clip(data[:,:,i])
			data_clipped[:,:,i]= data_ch_clipped

		return data_clipped

class MinMaxNormalizer(object):
	""" Normalize each image channel to range  """

	def __init__(self, norm_min=0., norm_max=1., exclude_zeros=True, **kwparams):
		""" Create a data pre-processor object """
			
		# - Set parameters
		self.norm_min= norm_min
		self.norm_max= norm_max
		self.exclude_zeros= exclude_zeros


	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			print("ERROR: Input data is None!")
			return None

		# - Normalize data
		#data_norm= np.copy(data)
		data_norm= data

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			if self.exclude_zeros:
				cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			else:
				cond= np.isfinite(data_ch)
			data_ch_1d= data_ch[cond]
			if data_ch_1d.size==0:
				print("WARN: Size of data_ch%d is zero, returning None!" % (i))
				return None

			data_ch_min= data_ch_1d.min()
			data_ch_max= data_ch_1d.max()
			
			#logger.info("data_ch_min=%f, data_ch_max=%f" % (data_ch_min, data_ch_max))
			
			data_ch_norm= (data_ch-data_ch_min)/(data_ch_max-data_ch_min) * (self.norm_max-self.norm_min) + self.norm_min
			data_ch_norm[~cond]= 0 # Restore 0 and nans set in original data
			data_norm[:,:,i]= data_ch_norm

		return data_norm	
     
class ImagePreProcessor:
    def __init__(self, cfg):
        """Add 3 normalized channels as a callable object.

        Args:None

        cfg= {
            "zscale_contrasts": [0, 0.25, 0.40],
            "sigma_clip_low": 5,
            "sigma_clip_up": 30,
            "clip_chid": 0,
        }
        """
        self.zscale_contrasts = cfg.get('zscale_contrasts', [0, 0.25, 0.40])
        self.zscaler= ZScaleTransformer(contrasts=self.zscale_contrasts)  
        self.sigma_clip_low = cfg.get('sigma_clip_low', 5)
        self.sigma_clip_up = cfg.get('sigma_clip_up', 30)
        self.clip_chid = cfg.get('clip_chid', 0)
        self.clipper= SigmaClipper(sigma_low=self.sigma_clip_low, sigma_up=self.sigma_clip_up, chid=self.clip_chid)
        self.normalizer= MinMaxNormalizer(norm_min=0.0, norm_max=1.0)

        self.preprocess_stages= [self.clipper.__call__, self.zscaler.__call__, self.normalizer.__call__]
        self.preprocess_stages.reverse()
        self.transform= self.__compose_fcns(*self.preprocess_stages)
    
    def __compose_fcns(self, *funcs):
        """ Compose a list of functions like (f . g . h)(x) = f(g(h(x)) """
        return functools.reduce(lambda f, g: lambda x: f(g(x)), funcs)
		

    def __call__(self, img: Image) -> Image:
        """ Add normalizations channels to the image. It works for radio continuum images extracted from fits files e.g.

            Args:
                img (Image): a single channel image in numpy array format

            Returns:
                Image: transformed cube.
        """

        return self.transform(np.dstack((img, img, img)))

class PercentileThr:
    def __init__(self, cfg):
        """...
        Args:None
        """
        self.percentile                 = cfg.get('percentile', 50)
        self.random_percentile          = cfg.get('random_percentile', False) 
        self.random_percentile_per_ch   = cfg.get('random_perc_per_ch', False)
        self.percentile_min             = cfg.get('percentile_min', 50)
        self.percentile_max             = cfg.get('percentile_max', 60)

    def __call__(self, img: Image) -> Image:
        """Add normalizations channels to the image. It works for radio continuum images extracted from fits files e.g.

        Args:
            img (Image): a single channel image in npy format

        Returns:
            Image: blurred image.
        """
        return self.__apply_thr_cube(img)
        #return stacked_image_pil
    		
    def __apply_thr_cube(self, data):
        """ Apply threshold to ndim=3 images """	

        nchans= data.shape[-1]
        cond= np.logical_and(data!=0, np.isfinite(data))

        # - Set percentiles threshold
        percentiles_thr= [self.percentile]*nchans

        if self.random_percentile:
            if self.random_percentile_per_ch:
                for k in range(nchans):
                    percentiles_thr[k]= np.random.uniform(low=self.percentile_min, high=self.percentile_max)
            else:
                percentile_rand= np.random.uniform(low=self.percentile_min, high=self.percentile_max)
                for k in range(nchans):	
                    percentiles_thr[k]= percentile_rand

        # - Threshold each channel
        for i in range(data.shape[-1]):
            percentile_thr= percentiles_thr[i]
            data_ch= data[:,:,i]
            
            cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))
            data_ch_1d= data_ch[cond_ch]
            
            p= np.percentile(data_ch_1d, percentile_thr)

            data_ch[data_ch<p]= 0
            data_ch[~cond_ch]= 0

            data[:,:,i]= data_ch

        # - Restore 0 and nans set in original data
        data[~cond]= 0 

        return data

class RemoveNaNs():
    def __init__(self):
        """...
        Args:None
        """
        
    def __call__(self, img) -> Image:
        # obtain the image histogram
        non_nan_idx = ~np.isnan(img)
        min_image = np.min(img[non_nan_idx])
        img[~non_nan_idx] = min_image # set NaN values to min_values
        return img
    
class Np2PIL:
    def __init__(self):
        """...
        Args:None
        """

    def __call__(self, img) -> Image:
        """Add normalizations channels to the image. It works for radio continuum images extracted from fits files e.g.

        Args:
            img (Image): a single channel image in npy format

        Returns:
            Image: blurred image.
        """
        return Image.fromarray(np.uint8(img*255))
        #return stacked_image_pil
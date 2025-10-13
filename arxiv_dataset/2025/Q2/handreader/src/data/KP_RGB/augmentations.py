import numbers
import random

import numpy as np
import PIL
from scipy import ndimage


class RandomRotation:
    """
    Rotate an entire clip randomly by a random angle within the specified bounds.

    This class applies random rotation to a list of images (clip) by selecting an angle
    randomly within the given degree bounds. The rotation can be applied to both
    NumPy arrays and PIL images.

    Parameters
    ----------
    degrees : tuple or int, optional
        A sequence of two values representing the lower and upper bounds for rotation in degrees.
        If a single integer is provided, the range will be (-degrees, +degrees). Default is (-10, 10).

    Methods
    -------
    __call__(clip)
        Rotates the input clip by a randomly selected angle within the given bounds.
    rotate(clip, angle)
        Rotates each image in the clip by the specified angle.
    """

    def __init__(
        self,
        degrees=(-10, 10),
    ):
        """
        Initializes the RandomRotation instance.

        Parameters
        ----------
        degrees : tuple or int, optional
            A sequence of two values representing the lower and upper bounds for rotation in degrees.
            If a single integer is provided, the range will be (-degrees, +degrees). Default is (-10, 10).

        Raises
        ------
        ValueError
            If degrees is a negative number or if the sequence length is not equal to 2.
        """

        super().__init__()

        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number," "must be positive")
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence," "it must be of len 2.")

        self.degrees_up = degrees[0]
        self.degrees_lower = degrees[1]

    def __call__(self, clip):
        """
        Rotates the input clip by a randomly selected angle.

        Parameters
        ----------
        clip : list of PIL.Image or numpy.ndarray
            A list of images (in NumPy ndarray or PIL Image format) to be rotated.

        Returns
        -------
        list of PIL.Image or numpy.ndarray
            A list of rotated images, in the same format as the input.

        Notes
        -----
        The angle is selected randomly within the bounds specified during initialization.
        """
        angle = random.uniform(self.degrees_up, self.degrees_lower)
        return self.rotate(clip, angle)

    def rotate(self, clip, angle):
        """
        Rotates each image in the clip by the specified angle.

        Parameters
        ----------
        clip : list of PIL.Image or numpy.ndarray
            A list of images (in NumPy ndarray or PIL Image format) to be rotated.
        angle : float
            The angle in degrees to rotate each image.

        Returns
        -------
        list of PIL.Image or numpy.ndarray
            A list of rotated images, in the same format as the input.

        Raises
        ------
        TypeError
            If the input is neither a list of NumPy arrays nor PIL Images.
        """
        if isinstance(clip[0], np.ndarray):
            rotated = [ndimage.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image" + f" but got list of {type(clip[0])}"
            )

        return rotated


class TemporalRescale:
    """
    Rescale the temporal length of a clip by a random factor within a specified range.

    This class applies temporal rescaling to a clip (list of images or frames) by adjusting its length
    based on a random sample rate. The new length is determined by scaling the original length of the clip
    by a factor within the provided bounds. The clip can either be truncated or padded (by repeating frames)
    to achieve the desired length.

    Parameters
    ----------
    sample_rate : tuple of float, optional
        A tuple representing the lower and upper bounds for the rescaling factor.
        The new length of the clip is chosen randomly within this range. Default is (0.8, 1.2).

    Methods
    -------
    __call__(clip)
        Rescales the input clip by a randomly selected sample rate.
    """

    def __init__(self, sample_rate=(0.8, 1.2)):
        """
        Initializes the TemporalRescale instance.

        Parameters
        ----------
        sample_rate : tuple of float, optional
            A tuple representing the lower and upper bounds for the rescaling factor.
            The new length of the clip is selected randomly within this range.
            Default is (0.8, 1.2).

        Raises
        ------
        ValueError
            If the lower bound of the sample rate is greater than the upper bound,
            or if the sample rate values are out of valid range.
        """

        super().__init__()

        self.min_len = 3
        self.max_len = 296
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError(
                "Invalid combination of rate_lower and rate_upper. Got: {}".format(
                    (rate_lower, rate_upper)
                )
            )

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def __call__(self, clip):
        """
        Rescales the input clip by a randomly selected sample rate.

        Parameters
        ----------
        clip : list of numpy.ndarray or PIL.Image
            A list of images or frames representing the clip to rescale.

        Returns
        -------
        list of numpy.ndarray or PIL.Image
            A rescaled clip, where the length of the clip has been modified by a random factor
            within the specified bounds.
        """

        sample_rate = random.uniform(self.rate_lower, self.rate_upper)
        vid_len = len(clip)
        new_len = max(int(vid_len * sample_rate), self.min_len)
        # new_len = min(new_len, self.max_len)

        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]

import numpy as np
from pyarchi.utils import create_logger

logger = create_logger("masks")


class Masks:
    """
        Class used to hold the mask for each iteration, as well as some of some important methods
    """

    def __init__(self, mask_factor, grid_increase, initial_mask, low_memory=0):

        self._mask_factor = mask_factor
        self.grid_increase = grid_increase

        self._masks = []
        self._low_mem = low_memory

        self._mask_size = len(np.where(initial_mask != 0)[0])
        self.add_mask(initial_mask)

    def add_mask(self, mask):
        """
        Stores a new mask. If the low memory mode is active, only two masks are stored in memory: the initial one
        and the latest.

        Parameters
        ----------
        mask:
            Mask to be added

        Returns
        -------

        """

        if self._low_mem and len(self._masks) == 2:
            self._masks[-1] = mask
        else:
            self._masks.append(mask)

    def update_mask(self, x_change, y_change, image_number):
        """
            Changes the mask to the correct position in the new image. If the grid_bg is not None, then
            the calculations are made with the bigger grid.

        Parameters
        ----------
        x_change
            change in the x direction
        y_change
            change in the y direction

        image_number
            Current image number.

        Returns
        -------

        """

        empty_mask = np.roll(self.first.copy(), x_change, axis=0)
        empty_mask = np.roll(empty_mask, y_change, axis=1)

        if image_number != 0:  # no need to add again the mask for the first frame
            self.add_mask(empty_mask)

    @property
    def latest(self):
        """
            Returns the latest mask stored in the class
        """
        return self._masks[-1]

    @property
    def first(self):
        """
            Returns the first mask
        """
        return self._masks[0]

    @property
    def number_masks(self):
        """
        Return the number of masks stored in the obejct
        """
        return len(self._masks)

    @property
    def all(self):
        """
        Return a list with all of the stored masks
        """
        return self._masks.copy()

    @property
    def factor(self):
        """
            Returns the mask's size  (i.e. circle radius or layers of pixels added to the mask)
        """
        return self._mask_factor

    @property
    def size(self):
        """ 
            Returns the number of pixels in the mask
        """
        return self._mask_size

    @property
    def normalized_points(self): 
        """ 
            Returns the number of points, normalized to the "normal" grid, with 200 by 200 px
        """
        return self._mask_size / self.grid_increase ** 2

    def __getitem__(self, item):
        return self._masks[item]  # delegate to li.__getitem__

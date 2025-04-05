from astropy.io import fits
import numpy as np

from .Mask import Masks

from pyarchi.utils import path_finder
from pyarchi.utils import create_logger, CDPP

logger = create_logger("star_track")


class Star:
    """
    Star class that holds the star information. We have a class attribute, number, that will be used to name the star.
     Each star increments this number by one.

    Parameters
    ---------------
        cdpp_type:
            CDPP algorithm; Can be a string or a function. If it is a string, it should be "K2". Otherwise, the function should have the format
                def foo(flux, time):
                    return metric
            
        pos=None:
            Initial position
            
        dist = None:
            Distance to center
    Notes
    -----------
        name:
            Designation of the star
        masks:
            Holds the mask used in each image.
        GP_data:
            Holds the GPs results
        positions:
            position, in pixels, of the centroid. Always on coordinates of the [200,200] grid
        init_pos:
            Initial position, in pixels, of the centroid. Always on coordinates of the [200,200] grid

        out_bound:
            An overlap between the mask and the empty region was found

        _active:
            Do we want to calculate the flux for this star

        photom:
            holds the light curve information
        debug:
            If 1 we compare the data obtained from our analysis with the one produced by the official pipeline
    """

    number = 0

    def __init__(self, cdpp_type: str, pos=None, dist=None):

        # General star information
        self.number = Star.number
        self.__class__.number += 1
        self.positions = [pos]
        self.init_pos = pos.copy()
        self.dist_center = dist

        # Star status:
        self.out_bound = False  # True if the mask overlaps the empty zone
        self._out_number = 0
        self._out_location = []
        self._active = True

        # Data storage
        self.masks = None
        self._GP_data = None

        # misc
        self._photom = []
        self._uncertainties = []
        self.debug = 0
        self.default_lightcurve = []

        self.DRP_uncert = None

        self.cdpp_type = cdpp_type

    def enable_debug(self, **kwargs):
        """
        Extracts the default data from  the FITS files
        Parameters
        ----------
        base_folder:
            path to folder in which the FITS files are located

        Returns
        -------

        """

        try:
            logger.info("Searching for the file")
            default_path = path_finder(mode="default", **kwargs)
        except:
            logger.fatal("Could not find the path to the file", exc_info=True)
            return -1

        with fits.open(default_path) as hdulist:
            self.default_lightcurve = hdulist[1].data["flux"]
            self.DRP_uncert = hdulist[1].data["FLUXERR"]
        self.debug = 1

    def calculate_cdpp(self, time=None):
        """
        Calculates the CDPP of the light curve, in order to quantify the results

        Returns
        -------
            CDPP, CDPP_def : noise for my light curve and for the official one, respectively
        """

        # TODO: allow to pass values for the CDPP
        if not self.is_active:  # If not active return stupid values
            return 2e7, 2e7

        sized = 41
        winlen = 10
        win = 30
        outl = True

        if time is None:
            logger.fatal("Missing the time to use DRP's CDPP")

        if self.cdpp_type == "K2":
            cv = CDPP(self.photom, time, sized, winlen, win, outl)
            cv_def = CDPP(self.default_lightcurve, time, sized, winlen, win, outl) if self.debug else None

        else:
            cv = self.cdpp_type(self.photom, time)
            cv_def = self.cdpp_type(self.default_lightcurve, time) if self.debug else None

        return cv, cv_def

    def out_bounds(self, index):
        """
            Take into account that star was out of bonds in a given frame
        """
        self._out_number += 1
        self.out_bound = True
        self._out_location.append(index)

    def add_photom(self, value):
        self.photom.append(value)

    def add_initial_mask(self, mask, factor, scaling_factor, low_memory=0):
        """
        Initializes the Masks class

        Parameters
        ----------
        mask
            Initial mask for the star
        factor
            Increase factor for the mask
        size_grid_change
            Size of the background grid
        low_memory:
            Mode in which only the necessary information is kept during the process
        Returns
        -------
            0
                If no error is found
            -1
                If the factor is a negative number
        """
        logger.info("{} - Adding initial masks".format(self.name))
        if type(factor) == dict:
            self.masks = Masks(factor[str(self.number)], scaling_factor, mask, low_memory)
            return 0
        if factor <= 0:
            logger.fatal("Invalid mask factor")
            return -1
        self.masks = Masks(factor, scaling_factor, mask, low_memory)

        return 0

    def change_init_pos(self, pos):
        """
        Change the star's initial position. Used when more than one initial center determination method is active
        and some information needs to be overwritten
        Parameters
        ----------
        pos:
            New position for the centroid of the star (for the 1st image)

        Returns
        -------

        """
        self.positions = [pos]
        self.init_pos = pos.copy()

    def update_mask(self, scaling_factor, index):
        """
        Updates the mask with the information from the current image
        Parameters
        ----------
        scaling_factor
            Size of the background grid.
        index
            Index of the current image

        Returns
        -------

        """

        if index == 0:
            return

        init_pos = self.init_pos
        positions = self.positions[-1]

        x_change = int(round(-init_pos[0] + positions[0]))
        y_change = int(round(-init_pos[1] + positions[1]))

        self.masks.update_mask(x_change, y_change, index)

    def remove_data(self):
        """
        Empty all information stored on this star. Used during the optimization process
        Returns
        -------

        """
        logger.info("Removing all data")
        del self.masks

        self.masks = None
        self.positions = [self.init_pos.copy()]
        self._photom = []
        self._uncertainties = []
        self.out_bound = False  # True if the mask uses the empty zone
        self._GP_data = None

    def calculate_errors(self, background, nstack, cron, dark):
        flux = self.photom[-1]
        npix = self.masks.normalized_points
        self._uncertainties.append(
            np.sqrt(flux + background * npix + npix * nstack * cron ** 2 + dark * npix)
        )

    def add_mask(self, mask):
        """
        Adds a mask to the Masks object

        Parameters
        ----------
        mask:
            mask to add

        Returns
        -------

        """

        self.masks.add_mask(mask)

    def add_center(self, point):
        self.positions.append(point)

    def change_center(self, index, new_point):
        self.positions[index] = new_point

    def disable(self):
        """
        Sets the flag to disable the calculation of the flux of this star.
        Returns
        -------

        """
        logger.info("Star {} was disabled".format(self.number))
        self._active = False

    def import_photom(self, photom):
        """
        Used when loading data from disk
        Parameters
        ----------
        photom

        Returns
        -------

        """
        self._photom = photom

    def all_masks(self):
        return self.masks.all

    def import_uncertainties(self, uncerts):
        self._uncertainties = uncerts

    @classmethod
    def reset_number(cls, new_number):
        cls.number = new_number

    @property
    def name(self):
        return "Star {}".format(self.number)

    # Properties:
    @property
    def is_active(self):
        return self._active

    @property
    def photom(self):
        return self._photom

    @property
    def uncertainties(self):
        return (
            self._uncertainties if self._uncertainties else [500e-6 for _ in self.photom]
        )

    @property
    def mask_factor(self):
        return self.masks.factor

    @property
    def mask_npoints(self):
        return self.masks.size

    @property
    def mask_norm_npoints(self):
        return self.masks.normalized_points

    @property
    def first_mask(self):
        return self.masks.first

    @property
    def latest_mask(self):
        """
        Returns the last mask added to the star.
        Returns
        -------

        """
        return self.masks.latest

    def __str__(self):
        return "{} - {}".format(self.name, self._active)

    def __repr__(self):
        return "{} - {}".format(self.name, self._active)

    @property
    def number_out_bounds(self):
        return self._out_number

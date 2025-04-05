from astropy.io import fits
import numpy as np

from pyarchi.masks_creation import create_circular_mask, create_shape_mask
from pyarchi.masks_creation import change_grid

from pyarchi.initial_detection import centers_from_fits, initial_dynam_centers

from pyarchi.utils.misc.rotation_mats import matrix_cnter_clock, matrix_clockwise
from pyarchi.utils import path_finder
from pyarchi.utils import get_optimized_mask
from pyarchi.utils import create_logger

from .Star_class import Star

logger = create_logger("Data")


class Data:
    """
        Data structure to hold the information of all the stars in the field.

        Parameters
        ---------------
        base_folder:
            location of the folder in which all the fits files are located.
        roll_ang:
            rotation angle for all the images.
        imgs:
            all the images in the fits file.
        stars:
            List in which all elements are a "Star" object that holds information of one star.
        mask_type:
            type of mask. either circle or shape 
        detect_mode:
            detection mode used
        init_detection_mode:
            initial detection mode
        mjd_time:
            mjd time of the observations
    """

    def __init__(self, filename):
        # error flag
        self._error_flag = 0

        # configuration loaded from the .yaml
        self.base_folder = ""
        self.mask_type = ""
        self.detect_mode = ""
        self.init_detection_mode = ""
        self.used_file = filename
        self.bg_grid = 0

        # data loaded from the fits file
        self.roll_ang = None
        self.mjd_time = None
        self._imgs = None
        self.offsets = None
        self.image_number = None
        self.forbidden_regions = []

        # misc
        self._stars = []
        self._image_dict = {}
        self.low_mem = 0
        self.calc_uncert = False
        self.uncertainties_params = {}  # to estimate the uncertainties for our data

    def _reset(self):
        """
        Resets the object to its initial state. "Cleaning" all data previously stored in here.

        It's called when the parameters are loaded on :function:`pyarchi.main.initial_loads.Data.load_parameters`.
        Keep in mind that the reset routine does not re-enable the disabled stars.
        Returns
        -------

        """

        # error flag
        self._error_flag = 0

        # configuration loaded from the .yaml
        self.base_folder = ""
        self.mask_type = ""
        self.detect_mode = ""
        self.bg_grid = 0

        # data loaded from the fits file
        self.roll_ang = None
        self.mjd_time = None
        self._imgs = None
        self.offsets = None
        self.forbidden_regions = []

        self._image_dict = {}

        # misc
        for star in self._stars:
            star.remove_data()

        Star.number = 0  # Reset the number of the stars every time this routine is called

    def _verify_validity(func):  # pylint: disable=no-self-argument
        """
        Used to validate the error flag before running a function.
        Decorator used inside this class

        Returns
        -------

        """

        def on_call(self, *args, **kwargs):
            if self._error_flag:
                logger.fatal(
                    "Error found. Method {} is not going to run. ".format(
                        func.__name__
                    )  # pylint: disable=no-member
                )
                return -1
            else:
                return func(self, *args, **kwargs)  # pylint: disable=not-callable

        return on_call

    def load_parameters(self, factor=None, **kwargs):
        """
        Loads all the necessary information from the .fits files. Launches the routines to find the initial
        position of the stars and create the corresponding masks.

        Before running the internal parameters, as well as the star's ones are removed. This was made in order to
        minimize loading data from disk. However, keep in mind that the reset routine does not re-enable stars. So, if a
        star was disabled in this object, then it will stay that way unless a new object is instantiated.
        Parameters
        ----------
        factor
        kwargs

        Returns
        -------
        -1:
            Errors were found during runtime
        0:
            Everything went without problems

        """
        self._reset()
        self.base_folder = kwargs["base_folder"]
        self.init_detection_mode = kwargs['initial_detect']
        self.detect_mode = kwargs["detect_mode"]
        self.mask_type = kwargs["method"]
        self.bg_grid = kwargs["grid_bg"]
        self.low_mem = kwargs["low_memory"]
        self.calc_uncert = kwargs["uncertainties"]

        load_flag = self._init_load(**kwargs)
        pos_flag = self._init_pos(**kwargs)
        mask_flag = self._initialize_masks(factor, **kwargs)

        if -1 in [load_flag, pos_flag, mask_flag]:
            self._error_flag = 1
            return -1
        else:
            init_image = self.get_image(0)

            self.forbidden_regions = np.where(np.isnan(init_image))
        return 0

    @_verify_validity
    def update_stars(self, image_number):
        """
        Controls the data treatment process for each image. Updates the masks for each star and calculates the flux
        passing through our mask. Afterwards the mask is validated to see if we have overlaps with the forbidden region,
        i.e., if the mask is over the  NaN areas of the image.
        Parameters
        ----------
        image_number

        store_stars:
            If parameter is set, store a 

        Returns
        -------

        """

        scaling_factor = self.bg_grid / 200 if self.bg_grid else 1
        for star in self._stars:
            if not star.is_active:
                continue

            star.update_mask(scaling_factor, image_number)

            final_img = self.get_image(image_number)
            try:
                final_result = np.multiply(star.latest_mask, final_img)
            except Exception as e:
                print(image_number)
                print(star, star.latest_mask.shape, final_img.shape)
                raise e

            photom = np.nansum(final_result)

            star.add_photom(photom)

        self._validate_forbidden_region(image_number)

    def get_image(self, number):
        """
        Ask for a given image. If the background grid is in use, returns the increased image
        
        Parameters
        ----------
        number : int
            Image number
        
        Returns
        -------
        image: numpy array
            Desired image
        """
        if number == "all":
            return list(self._image_dict.values()) if self._image_dict else self._imgs

        if self.low_mem or self.bg_grid:
            self._increase_imgs(number)
            return self._image_dict[number].copy()
        else:
            return self._imgs[number].copy()

    def reload_images(self, **kwargs):
        """
        Reload images from disk, in the case that the desired image has been deleted
        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        return self._init_load(**kwargs)

    def _init_load(self, **kwargs):
        """
                Opens the fits files and extracts the roll angle and the images

        Parameters
        ----------
        kwargs

        Returns
        -------

        """

        self._image_dict = {}

        logger.info("Extracting official pipeline Lightcurve information")
        try:
            default_path = path_finder(mode="default", **kwargs)
            subarray_path = path_finder(mode="subarray", **kwargs)
        except:
            self._error_flag = 1
            logger.critical("Could not find paths to DRP outputs")
            return -1
        try:
            hdulist = fits.open(default_path)
        except IOError:
            logger.error("Official Lightcurve file not found")
            self._error_flag = 1
            return -1
        else:
            with hdulist:
                self.roll_ang = hdulist[1].data["ROLL_ANGLE"]
                self.mjd_time = hdulist[1].data["MJD_TIME"]
                self.offsets = list(
                    zip(hdulist[1].data["CENTROID_X"], hdulist[1].data["CENTROID_Y"])
                )

                self.intended_loc = [hdulist[1].data["LOCATION_X"][0], hdulist[1].data["LOCATION_Y"][0]]

                if self.calc_uncert:
                    possible_curves = ["DEFAULT", "OPTIMAL", "RSUP", "RINF"]
                    possible_curves.remove(kwargs["official_curve"])
                    darks = []

                    def_points = np.pi * (hdulist[1].header["AP_RADI"]) ** 2
                    self.uncertainties_params["bg"] = (
                        hdulist[1].data["BACKGROUND"] / def_points
                    )
                    curr_dark = hdulist[1].data["DARK"] / def_points
                    darks.append(curr_dark)

                    for curve in possible_curves:
                        try:
                            default_path = path_finder(mode="default", off_curve=curve, **kwargs)
                        except:
                            self._error_flag = 1
                            logger.critical("Could not find paths to DRP outputs")
                            return -1
                        with fits.open(default_path) as file:
                            def_points = (
                                np.pi * (file[1].header["AP_RADI"]) ** 2
                            )  # pylint: disable=no-member
                            curr_dark = (
                                file[1].data["DARK"] / def_points
                            )  # pylint: disable=no-member
                            darks.append(curr_dark)

                    self.uncertainties_params["dark"] = np.median(darks, axis=0)
                    self.uncertainties_params["t_exp"] = hdulist[1].header["EXPTIME"]
                    self.uncertainties_params["nstack"] = hdulist[1].header["NEXP"]
        try:
            hdulist = fits.open(subarray_path)
        except IOError:
            logger.error("Subarray file not found")
            self._error_flag = 1
            return -1
        else:
            with hdulist:
                imgs = hdulist[1].data
                self._imgs = imgs
                self.image_number = self._imgs.shape[0]
                self.image_size = self._imgs[0].shape[::-1]  # to follow the X and Y convention
                if self.calc_uncert:
                    self.uncertainties_params["cron"] = 1.96 * hdulist[2].data["RON"]

    @_verify_validity
    def _init_pos(self, **kwargs):
        """
        Extracts the initial position of each star, creating a Star object for each of  those positions and
        storing them inside the self._stars. We can have the initial position determination with an image treatment
        approach (dynam), by loading data from the fits files and inferring their position (fits) or, specify a load
        type for the central star and the outer ones, separating them both with a plus sign..

        :return:
        """

        if "+" not in kwargs["initial_detect"]:
            primary = secondary = kwargs["initial_detect"]
        else:
            primary, secondary = kwargs["initial_detect"].split("+")

        disabled = [i.number for i in self.stars if not i.is_active]

        self._stars = initial_dynam_centers(self.get_image(0), self.bg_grid, **kwargs)
        
        for j, star in enumerate(self._stars):
            if j in disabled:
                star.disable()

        self._stars = centers_from_fits(
            primary, secondary, self._stars, self.roll_ang[0], self.offsets[0], **kwargs
        )


    @_verify_validity
    def _initialize_masks(self, factor, **kwargs):
        """
        Controls the process of creating the masks for each star. During the optimization process the factor is
        passed as an argument. Otherwise, it is loaded from the corresponding file.

        Parameters
        ----------
        factor:
            Passed during the optimization process. Mask is created using it it something other than None is passed
        kwargs

        Returns
        -------

        """

        scaling_factor = self.bg_grid / 200 if self.bg_grid else 1
        if factor is None:
            # If we have no manual input and no  command for optimization
            # then we will load optimized values from the json file

            optim_mask = get_optimized_mask(**kwargs)
            if optim_mask == -1:
                logger.fatal("Could not retrieve the optimized factors")
                self._error_flag = 1
                return -1

            if self._create_mask(optim_mask, scaling_factor, **kwargs) == -1:
                logger.fatal("Problem with the mask creation process")
                return -1

        else:  # during optimization process
            self._create_mask(factor, scaling_factor, **kwargs)

    @_verify_validity
    def _create_mask(self, factor, scaling_factor, **kwargs):
        """
        Creates a mask for each star, using the specified method on the config.yaml file

        Parameters
        ----------
        factor:
            Scaling factor - integer if this is called during the optimizaton process. Otherwise should be a list

        size_grid_change:
            size of the big grid


        Returns
        -------

        """

        if self.abort_process:
            logger.fatal("Process was aborted due to previous errors")
            return -1

        if "+" not in kwargs["method"]:
            primary = secondary = kwargs["method"]
        else:
            primary, secondary = kwargs["method"].split("+")

        circle_result = create_circular_mask(
            self.get_image(0), self._stars, factor, primary, secondary
        )
        if circle_result == -1:
            logger.fatal("Could not create mask using the 'CIRCLE' method. ")
            return -1
        shape_result = create_shape_mask(
            self.get_image(0), self._stars, factor, scaling_factor, primary, secondary, kwargs['grid_bg'], kwargs['repeat_removal']
        )

        if shape_result == -1:
            logger.fatal("Could not create mask using the 'SHAPE' method. ")
            return -1

        # Create a joint dict with the star number and the mask
        full_dict = circle_result.copy()
        full_dict.update(shape_result)

        for key, mask in full_dict.items():
            process_flag = self._stars[key].add_initial_mask(
                mask, factor, scaling_factor, kwargs["low_memory"]
            )

            if process_flag == -1:
                return -1

        return 0

    def calculate_uncertainties(self, index):
        """
        Trigger the calculation of the uncertainties
        
        Parameters
        ----------
        index : int
            Image number
        """
        if not self.calc_uncert:
            return
        else:
            params = self.uncertainties_params
            for star in self.stars:
                star.calculate_errors(
                    background=params["bg"][index],
                    nstack=params["nstack"],
                    cron=params["cron"][index],
                    dark=params["dark"][index],
                )

    def _validate_forbidden_region(self, image_number):
        """
        Checks for overlaps with the forbidden region
        Returns
        -------

        """

        for star in self._stars:

            if not star.is_active:
                continue

            if star.latest_mask[self.forbidden_regions].any() != 0:
                star.out_bounds(image_number)

    def disable_star(self, star_number):
        """
        Disables a star to avoid computing redundant information i.e. after the best value has been found during the
        optimization process for one star, but others are still not yet optimized.
        Parameters
        ----------
        star_number

        Returns
        -------

        """
        self._stars[star_number].disable()

    @_verify_validity
    def _increase_imgs(self, index):
        """
        Increases the size of all images before the process starts
        Returns
        -------

        """

        if index not in self._image_dict:
            ratio = self.bg_grid / 200
            if ratio == 0:
                im = self._imgs[0].copy()
            else:
                im = np.divide(change_grid(self._imgs[0].copy(), ratio), ratio ** 2)

            self._imgs = self._imgs[1:]

            self._image_dict[index] = im

            if self.low_mem:
                self._image_dict = {
                    key: val for key, val in self._image_dict.items() if key > index - 2
                }

    # Properties of the class
    @property
    def stars(self):
        """
        Returns the list with all of the :class:`~pyarchi.data_objects.Star_class.Star` objects.
        If an error has been raised, returns an empty list
        
        """
        if not self._error_flag:
            return self._stars
        else:
            return []

    @property
    def abort_process(self):
        """
        If the error_flag is set then the process should be aborted
        Returns
        -------

        """
        return self._error_flag

    @property
    def all_curves(self):
        """
        Light curve of all stars, in the star numbering order (from closest to center to farthest)
        Returns
        -------

        """
        if not self._error_flag:
            return np.asarray([star.photom for star in self.stars])
        else:
            return []

    @property
    def all_uncertainties(self):
        """
        Light curve of all stars, in the star numbering order (from closest to center to farthest)
        Returns
        -------

        """
        if not self._error_flag:
            return np.asarray([star.uncertainties for star in self.stars])
        else:
            return []

    @property
    def is_empty(self):
        """
        See if the object has been used in a run, i.e. if it has data
        
        Returns
        -------
        boolean:
            True if it is empty or an error has been raised, False otherwise
        """
        if not self.stars:
            return True
        return len(self.stars[0].photom) == 0 or self._error_flag

    # Static methods
    @staticmethod
    def get_rot_mat(angle, clockwise):
        """
        Returns a rotation matrix (around the origin) for a given angle
        :param angle: rotation angle, in degrees
        :param clockwise: True for a clockwise rotation and False for a counter clockwise rotation.
        :return: rotation matrix
        """
        if clockwise:
            return matrix_clockwise(angle)
        else:
            return matrix_cnter_clock(angle)

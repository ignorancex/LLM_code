from typing import List

import numpy as np

from astropy.io import fits
from pyarchi.utils import path_finder
from pyarchi.utils.misc.rotation_mats import matrix_cnter_clock
from pyarchi.utils import create_logger
from pyarchi.data_objects import Star

logger = create_logger("initial_detection")


def centers_from_fits(primary: str, secondary: str, stars: List, initial_angle: float, initial_offset: List, **kwargs):
    """
    Using information stored on the fits files, we determine the centers positions. The centers are determined using
    relations between the differences in RA and DEC of all stars in relation to the known point : the central star.

    After determining the center, we use the primary and secondary arguments to see if this function should change
    the initial position of the star.

    Parameters
    ----------
    primary: str
        Methodology to apply to the central star. If it's fits then the initial position of that star is changed to
        be the one determined here.
    secondary: str
        Methodology to apply to the outer stars. If it's fits then the initial position of those stars are changed
            to be the ones determined here.
    stars: list
        List with all the stars found with the dynam method
    initial_angle: float
        Rotation angle of the satellite for the first image
    initial_offset: list    
        DRP's estimation of the central star location
    kwargs
        kwargs
        
    Returns
    -------
        List:
            Updated list of stars, with the positions determined by the fits method

    """

    if primary != "fits" and secondary != "fits":
        return stars

    to_calculate = []

    if primary == "fits":
        to_calculate += [0]

    if secondary == "fits":
        to_calculate += [1]

    arcsec = 1 / 3600
    arcsec_per_pix = 1
    scaling_factor = kwargs['grid_bg'] / 200 if kwargs['grid_bg'] != 0 else 1

    r_ang = initial_angle

    stars_path = path_finder(mode="stars", **kwargs)

    try:
        hdulist = fits.open(stars_path)
    except IOError:
        logger.error("Star Catalogue file not found")
        return -1
    else:
        with hdulist:
            cent_ra = hdulist[1].header["CENT_RA"]
            cent_dec = hdulist[1].header["CENT_DEC"]
            first_RA = hdulist[1].data["RA"]
            first_DEC = hdulist[1].data["DEC"]
            mags = hdulist[1].data["MAG_CHEOPS"]

    logger.info(
        "Extracted StarCatalogue's information; Starting the position analysis process"
    )

    central_star = initial_offset

    valid_stars = (
        0
    )  # number of valid stars found so far. There are more than the usable ones
    distances = []
    positions = []

    if 0 in to_calculate:  # for the central star
        pos = [central_star[1] - 412, -412 + central_star[0]]
        pos = np.multiply(pos, scaling_factor) + np.floor(scaling_factor / 2)
        stars[valid_stars].change_init_pos(pos)

    for index in range(len(first_DEC)):
        star_ra, star_dec = first_RA[index], first_DEC[index]
        r_mat = matrix_cnter_clock((360 - r_ang) * np.pi / 180)

        # http://spiff.rit.edu/classes/phys373/lectures/astrom/astrom.html
        delta_RA = (cent_ra - star_ra) * np.cos((cent_dec * np.pi / 180)) / arcsec
        delta_DEC = (cent_dec - star_dec) / arcsec

        initial_coords = [[delta_RA], [delta_DEC]]
        new_coords = np.dot(r_mat, initial_coords)

        star_ra = new_coords[0][0]
        star_dec = new_coords[1][0]

        x_pos = 100 - (central_star[1] - 512 + arcsec_per_pix * star_dec)
        y_pos = 100 + central_star[0] - 512 + arcsec_per_pix * star_ra

        if 0 < x_pos < 200 and 0 < y_pos < 200:
            if mags[index] <= 13:
                distances.append(
                    np.sqrt(
                        (x_pos - (central_star[0] - 412)) ** 2
                        + (y_pos - (central_star[1] - 412)) ** 2
                    )
                )
                positions.append([x_pos, y_pos])

    valid_stars = 1
    warning_mismatch = 0
    if len(distances) != len(stars):
        logger.warning("Mismatch between fits and dynam initial detection methods. Possible non-visible stars inside the image!!")
        logger.warning("Creating all of the background stars from fits file; Disregarding data from 'dynam' method")
        warning_mismatch = 1
        stars = stars[:1]
        Star.reset_number(1)

    for dist in sorted(distances[1:]):
        pos = positions[distances.index(dist)]

        if 1 in to_calculate:  # for all other stars
            pos = np.multiply(pos, scaling_factor) + np.floor(scaling_factor / 2)

            if warning_mismatch:  # if the fits and dynam do not match, create from scratch
                stars.append(Star(kwargs["CDPP_type"], positions, dist))
            else:
                stars[valid_stars].change_init_pos(pos)
                valid_stars += 1

    return stars

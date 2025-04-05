from astropy.io import fits
import numpy as np
import os

from pyarchi.utils import get_optimized_mask, create_logger
from pyarchi.utils import path_finder

logger = create_logger("util")


def export_txt(Data_fits, path, **kwargs):
    """
    Stores the light curves on a text file with the following format:
        MJD_TIME;ROLL_ANGLE FLux <Star i> FLUX_ERR <Star i>

    Parameters
    ----------
    Data_fits
        :class:`~pyarchi.data_objects.Data.Data` object. 
    path:
        Path in which the file shall be stored
    kwargs
        Configuration values

    Returns
    -------

    """

    default_path = path_finder(mode="default", **kwargs)
    file_name = "/" + kwargs["base_folder"].split("/")[-2].split("/")[-1]
    try:
        hdulist = fits.open(default_path)
    except IOError:
        logger.error("Lightcurve-Default file not found", exc_info=True)
        return -1
    except Exception as e:
        logger.fatal("Unspecified error. Refer to previous log messages", exc_info=True)
    else:
        with hdulist:
            roll_ang = hdulist[1].data["ROLL_ANGLE"]
            mjd_time = hdulist[1].data["MJD_TIME"]

    off_curve = default_path.split("-")[-1].split("_")[0]

    with fits.open(default_path) as hdulist:
        default_lightcurve = hdulist[1].data["flux"]
        default_err = hdulist[1].data["FLUXERR"]

    header = "mjd_time; roll_ang; {} lightcurve".format(off_curve)
    np.savetxt(
        path + file_name + ".txt",
        np.c_[mjd_time, roll_ang, default_lightcurve, default_err],
        delimiter=" ",
        header=header,
    )

    results_file = os.path.join(path, "pyarchi_output.txt")

    header = "Method - {}\n".format(kwargs["method"])
    header += "Detect mode - {}\n".format(kwargs["detect_mode"])
    header += "Initial load - {}\n".format(kwargs["initial_detect"])
    header += "Background grid - {}\n".format(kwargs["grid_bg"])
    optim_mask = get_optimized_mask(**kwargs)
    header += "Stars information:\n"
    for key, val in optim_mask.items():
        header += "\t Star: {}; Factor: {}; Out of bounds: {}\n".format(
            key, val, Data_fits.stars[int(key)].out_bound
        )

    header += "mjd_time; roll_ang; -> Stars"

    a = Data_fits.all_curves
    b = Data_fits.all_uncertainties
    c = np.empty((a.shape[0] * 2, a.shape[1]), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b

    np.savetxt(
        results_file, np.vstack((mjd_time, roll_ang, c)).T, delimiter=" ", header=header
    )


if __name__ == "__main__":
    pass


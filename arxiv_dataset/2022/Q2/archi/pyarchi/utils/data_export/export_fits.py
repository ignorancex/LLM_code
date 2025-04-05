from astropy.io import fits
import os

from pyarchi.utils import create_logger
from pyarchi.utils import path_finder

logger = create_logger("util")


def create_fits(master_folder, data_fits, **kwargs):
    """
    Export the results to a fits file, containing the light curves and a correspondence of star to Cv and
     respective radius factor. 



    Parameters
    ----------
        master_folder:
            Path in which the data shall be stored
        data_fits
            :class:`~pyarchi.data_objects.Data.Data` object.
        kwargs


    Notes
    -----

        Data stored in the header unit of the file :
            Keyword      data
            
            method        type of mask used
            detect         tracking method 
            initial        initial detection method 
            grid           size of the background grid
            CDPP_TYPE       CDPP algorithm in use
        
       In the data unit of the file, we have each star, with the corresponding
        time, rotation angle, flux values and uncertainties
    """

    # TODO: store more info -> improve organization

    logger.info("Extracting data to FITS file")
    hdus = []
    default_path = path_finder(mode="default", **kwargs)
    try:
        hdulist = fits.open(default_path)
    except IOError:
        logger.fatal("File does not exist")
        return -1
    else:
        with hdulist:
            roll_ang = hdulist[1].data["ROLL_ANGLE"]
            mjd_time = hdulist[1].data["MJD_TIME"]

    col1 = fits.Column(name="MJD_TIME", format="E", array=mjd_time)
    col2 = fits.Column(name="Rotation", unit="deg", format="E", array=roll_ang)

    send_cols = [col1, col2]
    for star in data_fits.stars:
        send_cols.append(fits.Column(name=star.name, format="E", array=star.photom))
        send_cols.append(
            fits.Column(
                name="FLUX_ERR_{}".format(star.number),
                format="E",
                array=star.uncertainties,
            )
        )
    cols = fits.ColDefs(send_cols)

    hdr = fits.Header()
    hdr["method"] = kwargs["method"]
    hdr["detect"] = kwargs["detect_mode"]
    hdr["initial"] = kwargs["initial_detect"]

    hdr["grid"] = kwargs["grid_bg"]
    hdr["CDPPTYPE"] = kwargs["CDPP_type"]

    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdus.append(primary_hdu)
    hdu = fits.BinTableHDU.from_columns(cols, name="Photometry")
    hdus.append(hdu)

    col1 = fits.Column(
        name="Star", format="B", array=[star.number for star in data_fits.stars]
    )
    col2 = fits.Column(
        name="Cv",
        format="E",
        array=[star.calculate_cdpp(data_fits.mjd_time)[0] for star in data_fits.stars],
    )
    col3 = fits.Column(
        name="Factors", format="E", array=[star.mask_factor for star in data_fits.stars]
    )
    col4 = fits.Column(
        name="Out bounds", format="E", array=[star.out_bound for star in data_fits.stars]
    )
    cols = fits.ColDefs([col1, col2, col3, col4])
    hdu = fits.BinTableHDU.from_columns(cols, name="General")
    hdus.append(hdu)

    hdul = fits.HDUList(hdus)

    hdul.writeto(os.path.join(master_folder, "pyarchi_output.fits"), overwrite=True)

    logger.info("Fit file was created")

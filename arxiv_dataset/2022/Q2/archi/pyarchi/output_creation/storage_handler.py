import matplotlib.pyplot as plt
import os

from pyarchi.utils import export_txt, create_fits, photo_SaveInfo
from pyarchi.data_objects import Data

from .photometry_outputs import photom_plots
from pyarchi.utils import handle_folders

from pyarchi.utils import create_logger

logger = create_logger("Output creation")


def store_data(data_fits, job_number, singular=None, **kwargs):
    """
    Stores the data extracted from the entire pipeline.

    Like the controllers, it can work with a :class:`~pyarchi.data_objects.Data.Data` object that was used for both parts
    or only for one of them.

    If one wishes, it can also only create the folders and extract the data only for the chosen star.
   
    Parameters
    ----------
    data_fits:
        :class:`~pyarchi.data_objects.Data.Data` object.
    job_number:
        Job number, from the supernova server. Will be used to save the parent folder inside the
        kwargs["results_folder"] to create the entire directory structure.
    singular:
        If it's not None, then only that star's light curve is saved. Only works for the "show_results".

    kwargs: 
        Dictionary with the configuration values. Can be obtained with the  :class:`~pyarchi.routines.Photo_Controller.Photo_controller` parameters property
    """
    if kwargs["headless"]:
        plt.switch_backend("Agg")

    if not isinstance(data_fits, Data):
        logger.fatal("Data object found an error during run time. No data will be stored")
        return -1

    if data_fits.abort_process:
        logger.fatal("Data object came from an aborted run. No data will be extracted")
        raise RuntimeError(
            "Data object came from an aborted run. No data will be extracted"
        )
    if singular is not None and not isinstance(singular, int):
        logger.fatal("the 'singular' parameter should be an integer")
        raise ValueError("the 'singular' parameter should be an integer")

    master_folder = handle_folders(len(data_fits.stars), job_number, **kwargs)

    if not data_fits.is_empty:
        if data_fits.low_mem:
            data_fits.reload_images(**kwargs)
        photo_SaveInfo(master_folder, data_fits)
        photom_plots(data_fits, master_folder, singular, **kwargs)

        if kwargs["export_text"]:
            logger.info("Starting txt file creation")
            if export_txt(data_fits, master_folder, **kwargs) == -1:
                logger.fatal("File creation has failed")

        if kwargs["export_fit"]:
            logger.info("Starting fits file creation")
            if create_fits(master_folder, data_fits, **kwargs) == -1:
                logger.fatal("Problems were found during fit file creation")

    if kwargs["show_results"]:
        plt.show()

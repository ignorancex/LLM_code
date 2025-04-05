from matplotlib import pyplot as plt
import numpy as np
from pyarchi.star_track import star_tracking_handler
from pathlib import Path
from pyarchi.utils import my_timer
from pyarchi.utils import create_logger

logger = create_logger("main")


@my_timer
def photometry(data_fits, save_folder, **kwargs):
    """
    This function ensures that the entire process runs in the correct order. First the masks are updated and, afterwards,
    the centers for the next image are calculated.

    Parameters
    ------------------
    data_fits: Object of "Data" class that will hold the results of the analysis for each star

    kwargs:
        dictionary with all the necessary information. It is loaded from the config.yaml file

    Returns
    ------------------
        data_fits: Object of "Data" class that holds the results of the analysis for each star
        -1: If an error was found during the process
    """

    if not kwargs[
        "optimize"
    ]:  # avoid excessive writing to log file during optimization process
        logger.info(
            "Starting data analysis process with a grid of {}".format(kwargs["grid_bg"])
        )

    if kwargs["headless"]:
        plt.switch_backend("Agg")
    else:
        plt.switch_backend("TkAgg")

    for index in range(data_fits.image_number):
        data_fits.update_stars(index)
        data_fits.calculate_uncertainties(index)

        if data_fits.abort_process:
            logger.fatal("Errors found during run time")
            return -1

        if (kwargs["plot_realtime"] or kwargs['save_gif']) and not kwargs["optimize"]:
            pnt_mask = np.zeros(data_fits.stars[0].latest_mask.shape)
            for star in data_fits.stars:
                plt.contour(star.latest_mask)

                positions = star.positions[-1]

                pnt_mask[int(positions[0]), int(positions[1])] = 3e7

            img = data_fits.get_image(index).copy()
            plt.imshow(img)

            plt.contour(pnt_mask)

            if kwargs['save_gif']:
                save_folder = Path(save_folder)
                if not save_folder.exists():
                    logger.fatal("Save folder for the gif does not exist!")
                    return -1

                plt.savefig(save_folder / f'gif/images/{str(index)}.png')
            if kwargs['plot_realtime']:
                plt.pause(0.2)

            plt.clf()

        # Update the star's positions for the next frame ##############

        points = star_tracking_handler(data_fits, index, **kwargs)

        if points == -1:
            logger.fatal("Errors found during center determination")
            return -1

    return data_fits

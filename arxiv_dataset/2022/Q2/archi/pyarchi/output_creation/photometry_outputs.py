from matplotlib import pyplot as plt
import os
import numpy as np


from pyarchi.utils import create_logger, create_gif

logger = create_logger("Output creation")


def photom_plots(data_fits, master_folder, singular=None, **kwargs):
    """
    Processes the data collected from the images and outputs the desired graphs.

    Parameters
    ----------
    data_fits:
        :class:`~pyarchi.data_objects.Data.Data` object with all the stars information inside
    singular:
        If it's not None, then only that star's light curve is saved. Only works for the "show_results".
    master_folder:
        Root folder in which all the data shall be stored
        
    kwargs:
        Configuration values that are used to change the data processed and the output data:

    Returns
    -------

    """

    if kwargs["headless"]:
        plt.switch_backend("Agg")

    scaling_factor = kwargs["grid_bg"] / 200 if kwargs["grid_bg"] else 1

    logger.info("Starting the plot process")
    point_size = 3  # size of the dots on the graphs

    colors1 = plt.get_cmap("jet_r")
    x = data_fits.mjd_time

    if kwargs["report_pictures"]:
        fig, ax = plt.subplots(2, 2)
        fig_1, ax_1 = plt.subplots()

        pos = [[0, 0], [0, 1], [1, 0], [1, 1]]

        colors1 = plt.get_cmap("jet_r")

        for j in range(len(data_fits.stars)):
            cdpp = data_fits.stars[j].calculate_cdpp(data_fits.mjd_time)[0]

            ax[pos[j][0], pos[j][1]].scatter(
                x,
                data_fits.stars[j].photom,
                color="black",
                label="Cv = {:.2f}".format(cdpp),
                s=point_size,
            )

            ax[pos[j][0], pos[j][1]].set_title(data_fits.stars[j].name)
            ax[pos[j][0], pos[j][1]].legend(loc=4)

            ax[pos[j][0], pos[j][1]].ticklabel_format(axis="y", style="sci")
            ax[pos[j][0], pos[j][1]].yaxis.major.formatter.set_powerlimits((0, 0))

            # Figure 2:
            ax_1.scatter(
                x,
                data_fits.stars[j].photom,
                c=colors1((5 * j) / 40),
                label=data_fits.stars[j].name,
                zorder=-j,
                s=point_size,
            )

            ax_1.ticklabel_format(axis="y", style="sci")
            ax_1.yaxis.major.formatter.set_powerlimits((0, 0))

        ax_1.legend()
        fig.tight_layout()
        fig_1.tight_layout()

    fig_2, ax_2 = plt.subplots()

    img = data_fits.get_image(0)
    ax_2.imshow(img)
    circles = []
    names = []
    [
        logger.info(
            star.name + " : {} ppm".format(star.calculate_cdpp(data_fits.mjd_time)[0])
        )
        for star in data_fits.stars
    ]
    for j, star in enumerate(data_fits.stars):
        circle_rad = 10
        if kwargs["grid_bg"]:
            circle_rad = int(circle_rad * scaling_factor)

        positions = star.positions[0]
        circle1 = plt.Circle(
            (positions[1], positions[0]),
            circle_rad,
            edgecolor=colors1((5 * j) / 40),
            facecolor="none",
        )

        ax_2.add_artist(circle1)
        circles.append(circle1)
        names.append(data_fits.stars[j].name)
    ax_2.legend(circles, names)

    fig_2.savefig(os.path.join(master_folder, "star_names"))

    if kwargs["debug"]:
        data_fits.stars[0].enable_debug(**kwargs)
        cv, cv_def = data_fits.stars[0].calculate_cdpp(data_fits.mjd_time)
        fig = plt.figure()

        plt.scatter(
            x,
            data_fits.stars[0].default_lightcurve
            / np.max(data_fits.stars[0].default_lightcurve),
            label=kwargs["official_curve"],
            s=point_size,
        )

        plt.scatter(
            x,
            data_fits.stars[0].photom / np.max(data_fits.stars[0].photom),
            label=kwargs["method"],
            s=point_size,
        )

        plt.title(" Normalized lightcurves")
        plt.legend()
        fig_1 = plt.figure()

        plt.scatter(
            x,
            data_fits.stars[0].default_lightcurve,
            label="DRP; Cv = {}".format(round(cv_def, 2)),
            s=point_size,
        )
        plt.scatter(
            x,
            data_fits.stars[0].photom,
            label="pyarchi; Cv = {}".format(round(cv, 2)),
            s=point_size,
        )
        plt.title("Lightcurves")
        plt.legend(bbox_to_anchor=(0.6, 1), ncol=1)

        fig.savefig(os.path.join(master_folder, "Star_0/normalized_curves"))
        fig_1.savefig(os.path.join(master_folder, "Star_0/pyarchi_official_compare"))

    if kwargs["show_results"]:
        # PLot the ligthcurve
        for j in range(len(data_fits.stars)):

            if singular is not None and singular != j:
                continue

            plt.figure()
            cv, _ = data_fits.stars[j].calculate_cdpp(data_fits.mjd_time)

            plt.scatter(
                x,
                data_fits.stars[j].photom,
                label="Method - {}; Cv = {}".format(kwargs["method"], round(cv, 2)),
                s=point_size,
            )
            plt.title("Lightcurve of {}".format(data_fits.stars[j].name))
            plt.legend(loc=1)
            plt.savefig(os.path.join(master_folder, "Star_{}/pyarchi_curve".format(j)))

    if kwargs['save_gif']:
        create_gif(os.path.join(master_folder,'gif/images'),os.path.join(master_folder,'gif'), 'star_tracking.gif' )
    logger.info("All plots were completed")

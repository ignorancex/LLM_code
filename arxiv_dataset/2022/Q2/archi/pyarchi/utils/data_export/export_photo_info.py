import os


def photo_SaveInfo(path, data_fits):
    """
    Stores information related to the photometric run.

    In specific:
        - which file was used
        - mask type
        - detect mode
        - background grid
        - all star related info 
        
    Parameters
    ----------
        path:
            path in which the file will be stored

        data_fits:
             :class:`~pyarchi.data_objects.Data.Data` object.

    Returns
    -------

    """

    with open(os.path.join(path, "photo_info.txt"), mode='w') as file:
        file.write("Used file: {}\n ".format(data_fits.used_file))
        file.write("Mask type: {}\n".format(data_fits.mask_type))
        file.write("Initial mode: {}\n".format(data_fits.init_detection_mode))

        file.write("Detect mode: {}\n".format(data_fits.detect_mode))
        file.write("Background grid: {}\n".format(data_fits.bg_grid))

        file.write("Star \t Cv \t Factors \n")

        for j in data_fits.stars:
            file.write("{} \t {} \t {}\n".format(j.number, j.calculate_cdpp(data_fits.mjd_time)[0], j.mask_factor))

import numpy as np


def static_method(Data_fits, img_number, primary, secondary):
    """
    Rotates the points in the last know position by the corresponding rotation angle (difference between current angle
     and the  one from the last image).

    Parameters
    ----------
    Data_fits:
         :class:`pyarchi.main.initial_loads.Data` object.
    img_number:
         Number of the current image
    primary:
        Methodology to apply to the central star. If it's static then the central star is tracked using this method
    secondary:
        Methodology to apply to the outer stars. If it's static then they are tracked using this method

    Returns
    -------

    """
    if primary != "static" and secondary != "static":
        return

    to_calculate = []
    if primary == "static":
        to_calculate.append(0)

    if secondary == "static":
        to_calculate += [star.number for star in Data_fits.stars[1:]]

    if img_number + 1 < len(Data_fits.roll_ang):
        # FIXME:  no need to go get the rotation matrix from the Data class -> get from utils package
        rot_mat = Data_fits.get_rot_mat(
            ((Data_fits.roll_ang[img_number + 1] - Data_fits.roll_ang[0]) * np.pi / 180),
            clockwise=True,
        )

        scaling_factor = Data_fits.bg_grid / 200 if Data_fits.bg_grid != 0 else 1
        delta_pos = [100, 100]
        delta_pos = np.multiply(delta_pos, scaling_factor) + np.floor(scaling_factor / 2)

        for star in Data_fits.stars:

            pair = star.positions[0].copy()

            coords = np.dot(rot_mat, [[pair[0] - delta_pos[0]], [pair[1] - delta_pos[1]]])
            x = coords[0][0] + delta_pos[0]
            y = coords[1][0] + delta_pos[1]

            if star.number in to_calculate:
                star.positions.append([x, y])

    return 0

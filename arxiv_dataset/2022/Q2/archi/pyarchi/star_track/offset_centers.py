import numpy as np


def rotate_points(Data_fits, img_number, to_calculate):
    if img_number + 1 < len(Data_fits.roll_ang):
        rot_mat = Data_fits.get_rot_mat(
            ((Data_fits.roll_ang[img_number + 1] - Data_fits.roll_ang[0]) * np.pi / 180),
            clockwise=True,
        )

        scaling_factor = Data_fits.bg_grid / 200 if Data_fits.bg_grid != 0 else 1

        delta_pos = [ int(Data_fits.image_size[0]/2),  int(Data_fits.image_size[1]/2)]

        delta_pos = np.multiply(delta_pos, scaling_factor) + np.floor(scaling_factor / 2)

        for index, star in enumerate(Data_fits.stars):
            if index not in to_calculate:
                continue

            pair = star.positions[0].copy()
            coords = np.dot(rot_mat, [[pair[0] - delta_pos[0]], [pair[1] - delta_pos[1]]])

            x = coords[0][0] + delta_pos[0]
            y = coords[1][0] + delta_pos[1]

            if star.number in to_calculate:
                star.positions.append([x, y])


def offsets_method(Data_fits, index, primary, secondary):
    """
    This function expands the functionality of rotate_points. After rotating the points we calculate the offset
    experienced by the central star, by calculating the deviation between the center location from the last two images.
    The offset center for the central star is simply obtained from the fits files, without rotating the previous point.
    
    Parameters
    ---------------
    Data_fits
         :class:`pyarchi.main.initial_loads.Data` object.
    index
        index of the image
    primary:
        Methodology to apply to the central star. If it's offsets then the central star is tracked using this method
    secondary:
        Methodology to apply to the outer stars. If it's offsets then they are tracked using this method

    Returns
    -------

    """

    if primary != "offsets" and secondary != "offsets":
        return

    to_calculate = []

    if secondary == "offsets":
        to_calculate = to_calculate + [star.number for star in Data_fits.stars[1:]]

    scaling_factor = Data_fits.bg_grid / 200 if Data_fits.bg_grid != 0 else 1

    if index + 1 < len(Data_fits.roll_ang):

        offsets = Data_fits.offsets[index + 1]
        off_y = -Data_fits.intended_loc[0] + offsets[0]
        off_x = -Data_fits.intended_loc[1] + offsets[1]

        if primary == "offsets":
            central = [off_x + int(Data_fits.image_size[0]/2), off_y + int(Data_fits.image_size[1]/2)]
            central = np.multiply(central, scaling_factor) + np.floor(scaling_factor / 2)
            Data_fits.stars[0].add_center(central)
            positions_main = Data_fits.stars[0].positions

        else:
            positions_main = Data_fits.offsets

        center_shift = [
            positions_main[index + 1][0] - positions_main[0][0],
            positions_main[index + 1][1] - positions_main[0][1],
        ]

        rotate_points(Data_fits, index, to_calculate)

        for j in range(1, len(Data_fits.stars)):
            if j in to_calculate:
                Data_fits.stars[j].change_center(
                    -1,
                    [
                        Data_fits.stars[j].positions[-1][0] + center_shift[0],
                        Data_fits.stars[j].positions[-1][1] - center_shift[1],
                    ],
                )
    return 0

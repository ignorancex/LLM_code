import numpy as np 


def shape_increase(data, factor, fac=1):
    """
    Increases the boundary of the mask by one pixel. i.e., adds one layer of pixels around the mask present in the data
    passed in.

    Parameters
    --------------
    data:
        Array with the original shape that we wish to expand
    factor:
        number of pixels that we wish to increase
    fac:
        Used internally

    Returns
    -------
        numpy array:
            Increased image
    Notes
    -----

        This is a recursive function to expand by a number of pixels (factor) our image inside the data array. The shape increase
        avoids boundary issues, by avoiding leaving the image edges. 
        This function muss be refactored since it's inefficient. However, for the time being, the overhead that it introduces
        is not enough to justify optimizing it.
    """
    # ToDO: refactor this. IN other words: prettify this
    if factor < 1:
        return data

    positions = np.where(data != 0)  # array positions that we wish to increase

    new = np.zeros(data.shape)

    cases = [[0, 0], [0, 1], [1, 0], [1, 1], [0, -1], [-1, 0], [-1, -1], [1, -1], [-1, 1]]

    max_pos_x = data.shape[0]
    max_pos_y = data.shape[1]

    for pos in zip(positions[0], positions[1]):

        for j in cases:

            val_x = j[0]
            val_y = j[1]

            # avoid breaching image edges
            if pos[0] + val_x < 0:
                val_x = -pos[0]

            elif pos[0] + val_x >= max_pos_x:
                val_x = max_pos - pos[0] - 1

            if pos[1] + val_y < 0:
                val_y = -pos[1]

            elif pos[1] + val_y >= max_pos_y:
                val_y = max_pos - pos[1] - 1

            new[pos[0] + val_x, pos[1] + val_y] = 1

    if fac >= factor:
        return new
    elif fac > 1000:
        # Cap of 1000 iterations
        logger.fatal("Iteration cap was reached. Problems increasing the mask")
        return -1
    else:
        return shape_increase(new, factor, fac + 1)

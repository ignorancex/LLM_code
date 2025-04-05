import numpy as np


def create_circular_mask(img, stars, radius, primary, secondary):
    """
    Defines a circular mask for all the stars, with a pre determined radius. If a size_grid_change is different than
    zero
    it converts the mask to that grid size

    Parameters
    ----------
    stars:
          list of all the  :class:`pyarchi.star_track.Star_class.Star` objects
    img:
        First image
    radius:
        radius of the circles. Can be a general radius(same for all the stars) or a dict with different radii,
    in which  the keys are the indexes and the values are the radius

    size_grid_change:
        size of the bigger grid, to which we can convert the mask. If it's zero we don't convert. Otherwise it is
        converted

    primary:
        Methodology to apply to the central star. If it's fits then the initial position of that star is changed to
        be the one determined here.
    secondary:
        Methodology to apply to the outer stars. If it's fits then the initial position of those stars are changed to
        be the ones determined here.


    Returns
    -------
    masks_dict:
        Dictionary where the keys are the number of the star and the values the corresponding mask
    """

    if primary != "circle" and secondary != "circle":
        return {}

    to_calculate = []
    if primary == "circle":
        to_calculate.append(0)

    if secondary == "circle":
        to_calculate += [star.number for star in stars[1:]]

    xx, yy = np.mgrid[: img.shape[0], : img.shape[0]]

    masks_dict = {}
    for index in range(len(stars)):
        if index not in to_calculate:
            continue

        coords = stars[index].positions[0].copy()
        mask = np.zeros(img.shape)

        if not isinstance(radius, (dict)):
            # if the radius if an integer then we use the same value for all stars -> used for optimizing the process
            mask[
                np.where((xx - coords[0]) ** 2 + (yy - coords[1]) ** 2 <= (radius) ** 2)
            ] = 1

        else:
            # If we have an optimized radius we can use it to give value for each star in the list
            mask[
                np.where(
                    (xx - coords[0]) ** 2 + (yy - coords[1]) ** 2
                    <= (radius[str(index)]) ** 2
                )
            ] = 1

        masks_dict[index] = mask

    return masks_dict

from .dynamical_centers import dynam_method
from .offset_centers import offsets_method
from .static_method import static_method


def star_tracking_handler(Data_fits, index, **kwargs):
    """
    Handles the detection mode for the target star and the ones around it. Allows to have two different modes of
    detection active at the same time.

    Parameters
    ----------
    Data_fits:
         :class:`pyarchi.main.initial_loads.Data` object.
    index:
        Image number
    kwargs

    Returns
    -------
    int
        Error code of the different star tracking methods. 0 if everything as expected
        1 otherwise

    """

    if "+" not in kwargs["detect_mode"]:
        primary = secondary = kwargs["detect_mode"]
    else:
        primary, secondary = kwargs["detect_mode"].split("+")

    results = static_method(Data_fits, index, primary, secondary)

    if results == -1:
        return -1

    results = dynam_method(Data_fits, index, primary, secondary, kwargs['repeat_removal'])

    if results == -1:
        return -1

    results = offsets_method(Data_fits, index, primary, secondary)
    return results

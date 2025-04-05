import os


def parameters_validator(**kwargs):
    """
    Loads the configuration parameters from the .yaml file. After loading,
    it  checks some of the parameters to see if they have valid values.

    Parameters
    ----------
    parameters

    Returns
    -------

    """
    wrong_params = []
    warnings = []
    paths_to_test = ["base_folder", "optimized_factors", "results_folder"]

    for key in paths_to_test:
        if not os.path.exists(kwargs[key]):
            wrong_params.append(key)

    if kwargs["official_curve"] not in ["DEFAULT", "OPTIMAL", "RINF", "RSUP"]:
        wrong_params.append(kwargs["official_curve"])

    if kwargs['data_type'] not in ['real', 'simulated']:
        wrong_params.append(kwargs['data_type'])

    for param in ["method", "detect_mode", "initial_detect"]:
        if type(kwargs[param]) is not str:
            wrong_params.append(param)
            continue

        if "+" in kwargs[param]:
            modes = kwargs[param].split("+")
        else:
            modes = [kwargs[param]]

        if param == "method":
            valid_values = ["circle", "shape"]
        if param == "detect_mode":
            valid_values = ["dynam", "offsets", "static"]
        elif param == "initial_detect":
            valid_values = ["dynam", "fits"]

        for md in modes:
            if md not in valid_values and param not in wrong_params:
                wrong_params.append(param)

    if not isinstance(kwargs["val_range"], list):
        if kwargs["optimize"]:
            wrong_params.append("val_range")
        else:
            warnings.append("val_range")
    else:
        optim_vals = kwargs["val_range"]
        if optim_vals[1] <= optim_vals[0] < 0 or optim_vals[1] < 0:
            if kwargs["optimize"]:
                wrong_params.append("val_range")
            else:
                warnings.append("val_range")

    if (kwargs["grid_bg"] / 200) % 2 != 1 and kwargs["grid_bg"] != 0:
        wrong_params.append("grid_bg")

    return wrong_params, warnings, kwargs

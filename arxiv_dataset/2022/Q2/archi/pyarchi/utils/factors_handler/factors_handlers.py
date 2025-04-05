import numpy as np
import json
import os

from pyarchi.utils import create_logger

logger = create_logger("util")

def get_file_name(path, storage_location):
    """
    Extracts the last part of the path, i.e. the dataset folder name, to store the optimized masks

    Example
    ---------
        >>> get_file_name("kamuish/teste")  
        >>> "teste"

        >>> get_file_name("kamuish/teste/")  
        >>> "teste"
    Parameters
    ----------
    path : string
        path to the base folder
    """

    
    splitted_path = path.split("/")
    file_name = splitted_path[-2] if not splitted_path[-1] else splitted_path[-1]
    json_full_path = os.path.join(storage_location, "{}.json".format(file_name))

    return json_full_path

def store_optimized_radius(new_dict, **kwargs):  # TODO: improve save for method combination
    """
    Updates the json file in which the optimization info is stored

    :param new_dict: dict wth the format { star index : radius }
    :return: nothing
    """

    logger.info("Storing optimized radius in the json file")

    json_path = kwargs["optimized_factors"]
    method = kwargs["method"]
    detect_mode = kwargs["detect_mode"]
    size_bg_grid = str(kwargs["grid_bg"])
    initial_method = kwargs["initial_detect"]

    json_full_path = get_file_name(kwargs["base_folder"], kwargs["optimized_factors"])

    print(json_full_path)
    if os.path.exists(json_full_path):
        with open(json_full_path, "r") as data_file:
            try:
                data_loaded = json.load(data_file)
            except Exception as e:
                logger.critical(
                    "Problem opening the json file with the optimized radius. Could not save data to file",
                    exc_info=True,
                )
                return -1

        with open(json_full_path, "w") as data_file:
            try:
                for key, value in data_loaded[detect_mode][method][
                    size_bg_grid
                ].items():  # saving over old values
                    if not np.isnan(new_dict[key]):
                        data_loaded[detect_mode][method][size_bg_grid][initial_method][
                            key
                        ] = new_dict[key]
            except KeyError as e:
                try:  # initial_method does not exist
                    data_loaded[detect_mode][method][size_bg_grid][
                        initial_method
                    ] = new_dict

                except KeyError as e:  # bg_grid does not exist
                    try:
                        data_loaded[detect_mode][method][size_bg_grid] = {
                            initial_method: new_dict
                        }
                    except KeyError as e:
                        try:
                            data_loaded[detect_mode][method] = {
                                size_bg_grid: {initial_method: new_dict}
                            }
                        except KeyError as e:
                            data_loaded[detect_mode] = {
                                method: {size_bg_grid: {initial_method: new_dict}}
                            }

            json.dump(data_loaded, data_file, indent=4)

    else:
        logger.info("Json file with optimized radii not found. Creating new file")

        with open(json_full_path, "w") as data_file:
            json.dump(
                {detect_mode: {method: {size_bg_grid: {initial_method: new_dict}}}},
                data_file,
                indent=4,
            )

    return 0


def get_optimized_mask(**kwargs): # TODO: improve load for method combination
    """
    Retrieves the optimized radius from the json file. From this file we load a dict in which each key is
    the method name and the associated value is a dict with information about each star

    Parameters
    ----------
    kwargs
        - method : used to retrieve the optimization parameters from the json file

    Returns
    -------
    data_loaded:
        dict in which the keys are the index of each star and the values are the radius for it. If the method name does not exist then it's returned a empty dict
    """

    logger.info("Retrieving optimized factors")

    json_path = kwargs["optimized_factors"]
    json_full_path = get_file_name(kwargs["base_folder"],kwargs["optimized_factors"])

    if os.path.exists(json_full_path):
        with open(json_full_path, "r") as data_file:
            try:
                data_loaded = json.load(data_file)
            except Exception as e:
                logger.critical(
                    "Json file with optimized radius not found", exc_info=True
                )
                return -1
            else:

                try:
                    return data_loaded[kwargs["detect_mode"]][kwargs["method"]][
                        str(kwargs["grid_bg"])
                    ][kwargs["initial_detect"]]
                except KeyError as e:
                    logger.critical(
                        "Method {} with grid {} and detect mode not found in the json file \n "
                        "You should run the optimization process before analysing the star."
                        "".format(
                            kwargs["method"], kwargs["grid_bg"], kwargs["detect_mode"]
                        ),
                        exc_info=True,
                    )

                    return -1

    else:
        with open(json_full_path, "w") as data_file:
            logger.fatal("JSON file did not exist. created an empty one")
            return -1


if __name__ == "__main__":
    import yaml

    with open(
        "/home/andre/PycharmProjects/CHEOPS_mult_dtct/configs/config.yaml", "r"
    ) as stream:
        kwargs = yaml.safe_load(stream)

    x = get_optimized_mask(**kwargs)


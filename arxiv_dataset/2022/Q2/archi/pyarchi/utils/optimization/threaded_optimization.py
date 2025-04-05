import numpy as np
import os
import logging

from .optimizer import optimizer
from .circular_fine_tune import circular_tuner

from pyarchi.utils import create_logger

logger = create_logger("utils")


def general_optimizer(func, data_f, job_number, max_process, **kwargs):
    """
    Responsible for setting up the variables used during the optimization process.

    If the factor determined to be the best one is near the upper limit, then the range of values is extended and a new
    search for minimum noise starts. If the determined factor is inside a "safe" distance away from the highest possible
    value then the star is disabled and no further studies are made on it.

    Also responsible for creating .txt files with all the relevant information
    Parameters
    ----------
    func:
        function that launches the photometric process
    data_f:
        :class:`pyarchi.main.initial_loads.Data`  object with all the stars information inside
    job_number:
        JOb number assigned by the SLURM workload manager
    max_process:
        Maximum number of processes that can be launched during this routine
    Returns
    -------
    optimized_dict:
        Dictionary in which the keys are the star numbers and the values are the optimal factors
    """
    logger.info("Optimization with grid of: {}".format(kwargs["grid_bg"]))

    low, high = kwargs["val_range"]
    step = kwargs["step"]
    logging.disable(logging.INFO)

    value_range = np.arange(low, high + step, step)
    path = os.path.join(kwargs["results_folder"], str(job_number))
    if not os.path.exists(path):
        os.mkdir(path)

    file_path = os.path.join(path, "optimization_info.txt")

    with open(file_path, mode="w") as file:
        file.write("Method - {}\n".format(kwargs["method"]))
        file.write("Detect mode - {}\n".format(kwargs["detect_mode"]))
        file.write("Initial detection mode - {}\n".format(kwargs["initial_detect"]))
        file.write("Background grid - {}\n".format(kwargs["grid_bg"]))
        file.write("Factors Stars: -> \n")

    try:
        min_cvs, optimized_dict = optimizer(
            value_range, max_process, func, data_f, file_path, **kwargs
        )
    except:
        logger.fatal("Problem on the optimization routine. Refer to logs")
        return -1

    increase = high - low
    iterations = 0
    to_disable = []

    while len(to_disable) != len(optimized_dict.values()):
        to_disable = []

        for key, val in optimized_dict.items():
            if val < high - 2 * step:  # allow for some tolerance near the border
                to_disable.append(int(key))

        print(
            "DISABLING: {} - Last run in the interval: {} <-> {}".format(
                to_disable, low, high
            )
        )
        if len(to_disable) == len(optimized_dict):
            logger.fatal("All stars were disabled")
            break

        low, high = high - 2 * step, high + increase - 2 * step
        value_range = np.arange(low, high + step, step)
        min_cvs_tmp, optimized_dict_tmp = optimizer(
            value_range, max_process, func, data_f, file_path, to_disable, **kwargs
        )

        for (star,cdpp,) in min_cvs_tmp.items():  # update old values if the new ones are better

            if cdpp < min_cvs[star]:
                min_cvs[star] = cdpp
                optimized_dict[str(star)] = optimized_dict_tmp[str(star)]

        iterations += 1
        if iterations > kwargs["optimization_extensions"]:
            logger.fatal("Reached max iters. Last values: {} {}".format(low, high))
            break

    if kwargs["fine_tune_circle"] and "circle" in kwargs["method"]:
        vals = kwargs["method"].split("+")

        to_disable = []

        if len(vals) == 1 and vals[0] == "circle":
            to_disable = []

        elif vals[0] == "circle":
            to_disable = [i for i in data_f.stars[1:].number]
        elif len(vals) != 1 and vals[1] == "circle":
            to_disable = [0]

        min_cvs_tmp, optimized_dict_tmp = circular_tuner(
            best_values = optimized_dict.values(),
            max_process = max_process,
            func = func,
            data_f = data_f,
            file_path = file_path,
            to_disable = to_disable,
            **kwargs
        )

        for (star, cdpp,) in min_cvs_tmp.items():  # update old values if the new ones are better
            if cdpp < min_cvs[star]:
                min_cvs[star] = cdpp
                optimized_dict[str(star)] = optimized_dict_tmp[str(star)]

    logging.disable(logging.NOTSET)

    logger.info("Max factor used: {}".format(high))
    [
        logger.info(
            "Star {} -  Factor: {} - Noise: {} ppm".format(
                index, optimized_dict[index], min_cvs[int(index)]
            )
        )
        for index in optimized_dict
    ]

    return optimized_dict

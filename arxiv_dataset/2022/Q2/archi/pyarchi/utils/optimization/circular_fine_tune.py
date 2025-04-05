import numpy as np
from multiprocessing import Process, Queue

from pyarchi.utils import create_logger

logger = create_logger("utils")


def run_function(queue, func, factors, data_f, to_disable=[], **kwargs):
    """
    Run each interaction of the function and  returns the results ordered on a dictionary

    Parameters
    ----------
    queue
    func
    factors

    Returns
    -------

    """
    kwargs_optim = kwargs.copy()

    kwargs_optim["debug"] = 0
    kwargs_optim["show_results"] = 0
    kwargs_optim["plot_realtime"] = 0

    kwargs_optim["report_pictures"] = 0
    kwargs_optim["show_results"] = 0
    kwargs_optim["export_text"] = 0
    kwargs_optim["export_fit"] = 0
    kwargs_optim["low_memory"] = 1
    kwargs_optim["uncertainties"] = 0

    try:
        results_dict = {i: {} for i in range(len(factors[0]))}
    except:
        print("oh no")
        return
    vals_dict = {str(i): val for i, val in enumerate(factors[0])}
    data_f.load_parameters(vals_dict, **kwargs_optim)

    for star_ind in to_disable:
        data_f.disable_star(star_ind)

    for index, fac in enumerate(factors):

        if index != 0:
            vals_dict = {str(i): val for i, val in enumerate(fac)}

            data_f.load_parameters(vals_dict, **kwargs_optim)

        data_fits = func(DataFits=data_f, factor=vals_dict, **kwargs_optim)

        for ind, star in enumerate(data_fits.stars):

            star_noise = star.calculate_cdpp(data_fits.mjd_time)[0]
            if star.out_bound:
                star_noise = float("nan")

            results_dict[ind][star.mask_factor] = star_noise

    queue.put(results_dict)
    return


def circular_tuner(
    best_values, max_process, func, data_f, file_path, to_disable=[], **kwargs
):
    """
    Decides which factors will be used in each process that it spawns. After the processes are done, extracts the
    resulting data from the Queue and parses it, in order to find the noise values and mask factors. Writes to a .txt
    file the resulting values for each factor. Take note that the values are not guaranteed to be in order, since they
    are saved in the order "imposed" by the processes.
    Parameters
    ----------
    value_range:
        Lower and upper limit of the values to be tested
    max_process:
        Maximum number of processes that can be launched during this routine
    func:
        function that launches the photometric process
    data_f:
        :class:`pyarchi.main.initial_loads.Data`  object with all the stars information inside
    file_path:
        Path in which run time information shall be stored
    kwargs

    Returns
    -------
    min_cvs:
        list with the minimum noise found

    optimized_dict:
        Dictionary in which the keys are the star numbers and the values are the optimal factors
    """

    vals = [np.linspace(int(i) - 1, int(i) + 1, num=21) for i in best_values]
    value_range = np.asarray(list(zip(*vals)))

    comms_queue = Queue()
    process_to_spawn = (
        max_process if value_range.shape[0] >= max_process else value_range.shape[0]
    )

    logger.info(
        "Optimizer going to spawn {} processes, for values: {}".format(
            process_to_spawn, value_range
        )
    )
    # Divide into equal lists the factors to be used
    splitted_values = np.array_split(value_range, process_to_spawn)

    for k in range(process_to_spawn):
        p = Process(
            target=run_function,
            args=(comms_queue, func, splitted_values[k], data_f, to_disable),
            kwargs=kwargs,
        )
        p.start()

    factors = []
    cvs = []

    for ii in range(process_to_spawn):

        data = comms_queue.get()
        if (
            ii == 0
        ):  # For the first run, create list with list for each star, in which the noise values will be stored
            cvs = [[] for _ in range(len(data.keys()))]
            factors = [[] for _ in range(len(data.keys()))]

        for star, star_dict in data.items():
            for fac, noise in star_dict.items():
                factors[star].append(fac)
                cvs[star].append(noise)

    factors = np.array(factors)
    optimized_dict = {}

    min_cvs = {}
    for index, star_list in enumerate(cvs):
        min_cv = np.nanmin(star_list)

        if np.isnan(min_cv):
            optimized_dict[str(index)] = 0
            min_cvs[index] = 2e7
            continue

        min_cvs[index] = min_cv
        min_index = np.where(star_list == min_cv)

        optimal_factor = factors[index][min_index]

        optimized_dict[str(index)] = optimal_factor[0]

    return min_cvs, optimized_dict


if __name__ == "__main__":
    pass


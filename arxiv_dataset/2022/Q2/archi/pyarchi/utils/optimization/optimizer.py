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

    results_dict = {}
    data_f.load_parameters(factors[0], **kwargs_optim)

    for star_ind in to_disable:
        data_f.disable_star(star_ind)

    for index, fac in enumerate(factors):

        if index != 0:
            data_f.load_parameters(fac, **kwargs_optim)
        
        if data_f.abort_process:
            queue.put(-1)
            return -1 
        data_fits = func(DataFits=data_f, factor=fac, **kwargs_optim)

        star_results = {}
        for ind, star in enumerate(data_fits.stars):
            star_results[ind] = star.calculate_cdpp(data_fits.mjd_time)[0]

            if star.out_bound:
                star_results[ind] = float("nan")

        results_dict[fac] = star_results

    queue.put(results_dict)
    return


def optimizer(value_range, max_process, func, data_f, file_path, to_disable=[], **kwargs):
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

    # elements not inside the sublists

    logger.debug(splitted_values)
    
    workers = []
    for k in range(process_to_spawn):
        p = Process(
            target=run_function,
            args=(comms_queue, func, splitted_values[k], data_f, to_disable),
            kwargs=kwargs,
        )
        p.start()
        workers.append(p)

    factors = []
    cvs = []

    for ii in range(process_to_spawn):

        data = comms_queue.get()
        if data == -1:
            logger.fatal("Error in the worker. Shutting down optimization process")
            comms_queue.close()
            [p.terminate() for p in workers]
            raise Exception("Errors in the optimization routine")
        if (
            ii == 0
        ):  # For the first run, create list with list for each star, in which the noise values will be stored
            cvs = [[] for _ in range(len(data[list(data.keys())[0]]))]

        for factor, stars_dict in data.items():
            factors.append(factor)

            for key, CV in stars_dict.items():
                cvs[key].append(CV)

    factors = np.array(factors)
    optimized_dict = {}

    min_cvs = {}
    for index, star_list in enumerate(cvs):
        min_cv = np.nanmin(star_list)
        if np.isnan(min_cv):
            optimized_dict[str(index)] = 1
            min_cvs[index] = 2e7
            continue

        min_cvs[index] = min_cv
        min_index = np.where(star_list == min_cv)

        optimal_factor = factors[min_index]

        optimized_dict[str(index)] = int(optimal_factor[0])

    with open(file_path, mode="a") as file:

        for index, fac in enumerate(factors):
            cdpps = ""
            for star_list in cvs:
                cdpps += str(star_list[index]) + " "

            file.write("{} \t {} \n".format(fac, cdpps))

    return min_cvs, optimized_dict

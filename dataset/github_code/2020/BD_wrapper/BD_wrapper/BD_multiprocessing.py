import multiprocessing
import signal
import sys
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from BD_wrapper.BD_wrapper import BayesianDistance

#  With Python 3.8 the start method for multiprocessing defaults to 'spawn' for
#  MacOS systems. Here we change it back to 'fork' for compatibility reasons.
if sys.version_info[:2] >= (3, 8):
    multiprocessing.set_start_method('fork', force=True)


def init_worker():
    """ Worker initializer to ignore Keyboard interrupt """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def init(bd_list):
    global ilist, bd_object, bd_data
    bd_object, bd_data = bd_list
    ilist = np.arange(len(bd_data))


def determine_distance(i):
    result = BayesianDistance.determine(bd_object, bd_data[i], i)
    return result


def get_cartesian_coords(i):
    result = BayesianDistance.get_cartesian_coords(bd_object, bd_data[i])
    return result


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def func(task='determine_distance', use_ncpus=None):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    # p = multiprocessing.Pool(ncpus, init_worker)
    if use_ncpus is None:
        use_ncpus = int(0.75 * ncpus)
    print('Using {} of {} cpus'.format(use_ncpus, ncpus))
    try:
        if task is 'determine_distance':
            results_list = parallel_process(ilist, determine_distance,
                                            n_jobs=use_ncpus)
            # results_list = p.map(determine_distance, tqdm(ilist))
        elif task is 'get_cartesian_coords':
            results_list = parallel_process(ilist, get_cartesian_coords,
                                            n_jobs=use_ncpus)
            # results_list = p.map(get_cartesian_coords, tqdm(ilist))
    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        quit()
    return results_list

'''
This file allows the allocation an arbitrary number of GPUs
based on the ones available. If someone is using one or
many GPUs, then the remaining ones will be allocated for
our purpose. In order to check which ones are being used,
the memory allocation must be more than a given threshold,
defined by the variable 'threshold'.
'''
import os
import subprocess
import numpy as np

def list_used_memory():
    ''' Returns the memory allocated for each of the GPUs.
    '''
    gpus_memory_used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
    gpus_memory_used = list(map(int, gpus_memory_used.decode('ascii').split('\n')[:-1]))
    return np.array(gpus_memory_used)

def list_available_gpus(threshold=15):
    ''' Returns the IDs of the gpus available.
    Args:
        threshold: Smallest amount of memory to be considered used (MiB)
    '''
    gpus_memory = list_used_memory()
    # Which GPUs are under the threshold
    available_gpus, = np.nonzero(gpus_memory < threshold)
    return available_gpus

def reserve_gpus(count=1, threshold=15):
    '''Will reserve available GPUS.
    Args:
        count: number of GPUs to reserve.
    '''
    available_gpus = list_available_gpus(threshold)
    # Is there enough GPUs for our need?
    if count > len(available_gpus):
        print('Not enough GPUs available.')
        count = len(available_gpus)
    # Transformation of the list of available GPUs to a string
    gpu_id = ','.join(map(str, available_gpus[:count]))
    # Setting the available GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    return available_gpus[:count]
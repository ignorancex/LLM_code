import os
import subprocess
import time
from pathlib import Path
    
def get_gpu_uuid2id_map():
    try:
        # Execute the nvidia-smi command with the --query-gpu option
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_uuid,index', '--format=csv'])

        # Decode the output and split it into lines
        output = output.decode('utf-8')
        lines = output.strip().split('\n')

        # Extract the header and data
        header = lines[0].split(', ')
        data = [line.split(', ') for line in lines[1:]]

        # Create a dictionary to map GPU UUIDs to GPU IDs
        gpu_id_map = {gpu_info[0]: int(gpu_info[1]) for gpu_info in data}
        gpu_id_rev_map = {int(gpu_info[1]): gpu_info[0] for gpu_info in data}

        # Look up the GPU ID for the given GPU UUID
        return gpu_id_map, gpu_id_rev_map

    except subprocess.CalledProcessError as err:
        print("Failed to run nvidia-smi:", err)
        return -1

GPU_UUID2ID_MAP = get_gpu_uuid2id_map()[0]

def first_free_gpu(gpu_uuid2id_map=GPU_UUID2ID_MAP):
    try:
        used_gpus = get_used_gpus(gpu_uuid2id_map=gpu_uuid2id_map)

        num_gpus = len(gpu_uuid2id_map)
        for gpu_id in range(num_gpus):
            if gpu_id not in used_gpus:
                return gpu_id

        # If all GPUs are occupied, return None
        return None

    except subprocess.CalledProcessError as err:
        print("Failed to run nvidia-smi:", err)
        return None
    
def get_used_gpus(gpu_uuid2id_map=GPU_UUID2ID_MAP):
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv'])
        output = output.decode('utf-8')
        lines = output.strip().split('\n')[1:]  # Skip header

        used_gpus = set()
        for line in lines:
            gpu_uuid, pid = line.split(', ')
            gpu_id = gpu_uuid2id_map.get(gpu_uuid, None)
            if gpu_id is not None:
                used_gpus.add(gpu_id)
        
        return used_gpus

    except subprocess.CalledProcessError as err:
        print("Failed to run nvidia-smi:", err)
        return None
    
def wait_and_run(job_arr, outfile, errfile, 
                 gpu_id=None, gpu_uuid2id_map=GPU_UUID2ID_MAP, 
                 init_sleep=None, prev_gpu=None):
    if not Path(outfile).parent.exists():
        Path(outfile).parent.mkdir(parents=True)
    if not Path(errfile).parent.exists():
        Path(errfile).parent.mkdir(parents=True)
    
    if init_sleep is not None:
        time.sleep(init_sleep)
    if gpu_id is None:
        gpu_id = first_free_gpu(gpu_uuid2id_map=gpu_uuid2id_map)
        while gpu_id is None:
            time.sleep(10)
            gpu_id = first_free_gpu(gpu_uuid2id_map=gpu_uuid2id_map)
        
        if gpu_id == prev_gpu: # gpu_id is guaranteed to be not None here
            # Just give the previous process some more time to show up on gpu 
            return wait_and_run(job_arr, outfile, errfile, 
                                gpu_id=gpu_id, gpu_uuid2id_map=gpu_uuid2id_map, 
                                init_sleep=2*init_sleep, prev_gpu=None)
    else:
        used_gpus = get_used_gpus(gpu_uuid2id_map=gpu_uuid2id_map)
        if gpu_id in used_gpus:
            raise ValueError('GPU {} is already in use!'.format(gpu_id))
    
    curr_env = os.environ.copy()
    curr_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    proc = subprocess.Popen(
        job_arr, stdout=open(outfile, 'w'), stderr=open(errfile, 'w'), env=curr_env)
    
    return proc, gpu_id
    
def hold_gpu():
    import torch
    A = torch.tensor(0).cuda()
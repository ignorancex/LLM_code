from collections import namedtuple
import subprocess
from functools import lru_cache
from typing import Optional, Union

from psutil import Process


@lru_cache
def query_gpu_memory():
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc.check_returncode()
    return [int(line) for line in proc.stdout.strip().splitlines()]


@lru_cache
def query_gpu_uuid():
    proc_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=gpu_uuid",
            "--format=csv,noheader",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc_result.check_returncode()
    return [line for line in proc_result.stdout.strip().splitlines()]


def query_free_memory():
    proc_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc_result.check_returncode()
    return [int(line) for line in proc_result.stdout.strip().splitlines()]


def query_proc_used_memory(
    pid_or_proc: Optional[Union[int, Process]] = None,
    device_idx: Optional[int] = None,
    include_children: Optional[bool] = None,
) -> Union[int, list]:
    """
    Query the GPU memory usage of a specific process or all processes. Optionally, include the memory usage of the process's children.

    Args:
        pid_or_proc (int or psutil.Process, *optional*):
            The process ID (PID) or a Process object to query. If None, queries the memory usage for all processes. Defaults to None.
        device_idx (int, *optional*): 
            The GPU device index to filter by. If None, considers all GPUs. Defaults to None.
        include_children (bool, *optional*):
            If True, includes the memory usage of child processes. Defaults to False.

    Returns:
        Union[int, list]: 
            If a specific process is queried, returns the total GPU memory used by the process (and optionally its children) on the specified GPU. 
            If no process is specified, returns a list of all processes with their respective memory usage details.
    """
    proc_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc_result.check_returncode()

    uuid_to_gpu_idx = {
        gpu_uuid: index for index, gpu_uuid in enumerate(query_gpu_uuid())
    }
    ProcMemoUsage = namedtuple("ProcMemoUsage", ["pid", "device_idx", "used_memory"])

    all_proc_used_memory = [
        ProcMemoUsage(int(pid), uuid_to_gpu_idx[gpu_uuid.strip()], int(used_memory))
        for line in proc_result.stdout.strip().splitlines()
        for pid, gpu_uuid, used_memory in [line.split(",")]
    ]

    if pid_or_proc is None:
        return all_proc_used_memory

    proc = Process(pid_or_proc) if isinstance(pid_or_proc, int) else pid_or_proc
    if not isinstance(proc, Process):
        raise ValueError(
            f"pid_or_proc should be int or psutil.Process, but got {type(proc)}"
        )

    child_proc_id = (
        [p.pid for p in proc.children(recursive=True)] if include_children else []
    )

    return sum(
        memory_usage
        for pid, gpu_idx, memory_usage in all_proc_used_memory
        if (pid == proc.pid or pid in child_proc_id)
        and (device_idx is None or gpu_idx == device_idx)
    )

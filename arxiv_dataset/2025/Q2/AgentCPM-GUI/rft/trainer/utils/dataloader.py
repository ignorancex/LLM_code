import zmq
import threading
import multiprocessing
import queue
from typing import Iterator, Any, Callable, Optional, List
import pickle
from torch.utils.data import Sampler
from collections import defaultdict
import torch.distributed as dist

class GlobalDistributed0MQDataLoader:
    def __init__(
        self,
        dataset: Any,
        global_sync_address: str,
        batch_size: int,
        collate_fn: Callable,
        num_workers: int,
        sampler: Sampler,
        worker_init_fn: Callable,
        prefetch_factor: int,
        world_size:int,
        **kwargs: Any
    ):
        self.dataset = dataset
        self.global_sync_address = global_sync_address
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.sampler = sampler
        self.worker_init_fn = worker_init_fn
        self.prefetch_factor = prefetch_factor
        self.kwargs = kwargs
        self._init_kwargs = kwargs
        self.world_size = world_size
        self.rank = dist.get_rank()

        self.index_queue = multiprocessing.Queue(self.num_workers)
        self.result_queue = multiprocessing.Queue(self.prefetch_factor)
        
        self.workers = [ multiprocessing.Process(
            target=GlobalDistributed0MQDataLoader._load_data,
            args=(
                self.dataset,
                self.index_queue,
                self.result_queue,
                self.collate_fn,
                self.worker_init_fn
            ),
            daemon=True
        )  for _ in range(self.num_workers) ]
        
        for p in self.workers:
            p.start()
        
        if self.rank == 0:
            self.master_proc = multiprocessing.Process(
                target=GlobalDistributed0MQDataLoader._master_loop,
                args=(self.global_sync_address, self.batch_size,self.sampler),
                daemon=True
            )
            self.master_proc.start()
    
    def __len__(self):
        return len(self.sampler) // self.world_size
    
    @staticmethod
    def _master_loop(
        global_sync_address: str,
        batch_size: int,
        sampler: Sampler,
    ):
        '''Master loop to dispatch tasks to workers'''
        zctx = zmq.Context()
        task_dispatcher = zctx.socket(zmq.REP)
        task_dispatcher.bind(global_sync_address)
        
        it = None
        
        multiturn_cache = queue.PriorityQueue()
        cached_completions = defaultdict(list)
        
        while True:
            req: str | list[dict] | dict = task_dispatcher.recv_pyobj()
            if isinstance(req,str):
                if req == "REQ_TASK":
                    tasks = []
                    for _ in range(min(multiturn_cache.qsize(),batch_size)):
                        tasks.append(multiturn_cache.get()[1])

                    while len(tasks) < batch_size:
                        try:
                            tasks.append(next(it))
                        except StopIteration:
                            it = iter(sampler)
                            print("Restart Sampler During Epoch")
                
                    task_dispatcher.send(pickle.dumps(tasks))
                
                elif req == "RESTART":
                    it = iter(sampler)
                    task_dispatcher.send_string("RESTARTED")
                
                else:
                    raise NotImplementedError(f"Receive Unknown Request {req}")
            elif isinstance(req,dict):
                if "get" in req:
                    index = req["get"]
                    if len(cached_completions[index]) > 0:
                        d = cached_completions[index].pop(0)
                        if not req["pop"]:
                            cached_completions[index].append(d)
                        # print(f"Cache Completions: {d}")
                    else:
                        d = ""
                        print(f"Error: Illegal access of cache completions at index {index}.")
                    task_dispatcher.send_string(d)
                else:
                    raise NotImplementedError(f"Receive Unknown Request Type {type(req)}")

            elif isinstance(req,list):
                for d in req:
                    multiturn_cache.put((d["gid"],d["next_id"]))
                    cached_completions[d["id"]].append(d["completion"])
                task_dispatcher.send_string("Received")
                
                # counts = {}
                # for k,v in cached_completions.items():
                #     if len(v) > 0:
                #         counts[k] = len(v)
                # print(f"Cache Completions: {counts}")
                
            else:
                raise NotImplementedError(f"Receive Unknown Request Type {type(req)}")

    @staticmethod
    def _load_data(
        dataset: Any,
        index_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        collate_fn: Callable,
        worker_init_fn: Callable,
    ):
        worker_init_fn(None)
        
        while True:
            indices = index_queue.get()
            
            data = [dataset[index] for index in indices]

            result_queue.put(collate_fn(data))

    def __iter__(self):
        '''Get tasks from the master and load the data into the queue'''
        zctx = zmq.Context()
        task_receiver = zctx.socket(zmq.REQ)
        task_receiver.connect(self.global_sync_address)
        
        
        if self.rank == 0:
            task_receiver.send_pyobj("RESTART")
            task_receiver.recv()        
        dist.barrier()
        
        def get_task():
            while True:
                task_receiver.send_pyobj("REQ_TASK")
                tasks = pickle.loads(task_receiver.recv(copy=False))
                if tasks is None:
                    self.result_queue.put(None)
                    break
                self.index_queue.put(tasks)
        task_get_thd = threading.Thread(target=get_task,daemon=True)
        task_get_thd.start()
        
        
        while True:
            data = self.result_queue.get()
            if data is None:
                break
            yield data

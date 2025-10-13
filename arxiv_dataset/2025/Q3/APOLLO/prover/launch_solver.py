import os
import copy
import time
import warnings
import argparse

import torch

from prover.workers import DataLoader, Scheduler, ProcessScheduler, GeneratorProcess, SearchProcess
from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import get_datetime, load_config, AttrDict


def launch_llm(data_path, config, log_dir, node_rank=0, world_size=1):

    cfg = load_config(config)
    os.makedirs(log_dir, exist_ok=True)

    max_model_len = cfg.get('max_model_len', None)

    ngpus = torch.cuda.device_count()
    assert ngpus >= 1
    
    # create data loader
    data_loader = DataLoader(
        data_path=data_path,
        data_split=cfg.get('data_split', None),
        data_repeat=cfg.get('data_repeat', 1),
        node_rank=node_rank,
        world_size=world_size,
        log_dir=log_dir,
    )

    # build Lean verifier
    verifier_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=cfg.lean_max_concurrent_requests,
        memory_limit=cfg.lean_memory_limit,
        timeout=cfg.lean_timeout,
        name='verifier',
    )

    # load LLM models on gpus
    generator_scheduler = ProcessScheduler(batch_size=cfg.batch_size, name='generator')
    llm_processes = [
        GeneratorProcess(
            local_rank=local_rank,
            node_rank=node_rank,
            model_path=cfg.model_path,
            task_queue=generator_scheduler.task_queue,
            request_statuses=generator_scheduler.request_statuses,
            lock=generator_scheduler.lock,
            args=cfg.model_args,
            max_model_len=max_model_len,
        )
        for local_rank in range(ngpus)
    ]

    # create a unified scheduler interface
    scheduler = Scheduler(dict(
        verifier=verifier_scheduler,
        generator=generator_scheduler,
    ))

    # launch search processes
    search_processes = [
        SearchProcess(
            idx=i+node_rank*cfg.n_search_procs,
            log_dir=log_dir,
            tokenizer_path=cfg.model_path,
            scheduler=scheduler,
            data_loader=data_loader,
            cfg=cfg,
        )
        for i in range(min(cfg.n_search_procs, data_loader.size()))
    ]
    for p in search_processes:
        p.start()
    print(f'Complete launching {len(search_processes)} SearchProcesses')

    for p in llm_processes:
        p.start()
    print(f'Complete launching {len(llm_processes)} LLMProcesses')

    for p in search_processes:
        p.join()
    print(f'All {len(search_processes)} SearchProcesses stopped')

    scheduler.close()

    for p in llm_processes:
        p.join()
    print(f'All {len(llm_processes)} LLMProcesses stopped')
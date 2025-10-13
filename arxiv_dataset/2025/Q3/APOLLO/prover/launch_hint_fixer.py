import os
import sys
import copy
import time
import pickle
import random
import warnings
import argparse

import torch

from prover.workers import JsonlDataLoader, Scheduler, ProcessScheduler, GeneratorProcess, SearchProcess, AnalyzeProcess
from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import get_datetime, load_config, AttrDict
from tqdm import tqdm
import torch.multiprocessing as mp



def launch_parallel_hint_search(data_jsonl, config, log_dir, node_rank=0, world_size=1):
    cfg = load_config(config)
    os.makedirs(log_dir, exist_ok=True)

    ngpus = torch.cuda.device_count()
    assert ngpus >= 1
    
    # create data loader
    data_loader = JsonlDataLoader(
        data_jsonl=data_jsonl,
        data_split='train',
        data_repeat=1,
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

    # create a unified scheduler interface
    scheduler = Scheduler(dict(
        verifier=verifier_scheduler,
    ))

    # launch search processes
    search_processes = [
        AnalyzeProcess(
            idx=i+node_rank*cfg.n_search_procs,
            log_dir=log_dir,
            scheduler=scheduler, # maybe should put just verifier_scheduler???
            data_loader=data_loader,
            # stop_event = mp.Event(),
            cfg=cfg,
        )
        for i in range(min(cfg.n_search_procs, data_loader.size()))
    ]
    for p in search_processes:
        p.start()
    print(f'Complete launching {len(search_processes)} HintSolverProcesses')

    for p in search_processes:
        p.join()
    print(f'All {len(search_processes)} HintSolverProcesses stopped')

    scheduler.close()
    verifier_scheduler.close()

def fix_with_hint(code: str, verbose: bool) -> str:
        from utils.syntax_repair import SyntaxCorrector
        from utils.sorrify import Sorrifier, ProofTree, LeanServerSorrifier
        from utils.hint_repair import ProofRepairer, LeanServerProofRepairer
        from prover.lean.verifier import verify_lean4_file
        # Remove Syntax Errors from the code
        code_corrected = SyntaxCorrector(code).correct_text()
        
        # Begin Sorrification
        pt = ProofTree(code_corrected)
        pt.parse_lean_with_dot_subcases()

        checker = Sorrifier(pt, verify_lean4_file, clean_empty_lines=True, clean_comments=False,
                                      pbar=verbose)
        code_corrected_sorry = checker.verify_and_fix_tree()


        # Repair the Proof
        repairer = ProofRepairer(code_corrected_sorry, verify_lean4_file, verbose=verbose)
        final_code = repairer.repair_proof()

        return final_code

def _post_process(data: dict, proof_code: str):
        header = data.get('header', str())
        tailer = data.get('tailer', str())
        formal_statement = data['formal_statement']
        return dict(
            name = data.get('name', str()),
            formal_statement = formal_statement,
            header = header,
            statement_proposal=f'{header}{formal_statement}{proof_code}{tailer}',
            proof_code=proof_code,
        )

def launch_regular_hint_search(data_jsonl, config, log_dir, node_rank=0, world_size=1, verbose=True, sample_size=5):
    candidate_list, info_list, request_id_list = [], [], []

    cfg = load_config(config)
    os.makedirs(log_dir, exist_ok=True)

    verifier_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=cfg.lean_max_concurrent_requests,
        memory_limit=cfg.lean_memory_limit,
        timeout=cfg.lean_timeout,
        name='verifier',
    )

    # create a unified scheduler interface
    scheduler = Scheduler(dict(
        verifier=verifier_scheduler,
    ))

    # If HintRepair received empty data_jsonl
    if not len(data_jsonl):
        scheduler.close()
        verifier_scheduler.close()
        return

    # Sample from failed attempts for repair
    if sample_size > len(data_jsonl):
        sample_size = len(data_jsonl)
        
    data_jsonl = random.sample(data_jsonl, sample_size)

    for sample in (tqdm(data_jsonl, desc='Fixing with HintRepair') if verbose else data_jsonl):
        try:
            candidate = _post_process(sample, sample['formal_proof'])

            candidate['statement_proposal'] = fix_with_hint(candidate['statement_proposal'], verbose=verbose)

            candidate_list.append(candidate)
            request_id = scheduler.verifier_submit_request(candidate['statement_proposal'])
            request_id_list.append(request_id)
        except:
            continue

    result_list = scheduler.verifier_get_all_request_outputs(request_id_list)

    success_count = sum([int(result['complete']) for result in result_list])
    

    summary_dict = dict(success=[], failure=[])
    for _idx, (candidate, result) in enumerate(zip(candidate_list, result_list)):
        success_flag = 'success' if result['complete'] else 'failure'
        summary_dict[success_flag].append(dict(
            problem_name=candidate['name'],
            # sample_info=info,
            formal_statement=candidate['formal_statement'],
            proof_code=candidate['proof_code'],
            header=candidate['header'],
            result=result,
            verified_code=candidate['statement_proposal']
        ))
        print(candidate['statement_proposal'])
    
    prob_log_basedir = os.path.join(log_dir, 'hint_repair')
    os.makedirs(prob_log_basedir, exist_ok=True)
    for success_flag, summary_list in summary_dict.items():
        if len(summary_list) > 0:
            with open(os.path.join(prob_log_basedir, f'hint-solver-{success_flag}.pkl'), 'wb') as pkl_f:
                pickle.dump(summary_list, pkl_f)
    
    scheduler.close()
    verifier_scheduler.close()
    


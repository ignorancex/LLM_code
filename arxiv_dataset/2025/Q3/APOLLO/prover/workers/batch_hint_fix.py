'''
    This code is partially adopted from https://github.com/deepseek-ai/DeepSeek-Prover-V1.5
'''
import os
import sys
import time
import copy
import json
import pickle
from pathlib import Path

import torch
import torch.multiprocessing as mp
import numpy as np

from prover.utils import AttrDict, get_datetime


class AnalyzeProcess(mp.Process):
    def __init__(self, idx, log_dir, scheduler, data_loader, cfg):
        self.idx = idx
        self.log_dir = Path(log_dir)
        self.scheduler = scheduler
        self.data_loader = data_loader
        super().__init__()

        self._current_prob_idx = None
    def fix_with_hint(self, code: str) -> str:
        from utils.syntax_repair import SyntaxCorrector
        from utils.sorrify import Sorrifier, ProofTree, LeanServerSorrifier
        from utils.hint_repair import ProofRepairer, LeanServerProofRepairer
        from prover.lean.verifier import verify_lean4_file
        # Remove Syntax Errors from the code
        code_corrected = SyntaxCorrector(code).correct_text()
        
        # Begin Sorrification
        pt = ProofTree(code_corrected)
        pt.parse_lean_with_dot_subcases()

        tree = pt.tree
        checker = LeanServerSorrifier(pt, self.scheduler, clean_empty_lines=True, clean_comments=False,
                                      pbar=False)
        code_corrected_sorry = checker.verify_and_fix_tree()


        # Repair the Proof
        repairer = LeanServerProofRepairer(code_corrected_sorry, self.scheduler, verbose=True)
        final_code = repairer.repair_proof()

        return final_code
    
    def _post_process(self, data: dict, proof_code: str):
        header = data.get('header', str())
        tailer = data.get('tailer', str())
        formal_statement = data['formal_statement']
        return dict(
            statement_proposal=f'{header}{formal_statement}{proof_code}{tailer}',
            proof_code=proof_code,
        )
    
    def process_print(self, logs, **kwargs):
        print('Process ID: {:3d}    Problem ID: {}    {}'.format(self.idx, 0, logs), **kwargs)

    def run(self):
        while True:
            prob_idx, prob_runname, data = self.data_loader.get()
            if prob_idx is None: break
            
            sample_start_time = time.time()            

            # submit requests to the verification server when receiving from the generator
            candidate_list, info_list, request_id_list = [], [], []
            for sample in [data]:
                candidate = self._post_process(sample, sample['formal_proof'])

                candidate['statement_proposal'] = self.fix_with_hint(candidate['statement_proposal'])

                candidate_list.append(candidate)
                request_id = self.scheduler.verifier_submit_request(candidate['statement_proposal'])
                request_id_list.append(request_id)
            sample_timecost = time.time() - sample_start_time

            verification_start_wait_time = time.time()
            result_list = self.scheduler.verifier_get_all_request_outputs(request_id_list)
            verification_timecost = time.time() - verification_start_wait_time

            success_count = sum([int(result['complete']) for result in result_list])
            self.process_print('Success: {} / {}    Generation: {:.2f} secs    Verfication: {:.2f} secs'.format(
                success_count, len(candidate_list), sample_timecost, verification_timecost,
            ))
            

            summary_dict = dict(success=[], failure=[])
            for _idx, (candidate, result) in enumerate(zip(candidate_list, result_list)):
                success_flag = 'success' if result['complete'] else 'failure'
                summary_dict[success_flag].append(dict(
                    problem_name=data['name'],
                    # sample_info=info,
                    formal_statement=data['formal_statement'],
                    proof_code=candidate['proof_code'],
                    header=data['header'],
                    result=result,
                    verified_code=candidate['statement_proposal']
                ))
            
            prob_name, run_id = prob_runname.split('/')
            prob_log_basedir = self.log_dir / 'hint_repair'
            os.makedirs(prob_log_basedir, exist_ok=True)
            for success_flag, summary_list in summary_dict.items():
                if len(summary_list) > 0:
                    with open(prob_log_basedir / f'hint-solver-{success_flag}.pkl', 'wb') as pkl_f:
                        pickle.dump(summary_list, pkl_f)

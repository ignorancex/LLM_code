import argparse
import multiprocessing as mp
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch as t
from rich import print
from tqdm import tqdm

import experiments
from src.configs import SIMULATOR
from src.simulate import simulate_episode

t.set_num_threads(1)

seed = 1992
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.cuda.manual_seed_all(seed)

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
os.environ['OMP_NUM_THREADS'] = '1'

SOLVER = experiments.get_solver()


def evaluator(idx, params):
    summary, itr, success_itr, failed_itr, lock, args = params['summary'], params['itr'], params['success_itr'], params['failed_itr'], params['lock'], params['args']
    if t.cuda.is_available():
        t.cuda.set_device(idx % t.cuda.device_count())
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    solver = deepcopy(SOLVER)
    assert solver.load(args.model, strict=True)
    solver.to(device)
    solver.eval(trt=os.environ.get('USE_TRT', 'false').lower() == 'true')

    if args.levels_file:
        SIMULATOR.load_levels_from(args.levels_file)

    while True:
        lock.acquire()
        local_itr = itr.value
        itr.value += 1
        lock.release()

        if local_itr >= args.n_levels:
            return

        with t.no_grad():
            def get_simulator():
                simulator = SIMULATOR()
                state = simulator.reset(args.offset + local_itr)

                return simulator, state

            try:
                reward = simulate_episode(get_simulator,
                                          get_action=lambda state: int(solver.get_action(state)),
                                          pre_step=lambda simulator, state, action: solver.update(action),
                                          post_step=lambda *args: None,
                                          pre_episode=lambda *args: solver.reset(),
                                          post_episode=lambda *args: None)
            except Exception as e:
                print('Exception occurred for:', args.offset + local_itr)
                print(e)
                reward = SIMULATOR.failed

            lock.acquire()
            summary[args.offset + local_itr] = reward
            if reward == SIMULATOR.failed:
                failed_itr.value += 1
            else:
                success_itr.value += 1
            lock.release()


if __name__ == '__main__':
    mp.set_start_method('spawn', True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--levels-file', default=None)
    parser.add_argument('--summary', default=f'results/{os.environ.get("SOLVER", "model_free").lower()}-summary.npy')
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--n-levels', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()

    if args.levels_file:
        SIMULATOR.load_levels_from(args.levels_file)

    if not args.n_levels:
        args.n_levels = SIMULATOR.n_levels

    args.n_levels = min(args.n_levels, SIMULATOR.n_levels - args.offset)

    print(f'\n'
          f'{SOLVER}\n'
          f'model file -----------> {args.model}\n'
          f'total workers --------> {args.n_workers}\n'
          f'levels file ----------> {"Default" if not args.levels_file else args.levels_file}\n'
          f'summary file ---------> {args.summary}\n'
          f'total levels ---------> {args.n_levels}\n'
          f'levels offset --------> {args.offset}\n')

    summary = mp.Manager().dict({})
    lock = mp.RLock()
    itr = mp.Value('i', 0)
    success_itr = mp.Value('i', 0)
    failed_itr = mp.Value('i', 0)

    params = {
        'summary': summary,
        'itr': itr,
        'success_itr': success_itr,
        'failed_itr': failed_itr,
        'lock': lock,
        'args': args
    }

    print("starting evaluators...\n")
    processes = [mp.Process(target=evaluator, args=(idx, params)) for idx in range(args.n_workers)]
    [p.start() for p in processes]

    # Start the progress bar
    pbar = tqdm(total=args.n_levels, desc='Progress')
    local_itr, prev_itr = 0, 0
    success, failed = 0, 0
    while local_itr < args.n_levels:
        time.sleep(0.5)
        success = success_itr.value
        failed = failed_itr.value
        local_itr = success + failed

        pbar.set_description('Success: {} ({:.2f}%) | Failure: {} ({:.2f}%) | Progress'.format(success, success / (local_itr + 1e-20) * 100, failed, failed / (local_itr + 1e-20) * 100))
        pbar.update(local_itr - prev_itr)

        prev_itr = local_itr

    pbar.update(args.n_levels - prev_itr)
    [p.join() for p in processes]

    summary = dict(summary)
    t.save({
        "simulator": SIMULATOR.__name__,
        "solver": str(SOLVER),
        "rewards": summary,
        "success rate": success / (success + failed),
        "success": [k for k, v in summary.items() if v != SIMULATOR.failed],
        "failed": [k for k, v in summary.items() if v == SIMULATOR.failed]
    }, args.summary)

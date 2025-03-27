import argparse
import json
import os

from eval_bird import main_bird
from eval_turney import main_turney
from eval_ppdb_paws import main_ppdb_paws

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_run_mode", action='store_true')
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--task", type=str, default='paws_short', choices=['ppdb_exact', 'ppdb', 'paws_short'])
    parser.add_argument("--data_dir", type=str, default='data/')
    parser.add_argument("--result_dir", type=str, default='result/')

    parser.add_argument('--bird_fname', type=str, default='data/bird/data.txt')
    parser.add_argument('--turney_fname', type=str, default='data/turney/data.txt')
    args = parser.parse_args()

    tasks = ['paws_short', 'bird', 'turney', 'ppdb', 'ppdb_exact']

    for task in tasks:
        print(task)
        setattr(args, 'data_dir', f'data/{task}/')
        os.makedirs(args.result_dir, exist_ok=True)
        score_path = f'{args.result_dir}/{task}.json'
        if os.path.exists(score_path):
            with open(score_path, 'r') as score_file:
                score_dict = json.load(score_file)
        else:
            score_dict = {}
        if task == 'bird':
            scores = main_bird(args)
        elif task == 'turney':
            scores = main_turney(args)
        else:
            scores = main_ppdb_paws(args, task=task)
        score_dict.update(scores)

        print('=' * 20)
        print(task)
        print('=' * 20)
        for mname, sdict in scores.items():
            print(mname, ',', sdict['core_metric'])
        with open(score_path, 'w') as score_file:
            score_file.write(json.dumps(score_dict) + '\n')

    print('done')

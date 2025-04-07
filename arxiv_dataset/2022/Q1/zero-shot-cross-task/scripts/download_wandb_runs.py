import json
from collections import defaultdict
import argparse
import pandas as pd
import wandb
from pathlib import Path
from tqdm import tqdm
import shutil
from src.common.util import sanitize_name
import os

PROJECT_ROOT = Path.cwd()
if 'scripts' in str(PROJECT_ROOT):
    while 'scripts' in str(PROJECT_ROOT):
        PROJECT_ROOT = PROJECT_ROOT.parent


def download_wandb_runs(args):
    out_path = PROJECT_ROOT.joinpath('data')
    run_csv_path = out_path.joinpath('runs.csv')
    summary_keys_path = out_path.joinpath('summary_keys.txt')

    prediction_path = PROJECT_ROOT.joinpath('data', 'predictions')
    if prediction_path.exists():
        shutil.rmtree(prediction_path)
    prediction_path.mkdir(parents=True)
    print(f'Downloading to {out_path}')
    api = wandb.Api()

    print(f"Downloading runs from groups {args.dl_groups} and runs {args.dl_runs}")

    # Project is specified by <entity/project-name>
    runs = api.runs("gabeorlanski/zero-shot-eval")
    print(f"{len(runs)} runs found")
    records = []
    summary_keys = set()
    group_predictions = defaultdict(list)
    tmp_path = prediction_path.joinpath('tmp')
    for run in tqdm(runs, desc='Aligning data'):
        record_dict = {}
        run_summary = {k: v for k, v in run.summary._json_dict.items()
                       if not k.startswith('_')}
        summary_keys.update(set(run_summary))
        record_dict.update(run_summary)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        record_dict.update({k: v for k, v in run.config.items()
                            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        record_dict['name'] = run.name
        record_dict['group'] = run.group
        records.append(record_dict)

        if not args.dl_groups or not args.dl_runs:
            continue
        elif args.dl_groups and run.group not in args.dl_groups:
            continue
        elif args.dl_runs and run.config['run_name'] not in args.dl_runs:
            continue

        prediction_artifact = None
        for f in run.logged_artifacts():
            if f.type == 'predictions':
                prediction_artifact = f
                break
        if prediction_artifact is None:
            continue

        tmp_path.mkdir()
        prediction_artifact.download(str(tmp_path.resolve().absolute()))

        preds_file = tmp_path.joinpath('predictions.jsonl')
        group_predictions[sanitize_name(run.group)].extend([
            {"run_name": run.config['run_name'], 'name': run.name, **json.loads(line)}
            for line in preds_file.read_text('utf-8').splitlines(False)
        ])
        shutil.rmtree(tmp_path)

    for group, values in group_predictions.items():
        group_file = prediction_path.joinpath(f"{group}.jsonl")
        with group_file.open('w', encoding='utf-8') as f:
            f.write('\n'.join(map(json.dumps, values)))

    runs_df = pd.DataFrame.from_records(records)
    with summary_keys_path.open('w', encoding='utf-8') as f:
        for k in summary_keys:
            f.write(k + '\n')

    runs_df.to_csv(run_csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dl-groups",
        type=str,
        action="append",
        default=[],
        help="groups to download",
    )
    parser.add_argument(
        "--dl-runs",
        type=str,
        action="append",
        default=[],
        help="run names to download",
    )
    args = parser.parse_args()
    download_wandb_runs(args)

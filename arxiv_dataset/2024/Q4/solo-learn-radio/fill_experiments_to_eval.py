import argparse
import pathlib

#read_args
parser = argparse.ArgumentParser(description="Generate a bash file with commands to create experiments configurations")
parser.add_argument("--dataset", type=str, default="robin", help="Dataset name", choices=['robin', 'rgz', 'vlass', 'mirabest', 'frg'])
parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, help="Linear evaluation or finetuning")
parser.add_argument("--model_list_path", type=pathlib.Path, default="model_list.txt", help="Model list path")

args = parser.parse_args()

finetune = "--finetune" if args.finetune else ""

# Read run_ids from .txt file
with open(args.model_list_path, 'r') as file:
    run_ids = file.readlines()

with open("model_list_path.txt", 'r') as file:
    baseline_paths = file.readlines()

#switch dataset
if args.dataset == "robin":
    with open('gen_eval_exp_robin.sh', 'w') as output_file:
        for baseline_path in baseline_paths:
            baseline_path = baseline_path.strip()
            line = f"python generate_eval_experiments.py \
    --local_or_wandb local {finetune} --augmentations nomeanstd \
    --model_path {baseline_path} --dataset robin \
    --dataset_path ../2-ROBIN --devices 0 --num_workers 6 --datalist info.json \
    --balancing_strategy as_is --sample_size 1650 --K 3 --reps 1\n"
            # Scrivi la riga nel file di output
            output_file.write(line)
        for run_id in run_ids:
            run_id = run_id.strip()
            line = f"python generate_eval_experiments.py \
    --local_or_wandb wandb {finetune} --augmentations nomeanstd \
    --run_id {run_id} --dataset robin \
    --dataset_path ../2-ROBIN --devices 0 --num_workers 6 --datalist info.json \
    --balancing_strategy as_is --sample_size 1650 --K 3 --reps 1\n"
            # Scrivi la riga nel file di output
            output_file.write(line)
if args.dataset == "rgz":
    # Apre il file di output in modalità scrittura
    with open('gen_eval_exp_rgz.sh', 'w') as output_file:
        for baseline_path in baseline_paths:
            baseline_path = baseline_path.strip()
            line = f"python generate_eval_experiments.py \
    --local_or_wandb local {finetune} --augmentations nomeanstd \
    --model_path {baseline_path} --dataset rgz \
    --dataset_path ../RGZ-D1-smorph-dataset --devices 0 --num_workers 6 --datalist info_wo_nan.json \
    --balancing_strategy as_is --sample_size 4500 --K 3 --reps 1\n"
            # Scrivi la riga nel file di output
            output_file.write(line)
        # Ciclo attraverso ogni run_id
        for run_id in run_ids:
            # Rimuovi eventuali spazi bianchi e caratteri di nuova riga
            run_id = run_id.strip()
            # Crea la riga con il run_id corrente
            
            line = f"python generate_eval_experiments.py --run_id {run_id} --dataset rgz \
    --local_or_wandb wandb {finetune} --augmentations nomeanstd \
    --dataset_path ../RGZ-D1-smorph-dataset --devices 0 --num_workers 6 --datalist info_wo_nan.json \
    --balancing_strategy as_is --sample_size 4500 --K 3 --reps 1\n"
            # Scrivi la riga nel file di output
            output_file.write(line)
if args.dataset == "vlass":
    with open('gen_eval_exp_vlass.sh', 'w') as output_file:
        for baseline_path in baseline_paths:
            baseline_path = baseline_path.strip()
            line = f"python generate_eval_experiments.py \
    --local_or_wandb local {finetune} --augmentations nomeanstd \
    --model_path {baseline_path} --dataset vlass \
    --dataset_path ../vlass --devices 0 --num_workers 6 --datalist info_downloaded.json \
    --balancing_strategy as_is --sample_size 2900 --K 3 --reps 1\n"
            output_file.write(line)
        for run_id in run_ids:
            run_id = run_id.strip()
            line = f"python generate_eval_experiments.py --run_id {run_id} --dataset vlass \
    --local_or_wandb wandb {finetune} --augmentations nomeanstd \
    --dataset_path ../vlass --devices 0 --num_workers 6 --datalist info_downloaded.json \
    --balancing_strategy as_is --sample_size 2900 --K 3 --reps 1\n"
            output_file.write(line)
if args.dataset == "mirabest":
    with open('gen_eval_exp_mirabest.sh', 'w') as output_file:
        for baseline_path in baseline_paths:
            baseline_path = baseline_path.strip()
            line = f"python generate_eval_experiments.py \
    --local_or_wandb local {finetune} --augmentations nomeanstd \
    --model_path {baseline_path} --dataset mirabest \
    --dataset_path ../mirabest --devices 0 --num_workers 1 --datalist info.json \
    --balancing_strategy as_is --sample_size 0 --K 1 --reps 5\n"
            # Scrivi la riga nel file di output
            output_file.write(line)
        # Ciclo attraverso ogni run_id
        for run_id in run_ids:
            # Rimuovi eventuali spazi bianchi e caratteri di nuova riga
            run_id = run_id.strip()
            # Crea la riga con il run_id corrente
            line = f"python generate_eval_experiments.py --run_id {run_id} --dataset mirabest \
    --local_or_wandb wandb {finetune} --augmentations meanstd \
    --dataset_path ../mirabest --devices 0 --num_workers 1 --datalist info.json \
    --balancing_strategy as_is --sample_size 0 --K 1 --reps 5\n"
            # Scrivi la riga nel file di output
            output_file.write(line)
if args.dataset == "frg":
    with open('gen_eval_exp_frg.sh', 'w') as output_file:
        for baseline_path in baseline_paths:
            baseline_path = baseline_path.strip()
            line = f"python generate_eval_experiments.py \
    --local_or_wandb local {finetune} --augmentations nomeanstd \
    --model_path {baseline_path} --dataset frg \
    --dataset_path ../frg-FirstRadioGalaxies --devices 0 --num_workers 6 --datalist info.json \
    --balancing_strategy as_is --sample_size 924 --K 3 --reps 1\n"
            output_file.write(line)
        for run_id in run_ids:
            run_id = run_id.strip()
            line = f"python generate_eval_experiments.py --run_id {run_id} --dataset frg \
    --local_or_wandb wandb {finetune} --augmentations nomeanstd \
    --dataset_path ../frg-FirstRadioGalaxies --devices 0 --num_workers 6 --datalist info.json \
    --balancing_strategy as_is --sample_size 924 --K 3 --reps 1\n"
            output_file.write(line)



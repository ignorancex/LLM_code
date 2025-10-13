import json
import os
import pprint as pp
import random
from datetime import date

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb


def fix_random_seed_as(random_seed):
    if random_seed == -1:
        random_seed = np.random.randint(100000)
        print("RANDOM SEED: {}".format(random_seed))

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    return random_seed


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def create_experiment_export_folder(experiment_dir, experiment_description):
    print(os.path.abspath(experiment_dir))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    print(os.path.abspath(experiment_path))
    os.mkdir(experiment_path)
    print("folder created: " + os.path.abspath(experiment_path))
    return experiment_path


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def export_config_as_json(config, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=2)


def generate_tags(config):
    tags = []
    tags.append(config.get('clip_image_model', config.get('clip_text_model')))
    tags.append(config.get('blip_model_type'))
    tags.append(config.get('blip_model_name'))
    tags.append(config.get('evaluator_code'))
    tags = [tag for tag in tags if tag is not None]
    return tags


def set_up_gpu(device_idx):
    # os setup should be done before import torch
    if device_idx:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_idx
        return {
            'num_gpu': len(device_idx.split(","))
        }
    else:
        idxs = os.environ['CUDA_VISIBLE_DEVICES']
        return {
            'num_gpu': len(idxs.split(","))
        }


def setup_experiment(config):
    device_info = set_up_gpu(config['device_idx'])
    config.update(device_info)

    random_seed = fix_random_seed_as(config['random_seed'])
    config['random_seed'] = random_seed
    export_root = create_experiment_export_folder(config['experiment_dir'], config['experiment_description'])
    export_config_as_json(config, export_root)
    config['export_root'] = export_root

    pp.pprint(config, width=1)
    os.environ['WANDB_SILENT'] = "true"
    tags = generate_tags(config)
    project_name = config['wandb_project_name']
    wandb_account_name = config['wandb_account_name']
    experiment_name = config['experiment_description']
    experiment_name = experiment_name if config['random_seed'] != -1 else experiment_name + "_{}".format(random_seed)

    wandb.init(config=config, name=experiment_name, project=project_name,
               entity=wandb_account_name, tags=tags, settings=wandb.Settings(start_method="thread"))
    # save code, loss and augmenter is enough
    wandb.run.log_code(include_fn=lambda f: f.endswith(".py") and (("pipeline" in f) or ("models" in f) or ("pretrain" in f)) and "lavis" not in f)
    return export_root, config

if __name__ == "__main__":
    # Update metrics for a run, after the run has finished
    import wandb

    api = wandb.Api()

    run = api.run("junyang/zscir/l9u3i9ve")
    print(run.summary)
    # compute average of shirt, dress and toptee metric recall_@1,5,10,50
    run.summary["best_model/avg_fashionIQ_recall_@1"] = (run.summary["best_model/fashionIQ_shirt_recall_@1"] + run.summary["best_model/fashionIQ_dress_recall_@1"] + run.summary["best_model/fashionIQ_toptee_recall_@1"]) / 3
    run.summary["best_model/avg_fashionIQ_recall_@5"] = (run.summary["best_model/fashionIQ_shirt_recall_@5"] + run.summary["best_model/fashionIQ_dress_recall_@5"] + run.summary["best_model/fashionIQ_toptee_recall_@5"]) / 3
    run.summary["best_model/avg_fashionIQ_recall_@10"] = (run.summary["best_model/fashionIQ_shirt_recall_@10"] + run.summary["best_model/fashionIQ_dress_recall_@10"] + run.summary["best_model/fashionIQ_toptee_recall_@10"]) / 3
    run.summary["best_model/avg_fashionIQ_recall_@50"] = (run.summary["best_model/fashionIQ_shirt_recall_@50"] + run.summary["best_model/fashionIQ_dress_recall_@50"] + run.summary["best_model/fashionIQ_toptee_recall_@50"]) / 3
    print("recall_@1: ", run.summary["best_model/avg_fashionIQ_recall_@1"])
    print("recall_@1: ", round(run.summary["best_model/avg_fashionIQ_recall_@1"] * 100, 2))
    print("recall_@5: ", run.summary["best_model/avg_fashionIQ_recall_@5"])
    print('recall_@5', round(run.summary["best_model/avg_fashionIQ_recall_@5"] * 100, 2))
    print("recall_@10: ", run.summary["best_model/avg_fashionIQ_recall_@10"])
    print("recall_@10: ", round(run.summary["best_model/avg_fashionIQ_recall_@10"] * 100, 2))
    print("recall_@50: ", run.summary["best_model/avg_fashionIQ_recall_@50"])
    print("recall_@50: ", round(run.summary["best_model/avg_fashionIQ_recall_@50"] * 100, 2))
    run.summary.update()

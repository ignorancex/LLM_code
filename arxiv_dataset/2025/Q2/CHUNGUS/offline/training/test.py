import os
import json
import argparse
import logging
import torch
import torch.nn.functional as F
import random
import numpy as np
import prediction_heads

from pathlib import Path
from evaluator import Evaluator
from embedding_dataset import EmbeddingDataset
from losses import HDR
from experiment import Experiment


def main(arguments):

    # Read experiment
    experiment_path = Path(arguments['experiment'])
    experiment = Experiment(experiment_path, should_exist=True)
    settings = experiment.read_settings()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(experiment.test_log_file), logging.StreamHandler()]
    )

    # Log the settings and arguments
    for key in settings:
        logging.info("Read setting from experiment '{}': {}".format(key, settings[key]))
    for key in arguments:
        logging.info("Argument '{}': {}".format(key, arguments[key]))
    
    # Set seed
    seed = arguments['seed']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Create device
    device = torch.device('cuda')

    # Get datasets
    train_dataset = EmbeddingDataset(
        np.load(str(arguments['train_embeddings']), allow_pickle=True).item()['data'],
        rebalance=False, device=device
    )
    valid_dataset = EmbeddingDataset(
        np.load(str(arguments['valid_embeddings']), allow_pickle=True).item()['data'],
        rebalance=False, device=device
    )

    # Print dataset sizes
    logging.info("{} train samples and {} validation samples".format(len(train_dataset), len(valid_dataset)))

    # Create evaluation metrics
    eval_metrics = {
        'hdr01': HDR(0.1),
        'hdr015': HDR(0.15),
        'hdr02': HDR(0.2),
        'hdr025': HDR(0.25),
        'hdr03': HDR(0.3),
        'hdr035': HDR(0.35),
        'hdr04': HDR(0.4),
        'hdr045': HDR(0.45),
        'hdr05': HDR(0.5)
    }
    evaluator = Evaluator(eval_metrics)

    # Network
    predictor_network = prediction_heads.get_head(settings['prediction_head'], emb_dim=settings['emb_dim'])
    predictor_network.load_state_dict(torch.load(experiment.best_model_file))
    logging.info("Starting evaluation (with best model file for experiment)...")

    # Evaluate
    evaluation_results = {
        "train": evaluator.evaluate(predictor_network, train_dataset, device, use_loader=False),
        "valid": evaluator.evaluate(predictor_network, valid_dataset, device, use_loader=False)
    }
    
    # Finish
    logging.info("Evaluation finished with {}".format(evaluation_results))
    
    # Save evaluation outputs
    with open(experiment.test_results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # seed
    parser.add_argument("--seed", type=int, default=123)
    
    # data
    parser.add_argument("--train_embeddings", type=str, required=True)
    parser.add_argument("--valid_embeddings", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    
    args = parser.parse_args()
    main(vars(args)) # pass args as dictionary

import os
import json
import torch
import random
import logging
import argparse
import numpy as np
import prediction_heads

from pathlib import Path
from trainer import Trainer
from evaluator import Evaluator
from losses import get_loss, HDR
from experiment import Experiment
from embedding_dataset import EmbeddingDataset


def main(settings):

    # Create experiment
    experiment_path = Path(settings['experiment'])
    experiment = Experiment(experiment_path, should_exist=False)
    experiment.write_settings(settings)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(experiment.train_log_file), logging.StreamHandler()]
    )

    # Log the settings
    for key in settings:
        logging.info("Setting '{}': {}".format(key, settings[key]))
    
    # Set seed
    seed = settings['seed']
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
    rebalance = True if settings['rebalance'] else False
    train_dataset = EmbeddingDataset(
        np.load(str(settings['train_embeddings']), allow_pickle=True).item()['data'],
        rebalance=rebalance, device=device, num_embeddings=settings['train_num_embeddings']
    )
    valid_dataset = EmbeddingDataset(
        np.load(str(settings['valid_embeddings']), allow_pickle=True).item()['data'],
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

    # Optimizer
    if settings['weight_decay'] > 0:
        optimizer = torch.optim.AdamW(predictor_network.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    else:
        optimizer = torch.optim.Adam(predictor_network.parameters(), lr=settings['lr'])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, settings['lr_decay_step'], gamma=settings['lr_decay_gamma'])
    
    # Loss function
    loss_fn = get_loss(name=settings['loss_fn'], l=settings['lrizz_L'], loss_args=settings['loss_args'])

    # Create trainer and start training
    trainer = Trainer(network=predictor_network,
                      train_data=train_dataset,
                      valid_data=valid_dataset,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      evaluator=evaluator,
                      target_metric=settings['target_metric'],
                      best_save_path=experiment.best_model_file,
                      last_save_path=experiment.last_model_file,
                      device=device,
                      use_loader=False)
    logging.info("Starting training...")
    train_results = trainer.train(settings['epochs'], patience=settings['patience'])
    logging.info("Training has finished")
    
    # Run final evaluation of best model
    logging.info("Starting evaluation...")
    predictor_network.load_state_dict(torch.load(experiment.best_model_file))
    evaluation_results = evaluator.evaluate(predictor_network, valid_dataset, device, use_loader=False)
    logging.info("Evaluation finished with {}".format(evaluation_results))
    
    # Save evaluation outputs
    with open(experiment.train_results_file, 'w') as f:
        json.dump({"train": train_results, "eval": evaluation_results}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # seed
    parser.add_argument("--seed", type=int, default=123)
    
    # data
    parser.add_argument("--train_embeddings", type=str, required=True)
    parser.add_argument("--valid_embeddings", type=str, required=True)
    parser.add_argument("--train_num_embeddings", type=int, default=-1) # for capping number of training samples
    parser.add_argument("--experiment", type=str, required=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.25)
    parser.add_argument("--lr_decay_step", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--target_metric", type=str, default='hdr025')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=50)
    
    # model
    parser.add_argument("--prediction_head", type=str, required=True)
    parser.add_argument("--emb_dim", type=int, default=384)
    parser.add_argument("--loss_fn", type=str, default='lrizz')
    parser.add_argument("--lrizz_L", type=float, default=0.5)
    parser.add_argument("--loss_args", type=str, default=None)
    
    # if want to rebalance
    parser.add_argument("--rebalance", default=False, action='store_true')
    
    args = parser.parse_args()
    main(vars(args)) # pass args as dictionary

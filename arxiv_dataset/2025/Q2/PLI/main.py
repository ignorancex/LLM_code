from data import create_test_dataloaders, create_train_eval_dataloaders
from evaluators import get_evaluator_cls
from loggers.file_loggers import BestModelTracker
from loggers.wandb_loggers import WandbSimplePrinter, WandbSummaryPrinter, WandbBestPrinter
from models import create_models
from optimizers import create_optimizers, create_lr_schedulers
from options import get_experiment_config
from set_up import setup_experiment
from trainers import get_trainer_cls
from losses import loss_factory

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)
    models = create_models(configs)
    test_dataloaders = create_test_dataloaders(configs)  # only load raw image and text

    pretrained_dataloaders, criterions, optimizers, lr_schedulers, train_loggers, train_evaluators = None, None, None, None, None, None
    # pretrain setting
    if configs['pretrain_dataset'] != "None": # load pretrain dataset
        pretrained_dataloaders, test_dataloaders = create_train_eval_dataloaders(configs)
        criterions = loss_factory(configs)
        optimizers = create_optimizers(models=models, config=configs)
        lr_schedulers = create_lr_schedulers(optimizers, config=configs)
        train_loggers = [WandbSimplePrinter('train/')]

    val_loggers = [WandbSimplePrinter('val/'), WandbBestPrinter('best_model/', metric_key=configs['best_metric']),
                   BestModelTracker(export_root, metric_key=configs['best_metric'])]
    evaluators = {key: get_evaluator_cls(configs['evaluator_code'])(models, value, configs, top_k=configs['topk'])
                  for key, value in test_dataloaders.items()}

    trainer = get_trainer_cls(configs["trainer"])(models, val_loggers, evaluators, pretrained_dataloaders, criterions, optimizers, lr_schedulers,
                                                  configs['epoch'], train_loggers, train_evaluators, start_epoch=0, **configs)
    trainer.run()

def get_summary_keys(configs):
    summary_keys = ['recall_@{}'.format(k) for k in configs['topk'].split(",")]
    return summary_keys


if __name__ == '__main__':
    main()
    # test __call__ method
    # import torch, numpy as np
    # similarity_matrix = torch.randn(2, 3)
    # print(similarity_matrix)
    # top_k = (1, 2, 3)
    # ref_attribute_matching_matrix = torch.tensor([[True, False, False], [False, True, False]])
    # assert similarity_matrix.shape == ref_attribute_matching_matrix.shape
    # print((ref_attribute_matching_matrix == True))
    # similarity_matrix[ref_attribute_matching_matrix == True] = similarity_matrix.min()
    # print(similarity_matrix)
    #
    # # cosine_similarity
    # a = torch.randn(2, 3)
    # b = torch.randn(2, 3)
    # print(a, b)
    # # matrix is (2, 2)
    # print(torch.nn.functional.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2))


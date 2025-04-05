import torch
import numpy as np
import random
from copy import deepcopy

from dataset import create_fed_dataset
from algorithm import create_system
from utils import GloVe, pickle_save

from options import args_parser

def main(args):

    args_backup = deepcopy(args)

    # get embedding
    config = {}
    if args.dataset in ['agnews', ]:
        embed = GloVe(root=args.data_dir)
        config['embed'] = embed
        args.embed = embed

    # get dataset
    train_datasets, server_dataset, test_dataset = create_fed_dataset(args, config)

    # get system
    server = create_system(train_datasets, server_dataset, test_dataset, args)

    # run experiments
    server.run(args)

    content = {
        'args': args_backup,
        'history': server.history.data,
        'time': server.aggregator.running_times,
    }

    if args.history_path != 'none':
        pickle_save(content, args.history_path, mode='ab')

    if args.defense == 'boba_vis':
        c2 = {
            'args': args_backup,
            'ao_loss': server.aggregator.ao_loss,
            'es_loss': server.aggregator.es_loss,
            # 'best_loss': server.aggregator.best_loss,
            # 'worst_loss': server.aggregator.worst_loss,
        }
        print(c2)
        pickle_save(c2, '../history/ablation_vis/loss_record.pkl', mode='ab')

    if args.defense == 'boba_labeldist':
        c3 = {
            'args': args_backup,
            'fit_label_dist': server.aggregator.label_dists,
        }

        pickle_save(c3, f'../history/label_dist/{args.dataset}.pkl', mode='ab')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # the below three lines seem not necessary
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    setup_seed(args.seed)
    main(args)

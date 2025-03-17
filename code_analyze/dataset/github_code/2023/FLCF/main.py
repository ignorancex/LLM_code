#import algo_race
#import algo_rz
import numpy as np
import copy
import argparse
import torch
import central
import fair

DATASETS = [
    "Gowalla_m1",
    "Gowalla_100_m1",
    "Goodreads_100_m1",
    "AmazonProducts_100_m1",
    "debug_m1",
]

MODELS= [
    "NCFU",
    "NCF",
    "MF"
]


def parse_args():
    default_K = 10  # number of devices to train each round
    default_T = 500  # number of rounds
    default_E = 1  # number of epochs for all devices
    default_batch_size = 512  # batch size for all devices
    default_learn_rate = 0.001  # learning rate for all devices
    default_seed = 0  # random seed

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    req_grp = parser.add_argument_group(title="required arguments")
    req_grp.add_argument(
        "--algo",
        choices=["fair", "central"],
        required=True,
        help="algorithm to run: fedavg : Federated Average\n\n",
    )

    parser.add_argument("--early_stop_thold", action="store", type=str, default=None)
    parser.add_argument("--early_trend_window", action="store", type=int, default=100)
    parser.add_argument("--model", type=str, required=False, default="NCF", choices=MODELS)
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=32,
        help="embdding dim for user and books (default: {:d})".format(32),
    )
    # ncf
    parser.add_argument("--ncf_layers", type=int, required=False, default=1)
    parser.add_argument("--ncf_dropout", type=float, required=False, default=0)
    # 

    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--fair_compressions", type=str, required=False, default=None)
    parser.add_argument("--fair_data_loss_compression", type=str, required=False, default=None)
    parser.add_argument("--central_compression", type=str, required=False, default=None)
    parser.add_argument("--fair_randomize_user", action="store_true", default=False)
    parser.add_argument("--fair_use_fedhm", action="store_true", default=False)
    parser.add_argument("--fair_uniform_compression_sample", action="store_true", default=False)
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=1e-4,
        help="regularization of embeddings (default: {:f}".format(1e-4),
    )
    parser.add_argument(
        "--K",
        type=int,
        default=default_K,
        help="number of devices to train each round (default: {:d})".format(default_K),
    )
    parser.add_argument(
        "--T",
        type=int,
        default=default_T,
        help="number of rounds (default: {:d})".format(default_T),
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=20,
        help="number of rounds to eval after (default: {:d})".format(20),
    )
    parser.add_argument(
        "--E",
        type=int,
        default=default_E,
        help="number of epochs for all devices (default: {:d})".format(default_E),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_batch_size,
        help="batch size for all devices (default: {:d})".format(default_batch_size),
    )
    parser.add_argument(
        "--num_neg",
        type=int,
        default=1,
        help="num_neg sampled for each positive (default: {:d})".format(10),
    )
    parser.add_argument(
        "--learn_rate",
        type=float,
        default=default_learn_rate,
        help="learning rate for all devices (default: {:.4f})".format(
            default_learn_rate
        ),
    )

    parser.add_argument(
        "--max_compression",
        type=float,
        default=0,
        help="max_compression only config".format(0),
    )
    parser.add_argument('--compression', type=float, default=1.0, help="compression")
    parser.add_argument('--seed', type=int, default=default_seed, help="random seed (default: {:d})".format(default_seed))
    parser.add_argument('--hetero_roast',action='store_true', help='run roast heterogenous memory models')
    parser.add_argument('--no_consistent_hashing',action='store_true', help='consistent hashing', default=False)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.algo == "central":
        central.central(args)
    elif args.algo == "fair":
        fair.fair(args)
    #elif args.algo == "fedrz":
    #    algo_rz.fed_rz(args)


#
#
#

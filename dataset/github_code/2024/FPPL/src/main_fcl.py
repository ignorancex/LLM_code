import argparse
import json
import os

import setproctitle
from trainer_fcl import train
from utils.conf import config_path


def load_json(setting_path):
    with open(setting_path) as f:
        param = json.load(f)
    return param


def parse_args():
    parser = argparse.ArgumentParser("Federated continual learning", add_help=False)

    parser.add_argument("--exp", default="", type=str, help="Experiment name")

    parser.add_argument("--model", default="fppl", type=str, help="Experiment model")
    parser.add_argument(
        "--method", default="fppl", type=str, help="Federated learning method"
    )
    parser.add_argument("--dataset", default="imagenetr", type=str, help="Dataset")
    parser.add_argument(
        "--device_id", type=int, default=0, help="device to use for training / testing"
    )
    parser.add_argument("--seeds", default="2023,2024,2025", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.set_defaults(pin_memory=True)

    parser.add_argument("--save_checkpoint", action="store_true", default=False)

    parser.add_argument("--async_task", action="store_true")
    parser.add_argument("--sync_task", action="store_false", dest="--async_task")
    parser.set_defaults(async_task=False)

    parser.add_argument("--wo_pool", action="store_true")

    parser.add_argument("--wo_unified", action="store_true")
    parser.add_argument("--wo_fusion", action="store_true")
    parser.add_argument("--wo_debias", action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    default_param = load_json(
        os.path.join(config_path(), "datasets", f"{args.dataset}.json")
    )
    config_param = load_json(
        os.path.join(config_path(), "models", f"{args.model}.json")
    )
    args = vars(args)
    args.update(default_param)
    args.update(config_param)
    args = argparse.Namespace(**args)
    setproctitle.setproctitle(
        "{}_{}_{}".format(
            "async" if args.async_task else "sync", args.dataset, args.model
        )
    )
    train(args)


if __name__ == "__main__":
    main()

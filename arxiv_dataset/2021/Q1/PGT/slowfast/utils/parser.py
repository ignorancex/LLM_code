#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys

import slowfast.utils.checkpoint as cu
from slowfast.config.defaults import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--rank",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--world_size",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system.",
        default="auto",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_known_args()


def load_config(args, opts=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if opts is not None:
        # remove unkown args
        remove_lists = []
        for i in range(len(opts)):
            if opts[i].startswith("--"):
                remove_lists.append(opts[i])
                if "=" not in opts[i]:
                    remove_lists.append(opts[i+1])
            elif opts[i].startswith("-"):
                remove_lists.append(opts[i])
        for opt in remove_lists:
            print("Remove unkonwn args {}".format(opt))
            opts.remove(opt)
        cfg.merge_from_list(opts)

    # Inherit parameters from args.
    if hasattr(args, "world_size") and hasattr(args, "rank"):
        cfg.NUM_SHARDS = args.world_size
        cfg.SHARD_ID = args.rank
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.LOGS.DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.LOGS.DIR)
    cfg.freeze()
    return cfg

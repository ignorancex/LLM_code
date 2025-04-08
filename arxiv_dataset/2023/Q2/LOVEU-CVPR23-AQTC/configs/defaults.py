from yacs.config import CfgNode as CN
import argparse, sys

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide Q2A training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/example.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="other opts",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--wdb_project",
        type=str,
        default=None
    )
    parser.add_argument(
        "--wdb_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--wdb_offline",
        type=str2bool,
        default=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def load_config(args):
    # Setup cfg.
    cfg = CN(new_allowed=True)
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg.OUTPUT_DIR = args.cfg_file.replace("configs", "outputs").strip('.yaml')
    cfg.SAVEPATH = f'/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/outputs/cvpr_loveu2023/{args.wdb_name}/'
    return cfg, args

def build_config():
    cfg = load_config(parse_args())
    return cfg
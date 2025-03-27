from memesv4 import do_experiment

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("seed")
args = parser.parse_args()

seed = int(args.seed)

params = {
    "sigma": 4,
    "RES": 32,
    "mutation": 0.001,
    "select": False,
    "uniform_init": True,
    "output_dir": "no_sel_uni",
    "seed": seed,
}
do_experiment(params)

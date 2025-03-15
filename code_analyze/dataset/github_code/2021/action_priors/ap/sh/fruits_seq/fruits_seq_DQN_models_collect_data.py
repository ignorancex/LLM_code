import argparse
import json
import os
import subprocess
from ... import paths


def main(args):

    models_path = "data/fruits_seq_DQN_models"
    dataset_path = "data/fruits_seq_DQN_dsets"

    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    with open(paths.TASKS_FRUITS_SEQ, "r") as f:
        tasks = json.load(f)

    for c in tasks:

        model_name = "model_{:s}.pt".format(str(c))
        model_path = os.path.join(models_path, model_name)

        dataset_name = "dset_eps_0_5_{:s}.h5".format(str(c))
        tmp_dataset_path = os.path.join(dataset_path, dataset_name)

        print("{:s} goal".format(str(str(c))))
        print(model_path)
        subprocess.call([
            "python", "-m", "ap.scr.online.fruits_seq.run_DQN_dset",
            "with", "device={:s}".format(args.device), "goal={:s}".format(str(c)),
            "load_model_path={:s}".format(model_path), "dset_save_path={:s}".format(tmp_dataset_path),
            "eps=0.5", "num_exp=100000"
        ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)

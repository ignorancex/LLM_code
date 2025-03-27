import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from dntk import medimg
from tqdm import tqdm

from utils.metrics import foreground_dice_score


def _parse_cmd_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--gt_dir", required=True,
        help="Ground-truth directory.")
    arg_parser.add_argument("--pred_dir", required=True,
        help="Prediction directory.")
    arg_parser.add_argument("--df_path", required=True,
        help="Data info csv path.")
    args = arg_parser.parse_args()

    return args


def _evaluate(pid, gt_dir, pred_dir):
    gt_path = os.path.join(gt_dir, pid, f"{pid}_lungsegment.nii.gz")
    pred_path = os.path.join(pred_dir, f"{pid}_pred.nii.gz")
    airway_path = os.path.join(gt_dir, pid, f"{pid}_airway.nii.gz")
    artery_path = os.path.join(gt_dir, pid, f"{pid}_artery.nii.gz")
    vein_path = os.path.join(gt_dir, pid, f"{pid}_vein.nii.gz")
    interseg_path = os.path.join(gt_dir, pid, f"{pid}_interseg.nii.gz")
    gt = medimg.read_image(gt_path).array
    pred = medimg.read_image(pred_path).array
    airway = medimg.read_image(airway_path).array
    artery = medimg.read_image(artery_path).array
    vein = medimg.read_image(vein_path).array
    interseg_vein = medimg.read_image(interseg_path).array
    intraseg_vein = np.logical_and(vein > 0, interseg_vein == 0)

    total_dice = foreground_dice_score(gt, pred, 18)
    airway_dice = foreground_dice_score(gt[airway > 0], pred[airway > 0], 18)
    artery_dice = foreground_dice_score(gt[artery > 0], pred[artery > 0], 18)
    vein_dice = foreground_dice_score(gt[vein > 0], pred[vein > 0], 18)
    interseg_vein_dice = foreground_dice_score(gt[interseg_vein > 0],
        pred[interseg_vein > 0], 18)
    intraseg_vein_dice = foreground_dice_score(gt[intraseg_vein],
        pred[intraseg_vein], 18)
    result = {
        "pid": pid,
        "total": total_dice,
        "artery": artery_dice,
        "airway": airway_dice,
        "vein": vein_dice,
        "interseg_vein": interseg_vein_dice,
        "intraseg_vein": intraseg_vein_dice
    }

    return result


def main():
    args = _parse_cmd_args()

    info = pd.read_csv(args.df_path)
    info_val = info.loc[info.subset == "val"].reset_index(drop=True)
    info_test = info.loc[info.subset == "test"].reset_index(drop=True)

    cols = [
        "total",
        "airway",
        "artery",
        "vein",
        "interseg_vein",
        "intraseg_vein"
    ]

    result_val = []
    pid_val = sorted(info_val.pid.tolist())
    for pid in tqdm(pid_val):
        result_val.append(_evaluate(pid, args.gt_dir,
            os.path.join(args.pred_dir, "val")))
    result_val = pd.DataFrame(result_val)
    print(result_val[cols].mean(axis=0))
    result_val.to_csv(os.path.join(args.pred_dir, "results_val.csv"),
        index=False)

    result_test = []
    pid_test = sorted(info_test.pid.tolist())
    for pid in tqdm(pid_test):
        result_test.append(_evaluate(pid, args.gt_dir,
            os.path.join(args.pred_dir, "test")))
    result_test = pd.DataFrame(result_test)
    print(result_test[cols].mean(axis=0))
    result_test.to_csv(os.path.join(args.pred_dir, "results_test.csv"),
        index=False)


if __name__ == "__main__":
    main()

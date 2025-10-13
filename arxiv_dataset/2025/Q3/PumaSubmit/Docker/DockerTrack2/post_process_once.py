import os
import argparse
from process import main
import numpy as np
import matplotlib.pyplot as plt
def run(input_pth = '', pred_path = '',local_process=False):
    out_dir = pred_path
    prediction_path = pred_path#'/home/ntorbati/STORAGE/PumaDataset/preds/'#'/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/output/images/melanoma-tissue-mask-segmentation/'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=input_pth,
        help="path to wsi, glob pattern or text file containing paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=out_dir,
        help="output directory",
    )
    parser.add_argument(
        "--cp",
        type=str,
        default="/checkpoint" ,
        help="comma separated list of checkpoint folders to consider",
    )
    parser.add_argument(
        "--only_inference",
        action="store_true",
        help="split inference to gpu and cpu node/ only run inference",
    )
    parser.add_argument(
        "--metric", type=str, default="f1", help="metric to optimize for pp"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--tta",
        type=int,
        default=8,
        help="test time augmentations, number of views (4= results from 4 different augmentations are averaged for each sample)",
    )
    parser.add_argument(
        "--save_polygon",
        action="store_true",
        help="save output as polygons to load in qupath",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="tile size, models are trained on 256x256",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.96875,
        help="overlap between tiles, at 0.5mpp, 0.96875 is best, for 0.25mpp use 0.9375 for better results",
    )
    parser.add_argument(
        "--inf_workers",
        type=int,
        default=4,
        help="number of workers for inference dataloader, maximally set this to number of cores",
    )
    parser.add_argument(
        "--inf_writers",
        type=int,
        default=2,
        help="number of writers for inference dataloader, default 2 should be sufficient"
        + ", \ tune based on core availability and delay between final inference step and inference finalization",
    )
    parser.add_argument(
        "--pp_tiling",
        type=int,
        default=8,
        help="tiling factor for post processing, number of tiles per dimension, 8 = 64 tiles",
    )
    parser.add_argument(
        "--pp_overlap",
        type=int,
        default=256,
        help="overlap for postprocessing tiles, put to around tile_size",
    )
    parser.add_argument(
        "--pp_workers",
        type=int,
        default=16,
        help="number of workers for postprocessing, maximally set this to number of cores",
    )
    parser.add_argument(
        "--keep_raw",
        action="store_true",
        help="keep raw predictions (can be large files for particularly for pannuke)",
    )
    parser.add_argument(
        "--nuclei_dir",
        type=str,
        default=prediction_path,
        help="path to nuclei save folder",
    )
    parser.add_argument(
        "--p1",
        type=str,
        default=pred_path,
        help="path to prediction",
    )

    tissue_type = 'metastatis'


    parser.add_argument(
        "--tissue_type",
        type=str,
        default=tissue_type,
        help="path to prediction",
    )
    parser.add_argument(
        "--local",
        type=str,
        default=local_process,
        help="path to prediction",
    )
    parser.add_argument("--cache", type=str, default=None, help="cache path")
    params = vars(parser.parse_args())
    main(params)

if '__main__' == __name__:
    run()
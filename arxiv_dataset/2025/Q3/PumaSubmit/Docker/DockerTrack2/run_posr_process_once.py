import os
import argparse
from process import main as main1
from eval_nuclei import evaluate_files
import numpy as np
def run_process(images = '', pred_path=''):
    results = []
    counter = 0
    for image in images:
        new_path = image
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input",
            type=str,
            default=new_path,
            help="path to wsi, glob pattern or text file containing paths",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="/output/",
            help="output directory",
        )
        parser.add_argument(
            "--cp",
            type=str,
            default="/checkpoint",
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
            default=None,
            help="path to nuclei save folder",
        )
        parser.add_argument(
            "--p1",
            type=str,
            default=pred_path,
            help="path to prediction",
        )

        if 'primary' in pred_path:
            tissue_type = 'primary'
        else:
            tissue_type = 'metastatis'

        parser.add_argument(
            "--tissue_type",
            type=str,
            default=tissue_type,
            help="tissue type after classification",
        )

        # parser.add_argument(
        #     "--nuclei_pred_path",
        #     type=str,
        #     default=nuclei_pred_path,
        #     help="path to nuclei prediction",
        # )
        counter += 1

        parser.add_argument("--cache", type=str, default=None, help="cache path")
        params = vars(parser.parse_args())

        main1(params)
        GTs = "/home/ntorbati/STORAGE/PumaDataset/01_training_dataset_geojson_nuclei/"
        file_name = os.path.basename(image)
        gt_path = os.path.join(GTs, file_name)
        output_path = str(f"{gt_path}").replace(".tif", "_nuclei10.json")
        nuclei_metrics = evaluate_files(output_path,"/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/output/melanoma-10-class-nuclei-segmentation.json")
        print(nuclei_metrics)
        results.append(nuclei_metrics)
    # Compute average metrics (macro F1-score for nuclei and average DICE for tissue)
    f1_scores_per_class = {}
    TP_per_class = {}
    FP_per_class = {}
    FN_per_class = {}
    dice_scores_per_class = {}
    for result in results:
        nuclei_metrics = result
        for class_name, class_metrics in nuclei_metrics.items():
            if class_name not in ["micro", "macro"]:  # skip "micro" and "macro"
                if class_name not in f1_scores_per_class:  # initialize if not in dict
                    f1_scores_per_class[class_name] = 0
                    TP_per_class[class_name] = 0
                    FP_per_class[class_name] = 0
                    FN_per_class[class_name] = 0
                    dice_scores_per_class[class_name + "len"] = 0
                if class_name == np.str_('nuclei_epithelium'):
                    print(class_metrics['f1_score'])
                f1_scores_per_class[class_name] += class_metrics['f1_score']
                TP_per_class[class_name] += class_metrics['TP']
                FP_per_class[class_name] += class_metrics['FP']
                FN_per_class[class_name] += class_metrics['FN']
                dice_scores_per_class[class_name + "len"] += 1

    # Compute the average F1-score for each nuclei class


    for class_name in f1_scores_per_class:
        f1_scores_per_class[class_name] /= (counter)#dice_scores_per_class[class_name + "len"]
    print('f1_nuclei' , f1_scores_per_class)
    print('TP_per_class' , TP_per_class)
    print('FP_per_class' , FP_per_class)
    print('FN_per_class' , FN_per_class)

    # Compute overall macro F1-score by averaging the per-class F1-scores
    macro_f1 = np.mean(list(f1_scores_per_class.values()))
    print('macro_f1' , macro_f1)


if __name__ == '__main__':
    path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/test/'
    out_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/output/images/melanoma-tissue-mask-segmentation/'
    #

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # all_tissue_data = np.sort([path + image for image in os.listdir(path)])
    images = [path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    res = [out_path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    run_process(images,out_path)


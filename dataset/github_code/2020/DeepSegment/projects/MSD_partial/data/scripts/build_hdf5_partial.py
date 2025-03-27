# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import os
import sys

# search for utilities module under root dir
DIR_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
DIR_ROOT = os.path.abspath(DIR_ROOT)

sys.path.insert(0, DIR_ROOT)
import utilities

# quick hack: reuse functions from MSD_full's build_hdf5.py
from projects.MSD_full.data.scripts.build_hdf5 import process_data
sys.path.pop(0)

# define paths
DIR_NII = os.path.join(os.path.dirname(__file__), "..", "nii")
DIR_NII = os.path.abspath(DIR_NII)

DIR_HDF5 = os.path.join(os.path.dirname(__file__), "..", "hdf5")
DIR_HDF5 = os.path.abspath(DIR_HDF5)

# define parameters controlling script's performance
HDF5_RESIZE_CHUNK = 500
HDF5_WRITE_FREQ = 20  # 300

# define train percents
TRAIN_PERCENTS = [2.5, 5.0, 10.0, 20.0, 40.0]

# define list of slice directions and other image properties
RESAMPLE_CONFIGS = []
RESAMPLE_CONFIGS += [
    {
        "slice_dim": 0,
        "slice_dim_label": "sagittal",
        "target_size": (256, 256),
        "target_res": (3.5, 1.5, 1.5),
        "crop": True
    }
]
RESAMPLE_CONFIGS += [
    {
        "slice_dim": 1,
        "slice_dim_label": "coronal",
        "target_size": (256, 256),
        "target_res": (1.5, 3.5, 1.5),
        "crop": True
    }
]
RESAMPLE_CONFIGS += [
    {
        "slice_dim": 2,
        "slice_dim_label": "axial",
        "target_size": (256, 256),
        "target_res": (1.5, 1.5, 3.5),
        "crop": True
    }
]

# define datasets and the correspoding label map
DATASETS = []
DATASETS += [
    {
        "name": "Liver",
        "src_dir": os.path.join(DIR_NII, "Task03_Liver"),
        "dest_dir": DIR_HDF5,
        "label_map": {
            0: 0,
            1: 1,
            2: 2
        },
        "max_subj": {
            "train": None,
            "test": None
        }
    }
]
DATASETS += [
    {
        "name": "Pancreas",
        "src_dir": os.path.join(DIR_NII, "Task07_Pancreas"),
        "dest_dir": DIR_HDF5,
        "label_map": {
            0: 0,
            1: 3,
            2: 4
        },
        "max_subj": {
            "train": None,
            "test": None
        }
    }
]
DATASETS += [
    {
        "name": "Spleen",
        "src_dir": os.path.join(DIR_NII, "Task09_Spleen"),
        "dest_dir": DIR_HDF5,
        "label_map": {
            0: 0,
            1: 5
        },
        "max_subj": {
            "train": None,
            "test": None
        }
    }
]


def main():
    # create logger
    logger = utilities.log.create_or_get_logger()

    # loop over each resample configuration
    for resample_config in RESAMPLE_CONFIGS:
        # print/log info
        logger.info("processing images with configurations:")
        utilities.log.print_dict(resample_config, logger=logger, indent=1)

        # loop over each datasets:
        for dataset in DATASETS:
            # print/log info
            msg = "  " * 1 + "- processing dataset: {:}"
            msg = msg.format(dataset["name"])
            logger.info(msg)
            utilities.log.print_dict(dataset, logger=logger, indent=2)

            # extract src and dest dirs
            dataset_src_dir = str(dataset["src_dir"])
            dataset_dest_dir = str(dataset["dest_dir"])

            # loop over train percent
            for train_percent in TRAIN_PERCENTS:
                # modify src and dest dirs
                dataset["src_dir"] = os.path.join(
                    dataset_src_dir, "trainpercent_%s" % train_percent
                )
                dataset["dest_dir"] = os.path.join(
                    dataset_dest_dir, "trainpercent_%s" % train_percent
                )

                # actually process data
                process_data(resample_config, dataset)


if __name__ == "__main__":
    # create logger
    logger = utilities.log.create_or_get_logger(level="DEBUG")

    main()

# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import os
import sys

# search for utilities module under root dir
DIR_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
DIR_ROOT = os.path.abspath(DIR_ROOT)

sys.path.insert(0, DIR_ROOT)
import utilities
sys.path.pop(0)

# define paths
DIR_NII_MSD2018 = os.path.join(DIR_ROOT, "data", "MSD2018")

DIR_NII_PROJ = os.path.join(os.path.dirname(__file__), "..", "nii")
DIR_NII_PROJ = os.path.abspath(DIR_NII_PROJ)


def generate_nii_filepaths(dataset, train_percent):
    logger = utilities.log.create_or_get_logger()

    # checkings
    if (train_percent < 0.0) or (train_percent > 80.0):
        msg = "train_percent must be between 0 and 80"
        logger.error(msg)
        raise ValueError(msg)

    # obtain a list of all available feature/label images
    imgs = os.path.join(DIR_NII_MSD2018, dataset, "imagesTr")
    imgs = sorted(os.listdir(imgs))

    labels = os.path.join(DIR_NII_MSD2018, dataset, "labelsTr")
    labels = sorted(os.listdir(labels))

    for filename_img, filename_label in zip(imgs, labels):
        if filename_img != filename_label:
            msg = "unexpected filename mismatch "
            msg += "between feature and label images"
            logger.error(msg)
            raise ValueError(msg)

    # make fullpaths
    imgs = [
        os.path.join(DIR_NII_MSD2018, dataset, "imagesTr", img) for img in imgs
    ]
    labels = [
        os.path.join(DIR_NII_MSD2018, dataset, "labelsTr", label)
        for label in labels
    ]

    # shuffle images deterministically
    # (1) divide images into 40 sets
    nset_total = 40

    img_sets = [imgs[i::nset_total] for i in range(nset_total)]
    label_sets = [labels[i::nset_total] for i in range(nset_total)]

    # (2) train-test split; every 5th set will be the test set
    img_test = img_sets[4::5]
    label_test = label_sets[4::5]

    img_train = [x for i, x in enumerate(img_sets) if ((i + 1) % 5 != 0)]
    label_train = [x for i, x in enumerate(label_sets) if ((i + 1) % 5 != 0)]

    # (3) flatten 2D lists
    img_test = [i for s in img_test for i in s]
    label_test = [l for s in label_test for l in s]

    img_train = [i for s in img_train for i in s]
    label_train = [l for s in label_train for l in s]

    # (4) determine number of images needed to achieve train_percent
    nimg_train = round(train_percent / 80.0 * len(img_train))

    # (5) create train set
    img_train = img_train[:nimg_train]
    label_train = label_train[:nimg_train]

    # create dictionary and return
    filepaths = {}
    filepaths['image_train'] = img_train
    filepaths['image_test'] = img_test
    filepaths['label_train'] = label_train
    filepaths['label_test'] = label_test

    return filepaths


# main
def main():
    logger = utilities.log.create_or_get_logger()

    # define dataset names
    datasets = []
    datasets += ["Task03_Liver"]
    datasets += ["Task07_Pancreas"]
    datasets += ["Task09_Spleen"]

    # define train percents
    train_percents = [2.5, 5.0, 10.0, 20.0, 40.0]

    # loop over datasets
    for dataset in datasets:
        logger.info("  " * 1 + "- dataset: {:}".format(dataset))

        # loop over train_percents
        for train_percent in train_percents:
            logger.info("  " * 2 + "- train_percent: {:}".format(dataset))

            # get sets of nii filepaths
            filepaths = generate_nii_filepaths(dataset, train_percent)

            # loop over nii filepaths sets
            for nii_setname, nii_fullpaths in filepaths.items():
                logger.info("  " * 3 + "- set: {:}".format(nii_setname))

                # create destination directory if not exists
                dir_img_dest = os.path.join(
                    DIR_NII_PROJ,
                    dataset,
                    "trainpercent_%.1f" % train_percent,
                    nii_setname,
                )
                if not os.path.exists(dir_img_dest):
                    os.makedirs(dir_img_dest)

                # loop over filepaths
                for nii_fullpath in nii_fullpaths:
                    nii_filename = os.path.basename(nii_fullpath)

                    msg = "  " * 4 + "- image: {:}".format(nii_filename)
                    logger.info(msg)

                    # determine relative path to src
                    nii_src_relpath = os.path.relpath(
                        nii_fullpath, dir_img_dest
                    )

                    # define destination path
                    nii_dest_fullpath = os.path.join(
                        dir_img_dest, nii_filename
                    )

                    # delete links if already exists
                    if os.path.islink(nii_dest_fullpath):
                        os.remove(nii_dest_fullpath)

                    # make symlinks
                    os.symlink(nii_src_relpath, nii_dest_fullpath)


if __name__ == "__main__":
    logger = utilities.log.create_or_get_logger()
    logger.info("creating symlinks")

    main()

    logger.info("done!")

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

# create logger
logger = utilities.log.create_or_get_logger()

# define paths
DIR_NII_MSD2018 = os.path.join(DIR_ROOT, "data", "MSD2018")

DIR_NII_PROJ = os.path.join(os.path.dirname(__file__), "..", "nii")
DIR_NII_PROJ = os.path.abspath(DIR_NII_PROJ)

# define train/test split
TRAIN_TEST_SPLIT = 5

# define dataset names
datasets = []
datasets += ["Task03_Liver"]
datasets += ["Task07_Pancreas"]
datasets += ["Task09_Spleen"]

# main
logger.info("creating symlinks:")
for dataset in datasets:
    logger.info("  " * 1 + "- {:}".format(dataset))

    # loop over each subjects
    logger.info("  " * 2 + "- processing subjects")

    filenames_img = os.path.join(DIR_NII_MSD2018, dataset, "imagesTr")
    filenames_img = sorted(os.listdir(filenames_img))

    filenames_label = os.path.join(DIR_NII_MSD2018, dataset, "labelsTr")
    filenames_label = sorted(os.listdir(filenames_label))

    counter_subj = 0
    for filename_img, filename_label in zip(filenames_img, filenames_label):
        # verify that filenames are the same
        if filename_img != filename_label:
            msg = "  " * 3 + "- unexpected filename mismatch between "
            msg += "imageTr ({:}) and labelTr ({:}); skipping..."
            msg = msg.format(filename_img, filename_label)
            logger.error(msg)

            quit()

        # determine train/test
        if ((counter_subj + 1) % TRAIN_TEST_SPLIT) == 0:
            tt = 'test'
        else:
            tt = 'train'

        # define full paths
        dir_img_src = os.path.join(DIR_NII_MSD2018, dataset, "imagesTr")
        fullpath_img_src = os.path.join(dir_img_src, filename_img)
        dir_img_dest = os.path.join(DIR_NII_PROJ, dataset, "image_%s" % tt)
        fullpath_img_dest = os.path.join(dir_img_dest, filename_img)

        dir_label_src = os.path.join(DIR_NII_MSD2018, dataset, "labelsTr")
        fullpath_label_src = os.path.join(dir_label_src, filename_label)
        dir_label_dest = os.path.join(DIR_NII_PROJ, dataset, "label_%s" % tt)
        fullpath_label_dest = os.path.join(dir_label_dest, filename_label)

        # determine relative paths
        relpath_img_src = os.path.relpath(fullpath_img_src, dir_img_dest)
        relpath_label_src = os.path.relpath(fullpath_label_src, dir_label_dest)

        # create directories if not exists
        for d in [dir_img_dest, dir_label_dest]:
            if not os.path.exists(d):
                os.makedirs(d)

        # delete links if already exists
        for l in [fullpath_img_dest, fullpath_label_dest]:
            if os.path.islink(l):
                os.remove(l)

        # make symlinks
        logger.info("  " * 3 + "- [{:5s}] {:}".format(tt, filename_img))

        os.symlink(relpath_img_src, fullpath_img_dest)
        os.symlink(relpath_label_src, fullpath_label_dest)

        # increment counter
        counter_subj += 1

logger.info("done!")

# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import os
import sys
import tarfile

# search for utilities module under root dir
DIR_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
DIR_ROOT = os.path.abspath(DIR_ROOT)

sys.path.insert(0, DIR_ROOT)
import utilities
sys.path.pop(0)

# # download required data from the following URLs and place under the folder ../nii/
# URL_LIVER = "https://drive.google.com/uc?export=download&id=1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu"
# URL_PANCREAS = "https://drive.google.com/uc?export=download&id=1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL"
# URL_SPLEEN = "https://drive.google.com/uc?export=download&id=1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE"

# create logger
logger = utilities.log.create_or_get_logger()

# define paths
DIR_NII = os.path.join(os.path.dirname(__file__), "..")
DIR_NII = os.path.abspath(DIR_NII)

DIR_TAR = os.path.dirname(__file__)
DIR_TAR = os.path.abspath(DIR_TAR)

# unpack downloaded tarballs
filenames_tar = []
filenames_tar += ["Task03_Liver.tar"]
filenames_tar += ["Task07_Pancreas.tar"]
filenames_tar += ["Task09_Spleen.tar"]

logger.info("unpacking tarballs:")
for filename_tar in filenames_tar:
    logger.info("  " * 1 + "- {:}".format(filename_tar))

    filepath_tar = os.path.join(DIR_TAR, filename_tar)
    if not os.path.exists(filepath_tar):
        logger.error("  " * 2 + "- tarball missing")
        logger.error("  " * 2 + "- download from http://medicaldecathlon.com/")
        logger.error("  " * 2 + "- and place under {:}".format(DIR_NII))
        continue

    file_tar = tarfile.open(filepath_tar)

    # extract all except system files
    members = file_tar.getmembers()
    members = [m for m in members if os.path.basename(m.name)[0] != '.']
    file_tar.extractall(DIR_NII, members=members)

    file_tar.close()

logger.info("done!")

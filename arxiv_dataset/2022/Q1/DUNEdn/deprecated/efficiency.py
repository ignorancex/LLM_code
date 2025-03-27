# This file is part of DUNEdn by M. Rossi
import os
import logging
import argparse
from time import time as tm
import numpy as np
import torch
from skimage.feature import canny

# instantiate logger
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir_name",
    "-p",
    default="../datasets",
    type=str,
    help="Directory path to datasets",
)


def draw_results(a, b, c, d):
    tot = a + b + c + d
    logger.info("Over a total of %d pixels:\n", tot)
    logger.info("------------------------------------------------")
    logger.info("|{:>20}|{:>12}|{:>12}|".format("", "Signal", "Background"))
    logger.info("------------------------------------------------")
    logger.info(
        "|{:>20}|{:>12.4e}|{:>12.4e}|".format("Predicted signal", a / tot, b / tot)
    )
    logger.info("------------------------------------------------")
    logger.info(
        "|{:>20}|{:>12.4e}|{:>12.4e}|".format("Predicted background", c / tot, d / tot)
    )
    logger.info("------------------------------------------------")
    logger.info("{:>21}|{:>12}|{:>12}|".format("", "Sensitivity", "Specificity"))
    logger.info("                     ---------------------------")
    logger.info("{:>21}|{:>12.4e}|{:>12.4e}|".format("", a / (a + c), d / (b + d)))
    logger.info("                     ---------------------------\n")


def main(dir_name):

    for s in ["readout_", "collection_"]:
        for ss in ["train", "val", "test"]:
            clear = np.array(
                torch.load(os.path.join(dir_name, "clear_planes", "".join([s, ss])))
            )

            edges = []
            for c in clear:
                edges.append(canny(np.array(c)).astype(float))

            edges = np.stack(edges, 0)
            clear[clear != 0] = 1

            d = clear * 10 - edges

            tp = (d[d == 9].shape)[0]
            tn = (d[d == 0].shape)[0]
            fn = (d[d == 10].shape)[0]
            fp = (d[d == -1].shape)[0]

            logger.info("".join(["\n", s, ss]))
            draw_results(tp, fp, fn, tn)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = tm()
    main(**args)
    logger.info("\nProgram done in %f", (tm() - start))

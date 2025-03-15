# This file is part of DUNEdn by M. Rossi
""" This module returns the ROIs from recob::hit objects"""
import os
import argparse
import logging
import glob
from time import time as tm
import numpy as np

# instantiate logger
logger = logging.getLogger(__name__)

n_channels = 2560
n_induction = 800
n_collection = 960
n_apas = 6
n_ticks = 6000


def process_hits(fname):
    hits = np.load(os.path.join(fname))

    ROIs = np.zeros((n_apas, n_collection, n_ticks))

    for apa in range(n_apas):
        first_ch = n_channels * apa + 2 * n_induction
        last_ch = n_channels * (apa + 1)
        mask = np.logical_and(hits[:, 0] >= first_ch, hits[:, 0] < last_ch)

        for hit in hits[mask]:
            ch = hit[0] - first_ch
            ROIs[apa, ch, hit[1] : hit[2]] = 1
    return ROIs[:, None]


def process_wires(fname):
    wires = np.load(os.path.join(fname))

    coll_wires = []
    for apa in range(n_apas):
        coll_wire = np.zeros((n_collection, n_ticks))
        first_ch = n_channels * apa + 2 * n_induction
        last_ch = n_channels * (apa + 1)
        mask = np.logical_and(wires[:, 1] >= first_ch, wires[:, 1] < last_ch)

        valid = wires[mask]
        coll_wire[valid[:, 1] - first_ch] = valid[:, 2:]
        coll_wires.append(coll_wire[None])

    return coll_wires


def process_hits_and_dump(dirname):
    """
    Processes hits to cast them into an array of shape (n_channels, n_ticks).
    Saves an array named collection_hits in benchmark/hits folder with all
    collection region of interests in the datasetto be used as a benchmark.
    Shape: (ALL_APAS, n_collection, n_ticks)
    """
    input_dir = os.path.join(dirname, "pandora_out")
    output_dir = os.path.join(dirname, "hits")
    fnames = glob.glob(os.path.join(input_dir, "*recobhits*"))

    ROIs = []
    for fname in fnames:
        ROIs.append(process_hits(fname))

    ROIs = np.concatenate(ROIs)
    outname = os.path.join(output_dir, "pandora_collection_hits")
    np.save(outname, ROIs)


def process_wires_and_dump(dirname):
    """
    Saves all collection planes array into benchmark/wires folder.
    Saves arrays named ..._collection.npy into benchmark/pandora_out folder.
    Shape: (ALL_APAS, n_collection, n_ticks)

    Note: the ADC counts are normalized arbitrarily
    """
    input_dir = os.path.join(dirname, "pandora_out")
    output_dir = os.path.join(dirname, "wires")
    fnames = glob.glob(os.path.join(input_dir, "*recobwires*"))

    wires = []
    for fname in fnames:
        wires.extend(process_wires(fname))
    wires = np.stack(wires)
    outname = os.path.join(output_dir, "pandora_collection_wires")
    np.save(outname, wires)


def main(dirname):
    process_hits_and_dump(dirname)

    process_wires_and_dump(dirname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirname",
        "-p",
        default="../datasets/backup/test/benchmark",
        help="Directory path to datasets",
    )
    args = vars(parser.parse_args())
    start = tm()
    main(**args)
    logger.info("Program done in %f", tm() - start)

import os
import logging
import glob
import argparse
import time as tm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# instantiate logger
logger = logging.getLogger(__name__)

tdc_min = 0
tdc_max = 6000
channels = 2560 * 6


def process_depo(dir_name):
    f_wire = glob.glob(os.path.join(dir_name, "wire/*"))
    f_simch = glob.glob(os.path.join(dir_name, "simch/*"))
    for f_w, f_s in zip(f_wire, f_simch):
        simch = np.load(f_s)
        wire = np.load(f_w)

        # ensure energy deposits are inside tdc window
        simch = simch[simch[:, 2] < tdc_max]
        simch = simch[simch[:, 2] > tdc_min]

        # charge and energy deposits on wires
        ch_depo = np.zeros([channels, tdc_max - tdc_min])
        en_depo = np.zeros_like(ch_depo)

        for i in simch:
            ch_depo[int(i[1]), int(i[2])] += i[4]
            en_depo[int(i[1]), int(i[2])] += i[5]

        fname = f_s.split("/")
        fname[-1] = fname[-1].replace("simch", "charge")
        fname = "/".join(fname)
        np.save(fname, ch_depo)
        print("Save file at: %s" % fname)
        fname = fname.replace("charge", "energy")
        np.save(fname, en_depo)
        print("Save file at: %s" % fname)

        roi_depo = np.zeros_like(ch_depo)
        roi_depo[wire[:, 1]] = wire[:, 2:]
        fname = f_w.split("/")
        fname[-1] = fname[-1].replace("wire", "roi")
        fname = "/".join(fname)
        np.save(fname, roi_depo)
        print("Save file at: %s" % fname)


def main(dir_name):
    # process_depo(dir_name)

    # check if things were done right
    f_raw = glob.glob(os.path.join(dir_name, "raw/raw*"))[0]
    f_ch = glob.glob(os.path.join(dir_name, "simch/charge*"))[0]

    raw = np.load(f_raw)[:, 2:]
    ch = np.load(f_ch)

    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=4, wspace=1.5, hspace=1.5)
    ax = fig.add_subplot(gs[0, :2])
    z = ax.imshow(ch[1600:2560])
    plt.colorbar(z, ax=ax)
    ax.set_title("Charge Deposition")

    ax = fig.add_subplot(gs[1, :2])
    z = ax.imshow(raw[1600:2560])
    plt.colorbar(z, ax=ax)
    ax.set_title("Raw Digits")

    ax = fig.add_subplot(gs[0, 2:])
    ax.plot(ch[2500], lw=0.2)
    ax.set_title("Wire 2500, Energy")

    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(raw[2500], lw=0.2)
    ax.set_title("Raw Wire 2500, Raw")

    plt.savefig("preprocess/simch_charge.png", dpi=300, bbox_inches="tight")
    plt.close()

    f_roi = glob.glob(os.path.join(dir_name, "wire/roi*"))[0]
    roi = np.load(f_roi)

    fig = plt.figure()
    plt.plot(roi[2500], lw=0.2)
    plt.savefig("preprocess/wire_roi.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name",
        "-p",
        default="../datasets/20200901",
        type=str,
        help="Directory path to datasets",
    )
    args = vars(parser.parse_args())
    start = tm.time()
    main(**args)
    print("Program done in %f" % (tm.time() - start))

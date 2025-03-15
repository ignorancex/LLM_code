"""
    This module reproduces the results obtained in arXiv:2103.01596

    Usage:
    
    ```
    python compute_denoising_performance.py -i <input.npy> -o <output.npy> \
        -t <target.npy> -m <model> [--model-path]
    ```

    Available models options are:
        - cnn (Convolutional Neural Network)
        - gcnn (Graph Convolutional Neural Network)
        - uscg (U-shaped Self Constructing Graph Network)
        - id (Identity Network)
"""
import argparse
from pathlib import Path
from time import time as tm
import numpy as np
from dunedn.inference.inference import (
    add_arguments_inference,
    compare_performance_dn,
    thresholding_dn,
)


def main(args):
    """
    Parameters
    ----------
        - args: Namespace, the inference arguments
    """
    # inference pass
    evt_dn = args.func(args)

    target = np.load(args.target)[:, 2:]

    # denoised event can be thresholded, comment this line to compare bare waveforms
    evt_dn = thresholding_dn(evt_dn)

    compare_performance_dn(evt_dn, target, args.dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Denoising benchmark for arXiv:2103.01596",
        add_help="loads event from file, denoise it and computes performance metrics",
    )
    add_arguments_inference(parser)
    parser.add_argument(
        "-t",
        type=Path,
        required=True,
        metavar="TARGET",
        dest="target",
        help="path to the event file containing ground truths",
    )
    args = parser.parse_args()

    start = tm()
    main(args)
    print(f"Program done in {tm()-start} s")

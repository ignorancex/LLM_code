"""Combine the partial features files into a single file."""
# Python
import argparse
from glob import glob
import os
# Third-party
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--feats-file-prefix", type=str, required=True)
parser.add_argument("--outfile", type=str, default=None, help="Output file")
parser.add_argument("--overwrite", action="store_true")


def main(args):
    if args.outfile is None:
        args.outfile = f"{args.feats_file_prefix}.pkl"

    if os.path.isfile(args.outfile) and not args.overwrite:
        print(f"Output file {args.outfile} already exists."
              +"Add the overwrite flag if you wish to overwrite the existing file.")
        exit(0)

    # Detect all file parts
    feats_files = glob(f"{args.feats_file_prefix}*.part")
    feats, labels = [], []
    for fn in feats_files:
        curr = torch.load(fn, weights_only=False)
        feats.append(curr["feats"])
        labels.append(curr["labels"])
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    print(f"Saving combined features to {args.outfile}")
    torch.save(dict(feats=feats, labels=labels), args.outfile)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
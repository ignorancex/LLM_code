# pylint: disable=no-name-in-module

'''Run this file using 
    python run.py
'''
import sys

try:
    from flint import arb, ctx
except ModuleNotFoundError:
    print("\033[1;91mError:\033[0m python-flint not installed")
    sys.exit()

from argparse import ArgumentParser, SUPPRESS
from dir24isoperim import verify_all, b0, b1, c0, init_prec, Output, parse_aux, write_labels

if __name__ == "__main__":
    parser = ArgumentParser(description="Verify estimates in DIR24.")
    parser.add_argument("--beta", type=str, default=b0, dest="beta",
                help=f"Value for beta0; should be in [0.5, {float(b1):f}] (default: {float(b0):f})")
    parser.add_argument("--c", type=str, default=c0, dest="c",
                help=f"Value for c0; should be in [0,1] (default: {float(c0):f})")
    parser.add_argument("--prec", type=int, default=ctx.prec, dest="prec",
                help=f"Working precision in bits (default: {ctx.prec:d})")
    parser.add_argument("--filename", type=str, default="", dest="filename",
                help="Write partition data to a file.")
    parser.add_argument("--parse-aux", const=True, default=False, action="store_const",
                dest="parse_aux", help=SUPPRESS)
    args = parser.parse_args()

    if args.parse_aux:
        print("Parsing auxiliary file")
        labels = parse_aux()
        if len(labels) > 0:
            print(f"Writing {len(labels):d} labels to file")
            write_labels(labels)
        sys.exit()

    init_prec(args.prec)
    print(f"Working precision: {ctx.prec:d}")
    beta = arb(args.beta)
    c = arb(args.c)
    print(f"beta0 = {beta}")
    print(f"c0 = {c}")
    if args.filename:
        if Output.get_instance().open(args.filename):
            print(f"Partition data will be written to '{args.filename}'")

    verify_all(beta, c)

    Output.get_instance().close()

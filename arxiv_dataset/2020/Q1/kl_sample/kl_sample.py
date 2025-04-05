"""
kl_sample: a code that sample cosmological parameters using
lensing data (for now CFHTlens). It performs a KL transform
to compress data. There are three main modules:
a - prep_real: prepare data in real space and store them
    inside the repository. Once they are there it is no
    longer needed to rerun it;
b - prep_fourier: prepare data in fourier space and store
    them inside the repository. Once they are there it is
    no longer needed to rerun it;
c - run: given the data (either in real or fourier space)
    and some additional input parameter do the actual run.
    Implemented samplers: emcee or single_point
"""

import sys
from kl_sample.io import argument_parser

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Call the parser
    args = argument_parser()

    # Redirect the run to the correct module
    if args.mode == 'prep_real':
        from kl_sample.prep_real import prep_real
        sys.exit(prep_real(args))
    if args.mode == 'prep_fourier':
        from kl_sample.prep_fourier import prep_fourier
        sys.exit(prep_fourier(args))
    if args.mode == 'plots':
        from kl_sample.plots import plots
        sys.exit(plots(args))
    if args.mode == 'run':
        from kl_sample.run import run
        sys.exit(run(args))
    if args.mode == 'get_kl':
        from kl_sample.get_kl import get_kl
        sys.exit(get_kl(args))

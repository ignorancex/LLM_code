# -*- coding: utf-8 -*-
import argparse

from .gui import plot_plotly
from .sme import SME_Structure
from .solve import SME_Solver
from .synthesize import Synthesizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "PySME",
        description=(
            "Synthesizes stellar spectra and determines best fit parameters "
            "to observations. The results are stored in the same file as the input."
        ),
    )
    parser.add_argument("file", help=".sme input file")
    parser.add_argument(
        "-s",
        "--synthesize",
        action="store_true",
        help="Only synthesize the spectrum, no fitting",
    )
    parser.add_argument("-o", "--output", help="Store the output to this file instead")
    parser.add_argument(
        "-p",
        "--plot",
        nargs="?",
        const=True,
        default=False,
        help="Create a plot with the results",
    )
    args = parser.parse_args()
    print(args)

    synthesize_only = args.synthesize
    filename = args.file
    output = args.output
    plot = args.plot
    if output is None:
        output = filename

    sme = SME_Structure.load(filename)
    if synthesize_only:
        syn = Synthesizer()
        sme = syn.synthesize_spectrum(sme)
    else:
        solver = SME_Solver()
        sme = solver.solve(sme, sme.fitparameters)
    sme.save(output)

    if plot:
        fig = plot_plotly.FinalPlot(sme)
        if plot is True:
            fig.show()
        else:
            fig.save(filename=plot, auto_open=False)

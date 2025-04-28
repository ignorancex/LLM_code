# -*- coding: utf-8 -*-
"""
Example of a parallel execution script

The script is called with command line parameters that are
then used to modify the runs. Here those are Teff, logg, and [M/H], but they
can be anything, e.g. names of different input spectra.

Here we simply synthesize different spectra for different input paramters,
but it can also be used to analyse different spectra, etc.
"""
import sys
from os import makedirs
from os.path import dirname, join, realpath

from pysme import sme as SME
from pysme import util
from pysme.gui import plot_plotly
from pysme.synthesize import synthesize_spectrum

if __name__ == "__main__":

    # Read the command line parameters
    # the first parameter is always the command used to call this script
    # i.e. the name of this file, thus we ignore it
    # The expected number of parameters is then 1 + 3 = 4
    if len(sys.argv) == 4:
        # Here we parse the input parameters
        # since they are supposed to be numbers, we convert them from strings
        teff, logg, monh = sys.argv[1], sys.argv[2], sys.argv[3]
        teff = float(teff)
        logg = float(logg)
        monh = float(monh)
    else:
        print(
            "No (or the wrong number of) command line "
            "parameters passed to the script"
        )
        raise RuntimeError

    # Define the location of all your files
    # here we use different directories for the different
    # products
    target = f"sun_{teff}_{logg}_{monh}"
    examples_dir = dirname(realpath(__file__))
    in_file = join(examples_dir, "sun_6440_grid.inp")
    out_file = join(examples_dir, f"out/{target}.sme")
    plot_file = join(examples_dir, f"plot/{target}.html")
    log_file = join(examples_dir, f"log/{target}.log")

    # Start the logging to the file
    makedirs(dirname(log_file), exist_ok=True)
    util.start_logging(log_file)

    # Load your existing SME structure or create your own
    sme = SME.SME_Structure.load(in_file)

    # Change parameters if your want
    sme.vrad = 0.35
    sme.vrad_flag = "whole"
    sme.cscale_flag = "linear"
    sme.cscale_type = "match"

    # Apply the parameters set via the command line
    sme.teff = teff
    sme.logg = logg
    sme.monh = monh

    # Start SME solver
    sme = synthesize_spectrum(sme)

    # Save results
    makedirs(dirname(out_file), exist_ok=True)
    sme.save(out_file)

    # Plot results
    makedirs(dirname(plot_file), exist_ok=True)
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=plot_file)

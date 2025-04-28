# -*- coding: utf-8 -*-
""" Minimum working example of an SME script
"""
from os.path import dirname, join

from pysme.gui import plot_plotly
from pysme.linelist.vald import ValdFile
from pysme.sme import SME_Structure
from pysme.synthesize import synthesize_spectrum

if __name__ == "__main__":

    # Create an empty SME Structure
    sme = SME_Structure()

    # Optional: set any stellar parameters
    sme.teff = 6000

    # Load the linelist
    examples_dir = dirname(__file__)
    vald_file = join(examples_dir, "sun.lin")
    sme.linelist = ValdFile(vald_file)

    # Set the wavelength region to match the range of the linelist
    wmin = sme.linelist.wlcent[0]
    wmax = sme.linelist.wlcent[-1]
    sme.wran = [wmin, wmax]

    # Calculate the synthetic spectrum
    sme = synthesize_spectrum(sme)

    # Save results
    out_file = join(examples_dir, "minimum_result.sme")
    sme.save(out_file)

    # Plot results
    plot_file = join(examples_dir, "minimum_plot.html")
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=plot_file)

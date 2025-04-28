# -*- coding: utf-8 -*-
""" Minimum working example of an SME script
"""
import os.path
import time
from ctypes import resize

import matplotlib.pyplot as plt
import numpy as np

from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.gui import plot_plotly
from pysme.linelist.vald import ValdFile
from pysme.persistence import save_as_idl
from pysme.solve import solve
from pysme.synthesize import Synthesizer, synthesize_spectrum

if __name__ == "__main__":

    # Define the location of all your files
    # this will put everything into the example dir
    target = "sun"
    examples_dir = os.path.dirname(os.path.realpath(__file__))
    in_file = os.path.join(examples_dir, "sun_6440_grid.inp")
    out_file = os.path.join(examples_dir, f"{target}.sme")
    plot_file = os.path.join(examples_dir, f"{target}.html")
    log_file = os.path.join(examples_dir, f"{target}.log")

    # Start the logging to the file
    # util.start_logging(log_file)

    # Load your existing SME structure or create your own
    sme = SME.SME_Structure.load(in_file)
    sme.save("test.sme")
    sme = SME.SME_Structure.load("test.sme")
    sme.abund = Abund(0, "asplund2009")
    sme.linelist = ValdFile(os.path.join(examples_dir, "sun.lin"))

    # Change parameters if your want
    sme.vsini = 0
    sme.vrad = 0.35
    sme.vrad_flag = "whole"
    sme.cscale_flag = "linear"
    sme.cscale_type = "match"

    # Define any fitparameters you want
    # For abundances use: 'abund {El}', where El is the element (e.g. 'abund Fe')
    # For linelist use: 'linelist {Nr} {p}', where Nr is the number in the
    # linelist and p is the line parameter (e.g. 'linelist 17 gflog')
    fitparameters = ["teff", "logg", "monh"]

    # sme.wave = [sme.wave[0, :200], sme.wave[0, 200:]]
    # sme.spec = [sme.spec[0, :200], sme.spec[0, 200:]]
    # sme.uncs = [sme.uncs[0, :200], sme.uncs[0, 200:]]
    # # sme.synth = [sme.synth[0, :200], sme.synth[0, 200:]]
    # sme.mask = [sme.mask[0, :200], sme.mask[0, 200:]]
    # sme.mask[0][100:110] = 0

    sme.uncs = np.sqrt(sme.spec)
    # sme.uncs = None

    sme.teff = 5000
    sme.logg = 4.0
    sme.monh = 0.2
    # 4555, 3.38, -0.89
    # 4559, 3.39, -0.89

    sme.nlte.set_nlte("Ba", "marcs2012p_t1.0_Ba.grd")
    sme.nlte.set_nlte("Ca", "marcs2012p_t1.0_Ca.grd")

    # sme.linelist = sme.linelist[::3]
    # sme.accrt = np.finfo(float).eps
    # sme.accrt = 1e-3
    # dogbox: fev = 11
    # lm: fev = 12

    # Start SME solver
    # sme = synthesize_spectrum(sme)
    sme.fitresults.maxiter = 100 * len(fitparameters)
    sme = solve(sme, fitparameters, segments=[0])
    chi2 = np.mean(sme.fitresults.residuals ** 2 / sme.uncs[sme.mask_line].ravel() ** 2)
    print(sme.citation())
    # monh = 0.0 -> chi2 = 0.00057
    # monh = 0.4 -> chi2 = 0.00037

    # Save results
    sme.save(out_file)

    # Plot results
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=plot_file)

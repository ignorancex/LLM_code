# -*- coding: utf-8 -*-
""" Main entry point for an SME script """
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.continuum_and_radial_velocity import match_rv_continuum
from pysme.gui import plot_plotly, plot_pyplot
from pysme.linelist.vald import ValdFile
from pysme.solve import SME_Solver
from pysme.synthesize import synthesize_spectrum

if __name__ == "__main__":
    target = "k2-3_2"
    util.start_logging(f"{target}.log")

    # Get input files
    if len(sys.argv) > 1:
        in_file, vald_file, fitparameters = util.parse_args()
    else:
        examples_dir = "/DATA/ESO/HARPS/K2-3/"
        in_file = os.path.join(examples_dir, "K2-3_red_c.ech")
        vald_file = os.path.expanduser("~/Documents/IDL/SME/harps_red.lin")
        atmo_file = "marcs2012p_t1.0.sav"
        fitparameters = []

    # Load files
    sme = SME.SME_Structure.load(in_file)

    if vald_file is not None:
        vald = ValdFile(vald_file)
        sme.linelist = vald

    if atmo_file is not None:
        sme.atmo.source = atmo_file
        sme.atmo.method = "grid"
        sme.atmo.geom = "PP"

    # Choose free parameters, i.e. sme.pname
    if len(fitparameters) == 0:
        # ["teff", "logg", "monh", "abund Mg", "abund Y"]
        if sme.fitparameters is not None and len(sme.fitparameters) != 0:
            fitparameters = sme.fitparameters
        else:
            fitparameters = ["teff", "logg", "monh"]

    sme.teff = 3800
    sme.logg = 4.86
    sme.monh = -0.4
    sme.vsini = 0
    sme.vmic = 1
    sme.vmac = 1
    sme.h2broad = True

    sme.nlte.set_nlte("Fe", "marcs2012_Fe2016.grd")

    sme.cscale_flag = "none"
    sme.cscale = 1
    sme.vrad_flag = "whole"
    sme.vrad = 31.82

    fitparameters = ["logg", "teff", "monh"]

    # Start SME solver

    sme = synthesize_spectrum(sme)

    # solver = SME_Solver(filename=f"test2.sme")
    # sme = solver.solve(sme, fitparameters)

    try:
        # Calculate stellar age based on abundances
        solar = Abund.solar()
        y, mg = sme.abund["Y"], sme.abund["Mg"]
        sy, smg = (
            sme.fitresults.punc["Y abund"],
            sme.fitresults.punc["Mg abund"],
        )
        x = y - mg - (solar["Y"] - solar["Mg"])
        sx = np.sqrt(sy ** 2 + smg ** 2)

        # Values from paper
        a = 0.175
        sa = 0.011
        b = -0.0404
        sb = 0.0019
        age = (x - a) / b
        sigma_age = 1 / b * np.sqrt(sx ** 2 + sa ** 2 + ((x - a) / b) ** 2 * sb ** 2)
        sigma_age = abs(sigma_age)
        logging.info("Age       \t%.3f +- %.3f Gyr", age, sigma_age)

        p = np.linspace(0, 10, 1000)
        g = norm.pdf(p, loc=age, scale=sigma_age)
        # Rescale to area = 1
        area = np.sum(g * np.gradient(p))  # Cheap integral
        g *= 1 / area
        plt.plot(p, g)
        plt.xlabel("Age [Gyr]")
        plt.ylabel("Probability")
        plt.show()
    except:
        pass

    # Plot results
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=f"{target}.html")

    # # if "synth" in sme:
    # #     plt.plot(sme.wob, sme.sob - sme.smod, label="Residual Python")
    # #     # plt.plot(sme.wave, sme.sob - orig, label="Residual IDL")
    # #     plt.legend()
    # #     plt.show()

    # mask_plot = plot_pyplot.MaskPlot(sme)
    # input("Wait a second...")

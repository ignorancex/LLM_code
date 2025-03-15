#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Example of use
###, L. Darme 02/07/2019

import matplotlib.pyplot as plt
import numpy as np

# Importing additional user-defined function

import UsefulFunctions as uf
import Amplitudes as am
import Production as br
import Detection as de
import LimitsList as lim
#############################################################################
###############        Several limits example    ##############################
#############################################################################


# This example create limits for the quark flavour violating operators

if __name__ == "__main__":

    ExperimentsList=np.array(["babar_invisibledecayBmtoKm","belle2_invisibledecayBmtoKm", \
                              "belle_invisibledecayB0toPi0", "belle_invisibledecayB0toK0","babar_invisibledecayBmtoPim",\
                               "e391a_invisibledecayKL0toPi0","e949_invisibledecayKptoPip","na62_invisibledecayKL0toPi0","na62_invisibledecayKptoPip",\
                               "ship_heavymesondecay"])
    ###invisible heavy meson decays

    geff={"gu11":2/3.,"gu22":2/3.,"gd11":-1/3.,"gd22":-1/3.,"gd33":-1/3.,"gl11":-1.,"gl22":-1.,"gd31":1,"gd32":1,"gd21":1}

    Lim,LabelLimit = lim.GetLimits(ExperimentsList,10.,geff,"V",True, ReadFromFile=False)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Example of use
###, L. Darme 02/07/2019
import numpy as np

# Importing additional user-defined function

import UsefulFunctions as uf
import Amplitudes as am
import Production as br
import Detection as de
import LimitsList as lim
#############################################################################
###############        Single limit example    ##############################
#############################################################################

if __name__ == "__main__":


    ###### Create a naive 10-limits line for SHIP
    #     xi_SHIP, EffLim_SHIP = GetHeavyMesonNaiveLimits(0.1, "ship", 10, (2./3.,-1./3.,-1.), "V")
    #     xi_sat_SHIP, EffLim_sat_SHIP, EffLim_sat_SHIP_low = RecastHeavyMesonDecayLimit(xi_SHIP, EffLim_SHIP, 0.1, 0.1, "ship", (2./3.,-1./3.,-1.), "V")
    #     xi_full, Lim_full = uf.CombineUpDown(xi_sat_SHIP, EffLim_sat_SHIP_low, EffLim_sat_SHIP, 7, 1.1)


    ###### Recast the scattering limits from MiniBooNE
        geffem={"gu11":2/3.,"gd11":-1/3.,"gl11":-1.} ## Define the effective coupling
        geffZal={"gu11":1/2.,"gd11":-1/2.,"gd22":-1/2.,"gu22":1/2.,"gd33":-1/2.,"gl11":-1/2,"gl22":-1/2}

        xi_full, Lim_full = lim.faser_decay.recast(0.1,geffem,"AV")
    #     xi_full, Lim_full = lim.babar_invisibledecayUpsilon.recast(0.1,geffZal,"AV")


        print( "-----  Recasted limit on Lambda (in GeV) -------")
        print(np.dstack((xi_full,Lim_full)))
        print( "------------")

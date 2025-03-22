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


# This example create limits for a dark photon model with non-negligible mixing with the Z, it uses the inner functions
# to generate dark sector field through both the axial and vector portals, and then decay them through the most advantageous one
# (the Vector portal in this case)

if __name__ == "__main__":

    geffem={"gu11":2/3.,"gu22":2/3.,"gd11":-1/3.,"gd22":-1/3.,"gd33":-1/3.,"gl11":-1.,"gl22":-1.}


    fracAVtoV=-0.016;Mzp=20.
    geffAV={"gu11":fracAVtoV,"gd11":-fracAVtoV,"gd22":-fracAVtoV,"gl11":-fracAVtoV,"gl22":-fracAVtoV}


    # Let us use a bit more the inner functions and make the decay limits for a combined V and AV operator for CHARM
    exp='charm';Delini=0.1;Delf=5;
    xeff,Limeff = de.MakeEffLimits(exp, lim.charm_decay.mx_ini, lim.charm_decay.lim_ini, geffem, "V", Delini )
    xprod, Nprod = br.NProd(Delini,exp,geffem,"V") #
    xifV,EffLimV = de.ShiftLimDel(xprod, Nprod, xeff, Limeff, Delf,Delini,exp,"V",geffem)
    xeff,Limeff = de.MakeEffLimits(exp, lim.charm_decay.mx_ini, lim.charm_decay.lim_ini, geffAV, "AV", Delini )
    xprod, Nprod = br.NProd(Delini,exp,geffAV,"AV") #
    xifAV,EffLimAV = de.ShiftLimDel(xprod, Nprod, xeff, Limeff, Delf,Delini,exp,("AV","V"),(geffAV,geffem))
    xif,EffLim=uf.CombineLimits((xifV,xifAV),(EffLimV,EffLimAV))
    Dexp,Lexp,beamtype=de.GeomForExp(exp) # Need to clean the above functions
    EffLimFin_Low=de.ShortLamLimit(xif, Delf,EffLim,Dexp,Lexp,beamtype,(geffAV,geffem),("AV","V"))
    xi_charm, Lim_charm=uf.CombineUpDown(xif,EffLimFin_Low,EffLim,1.1)

    print( "-----  Recasted limit on Lambda (in GeV) -------")
    print(np.dstack((xif,EffLimFin_Low,EffLim)))
    print( "------------")
    print( "-----  Combined up and down limit on Lambda (in GeV) -------")
    print(np.dstack((xi_charm, Lim_charm)))
    print( "------------")

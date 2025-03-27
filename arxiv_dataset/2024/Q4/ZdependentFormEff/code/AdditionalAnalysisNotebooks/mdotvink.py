#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Mass-loss recipe for massive stars including a metallicity dependence.
# Based on the IDL routine by Jorick S. Vink for his 2001 recipe
# with an (optional) update of the Z-dependence according to the 2020 paper   
# Python script written by Andreas A. C. Sander  
#
# This script can either be called as a stand-alone from the command line
# ./mdotvink.py Z Teff logL Mass [vinf] [-2001] [-silent]
# or by importing it as a Python module and calling the method
# mdotvink.mdot(Z, Teff, logL, M, vinf = -1, use2001 = False, silent = False)
#
# The calculations perfomed in this script are based on the following papers:
#
#   1. " to be titled properly "
#       Jorick S. Vink and A.A.C. Sander, 2020, MNRAS, submitted
#
#   2. "Mass-loss predictions for O and B stars as a function of metallicity"
#       Jorick S. Vink, A. de Koter, and H.J.G.L.M. Lamers, 2001, A&A 369, 574
#
#   3. "New theoretical mass-loss rates of O and B stars"
#       Jorick S. Vink, A. de Koter, and H.J.G.L.M. Lamers, 2000, A&A 362, 295
#
#   4. "On the nature of the bi-stability jump in the winds of early-type supergiants"
#       Jorick S. Vink, A. de Koter, and H.J.G.L.M. Lamers, 1999, A&A 350, 181  
#
#

import sys
import math
import numpy as np

#----------------------------------------------------------------
def info():
    print "Syntax: ./mdotvink.py Z Teff logL Mass [vinf] [-2001] [-silent]"
    print ""
    print "Parameters:"
    print "  Z       -  metallicity relative to Zsun (Zsun from Anders, E. & Grevesse N., 1989, GeCoA 53, 197)"
    print "  Teff    -  effective Temperature (in Kelvin)"
    print "  logL    -  log of Luminosity (in solar units)"
    print "  Mass    -  stellar mass (in solar masses)" 
    print "  vinf    -  (optional) known terminal wind velocity (in km/s)"
    print "  -2001   -  (optional) use Vink et al. (2001) metallicity scaling instead of 2020 result"
    print "  -silent -  (optional) suppress additional information, only return mass-loss rate"
    print ""
    print "Output:"  
    print "  Mdot   -   predicted logarithmic mass-loss rate (in log Msun/yr)"
    print ""
    print "Examples:"
    print "  ./mdotvink.py 0.5 30000 5.5 30" 
    print "  ./mdotvink.py 1.5 17000 5.7 45 2450 -silent" 
    print ""
    return

#----------------------------------------------------------------
def getZexponent(use2001):
    if (use2001):
        return 0.85
    else:
        return 0.42
        
#----------------------------------------------------------------
def getJumpTemperatures(Gamma, Z, use2001):
    #characteristic density for the bi-stability jump
    dZ = getZexponent(use2001)
    # calculated via Eq. (23) from Vink et al. (2001)
    charrho = -14.94 + (3.1857 * Gamma) + (dZ * np.log10(Z)) ; 
    #Jump temperatures via Eqs. (15) from Vink et al. (2001) and (6) from Vink et al. (2000)
    T1 = ( 61.2 + (2.59 * charrho) ) * 1000.
    T2 = ( 100. + (6.0 * charrho) ) * 1000.
    return [T1, T2]
    
#----------------------------------------------------------------
def mdotformula(zone, Teff, logL, M, ratio, Z, use2001 = False):
    
    offset = {"cold":-5.99,"inter":-6.688,"hot":-6.697}
    logL5  = logL - 5.
    logM30 = np.log10(M/30.)
    lograt = np.log10(ratio/2.)
    logT40 = np.log10(Teff/40000)
    logT20 = np.log10(Teff/20000)

    if (zone == "cold" or zone == "inter"):
        #Below the hotter bi-stability jump, we always use the 2001 exponent
        dZ = getZexponent(True)
        logMdot = offset[zone] + 2.210 * logL5 - 1.339 * logM30 \
            - 1.601 * lograt + 1.07 * logT20 + dZ * np.log10(Z)
    else:
        #Above the hotter bi-stability jump, dZ depends on user's choice
        dZ = getZexponent(use2001)
        logMdot = offset[zone] + 2.194 * logL5 - 1.313 * logM30 \
            - 1.226 * lograt + 0.933 * logT40 - 10.92 * (logT40**2) + dZ * np.log10(Z)

    return logMdot

#----------------------------------------------------------------
def mdot(Z, Teff, logL, M, vinf = -1, use2001 = False, silent = False):

    Gamma = 7.66E-5 * 0.325 * (10**logL)/M    
    Tjump = getJumpTemperatures(Gamma, Z, use2001)
    if (Tjump[0] < Tjump[1]):
        #unrealistic jump temperatures: do not calculate mdot
        return False

    MSUN = 1.989E33
    LSUN = 3.827E33
    STEBOL = 5.670E-5
    GGRAV  = 6.670E-8 
    rstar = np.sqrt(10**logL * LSUN/(4. * math.pi * STEBOL * (Teff**4.)))
    vesc  = np.sqrt(2.0 * GGRAV * M * (1.-Gamma) * MSUN/rstar) * 1.E-5
    ratio = vinf/vesc

    if (Teff == Tjump[0]):
        print 'The star is AT the first jump '
        print 'No Mass-loss rate can be provided'
        return False
    if (Teff == Tjump[1]):
        print 'The star is AT the second jump ' 
        print 'No Mass-loss rate can be provided'    
        return False

    if (Teff < Tjump[0]):
        if (not silent):
            print 'The star is located below the first jump '
        if (Teff < Tjump[1]):
            if (not silent):
                print 'The star is located below the second jump '
            zone = "cold"
            if (vinf == -1):
               ratio = 0.7
        else:
            if (not silent):
                print 'The star is located between the two bi-stability jumps'
            zone = "inter"
            if (vinf == -1):    
                ratio = 1.3                   
    if (Teff > Tjump[0]):
        if (not silent):
            print 'The star is located above the first jump '
        if (vinf == -1):        
            ratio = 2.6                   
        zone = "hot"
    logMdot = mdotformula(zone, Teff, logL, M, ratio, Z, use2001)

    return logMdot

#----------------------------------------------------------------

if __name__ == "__main__":
    # command line call handler
    if (len(sys.argv) > 4):
        zarg = float(sys.argv[1])
        targ = float(sys.argv[2])
        larg = float(sys.argv[3])
        marg = float(sys.argv[4])
        varg = -1.
        silent = False
        use2001 = False
        for carg in sys.argv[5:]:
            if (carg == "-silent"):
                silent = True
            else: 
                if (carg == "-2001"):
                    use2001 = True
                else:
                    #must be vinf instead
                    varg = float(carg)
        result = mdot(zarg, targ, larg, marg, varg, use2001, silent)
        if (result is False and not silent):
            print 'Error: Failed to calculate Mdot'
            print result 
        else:
            print '{0:.5f}'.format(result)
    else:
        if (len(sys.argv) > 1):
            print "Error: Insufficient number of arguments"
            print ""            
        info()

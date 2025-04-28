from xspec import *
import numpy as np
import constants as c
import bhtools

# Must run the HEADAS script first!
# subprocess.call("$HEADAS/headas-init.sh", shell=True)

#====================================================================================================
def spectrum(a=0.0, inc=0.0, m=7.5, d=8.5, fcol=1.7, l=0.1, lflag=1.0, \
             eta=0, rflag=-1.0, norm=1.0, E_min=0.1, E_max=50.0, NE=50, chat=0):
    '''
    Purpose:
    --------
    Accounting for general relativistic effects on the photon propagation, calculate the observed spectrum (specific flux [erg s^-1 cm^-2 keV^-1] vs. photon energy [keV]) from a geometrically thin, optically thick accretion disk with the effective temperature radial profile of Page & Thorne (1974).

    Inputs:
    -------
    a     - Black hole spin parameter [-]
    inc   - Disk inclination angle [degrees]
    m     - Black hole mass [M_sun]
    d     - Distance to source [kpc]
    fcol  - Color correction factor [-]
    l     - Disk luminosity [L_Edd]
    lflag - (>0: limb-darkened emission; <=0: isotropic emission) <-- same as KERRBB
    eta   - Inner boundary torque parameter [-] (0 <= eta <= 1)
    rflag - (>0: include self-irradiation; <=0: no self-irradiation)
    norm  - Normalization (should be set to 1)
    E_min - Minimum observed photon energy [keV]
    E_max - Maximum observed photon energy [keV]
    NE    - Number of elements in the array of observed photon energies
    chat  - Chatter level in XspecSettings (default to 0 to minimize terminal output)

    Outputs:
    --------
    Dictionary:
        'E'      - Array of photon energies (bin centered) [keV]
        'F_E'    - Array of specific fluxes [erg s^-1 cm^-2 keV^-1]
        'F_disk' - Integrated disk flux [erg s^-1 cm^-2]
    
    Notes:
    ------
    We could loop over KERRBB parameters, changing them within a session, rather than re-launching XSPEC every time. But, this is less transparent
    '''
    print("....KERRBB: (a,i) = ", a, inc)

    # Set the Xspec chatter level
    XspecSettings.chatter = int(chat)
    
    # From the input l = L/L_Edd, calculate the mass accretion rate for each black hole spin
    rad_eff = bhtools.radiative_efficiency(a=a)          # Array of radiative efficiencies [-]
    L_Edd   = bhtools.Eddington_luminosity(M=m)          # Eddington luminosity [erg s^-1]
    Mdd     = (l * L_Edd) / (rad_eff * c.c**2) / 1.0e18  # Array of mass accretion rates [10^18 g s^-1]
    
    # Load in the KERRBB model with default parameters
    Model("kerrbb")
    m1 = AllModels(1)

    # Set the input KERRBB parameters [value, fit delta, min, bot, top, max]
    # (not setting spin or inclination here)
    m1(1).values  = eta    # [-]
    m1(2).values  = a      # [-]
    m1(3).values  = inc    # [degrees]
    m1(4).values  = m      # [M_sun]
    m1(5).values  = Mdd    # [10^18 g s^-1]
    m1(6).values  = d      # [kpc]
    m1(7).values  = fcol   # [-]
    m1(8).values  = rflag  # Self-irradiation flag
    m1(9).values  = lflag  # Limb-darkening flag
    m1(10).values = norm   # [-]

    # Freeze the KERRBB parameters
    m1(1).frozen  = True
    m1(2).frozen  = True
    m1(3).frozen  = True
    m1(4).frozen  = True
    m1(5).frozen  = True
    m1(6).frozen  = True
    m1(7).frozen  = True
    m1(8).frozen  = True
    m1(9).frozen  = True
    m1(10).frozen = True

    # Adjust the energy range for the model according to the input E_min, E_max, NE
    setEnergies_str = str(E_min) + " " + str(E_max) + " " + str(NE) + " " + "log"
    AllModels.setEnergies(setEnergies_str)
    
    # Collect the KERRBB spectrum <-- what does m1.values(0) return?
    Plot("model")
    E    = np.array(Plot.x())      # [keV]
    F_E  = np.array(Plot.model())  # [photons s^-1 cm^-2 keV^-1]
    F_E *= E * c.keV2erg           # [erg s^-1 cm^-2 keV^-1]
        
    # Calculate the KERRBB disk flux
    calcFlux_str = str(E_min) + " " + str(E_max) + " " + "noerr"
    AllModels.calcFlux(calcFlux_str)
    F_disk = m1.flux[0]  # [erg s^-1 cm^-2]
    
    #return {'E':E, 'F_E':F_E, 'F_disk':F_disk}
    return {'F_disk':F_disk}

#====================================================================================================

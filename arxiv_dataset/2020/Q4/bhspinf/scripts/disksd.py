import numpy as np
import constants as c
import bhtools

#====================================================================================================
def spectrum(a=0.0, inc=0.0, m=7.5, d=8.5, fcol=1.7, l=0.1, lflag=1.0, \
             E_min=0.1, E_max=50.0, NE=50, r_out=1.0e6, NrIgral=1000, NTSS='NT'):
    '''
    Purpose:
    --------
    NOT accounting for general relativistic effects on the photon propagation, calculate the observed spectrum (specific flux [erg s^-1 cm^-2 keV^-1] vs. photon energy [keV]) from a geometrically thin, optically thick accretion disk with the effective temperature radial profile of Page & Thorne (1974).

    Inputs:
    -------
    a     - Black hole spin parameter [-]
    inc   - Disk inclination angle [degrees]
    m     - Black hole mass [M_sun]
    d     - Distance to source [kpc]
    fcol  - Color correction factor [-]
    l     - Disk luminosity [L_Edd]
    lflag - (>0: limb-darkened emission; <=0: isotropic emission) <-- same as KERRBB
    E_min - Minimum observed photon energy [keV]
    E_max - Maximum observed photon energy [keV]
    NE    - Number of elements in the array of observed photon energies
    r_out - Outer disk radius [R_g]
    Nr    - Number of disk radii elements to use for the r-integral
    NTSS  - Specify the radial disk temperature profile ('PT' for Page & Thorne; 'SS' for Shakura & Sunyaev)
    
    Outputs:
    --------
    Dictionary:
        'E'      - Array of photon energies (bin centered) [keV]
        'F_E'    - Array of specific fluxes [erg s^-1 cm^-2 keV^-1]
        'F_disk' - Integrated disk flux [erg s^-1 cm^-2]
    '''
    print("....DISKSD: (a,i) = ", a, inc)
    
    # Calculate some useful things
    rad_eff = bhtools.radiative_efficiency(a=a)  # Radiative efficiency [-]
    L_Edd   = bhtools.Eddington_luminosity(M=m)  # Eddington luminosity [erg s^-1]

    # Convert input parameters to CGS units
    M    = m * c.sun2g              # Black hole mass [g]
    R_g  = c.G * M / c.c**2         # Gravitational radius [cm]
    D    = d * c.kpc2cm             # Distance [cm]
    L    = l * L_Edd                # Disk luminosity [erg s^-1]
    Mdot = L / (rad_eff * c.c**2)   # Mass accretion rate [g s^-1]
    cosi = np.cos(inc * c.deg2rad)  # Cosine of the disk inclination angle
    
    # Location of the ISCO for the input black hole spin parameter: r_isco = R_isco / R_g
    r_isco = bhtools.spin2risco(a=a)  # [dimensionless in terms of R_g]
    R_isco = r_isco * R_g             # ISCO [cm]

    # Flux normalization (Newtonian)
    Kflux = (1.0 / fcol**4) * (R_isco / D)**2 * cosi  # [-]

    # Array of photon energies
    logE_min = np.log10(E_min)  # [keV]
    logE_max = np.log10(E_max)  # [keV]
    E_bedges = np.logspace(logE_min, logE_max, NE+1) * c.keV2erg  # Photon energies array [erg]
    E_arr    = np.zeros(NE)
    for i in np.arange(NE):
        E_arr[i] = 0.5 * (E_bedges[i] + E_bedges[i+1])  # [erg]

    #----------------------------------------------------------------------------------------------------
    # Effective temperature radial profile, r given in gravitational radius units
    def Teff_r(r):
        '''
        Inputs:
        -------
        r - Disk radius [Rg]
        '''
        # Novikov & Thorne (1973)
        if (NTSS == 'NT'):
            x  = np.sqrt(r)
            x0 = np.sqrt(r_isco)
            x1 = 2.0 * np.cos(np.arccos(a)/3.0 - np.pi/3.0)
            x2 = 2.0 * np.cos(np.arccos(a)/3.0 + np.pi/3.0)
            x3 = -2.0 * np.cos(np.arccos(a)/3.0)
            f  = 1.0 / (x**2 * (x**3 - 3.0*x + 2.0*a)) \
               * (x - x0 - 1.5*a*np.log(x/x0) \
               - 3.0*(x1-a)**2/(x1*(x1-x2)*(x1-x3))*np.log((x-x1)/(x0-x1)) \
               - 3.0*(x2-a)**2/(x2*(x2-x1)*(x2-x3))*np.log((x-x2)/(x0-x2)) \
               - 3.0*(x3-a)**2/(x3*(x3-x1)*(x3-x2))*np.log((x-x3)/(x0-x3)))
            Tstar = (3.0 * Mdot * c.c**6 / (8.0 * np.pi * c.sigmaSB * M**2 * c.G**2 * x**2))**0.25
            Teff  = Tstar * f**0.25

        # Shakura & Sunyaev (1973)
        if (NTSS == 'SS'):
            r_rin = r * (R_g / R_isco)
            Tstar = (3.0 * c.G * M * Mdot / (8.0 * np.pi * c.sigmaSB * R_isco**3.0))**0.25
            Teff  = Tstar * r_rin**(-0.75) * (1.0 - r_rin**(-0.5))**0.25

        return Teff

    #----------------------------------------------------------------------------------------------------
    # Calculate the disk spectrum w/ np.trapz
    def Igral_trapz(E):
        rscr_min = np.log10(1.0)
        rscr_max = np.log10(r_out/r_isco)
        rscr     = np.linspace(rscr_min, rscr_max, NrIgral)
        rx       = 10.0**rscr   # rx = r / r_isco
        r        = rx * r_isco  # [Rg] <-- required by Teff_r()
        Teff     = Teff_r(r=r)  # [K]
        Igrand   = (10.0**(2.0 * rscr) / np.log10(np.e)) / np.expm1(E / (c.kB * fcol * Teff))  # [-]
        Igral    = np.trapz(y=Igrand, x=rscr)
        return Igral
        
    #----------------------------------------------------------------------------------------------------

    # Calculate the standard disk spectrum without GR effects on the photon propagation
    F_E_arr = np.zeros(NE)
    options = {'epsabs':0, 'epsrel':1e-03, 'limit':100}
    for i in np.arange(NE):
        Igral      = Igral_trapz(E=E_arr[i])
        F_E_arr[i] = Kflux * 4.0 * np.pi * E_arr[i]**3 / (c.h**3 * c.c**2) * Igral  # [erg s^-1 cm^-2 erg^-1]

    # Convert to specific flux array to [erg s^-1 cm^-2 keV^-1] units
    F_E_arr /= c.erg2keV  # [erg s^-1 cm^-2 keV^-1]
    
    # Apply limb-darkening law for a semi-infinite electron scattering slab <-- Chandrasekhar (1960)?
    if (lflag >= 0.0): F_E_arr *= 0.5 + 0.75*cosi  # [erg s^-1 cm^-2 keV^-1]

    # Calculate the standard integrated disk flux
    EkeV_arr = E_arr * c.erg2keV                # [keV]
    F_disk   = np.trapz(y=F_E_arr, x=EkeV_arr)  # [erg s^-1 cm^-2]

    #return {'E':EkeV_arr, 'F_E':F_E_arr, 'F_disk':F_disk}
    return {'F_disk':F_disk}

#====================================================================================================

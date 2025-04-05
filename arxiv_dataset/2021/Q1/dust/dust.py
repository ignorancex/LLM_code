#! /usr/bin/env python
import numpy as np
import pyfits
from astroML.cosmology import Cosmology

#------------------------------------------------------------------------------------
#   Cardelli et al. 1989 extinction laws in FIR and IR/OPT:
#------------------------------------------------------------------------------------
def cardelliIR(waveA):
    Rv=3.1 #Mathis et al. 1989  
    wave=waveA/10000.
    x=1./wave
    ax=0.574*(x**1.61) 
    bx=-0.527*(x**1.61)
    return ax+bx/Rv

def cardelliOPT(waveA):
    Rv=3.1 
    wave=waveA/10000.
    x=1./wave
    y=x-1.82
    ax=1.+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7 
    bx=1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
    return ax+bx/Rv

#-------------------------------------------------------------------------------------
#  Attenuated Luminosity and Flux (log scale)
#-------------------------------------------------------------------------------------
def attenuatedQ(logL, Mcold_disc, rhalf_mass_disc, Z_disc):
    Zsun=0.0134 #Asplund et al. 2009
    s=1.6 #if lambda>2000A
    a_disc=1.68
    mp=1.67e-27 #kg
    Al_Av=cardelliOPT(waveHa)
    costheta=0.30 #scattering forward oriented
    sectheta=1./costheta
    albedo=0.56

    mean_col_dens_disc_log=np.log10(Mcold_disc)+np.log10(Msun_to_kg)-np.log10(1.4*mp*np.pi)-2.*np.log10(a_disc*rhalf_mass_disc*Mpc_to_cm)
    tau_disc=np.log10(Al_Av)+np.log10((Z_disc/Zsun)**s)+mean_col_dens_disc_log-np.log10(2.1e+21)
    tau_disc=10.**tau_disc
    al_disc=(np.sqrt(1.-albedo))*tau_disc
    attenuation_disc=-2.5*np.log10((1.-np.exp(-al_disc*sectheta))/(al_disc*sectheta))
    logL_attenuated=logL-0.4*attenuation_disc
    logflux_att=logL_attenuated-np.log10(4.*np.pi)-2.*np.log10(cosmo.Dl(redshift)*Mpc_to_cm) 
    logflux=logL-np.log10(4.*np.pi)-2.*np.log10(cosmo.Dl(redshift)*Mpc_to_cm) 
    return logL_attenuated, logflux_att, logflux, attenuation_disc       



#------------------------------------------------------------------------------------
#   Emission lines:
#------------------------------------------------------------------------------------

waveHa=6563. #A

#------------------------------------------------------------------------------------
#   Cosmology:
#------------------------------------------------------------------------------------
h0 = 0.6777
omega0=0.307
omegab = 0.0482
lambda0=0.693
cosmo = Cosmology(h=h0,omegaM=omega0, omegaL=lambda0)

#------------------------------------------------------------------------------------
#   Conversion factors and constants:
#------------------------------------------------------------------------------------
kg_to_Msun=1./(1.989e30)
Msun_to_kg=1.989e30
Mpc_to_cm=3.086e24


#------------------------------------------------------------------------------------
#   Inputs to be changed:
#------------------------------------------------------------------------------------

path = '/dust-master/' #working directory
redshift=0.09 #redshift of the SAM catalogue
#------------------------------------------------------------------------------------
#   Read input file and implement attenuation:
#   the input file must include: 
#   - Ha luminosity logscale:        log(L/ergs^-1) 
#   - Cold gas mass of the disc:     Mcold_disk [Msun] 
#   - Half-mass radius of the disc:  rhalf_mass_disc [Mpc], 
#   - Metallicity of the disc:       Z_disc
#
#   !!WARNING!! SAG provides 'rhalf_disc' which is the scale radius r0 with wrong name.
#   In order to get rhalf_mass_disc, apply the correction:
#   rhalf_mass_disc=1.68*r0,  see Gonzalez et al. 2009
#------------------------------------------------------------------------------------

hdulista = pyfits.open(path+'SAGinput_z=0.09.fits')
Mcold_disc=hdulista[1].data.field('Mcold_disc')  #Msun
rhalf_mass_disc=hdulista[1].data.field('rhalf_mass_disc')  #Mpc
logL=hdulista[1].data.field('logL') 
Z_disc=hdulista[1].data.field('Z_disc') 

logLatt, logFatt, logF, attenuation_disc = attenuatedQ(logL, Mcold_disc, rhalf_mass_disc, Z_disc)

#------------------------------------------------------------------------------------
#   Write output file with columns:  
#   - non-attenuated ELG luminosity: logL 
#   - attenuated ELG luminosity:     logLatt 
#   - non-attenuated ELG flux:       logF 
#   - attenuated ELG flux:           logFatt
#   - attenuation coefficient:       attenuation_disc
#------------------------------------------------------------------------------------
out=np.array([logL, logLatt, logF, logFatt, attenuation_disc])
out=np.transpose(out)      
np.savetxt(path+'SAGoutput_z=0.09_dust.txt', out, fmt='%12.9f', header='#log10(L/ergs^-1)    log10(Latt/ergs^-1)  log10(F/ergs^-1cm^-2)     log10(Fatt/ergs^-1cm^-2)    att_coeff') 



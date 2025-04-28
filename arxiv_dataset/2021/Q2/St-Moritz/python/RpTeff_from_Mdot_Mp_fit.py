#  AMDG
# JMJ-V!
# FdS 24.01.2020

"""This file contains two functions giving the radius and the Teff of forming
planets. They are defined in Aoyama et al. (2020).

A.)
The function Rp_from_Mdot_Mp(Mdot,Mp,Pop) returns the fit of the radius as a 
function of accretion rate Mdot and planet mass Mp for the cold- (Pop='cold')
or warm-start (Pop='warm') populations of Mordasini et al. (2012). Plotted in
Fig. 1 of Aoyama et al. (2020).

B.)
The function Teff_from_Mdot_Mp(Mdot,Mp,Tint,Pop,ffill) returns the fit of
of the Teff of the accreting region of the planet's surface. It ensures
approximate energy conservation. The details of the fit might change in 
future iterations of the model. See §2.2.2 and Figs. 2 and 3.

The non-accreting region is in principle emitting at Teff = Tint
but this is not relevant for this code.

The fits are valid for approximately (please see the paper):

         1 MJ <  Mp  < 30 MJ
   1e-5 ME/yr < Mdot < 2e-2 ME/yr

-- Input --
  Mdot: in ME/yr = 1.893e20 g/s
    Mp: in MJ = 1.898e30 g
   Pop: 'warm' or 'cold'

-- Output --
    Rp: in RJ = 7.15e9 cm
  Teff: in K

Works with python 3 and not hard to adapt for python 2.

test_RpTeff_fit(): lets one easily test these functions

If you use these functions, please cite:

  Aoyama, Marleau, Mordasini & Ikoma (2020)
  Spectral appearance of the planetary-surface accretion shock:
    Global spectra and hydrogen-line profiles and luminosities
  https://arxiv.org/abs/2011.06608

Comments and requests welcome!

  Gabriel-Dominique Marleau (c) 2021
  @uni-tuebingen.de with gabriel.marleau in front
"""

# -------------------------------------------------------------
# 
import numpy as np

def Rp_from_Mdot_Mp(Mdot,Mp,Pop):
	"""Radius of an accreting planet.
	
	Returns the radius (in RJ) of an accreting planet
	as function of Mdot (in ME/yr) and Mp (in MJ).
	Based on the fits of Aoyama et al. (2020)
	to the Bern planet structure model.
	The 'Pop' (population) argument is either 'warm'
	or 'cold', depending on which fit is preferred.
	The case 'warm' might be more realistic.
	"""
	
	lm2 = np.log10(Mdot/1e-2)
	if Pop == 'warm':
		return 0.411-0.244*lm2+3.45*np.exp(0.762*lm2) \
		      + (-0.489-0.0961*lm2+0.652*np.exp(0.353*lm2))*(Mp-1.0) \
		      + (-0.228-0.00106*lm2+0.226*np.exp(0.000220*lm2))*(Mp-1.0)**2.
	elif Pop == 'cold':
		return 1.53+0.111*lm2+1.06*np.exp(0.906*lm2) \
		      + (-0.195-0.0307*lm2+0.0977*np.exp(0.000695*lm2))*(Mp-1.0) \
		      + (-0.250+0.000276*lm2+0.254*np.exp(0.000214*lm2))*(Mp-1.0)**2.
	else:
		raise Exception('Pop should be "warm" or "cold" but given "{}"'.format(Pop))

def Teff_from_Mdot_Mp(Mdot,Mp,Tint,Pop,ffill):
	""""Effective temperature of the heated photosphere of an accretion spot.
	
	Returns the effective temperature of the accretion region
	(of the hot spot, if the filling factor ffill != 1) as a function
	of Mdot (in ME/yr), Mp (in MJ), the internal temperature Tint (in K),
	and ffill.  This function uses a semianalytical fit of the fraction
	of the incoming energy going into the planet.
	"""
	
	return (Tint**4. + \
	     fdown_Popsynthfit(np.log10(n_MdotMp(Mdot,Mp,Pop,ffill)),v_MdotMp(Mdot,Mp,Pop)/-1e5) \
	     * Tacc_MdotMp(Mdot,Mp,Pop,ffill)**4.\
	     )**0.25

# -----------------------------------------------------------------------------

def Tacc_MdotMp(Mdot,Mp,Pop,ffill):
	"""The 'accretion temperature' of the accreting region.
	
	The rest of the planet's surface is emitting only with Tint.
	"""

	#   Here it is correct to divide by sigma_SB=5.67e-5 cgs and not 4*sigma_SB = a*c
	#     because we want the effective temperature, not the gas temperature
	#     (see Marleau et al. 2019; http://adsabs.harvard.edu/abs/2019ApJ...881..144M)
	#
	return ( 6.67e-8*Mp*1.898e30*Mdot*1.893e20/ \
	     (ffill * 4*np.pi*(Rp_from_Mdot_Mp(Mdot,Mp,Pop)*7.15e9)**3. \
	     * 5.67e-5) )**0.25

def fdown1_Popsynthfit(lgn0,v0):
	"""Fit of the downward-travelling fraction of the flux.
	
	lgn0: log(n_0 / (cm^-3))
	v0: v_0 (km/s)
	
	Based on the data of Aoyama et al. (2018)
	and Aoyama et al. (2020; https://arxiv.org/abs/2011.06608).
	Fit given in the latter paper.
	"""
	
	return 0.703752-0.0967987*(lgn0-12)-0.0254579*abs(lgn0-12)**2. \
	     +  (-0.00527886-0.00146833*(lgn0-12)-0.000321504*abs(lgn0-12)**2.)*(v0-100) \
	     - 9.91492e-06*abs(v0-100)**2.

def fdown_Popsynthfit(lgn0,v0):
	"""Limit fdown to [0,1] for physical reasons.
	"""
	
	return np.fmax( np.fmin( fdown1_Popsynthfit(lgn0,v0), 1) , 0 )

def n_MdotMp(Mdot,Mp,Pop,ffill):
	"""Preshock number density and velocity as a function of
	accretion rate and mass (using the corresponding radius).
	
	Output: in cm^-3
	Mdot: in ME/yr
	Mp: in MJ
	"""
	
	# hydrogen mass fraction of the incoming gas
	X_H_MASSFRAC = 0.738
	
	return Mdot*1.893e20/(4*np.pi*( Rp_from_Mdot_Mp(Mdot,Mp,Pop)*7.15e9)**2.*ffill * -v_MdotMp(Mdot,Mp,Pop) * 1.673e-24/X_H_MASSFRAC)

def v_MdotMp(Mdot,Mp,Pop):
	"""Freefall velocity as a function of
	accretion rate and mass (using the corresponding radius).
	The accretion radius is assumed to be negligibly large.
	"""
	
	return v0simple(Mp, Rp_from_Mdot_Mp(Mdot,Mp,Pop))

def v0simple(Mp,Rp):
	"""Free-fall velocity for free-fall from infinity
	('simple': Racc = \infty)"""
	
	return -1 * np.sqrt(2*6.67e-8*Mp*1.898e30/(Rp*7.15e9))


def test_RpTeff_fit(mode, Pop='warm', ffill=1.0, Tint=1000.0):
	"""Test the radius and Teff fit; check by hand that we get reasonable values.
	"""
	
	## on the command line:
	#import RpTeff_from_Mdot_Mp_fit as RTfit
	# ...
	#import importlib
	#importlib.reload(pf)
	
	# make an array for independent variable
	#   and fix the other one
	if mode == 'Mdot':
		
		# constant value
		Mp = 20.
		
		Mdot = np.logspace(-5,-1,num=70)
		Mp = Mp *np.ones_like(Mdot)
		print("   Mp = {} MJ".format(Mp[0]))
		
	if mode == 'Mp':
		
		# constant value
		Mdot = 1e-3   # ME/yr (not MJ/yr!)
		
		Mp = np.linspace(0.5,20,num=70)
		Mdot = Mdot *np.ones_like(Mp)
		print("   Mdot = {} ME/yr".format(Mdot[0]))
	
	# call the functions
	Rp   = Rp_from_Mdot_Mp(Mdot,Mp,Pop)
	Teff = Teff_from_Mdot_Mp(Mdot,Mp,Tint,Pop,ffill)
		
	# save for plotting with gnuplot ;)
	out = 'test_RpTeff_fit_'+mode+'_'+Pop+'_ffill{}'.format(ffill)+'.dat'
	np.savetxt(out, np.c_[Mdot, Mp, Rp, Teff])
	print("-> "+out)
	

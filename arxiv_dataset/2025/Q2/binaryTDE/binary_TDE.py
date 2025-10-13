#
# Light curve model of newborn compact object (CO) tidally disrupting a main-sequence companion
# Reference: D. Tsuna, W. Lu (arXiv:2501.03316; ApJ accepted)
#

import numpy as np
import math
import itertools

import constants as con # physical constants

#
# key params of the light curve model
#

Mej_Eexp_MNi_array = [(3, 1e51, 0.15), (10, 3e51, 0.4), (10, 1e51, 0.15), (1, 5e50, 0.03)] # in [Msun, erg, Msun]
t_TDE_array = [2., 30., 10.] # time from SN explosion to TDE [days]
Mstar_array = [3., 10., 20.] # stellar companion mass [Msun]
visc_param_array = [1e-2] # viscosity parameter (alpha * (H/R)^2) 


# compact object/disk params
CO = 'NS' # compact object: BH or NS
MCO = 1.4*con.Msun # compact object mass
p_BB = 0.5  # power-law index of r-dependent accretion rate in Blandford & Begelman 99

# light curve params
Rej_init = 7e10 # initial ejecta radius (= He star radius. in cm)
kap_bol = 0.07 # ejecta opacity for bolometric light curve (not for gamma-rays from radioactive decay)
kap_gamma = 0.03 # gamma-ray opacity for radioactive decay
delta_ej = 1.0 # ejecta density profile (rho \propto r^(-delta))
alpha_Im = 3.204 # for delta_ej=1. For delta_ej=0, change to alpha_Im=pi^2/3
T_eff_min = 6e3 # floor value for effective temperature (in K)
tLC_end = 500. # end of light curve (in days)

# params for cooling in the wind nebula
evolve_cooling = True # if False, always assume 100% efficiency of radiation conversion 
# following params are used if evolve_cooling = True
gamma = 5./3. # adiabatic index of shocked wind
X_H = 0.7 # hydrogen mass fraction (in disrupted companion)
Y_He = 0.28 # helium mass fraction (in disrupted companion)
mu = 1./(2.*X_H+0.75*Y_He+0.5*(1.-X_H-Y_He)) # mean molecular weight (e.g. Kippenhahn textbook eq. 4.27)
mu_e = 2./(1.+X_H) # mean molecular weight per free e- (e.g. Kippenhahn eq. 4.30)
kappa_scat = 0.2*(1+X_H) # scattering opacity
Lambda_ff = lambda T: 1e-23*np.sqrt(T/1e7) # free-free cooling rate (Sutherland+93)


#
# functions
#

# Ni56 + Co56 heating rate (Wygoda+19), with gamma-ray trapping fraction.
def L_Ni56(time, M_Ni, M_ej, R_ej):
	f_trap_gamma = 1. - math.exp(-(3.-delta_ej)*kap_gamma*M_ej/4./math.pi/R_ej**2)
	Lgamma = f_trap_gamma * M_Ni/con.Msun * (6.45e43*math.exp(-time/con.t_Ni) + 1.38e43*math.exp(-time/con.t_Co))
	Lpos = M_Ni/con.Msun *  4.64e41*(math.exp(-time/con.t_Co) - math.exp(-time/con.t_Ni))
	return Lgamma + Lpos


# luminosity of shocked wind by gas cooling (Appendix of Tsuna & Lu 25)
def wind_luminosity(t_dyn, R_neb, v_array, dMdotdv, rhosh_w, Tsh_w, U_rad):
	# free-free cooling time
	t_cool_ff = rhosh_w*con.k_B*Tsh_w/(gamma-1)/mu/con.m_p/Lambda_ff(Tsh_w)/(X_H*rhosh_w/con.m_p)**2
	# IC cooling time for fixed U_rad
	Theta_e = (4.*con.k_B*Tsh_w)/(con.m_e*con.c**2)
	t_cool_IC = (3.*mu_e*con.m_e*con.c)/(8.*mu*U_rad*con.sigma_T) / (1.+Theta_e)
	# obtain Compton y-parameter
	betagam_max = math.sqrt(12.) * con.k_B*max(Tsh_w)/con.m_e/con.c**2
	betagam_min = math.sqrt(3.) * math.sqrt(con.k_B*min(Tsh_w)/con.m_e/con.c**2)
	if betagam_min < 1.0:
		A_norm = 1./( ((betagam_min)**(-2.*p_BB)-1.)/(2.*p_BB) + (1.-(betagam_max)**(-p_BB))/p_BB )
		e_gain_per_scat = 4.*A_norm/3. * ( (1.- (betagam_min)**(2.-2.*p_BB))/(2.-2.*p_BB) + ((betagam_max)**(2.-p_BB)-1.)/(2.-p_BB) )
	else: # corner case when all shocked electrons are relativistic
		A_norm = 1./(((betagam_min)**(-p_BB)-(betagam_max)**(-p_BB))/p_BB )
		e_gain_per_scat = 4.*A_norm/3. * ((betagam_max)**(2.-p_BB)- (betagam_min)**(2.-p_BB))/(2.-p_BB)
	# scattering optical depth
	tau_scat = kappa_scat/(4.*math.pi*R_neb**2/t_dyn) * np.trapz(dMdotdv, v_array)
	y = (1.+tau_scat) * tau_scat * e_gain_per_scat
	# enhancement of U_rad by y-parameter. cap this by Tsh(v_crit)/T_SN.
	min_temp_ratio = max(Tsh_w)/(U_rad/con.a_rad)**(1/4)
	t_cool_IC = np.maximum(t_cool_IC*math.exp(-y), t_cool_IC/min_temp_ratio) 
	# obtain using cooling time
	assert min(t_cool_ff)>=0.0
	assert min(t_cool_IC)>=0.0
	t_cool = 1./(1./t_cool_ff + 1./t_cool_IC)
	eps_rad = 1./(1.+t_cool/t_dyn)
	Lwind_rad = np.trapz(eps_rad * 0.5*dMdotdv*v_array**2, v_array)
	Lwind_kin = np.trapz(0.5*dMdotdv*v_array**2, v_array)
	Lwind_gas = Lwind_kin - Lwind_rad
	return Lwind_rad, Lwind_gas, Lwind_kin

#
# main
#
def evolve_ejecta(Mej, MCO, MNi, Mstar, Eexp, kap_bol, t_TDE, R_disk_in, visc_param):
	# initialize arrays
	t_arr = np.linspace(0., tLC_end, 50000)*86400
	rej_arr = np.zeros(len(t_arr))
	rneb_arr = np.zeros(len(t_arr))
	vej_arr = np.zeros(len(t_arr)) 
	Egas_arr = np.zeros(len(t_arr)) 
	Erad_arr = np.zeros(len(t_arr)) 
	epsrad_arr = np.zeros(len(t_arr)) 
	L_arr = np.zeros(len(t_arr)) 
	Lwind_arr = np.zeros(len(t_arr))
	Mwind_arr = np.zeros(len(t_arr))
	x_arr = np.zeros(len(t_arr))
	tauej_arr = np.zeros(len(t_arr))
	Teff_arr = np.zeros(len(t_arr)) 

	# initial condition for ejecta
	# (3-delta)*Mej*v^2/2(5-delta) = Ekin = Eint
	i = 0
	t = 0.0e0
	r_ej = Rej_init
	x = 1.0e0
	Erad = 0.5 * Eexp 
	Egas = 0.0 
	Ekin = 0.5 * Eexp
	v_ej = (2.*(5.-delta_ej)/(3.-delta_ej)*Ekin/Mej)**0.5

	# initial accretion disk properties
	if Mstar > 0.0:
		Rstar = con.Rsun * (Mstar/con.Msun)**0.6
		# can't be lower than the inner edge of disk, although this never happens for a normal star
		Rdisk_init = max(2.*Rstar*(MCO/Mstar), R_disk_in)
		Mdisk_init = MCO 
	else: # no companion, just put random value for disk radius
		Rdisk_init = 1e12
		Mdisk_init = 0.0
	Mwind = 0.0
	# disk angular momentum. take into account finite Mdisk_init as Mdisk_init ~ MCO
	Jdisk_init = 2./3. * math.sqrt(con.G*MCO*Rdisk_init) * MCO * (math.sqrt(1.+Mdisk_init/MCO)**3 - 1.)
	# viscous time. 1/(alpha*(H/R)**2) is absorbed as a single free parameter visc_param
	if Mdisk_init > 0.0:
		tvisc_init = 1./visc_param * math.sqrt(Rdisk_init**3/con.G/MCO) * (2.*MCO/Mdisk_init) * (math.sqrt(1.+Mdisk_init/MCO) - 1.) 
	else:
		tvisc_init = 1./visc_param * math.sqrt(Rdisk_init**3/con.G/MCO)

	# initialize disk parameters
	Mdisk = Mdisk_init
	Jdisk = Jdisk_init
	Rdisk = Rdisk_init
	tvisc = tvisc_init

	# loop over time...
	while t<t_arr[-1]:
		# photon diffusion time (alphaI_m from Arnett80,82)
		tdiff = 3.*kap_bol*Mej/4./math.pi/con.c/r_ej / alpha_Im
		# timestep. limit by diffusion, dynamical, viscous time
		dt = 0.01*min(tdiff, r_ej/v_ej, tvisc, 20.*(t_arr[1]-t_arr[0]))

		if (t > t_TDE and Mdisk>0.0): 
			# disk has formed. calculate disk wind
			Mdot = Mdisk/tvisc
			F_w = (p_BB)/(p_BB+0.5) # AM lever arm (Shen & Matzner 14)

			# critical radius
			R_crit = con.G*MCO / (v_neb)**2 
			R_crit = min(R_crit, Rdisk) 
			Lwind = 0.5*p_BB/(1.-p_BB) * con.G*MCO*Mdot/Rdisk * ((R_disk_in/Rdisk)**(p_BB-1.) - (R_crit/Rdisk)**(p_BB-1.))
			vneb_old = v_neb
			#
			# evolve wind nebula
			#
			# averaged wind
			Mdot_wind = Mdot * ((R_crit/Rdisk)**p_BB - (R_disk_in/Rdisk)**p_BB)
			vwind = math.sqrt(2.*Lwind/Mdot_wind)
			# densities at termination shock upstream
			rho_ej = (3.-delta_ej)/(4.*math.pi) * (Mej/r_ej**3) * (r_neb/r_ej)**(-delta_ej)
			rho_wind = Mdot_wind/(4.*math.pi*r_neb**2*vwind)
			# obtain nebula velocity. second term is at shocked wind rest frame (tilde{v}_neb)
			v_neb = r_neb/t + (vwind-r_neb/t)/(1.+math.sqrt(rho_ej/rho_wind)) 

			if evolve_cooling:
				#
				# Calculate radiative power of disk wind
				#
				# dynamical time
				t_dyn = r_neb/vneb_old
				# array of wind velocities
				v_max = math.sqrt(con.G*MCO/R_disk_in)
				vwind_array = np.logspace(np.log10(vneb_old), np.log10(v_max), 100)[1:]
				dMdotdv = (2.*p_BB) * (Mdot/vwind_array) * (vwind_array/math.sqrt(con.G*MCO/Rdisk))**(-2*p_BB)
				# shocked wind temperature and density from jump conditions.
				# for density, unshocked wind mass-loss rate scales with r^(-p) \propto (v_wind)^(-2p)
				Tsh_w = 2.*(gamma-1)/(gamma+1)**2 * mu*con.m_p*(vwind_array)**2/con.k_B
				rhosh_w = (gamma+1)/(gamma-1) * (dMdotdv*vwind_array) / (4.*math.pi*r_neb**2*vneb_old) 
				# radiation energy density
				u_rad = Erad / (4.*math.pi*r_ej**3/3.)
				Lwind_rad, Lwind_gas, Lwind_kin = wind_luminosity(t_dyn, r_neb, vwind_array, dMdotdv, rhosh_w, Tsh_w, u_rad)
				epsilon_rad = Lwind_rad/Lwind_kin
				# sanity check
				if abs(Lwind_kin-Lwind)/(Lwind)>0.03:
					print("warning -- large error in Lwind: Lwind=%g, Lwind_kin=%g, vneb=%g, last vneb=%g, dt=%g" % (Lwind, Lwind_kin, v_neb, vneb_old, dt))
			else: # simply assume everything launched from disk thermalizes and goes to radiation (most optimistic)
				Lwind_rad = 0.5*p_BB/(1.-p_BB) * con.G*MCO*Mdot/Rdisk * ((R_disk_in/Rdisk)**(p_BB-1.) - (R_crit/Rdisk)**(p_BB-1.))
				Lwind_gas = 0.0
				epsilon_rad = 1.0

			#
			# timestep control
			# limit timestep so that r_neb doesn't increase by more than 30% 
			#
			dt = min(dt, 0.3*r_neb/v_neb)
			r_neb += v_neb * dt
			# cap by r_ej (this happens if the wind energy exceeds ejecta kinetic energy)
			r_neb = min(r_neb, r_ej)
			if r_neb == r_ej:
				v_neb = v_ej

			Mwind += Mdot_wind * dt

			# evolve disk
			# update Mdisk and Jdisk
			Mdisk -= Mdisk/tvisc * dt
			Jdisk -= F_w*Jdisk/tvisc * dt
			# update rdisk based on new Jdisk and Mdisk
			Rdisk = 2.25*Jdisk**2/(con.G*MCO**3) / ((1.+Mdisk/MCO)**1.5 -1)**2 
			# update viscous time based on new Rdisk and Mdisk
			tvisc = 1./visc_param * math.sqrt(Rdisk**3/con.G/MCO) * (2.*MCO/Mdisk) * (math.sqrt(1.+Mdisk/MCO) - 1.)

		else:
			# wind hasn't been launched yet
			Lwind_rad = 0.0
			Lwind_gas = 0.0
			epsilon_rad = 0.0
			# nebula parameters
			r_neb = Rdisk
			v_neb = Rdisk/max(t, 1.e-10) 

		# evolve ejecta thermodynamics (Sec 2.3)
		pdV_rad = (Erad+2.*Egas)/(Erad+Egas) * Erad*v_ej/r_ej
		pdV_gas = (Erad+2.*Egas)/(Erad+Egas) * Egas*v_ej/r_ej
		Lrad = Erad/tdiff
		LNi = L_Ni56(t, MNi, Mej, r_ej)
		tau_ej = (3.-delta_ej)*kap_bol*Mej/4./math.pi/r_ej**2
		Erad += dt * (-pdV_rad + (1.-math.exp(-tau_ej))*Lwind_rad + LNi - Lrad)
		Egas += dt * (-pdV_gas + Lwind_gas)
		r_ej += dt * v_ej
		v_ej += dt * (pdV_rad + pdV_gas) * (5-delta_ej)/(3-delta_ej)/Mej/v_ej # E_ej = (3-delta)/2(5-delta) Mej v_ej**2
		x = min(1.0, math.sqrt(Lrad/4./3.14/r_ej**2/con.sigma_SB/T_eff_min**4))
			
		t += dt

		if t > t_arr[i]:
			if i%5000==0:
				print("    done %.2g[day]/%.2g[day]" % (t/86400., tLC_end))
			# append all parameters
			rej_arr[i] = r_ej
			rneb_arr[i] = r_neb
			vej_arr[i] = v_ej
			Erad_arr[i] = Erad
			epsrad_arr[i] = epsilon_rad 
			Egas_arr[i] = Egas
			L_arr[i] = Lrad
			Lwind_arr[i] = Lwind_rad + Lwind_gas
			Mwind_arr[i] = Mwind/con.Msun 
			x_arr[i] = x
			tauej_arr[i] = tau_ej 
			Teff_arr[i] = max(T_eff_min, (Lrad/4./math.pi/r_ej**2/con.sigma_SB)**(0.25))
			i += 1

	return t_arr, L_arr, Lwind_arr, rej_arr, vej_arr, rneb_arr, x_arr, Erad_arr, Egas_arr, epsrad_arr, tauej_arr, Teff_arr, Mwind_arr


###############################################
#                                             # 
#         parameter grid study                #
#                                             # 
###############################################

# parameters regarding CO and accretion wind
print('Compact object: %s' % CO)
if CO == 'BH':
	r_disk_in = 6.*con.G*MCO/con.c**2 # ISCO
elif CO == 'NS':
	r_disk_in = 1.2e6 # NS radius 
else:
	raise ValueError('undefined compact object')

# varying ejecta models...
for (Mej, Eexp, MNi) in Mej_Eexp_MNi_array:
	Mej *= con.Msun # Msun -> g
	MNi *= con.Msun # Msun -> g
	print("SN Ejecta model: Mej=%d Msun, Eej=%s erg" % (int(Mej/con.Msun), str(Eexp).replace('+','')))

	# varying other params...
	for (Mstar,t_TDE,visc_param) in itertools.product(Mstar_array, t_TDE_array, visc_param_array):
		t_TDE *= 86400. # day -> sec
		Mstar *= con.Msun # Msun -> g
		print("working on Mstar=%g Msun, t_TDE=%g day, visc_param=%g" % (Mstar/con.Msun, t_TDE/86400., visc_param))
		t_arr, L_arr, Lwind_arr, r_arr, v_arr, rneb_arr, x_arr, Erad_arr, Egas_arr, epsrad_arr, tauej_arr, Teff_arr, Mwind_arr = evolve_ejecta(Mej, MCO, MNi, Mstar, Eexp, kap_bol, t_TDE, r_disk_in, visc_param) 
		np.savetxt('Mej%g_Eej%s_Mstar%gMsun_tTDE%gday_visc%.1e.txt' % (int(Mej/con.Msun), str(Eexp).replace('+',''), Mstar/con.Msun, t_TDE/86400., visc_param), np.c_[t_arr/86400., L_arr, Teff_arr, Lwind_arr, r_arr, v_arr/1e5, rneb_arr, Erad_arr, Egas_arr, epsrad_arr, tauej_arr, Mwind_arr], header='time [day], luminosity [erg/s], T_eff [K], L_wind [erg/s], r_ej [cm], v_ej [km/s], r_neb [cm], Erad [erg], Egas [erg], epsilon_rad, tau_ej, M_wind [Msun]', fmt='%.6g')

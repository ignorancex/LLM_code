import math
import numpy as np

import constants


s = 0.0 # CSM density profile power-law index (s=0.0: uniform, as implicitly assumed in light curve calculation)


# calculate light curve properties, in the region where 10-90 % of energy is radiated
# this assumes time array is evenly sampled
def calc_LC_Lt(time, lum, t_BH):
	# checks
	assert min(lum) >= 0.0, "Negative luminosity"
	assert abs((time[1]-time[0])-(time[-1]-time[-2]))/(time[-1]-time[-2]) < 1e-6, "uneven timestep"
	# extract only when t > tBH
	lum = lum[time > t_BH]
	time = time[time > t_BH]
	# get cumulative sum
	cum_lum = np.cumsum(lum)
	index_10 = np.argmin(np.abs(cum_lum-0.1*cum_lum[-1]))
	index_90 = np.argmin(np.abs(cum_lum-0.9*cum_lum[-1]))
	# compute characteristic timescale and luminosity
	char_time = (time[index_90]-time[index_10])/86400
	char_lum = (cum_lum[index_90] - cum_lum[index_10]) * (time[1]-time[0])/(time[index_90]-time[index_10])
	return char_time, char_lum 


# rho_CSM at (a_bin, t)
def rho_CSM(MCSM, a_bin, t, v_esc, xi, chi):
	return (3.-s)*MCSM*t**(-3)*(a_bin/t)**(-s)/(4.*math.pi*v_esc**(3.-s)*(xi**(3.-s)-chi**(3.-s))) 

# CSM evolution script
def evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion, xi, x_ion_floor, p_BB, r_disk_in):
	chi = (1.-Rstar/a_bin)**0.5
	v_esc = (2.*constants.G*Mstar/Rstar)**0.5
	v_orb = (constants.G*(Mstar+MCO)/a_bin)**(0.5)
	t_BH = a_bin/v_esc/xi
	BH_flag = 0
	print("MCSM=%g Msun, a_bin=%g Rstar, t_BH=%g day" % (MCSM/constants.Msun, a_bin/Rstar, t_BH/86400))
	# initialize arrays
	if Rstar > 30.*constants.Rsun:
		t_arr = np.linspace(0., min(30.*t_BH/86400., 3000), 100000)*86400
	else:
		t_arr = np.linspace(0., 100, 100000)*86400
	r_arr = np.zeros(len(t_arr)-1) 
	v_arr = np.zeros(len(t_arr)-1) 
	Eint_arr = np.zeros(len(t_arr)-1) 
	L_arr = np.zeros(len(t_arr)-1) 
	Lwind_arr = np.zeros(len(t_arr)-1) 
	x_arr = np.zeros(len(t_arr)-1)

	# initial condition
	i = 0
	t = 0.0e0
	t_rec = 1e10 # time when recombination starts
	r = Rstar
	x = 1.0e0
	x_sat = 1.0e0
	# 0.5*MCSM*v^2 = Ekin = Eint = 0.15*MCSM*(xi*v_esc)^2 # 0.5*(0.5*MCSM*(xi*v_esc)^2)
	v = xi*v_esc*math.sqrt(0.3)
	Eint = 0.15*MCSM*(xi*v_esc)**2

	while t<t_arr[-1]:
		if (t > t_BH and t < a_bin/v_esc/chi): 
			Rbondi = constants.G * MCO / ((a_bin/t)**2 + v_orb**2)
			Mdotacc = 4.*math.pi*Rbondi**2 * ((a_bin/t)**2 + v_orb**2)**0.5 * rho_CSM(MCSM, a_bin, t, v_esc, xi, chi)
			# obtain average vwind, and then efficiency is 0.5*(vwind/c)^2.
			r_disk_out = min(Rbondi, (Rbondi**2/t)**2/(constants.G*MCO))
			if r_disk_out > r_disk_in:
				vwind = math.sqrt(p_BB/(1.-p_BB) * constants.G*MCO/r_disk_in * ((r_disk_in/r_disk_out)**p_BB - (r_disk_in/r_disk_out))) 
			else: # disk does not form outside ISCO, so no wind
				vwind = 0.0

			Lwind = 0.5 * vwind**2 * Mdotacc
		else:
			Lwind = 0.0
		# diffusion time when fully ionized 
		tdiff = 3.*kappa*MCSM/4./math.pi/constants.c/r
		# timestep when fully ionized
		dt = 0.01*min(tdiff, r/v, 10.*(t_arr[1]-t_arr[0]))

		if abs(x-1.0e0) < 1e-3:
			pdV = v*Eint/r
			r += dt * v
			v += dt * Eint/MCSM/r / 0.6 # 0.6 comes from Ekin=0.3MCSM v^2 
			Lrad = Eint/tdiff/x**2
			Eint += dt * (-pdV + Lwind - Lrad)
			x = min(1.0, math.sqrt(Lrad/4./math.pi/r**2/constants.sigma_SB/T_ion**4))
			if t > 0.0:
				t_rec = t
			if BH_flag==0 and Lwind > 0.0:
				BH_flag = 1
		elif x > 0.0e0:
			if BH_flag==0 and Lwind > 0.0:
				BH_flag = 1
				param = r/(4.*v*tdiff)
				x_sat = -param+math.sqrt(param**2 + 2.*param*Lwind/4./math.pi/r**2/constants.sigma_SB/T_ion**4)
				x_sat = max(x_ion_floor, x_sat)

			# limit timestep by not making x_i rise too fast, when wind is turned on
			#print(t, Lwind, dt, 0.01*tdiff, 0.01*r/v)
			if Lwind > 0.0:
				# keep update x_sat
				param = r/(4.*v*tdiff)
				x_sat = -param+math.sqrt(param**2 + 2.*param*Lwind/4./math.pi/r**2/constants.sigma_SB/T_ion**4)
				x_sat = max(x_ion_floor, x_sat)
				dt = min(dt, 0.05*min(1.,x_sat)*x*tdiff / abs(Lwind/(4.*math.pi*x**2*r**2*constants.sigma_SB*T_ion**4) - 1.))
			xr_old = x*r
			x -= (dt/5.)*(2.*v/r + (1.0 - Lwind/4./math.pi/xr_old**2/constants.sigma_SB/T_ion**4)/x/tdiff)
			# as ionizations goes as runaway, set a floor value
			x = min(max(x_ion_floor, x),1.0)

			r += dt * v
			xr = x*r
			eps_int = 3.*x*tdiff/r * constants.sigma_SB*T_ion**4
			v += 4.*math.pi/3.*(xr**3-xr_old**3)*eps_int/(3.*MCSM*v) / 0.6 # 0.6 comes from Ekin=0.3MCSM v^2 
			Eint = eps_int*(4.*math.pi/3.)*(xr)**3
			Lrad = 4.*math.pi*(xr)**2*constants.sigma_SB*T_ion**4 
		else:
			raise ValueError("Invalid ionization locaton: x=%g at t=%g day" % (x, t/86400.))
		
		if t > t_arr[i]:
			# append all parameters
			r_arr[i] = r
			v_arr[i] = v
			Eint_arr[i] = Eint
			L_arr[i] = Lrad
			Lwind_arr[i] = Lwind
			x_arr[i] = x
			i += 1
		t += dt
	return t_arr[:-1], L_arr, Lwind_arr, v_arr, x_arr, t_BH

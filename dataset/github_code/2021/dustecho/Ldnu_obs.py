import numpy as np
import const
from fpath import *
from math import pi, log10, sqrt, log, exp, sqrt, atan, cos, sin, acos, asin
import sys

# take 3 arguments from the command line
nH0 = float(sys.argv[1])        # [cm^-3] H number density
lamobs = float(sys.argv[2])     # [um] observer's wavelength
theobs = float(sys.argv[3])      # [rad] observer's viewing angle wrt. jet axis

# adjustable parameters
thej = 4*pi/180     # [rad] jet opening angle

# fixed ones
n0_over_nH = 1.45e-15    # dust number density over H number density
lam0 = 2.       # [um] critical wavelength for Qabs_lambda

# observer's time grid [sec]
Ntobs = 100
tobsmin = 0.1*const.pc2cm/const.C_LIGHT*(1-cos(max(theobs, thej)))
tobsmax = 4e3*tobsmin     # [sec]
tobsarr = np.logspace(log10(tobsmin), log10(tobsmax), Ntobs)
Ldnuarr = np.zeros(Ntobs, dtype=float)
xcentrarr = np.zeros(Ntobs, dtype=float)


def func_nH(r):      # gas density profile (r in pc)
    return nH0


def jdnu_intp(t, j_r, jdnuarr, tarr):
    # linear interpolation in time for a given r (given by index j_r)
    i_floor = np.argmin(np.abs(t-tarr))
    if t < tarr[i_floor] and i_floor != 0:
        i_floor -= 1
    if i_floor == Nt - 1:
        i_floor -= 1
    slope = (jdnuarr[i_floor+1, j_r] - jdnuarr[i_floor, j_r])\
            /(tarr[i_floor+1] - tarr[i_floor])
    return max(jdnuarr[i_floor, j_r] + slope * (t - tarr[i_floor]), 0)


# read the data for Td, asub (generated from 'generate_Td_asub')
savelist = ['Td', 'asub']   # no need for taud
for i_file in range(len(savelist)):
    fname = 'nH%.1e_' % nH0 + savelist[i_file]
    with open(savedir+fname + '.txt', 'r') as f:
        if savelist[i_file] == 'Td':
            row = f.readline().strip('\n').split('\t')
            tmin, tmax, Nt = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            rmin, rmax, Nr = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            amin, amax, Na = float(row[3]), float(row[4]), int(row[5])

            Tarr = np.zeros((Nt, Nr, Na), dtype=float)

            f.readline()    # skip this line
            for i in range(Nt):
                f.readline()    # skip this line
                for j in range(Nr):
                    row = f.readline().strip('\n').split('\t')
                    for k in range(Na):
                        Tarr[i, j, k] = float(row[k])
        # elif savelist[i_file] == 'taud':      # this file is not used for echo lightcurve
        #     row = f.readline().strip('\n').split('\t')
        #     numin, numax, Nnu = float(row[3]), float(row[4]), int(row[5])
        #     row = f.readline().strip('\n').split('\t')
        #     rmin, rmax, Nr = float(row[3]), float(row[4]), int(row[5])
        #
        #     taudarr = np.zeros((Nnu, Nr), dtype=float)
        #
        #     f.readline()    # skip this line
        #     for m in range(Nnu):
        #         row = f.readline().strip('\n').split('\t')
        #         for j in range(Nr):
        #             taudarr[m, j] = float(row[j])
        else:   # asubarr
            row = f.readline().strip('\n').split('\t')
            tmin, tmax, Nt = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            rmin, rmax, Nr = float(row[3]), float(row[4]), int(row[5])

            asubarr = np.zeros((Nt, Nr), dtype=float)

            f.readline()  # skip this line
            for i in range(Nt):
                row = f.readline().strip('\n').split('\t')
                for j in range(Nr):
                    asubarr[i, j] = float(row[j])

rarr = np.logspace(log10(rmin), log10(rmax), Nr)
r_ratio = rarr[1]/rarr[0]

tarr = np.linspace(tmin, tmax, Nt, endpoint=False)
dt = tarr[1] - tarr[0]
tarr += dt/2.

aarr = np.logspace(log10(amin), log10(amax), Na)
a_ratio = aarr[1]/aarr[0]

# nuarr = np.logspace(log10(numin), log10(numax), Nnu)    # frequency bins
# nu_ratio = nuarr[1]/nuarr[0]

jdnuarr = np.zeros((Nt, Nr), dtype=float)    # volumetric emissivity at lamobs


# then we calculate the volumetric emissivity jdnuarr(t, r)
# by integrating over dust size distribution
nuobs = const.C_LIGHT/(lamobs*1e-4)  # [Hz], observer's frequency
h_nu_over_k = const.H_PLANCK*nuobs/const.K_B    # a useful constant
for i in range(Nt):
    for j in range(Nr):
        r = rarr[j]
        j_pre_factor = 2*pi*const.H_PLANCK * nuobs/lamobs**2\
                       * n0_over_nH * func_nH(r)
        j_integ = 0.
        a = asubarr[i, j]
        while a <= amax:
            k = round(log10(a/amin)/log10(a_ratio))
            da = a * (sqrt(a_ratio) - 1./sqrt(a_ratio))
            if Tarr[i, j, k] < 100:  # ignore extremely cold dust
                a *= a_ratio
                continue
            j_integ += da * a**-0.5 /(a + (lamobs/lam0)**2) \
                       / (exp(h_nu_over_k/Tarr[i, j, k]) - 1)
            a *= a_ratio
        jdnuarr[i, j] = j_pre_factor*j_integ

# then we calculate observed lightcurve taking into account light-travel delay
Nmu = 100    # resolution of the bright stripe
percent = 0
for i_tobs in range(Ntobs):
    if 100*i_tobs/Ntobs > percent:
        print('%d %%' % percent)
        percent += 10
    tobs = tobsarr[i_tobs]
    Ldnu = 0.
    Ldnu_xcentr = 0.
    for j in range(Nr):
        r = rarr[j]
        dr = r * (sqrt(r_ratio) - 1/sqrt(r_ratio))
        if r*const.pc2cm < max(0, const.C_LIGHT*(tobs-tmax))/(1 - cos(theobs+thej)):
            continue    # light echo has already passed
        if theobs > thej and r*const.pc2cm > const.C_LIGHT*(tobs-tmin)/(1 - cos(theobs-thej)):
            # print('light echo hasnt arrived yet')
            continue    # light echo hasn't arrived yet
        mumin = max([cos(theobs+thej), 1 - const.C_LIGHT*(tobs-tmin)/(r*const.pc2cm)])
        mumax = min([cos(max(1e-10, theobs-thej)),
                     1 - const.C_LIGHT*(tobs-tmax)/(r*const.pc2cm)])
        if mumax < mumin:   # the region is outside the jet cone
            # print(mumax, mumin, 'mumax < mumin')
            continue
        # print('mumax-mumin', mumax-mumin)
        dmu = (mumax - mumin)/Nmu
        mu = mumin + dmu/2
        mu_integ = 0.
        mu_integ_xcentr = 0.
        while mu < mumax:
            if mu >= cos(max(1e-10, thej-theobs)):
                tphimax = pi
            else:
                phi = acos((mu - cos(theobs)*cos(thej))/(sin(theobs)*sin(thej)))
                if abs(sin(thej)*sin(phi)/sqrt(1-mu*mu)) > 1:  # avoid round-off errors
                    tphimax = pi
                else:
                    tphimax = asin(sin(thej)*sin(phi)/sqrt(1-mu*mu))
            t = tobs - r*const.pc2cm/const.C_LIGHT*(1-mu)
            jdnu = jdnu_intp(t, j, jdnuarr, tarr)
            mu_integ += dmu * jdnu * 2 * tphimax
            mu_integ_xcentr += dmu * r * sqrt(1-mu*mu) * jdnu * 2 * sin(tphimax)
            mu += dmu
        # print(mu_integ)
        Ldnu += 4*pi*dr*r*r * const.pc2cm**3 * mu_integ
        Ldnu_xcentr += 4*pi*dr*r*r * const.pc2cm**3 * mu_integ_xcentr
    Ldnuarr[i_tobs] = Ldnu
    if Ldnu < 1e10:    # ~zero flux
        xcentrarr[i_tobs] = np.NAN
    else:
        xcentrarr[i_tobs] = Ldnu_xcentr/Ldnu
    # print('tobs=%.1e yr' % (tobs / const.yr2sec),
    #       'Ldnu=%.1e' % Ldnu, 'xcentr=%.1e' % xcentrarr[i_tobs])


# write the data into files
param = 'nH%.1e_lam%.2fum_theobs%.2f_' % (nH0, lamobs, theobs)
with open(savedir+param+'Ldnu_xcentr.txt', 'w') as f:
    f.write('tobsmin\ttobsmax\tNtobs\t%.8e\t%.8e\t%d' % (tobsmin, tobsmax, Ntobs))
    f.write('\nLdnu[cgs]\txcentr[pc]')
    f.write('\n')
    for i_tobs in range(Ntobs):
        if i_tobs == 0:
            f.write('%.8e' % Ldnuarr[i_tobs])
        else:
            f.write('\t%.8e' % Ldnuarr[i_tobs])
    f.write('\n')
    for i_tobs in range(Ntobs):
        if i_tobs == 0:
            f.write('%.8e' % xcentrarr[i_tobs])
        else:
            f.write('\t%.8e' % xcentrarr[i_tobs])
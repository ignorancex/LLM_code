import numpy as np
import cst
from load_kap_table import *
import pickle
from math import pi, sqrt, sin, cos, tan, log, log10, floor, ceil

savedir = './disk_sltns/'
nofL2 = False   # turn on/off fL2

# adjustable parameters
# also remember to change "case" in "load_kap_table"!!
M2_in_Msun = 10.   # [Msun]
q = .5     # mass ratio accretor/donnor = M2/M1

savename = 'fL2grid_M%.1f_q%.1f_case%d' % (M2_in_Msun, q, case)

# grid for mass transfer rate and binary separation
NM1dot = 200   # (100, 100) is sufficiently accurate
Na = 200
logM1dotmin, logM1dotmax = -5.3, -1.3   # [Rsun]
logamin, logamax = 0.15, 3.25   # [Msun/yr]
logM1dotgrid = np.linspace(logM1dotmin, logM1dotmax, NM1dot)
logagrid = np.linspace(logamin, logamax, Na)

# ---caution:
# for very large binary separation a >> 1 AU and low Mdot << 1e-5 Msun/yr,
# the disk temperature becomes much less than 1e4 K, then some of the opacities tables
# are less reliable

# other parameters (that are not part of the grid)
alpha_ss = 0.1   # viscosity
tol = 1e-8   # fractional tolerance for bisection method
eps_small = 1e-12   # a very small number

# below is for grid search for disk thickness [only used at the beginning]
Nthe = 50   # ~50 is accurate enough
thegrid_min, thegrid_max = 0.1, 1.   # 0.1 to 1 is sufficient, we use analytic result below 0.1
thegrid = np.logspace(log10(thegrid_min), log10(thegrid_max), Nthe, endpoint=True)
Tfloor = 3e3   # [K] --- the minimum value for disk temperature solution

if case in [1, 3]:   # mean molecular weight
    mug = 1.3/2.4   # H-rich
else:
    mug = 4./3    # H-poor (fully ionized He)

# record the solutions in 2D grid
solu_the = np.empty((NM1dot, Na), dtype=float)  # outer disk thickness H/R
solu_T = np.empty((NM1dot, Na), dtype=float)   # outer disk temperature
solu_rho = np.empty((NM1dot, Na), dtype=float)
solu_tau = np.empty((NM1dot, Na), dtype=float)
solu_fL2 = np.empty((NM1dot, Na), dtype=float)      # outer disk
solu_fL2_inner = np.empty((NM1dot, Na), dtype=float)      # inner disk
Lacc_over_LEdd = np.empty((NM1dot, Na), dtype=float)
QL2Qadv_over_Qrad = np.empty((NM1dot, Na), dtype=float)


lgq = log10(q)
# positions of Lagrangian points (based on analytic fits)
xL1 = -0.0355 * lgq**2 + 0.251 * abs(lgq) + 0.500   # [a = SMA]
xL2 = 0.0756 * lgq**2 - 0.424 * abs(lgq) + 1.699  # [a]
if lgq > 0:   # m2 is more massive
    xL1 = 1 - xL1
    xL2 = 1 - xL2
mu = q/(1 + q)
# outer disk radius
Rd_over_a = (1-xL1)**4/mu
# relavent potential energies
PhiL1_dimless = -((1-mu)/abs(xL1) + mu/abs(1-xL1) + 0.5*(xL1-mu)**2)   # [G(M1+M2)/a]
PhiL2_dimless = -((1-mu)/abs(xL2) + mu/abs(1-xL2) + 0.5*(xL2-mu)**2)    # [G(M1+M2)/a]
PhiRd_dimless = -(1 - mu + mu/Rd_over_a + 0.5 * (1-mu)**2)

# recover CGS units
M2 = M2_in_Msun * cst.msun
GM2 = cst.G*M2

# --- checks
# print('PhiL1=%.3e, PhiL2=%.3e, PhiRd=%.3e, Rd=%.3e'
#       % (PhiL1_dimless, PhiL2_dimless, PhiRd_dimless, Rd_over_a))
# exit()


# --- below we predefine some useful numerical functions


def f1_the_T_fL2(the, T, fL2):
    return c1*T**4 * the**3 / (1-fL2) - the**2 + c2*T


def kap(rho, T):
    # return 0.34 + 3e24*rho*T**-3.5   # Kramer's opacity [not very accurate]
    lgT = log10(T)
    lgrho = log10(rho)
    lgR = lgrho - 3*lgT + 18
    if lgR > lgRgrid[-1]:  # very high density region (use Kramer's law extrapolation)
        lgrhomax = lgRgrid[-1] + 3*lgT - 18
        return 10**intp_lgkapgrid(lgRgrid[-1], lgT) + lgrho - lgrhomax
    # ---- Warning signs outside boundaries (actual solutions are unaffected)
    # if lgR < lgRgrid[0]:
    #     print('Warning! lgR=%.3f < lgRmin' % lgR)
    # if lgT < lgTgrid[0] or lgT > lgTgrid[-1]:
    #     print('Warning! lgT=%.3f outside the range (lgTmin, lgTmax)' % lgT)
    return 10**intp_lgkapgrid(lgR, lgT)


def f2_the_T_fL2(the, T, fL2):
    x = c4 * (T*the)**3 / (1-fL2)
    U_over_P = (1.5 + x)/(1 + 1./3 * x)
    rho = (1-fL2) * M1dot / (2*pi * alpha_ss * omgK * Rd**3) / the**3
    return 7./4 - (1.5 * U_over_P + c3*T**4 / kap(rho, T) / (1-fL2)**2) * the**2 \
        - PhiRd/(GM2/Rd) + (PhiL1 - fL2*PhiL2)/(GM2/Rd)/(1-fL2)


def T_the_nofL2(the):   # under the assumption fL2=0
    # return ((the**2/c2)**(-s) + ((c1*the)**-0.25)**(-s))**(-1./s)  # analytic (not perfect)
    logthe = log10(the)
    if logthe > logthegrid[-2]:  # use analytic extrapolation
        return 10**(logTarr[-2] - 0.25 * (logthe - logthegrid[-2]))
    if logthe < logthegrid[0]:  # analytic extrapolation
        return 10**(logTarr[0] + 2 * (logthe - logthegrid[0]))
    ithe = floor((logthe - logthegrid[0])/dlogthe)
    slope = (logTarr[ithe+1] - logTarr[ithe])/dlogthe
    logT = logTarr[ithe] + (logthe - logthegrid[ithe]) * slope
    return 10**logT


def T_the(the):   # only for non-zero fL2
    return (8*the**2 + 1 - 2 * (PhiL2-PhiRd)/(GM2/Rd))/(3*c2)


def fL2_the(the):  # only for non-zero fL2
    T = T_the(the)
    return 1 - c1 * T**4 * the**3/(the**2 - c2*T)


percent = 5  # progress bar
for n1 in range(NM1dot):
    if 100*n1/NM1dot > percent:
        print('%d percent' % percent)
        percent += 5
    M1dot = 10**logM1dotgrid[n1] * cst.msun/cst.yr
    for n2 in range(Na):
        a = 10**logagrid[n2] * cst.rsun
        Rd = Rd_over_a * a
        Phi_units = cst.G*(M2/mu)/a
        PhiL1 = PhiL1_dimless * Phi_units
        PhiL2 = PhiL2_dimless * Phi_units
        PhiRd = PhiRd_dimless * Phi_units
        # Keplerian frequency at Rd
        omgK = sqrt(GM2/Rd**3)

        # constants involved in numerical solutions
        c1 = 2*pi * cst.arad * alpha_ss * Rd / (3 * omgK * M1dot)
        c2 = cst.kB * Rd / (GM2 * mug * cst.mp)
        c3 = 8*pi**2 * cst.arad * alpha_ss * cst.c * Rd**2 / (M1dot**2 * omgK)
        c4 = 2*pi * mug * cst.arad * alpha_ss * omgK * cst.mp * Rd**3 / (cst.kB * M1dot)

        # only T < Tmax is possible
        Tmax = (4./(27*c1**2*c2))**(1./9)

        Tarr = np.zeros(Nthe, dtype=float)
        for i in range(Nthe):
            the = thegrid[i]
            # use bisection method
            Tleft = 0.1 * min(the**2 / c2, Tmax)
            f1left = f1_the_T_fL2(the, Tleft, fL2=0)
            Tright = Tmax
            while abs((Tleft-Tright)/Tright) > tol:
                T = (Tleft + Tright)/2
                f1 = f1_the_T_fL2(the, T, fL2=0)
                if f1 * f1left > 0:
                    Tleft = T
                    f1left = f1
                else:
                    Tright = T
            Tarr[i] = (Tleft + Tright)/2
        # now we have obtained numerical relation between the and T
        logTarr = np.log10(Tarr)
        logthegrid = np.log10(thegrid)
        dlogthe = logthegrid[1] - logthegrid[0]

        # bisection to find the numerical solution to f2(the, T, fL2=0)=0
        theright = 1.
        f2right = f2_the_T_fL2(theright, T_the_nofL2(theright), fL2=0)
        separation_factor = 0.95
        theleft = separation_factor * theright
        f2left = f2_the_T_fL2(theleft, T_the_nofL2(theleft), fL2=0)
        while f2left*f2right > 0:  # need to decrease theleft
            theright = theleft
            f2right = f2left
            theleft *= separation_factor
            f2left = f2_the_T_fL2(theleft, T_the_nofL2(theleft), fL2=0)
        # now the solution is between theleft and theright
        while abs((theleft-theright)/theright) > tol:
            the = (theleft + theright)/2
            f2 = f2_the_T_fL2(the, T_the_nofL2(the), fL2=0)
            if f2 * f2left > 0:
                theleft = the
                f2left = f2
            else:
                theright = the
        # solution
        the = (theleft + theright)/2
        T = T_the_nofL2(the)
        rho = M1dot / (2*pi * alpha_ss * omgK * Rd**3) / the**3
        themax = sqrt(3./8 * c2 * T + 1./4 * (PhiL2-PhiRd)/(GM2/Rd) - 1./8)

        if the < themax or nofL2:   # this is the correct solution
            tau = rho*kap(rho,T)*Rd*the / 2
            Qrad = 2*pi*Rd**2*cst.arad*T**4*cst.c/tau
            U = cst.arad*T**4 + 1.5*rho*cst.kB*T/mug/cst.mp
            P = 1./3*cst.arad*T**4 + rho*cst.kB*T/mug/cst.mp
            tvis = 1./(alpha_ss * the**2 * omgK)
            Md = 2*pi*rho*Rd**3*the
            Qadv = 1.5*U/P * the**2 * GM2/Rd / tvis * Md
            QL2 = 0.
            solu_rho[n1, n2] = rho
            solu_the[n1, n2] = the
            solu_T[n1, n2] = T
            solu_tau[n1, n2] = tau
            solu_fL2[n1, n2] = eps_small
            QL2Qadv_over_Qrad[n1, n2] = (QL2+Qadv)/Qrad
            Lacc_over_LEdd[n1, n2] = M1dot * kap(rho, T) / (4*pi * Rd * cst.c)
            continue

        # ---- below is for fL2 \neq 0

        themin = 1./2 * sqrt((PhiL2-PhiRd)/(GM2/Rd) - 1./2)   # corresponding to fL2=1, T=0
        # need to find the maximum corresponding to fL2=0
        # this is given by the intersection between T_the(the), T_the_nofL2(the)
        theleft = themin
        theright = 1.
        fleft = T_the(theleft) - T_the_nofL2(theleft)
        while abs((theleft - theright)/theright) > tol:
            the = (theleft + theright)/2
            f = T_the(the) - T_the_nofL2(the)
            if f * fleft > 0:
                theleft = the
                fleft = f
            else:
                theright = the
        themax = (theleft + theright)/2   # this corresponds to fL2=0

        # --- numerical solution for f2(the, T, fL2)=0 under non-zero fL2

        # -- do not use exactly themin (corresponding to T = 0, bc. kap table breaks down)
        # -- define another themin based on Tfloor (kap table won't be a problem)
        themin = sqrt(3./8*c2*Tfloor + 1./4 * (PhiL2-PhiRd)/(GM2/Rd) - 1./8)
        theleft = themin  #
        f2left = f2_the_T_fL2(theleft, T_the(theleft), fL2_the(theleft))
        theright = themax / (1 + eps_small)
        # bisection again
        while abs((theleft-theright)/theright) > tol:
            the = (theleft + theright)/2
            f2 = f2_the_T_fL2(the, T_the(the), fL2_the(the))
            if f2 * f2left > 0:
                theleft = the
                f2left = f2
            else:
                theright = the
        # solution
        the = (theleft + theright)/2
        T = T_the(the)
        fL2 = fL2_the(the)
        rho = (1-fL2) * M1dot / (2*pi * alpha_ss * omgK * Rd**3) / the**3

        tau = rho*kap(rho, T)*Rd*the / 2
        Qrad = 2*pi*Rd**2*cst.arad*T**4*cst.c/tau
        U = cst.arad*T**4 + 1.5*rho*cst.kB*T/mug/cst.mp
        P = 1./3*cst.arad*T**4 + rho*cst.kB*T/mug/cst.mp
        tvis = 1./(alpha_ss * the**2 * omgK)
        Md = 2*pi*rho*Rd**3*the
        Qadv = 1.5*U/P * the**2 * GM2/Rd / tvis * Md
        QL2 = fL2 * M1dot * (PhiL2 - PhiRd - 0.5*GM2/Rd)
        QL2Qadv_over_Qrad[n1, n2] = (QL2+Qadv)/Qrad
        solu_rho[n1, n2] = rho
        solu_the[n1, n2] = the
        solu_T[n1, n2] = T
        solu_tau[n1, n2] = tau
        solu_fL2[n1, n2] = fL2
        Lacc_over_LEdd[n1, n2] = M1dot * kap(rho, T) / (4*pi * Rd * cst.c)


# below is for the contribution to L2 mass by the inner disk
# fL2_inner is typically not significant and can be ignored for practical purposes

# compute T(Rsph, rho)
def func_T_Rsph_rho(T, Rsph, rho):
    return cst.arad*T**4 + 1.5*rho*cst.kB*T/mug/cst.mp - GM2*rho/Rsph


# compute fL2 contribution from the inner disk outflow
def func_Rsph(lgRsph, Mdot):
    Rsph = 10**lgRsph
    omgK_Rsph = sqrt(GM2/Rsph**3)
    rho = Mdot/(2*pi*alpha_ss*omgK_Rsph*Rsph**3)
    # T = (1./cst.arad * rho * GM2/Rsph)**0.25
    # get the temperature near Rsph by solving aT^4 + 1.5*rho*kB*T/mug/mp - GM2*rho/Rsph = 0
    Tleft, Tright = 0., 1e7   # this range should be wide enough
    fleft = func_T_Rsph_rho(Tleft, Rsph, rho)
    while abs((Tright-Tleft)/Tright) > tol:
        Tmid = 0.5 * (Tleft + Tright)
        fmid = func_T_Rsph_rho(Tmid, Rsph, rho)
        if fmid * fleft < 0:
            Tright = Tmid
        else:
            Tleft = Tmid
            fleft = fmid
    T = 0.5 * (Tleft + Tright)
    return Rsph - Mdot*kap(rho, T)/(4*pi*cst.c)
    # return Rsph - Mdot * 0.34 / (4 * pi * cst.c)   # simple Thomson opacity prescription


def solve_Rsph(Mdot):
    lgRleft = 7.   # this should be small enough
    lgRright = 15.
    fleft = func_Rsph(lgRleft, Mdot)
    # fleft should be negative
    while abs(lgRleft-lgRright) > tol:
        lgR = (lgRleft + lgRright)/2
        fmid = func_Rsph(lgR, Mdot)
        if fleft * fmid < 0:
            lgRright = lgR
        else:
            lgRleft = lgR
            fleft = fmid
    lgR = (lgRleft + lgRright)/2
    return 10**lgR


for n1 in range(NM1dot):
    M1dot = 10**logM1dotgrid[n1]*cst.msun/cst.yr
    for n2 in range(Na):
        a = 10**logagrid[n2]*cst.rsun
        # auxiliary parameters
        Rd = Rd_over_a * a
        PhiL2 = PhiL2_dimless * cst.G*(M2/mu)/a
        fL2_out = solu_fL2[n1, n2]
        # solve for Rsph
        Rsph = min(Rd, solve_Rsph((1-fL2_out)*M1dot))
        solu_fL2_inner[n1, n2] = (1-fL2_out) * abs(PhiL2)*Rsph/GM2

fL2_tot = solu_fL2 + solu_fL2_inner



# write the data into a pickle file
info = ['info', 'logM1dotgrid', 'logagrid', 'logfL2out', 'logfL2in', 'logtheta(H/R)',
        'logT(outer_disk_temperature)', 'logLaccLEdd(luminosity)', 'logQQoQ(radiative_efficiency)',
        'logtau(vertical_optical_depth)', 'logrho(outer_disk_density)']
data_all = [info, logM1dotgrid, logagrid, np.log10(solu_fL2), np.log10(solu_fL2_inner),
            np.log10(solu_the), np.log10(solu_T), np.log10(Lacc_over_LEdd),
            np.log10(QL2Qadv_over_Qrad), np.log10(solu_tau), np.log10(solu_rho)]
with open(savedir+savename+'.pkl', 'wb') as f:
    pickle.dump(data_all, f)
    print('data saved at:' + savedir+savename+'.pkl')


exit()

# write the data into txt files [old formats]
name_list = ['fL2out', 'fL2in', 'T', 'LaccLEdd', 'QQoQ']
data_list = [np.log10(solu_fL2), np.log10(solu_fL2_inner), np.log10(solu_T),
             np.log10(Lacc_over_LEdd), np.log10(QL2Qadv_over_Qrad)]
for n in range(len(name_list)):
    name = name_list[n] + 'M%.2f_q%.3f' % (M2_in_Msun, q)
    savedata = data_list[n]
    with open(savedir+name+'.txt', 'w') as f:
        f.write('logM1dotmin\tlogM1dotmax\tNlogM1dot\t%.5f\t%.5f\t%.5f\n'
                % (logM1dotmin, logM1dotmax, NM1dot))
        f.write('logamin\tlogamax\tNa\t%.5f\t%.5f\t%.5f\n'
                % (logamin, logamax, Na))
        for i in range(NM1dot):
            f.write('\n')
            for j in range(Na):
                if j == 0:
                    f.write('%.8f' % savedata[i, j])
                else:
                    f.write('\t%.8f' % savedata[i, j])

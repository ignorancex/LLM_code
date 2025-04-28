import numpy as np
from properties import cp_steel, beta_steel, my_steel, Lam_steel, Rho_steel
from utils import find_nearest_idx


def h_rad(Ti, Text):
    eps = 1.0  # Value for stainless steel(0.12) (carbon steel ~0.8) Close to 1 for the ladle is likely
    sigma = 5.670373e-8  # W·m−2·K−4 Stefan–Boltzmann constant
    return sigma*eps*(Ti**2 + Text**2)*(Ti + Text)


def h_steel_inner(Ts, Tw, H, dy, utau):  # NB dy / STJ
    tunefac = 1.0
    lam = Lam_steel(Ts)
    tmp = lam/H
    hflow = h_natural(Ts, Tw, H)
    hstirr = h_stirr(Ts, utau)
    hinner = (np.sqrt(hflow) + np.sqrt(hstirr))**2
    return min(tunefac*(tmp + hinner),5000.0) # Too high value only create instabilities


def h_slag_inner(lam, dy):  # NB dy / STJ
    tmp = 2*lam/dy
    hslagflow = h_slag_flow()
    return tmp*hslagflow/(tmp + hslagflow)


def h_gas_inner(Ti, Text, lam, dy):  # NB dy / STJ
    tmp = 2*lam/dy
    hrad = h_rad(Ti, Text)
    return tmp*hrad/(tmp + hrad)


def h_steel_flow(Ts, Tw, H):
    return 10000000.0  # model above not activated


def h_slag_flow():
    return 1000.0  # Need a model


def h_slag_metal():
#     return 000.0 # Need a model
    return 5000.0  # Need a model


def h_slag_lid(Ts, Tlid):
    return 0.0
#     return h_rad(Ts, Tlid) # Assume radiation


def h_stirr(T, utau):
    rho = Rho_steel(T)
    cp  = cp_steel(T)
    my  = my_steel(T)
    Pr = prantl(T)
    s = 1e-3
    spluss = s*utau*rho/my
    Tpluss = ((5.95+13.6*Pr**(0.596))+(0.117+0.235*Pr**0.893)*spluss)/(1.0+(0.011+0.0939*Pr**0.676)*spluss + (0.00005 + 0.0000683*Pr**0.62)*spluss*spluss)
#     print("tpluss", Tpluss)
#     print("Pr", Pr)
#     print("Cp", cp_steel(T))
#     print("my", my_steel(T))
#     print("Lam", Lam_steel(T))
    retval = rho*cp*utau/Tpluss
    if retval < 0.0:
        print('!! Warning NEG retval (utau,Tpluss,Pr,spluss,rho,cp,my):', utau,Tpluss,Pr,spluss,rho,cp,my)
    return retval


def h_natural(Ts, Tw, H):
    return nusselt(Ts, Tw, H)*Lam_steel(Ts)/max(H, 0.1)


def prantl(T):
    return my_steel(T)*cp_steel(T)/Lam_steel(T)


def grashof(Ts, Tw, H):
    x_c = 0.25*H**3
    vis = Lam_steel(Ts)/Rho_steel(Ts)
    return 9.81*beta_steel(Ts)*x_c*abs(Tw-Ts)/vis**2


def nusselt(Ts, Tw, H):
    pr = prantl(Ts)
    gr = grashof(Ts, Tw, H)
    ra = gr*pr
    if ra > 1.0e15 :
        print('ra out of bounds: (Ts,Tw,H) = ',Ts,Tw,H)
    ra0 = 2.0e10
    A = 0.75*(2.0*pr/(5.0*(1 + 2.0*pr**0.5 + 2.0*pr)))**0.25*(gr*pr)**0.25
    B = (gr*pr/(300.0*(1.0 + (0.5/pr)**(9.0/16.0))**(16.0/9.0)))**(1.0/3.0)
    return A + (B-A)/(1+np.exp(-ra/ra0-1.0))

def q_electric3(time, tt, p, dt):
    idx1 = find_nearest_idx(tt, time-dt)
    idx2 = find_nearest_idx(tt, time)
    if (idx1 == idx2):
        if (tt[idx1] > time):
            idx1 -= 1
        else:
            idx2 +=1
    if (idx2 >= len(tt)):
        return 0.0
    der = (p[idx2]-p[idx1])/(tt[idx2]-tt[idx1])
    if der > 40: # This is probably an error in the data, we need to restrict the power input
        der = 40.0
    pw = max(der,0.0)
    return pw*3600.0*1000.0
    for n in range(2,len(tt)):
        if (time <= tt[n] and time >= tt[n-1]):
            derivative = (p[n] - p[n-1])/(tt[n] - tt[n-1])
            pw = derivative*dt
            return pw*3600.0*1000.0
    if (time >= tt[-1]):
        pw = 0.0
    return pw*3600.0*1000.0

def q_electric2(time, dt, power_data):
    # We assume each measurement is in 1 second intervals, and that time is the final time., the time is then the index.
    time0 = time-dt
    idx = int(np.floor(time))
    if (idx < power_data.size):
        retval = 3600.0*1000.0*(power_data[idx] - power_data[int(np.floor(time0))])/dt
    #   print("time, retval, idx = ", time, retval, idx)
        retval = max((0.0, retval))
    else:
        retval = 0.0
    return retval


def q_electric(time, dt, power_data):
    # We assume each measurement is in 1 second intervals, and that time is the final time., the time is then the index.
    time0 = time-dt
    retval = 3600.0*1000.0*(sum(power_data[int(np.floor(time0)):int(np.floor(time))]))/dt
    retval = max((0.0, retval))
    return retval
#    if (time <=110 or time >=3000):
#        retval = 0.0
#    else:
#        # As a ballpark, 6 kwh/s
#        retval = 3600*1000*6.0
#    return retval


def q_radlid(T1, T2, R, dx, x1, x2, eps1, eps2):
    sigma = 2.670373e-8
    retval = (T2**4.0 - T1**4.0)/(1.0/eps2 + 1.0/eps1 - 1.0)*sigma*((R*(1.0-1.0/np.sqrt(2.0)))*(x2-x1))/(np.pi*((R*(1.0-1.0/np.sqrt(2.0)))**2.0+(x2-x1)**2.0)**2.0)*R*2.0*np.pi*dx*R*R/2.0*2.0*np.pi
    return retval


def q_rad(T1, T2, R, dx, x1, x2, lam, dy, eps1, eps2):
    # T1 = slag-metal, T2=wall
    sigma = 2.670373e-8
    dtheta = 2.0*np.pi
    retval = R*dtheta*dx*(T2**4.0 - T1**4.0)/(1.0/eps2 + 1.0/eps1 - 1.0)*sigma*((R*(1.0-1.0/np.sqrt(2.0)))*(x2-x1))/(np.pi*((R*(1.0-1.0/np.sqrt(2.0)))**2.0+(x2-x1)**2.0)**2.0)*R*R/2.0*dtheta
    return retval


def lidT(Tslag, xslag, xcell, Twall, L, R):
    ewall = 0.9
    eceil = 0.2
    xceil = L
    tmp = R*(1.0 - 1.0/np.sqrt(2.0))
    tmp1 = R/2.0*1.0/(np.pi*(xceil-xslag)**2)*Tslag**4/(1.0/ewall + 1.0/eceil - 1.0)
    tmp2 = R/2.0*1.0/(np.pi*(xceil-xslag)**2)*1.0/(1.0/ewall + 1.0/eceil - 1.0)
    for i in range(len(Twall)):
        tmp1 += Twall[i]**4/(1.0/ewall + 1.0/eceil - 1.0)*(tmp*max(0, xceil - xcell[i]))/(np.pi*(tmp**2 + max(0, xceil - xcell[i])**2)**2)
        tmp2 += 1.0/(1.0/ewall + 1.0/eceil - 1.0)*(tmp*max(0, xceil - xcell[i]))/(np.pi*(tmp**2 + max(0, xceil - xcell[i])**2)**2)
    return (tmp1/tmp2)**0.25


def enthalpy(Cps, Cpl, T):
    T1 = 1400.0+273
    T2 = 1500.0+273
    hfus = 1386.66e3  # J/kg #https://webbook.nist.gov/cgi/cbook.cgi?ID=C1305788&Mask=2 => 77.76 kJ/mol => 1386.66 kJ/kg
    if (T < T1):
        h = Cps*T
    elif (T < T2):
        h = Cps*T1 + hfus*(T-T1)/(T2-T1)
    else:
        h = Cps*T1 + hfus + Cpl*(T-T2)

    return h

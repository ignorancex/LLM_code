import numpy as np
from scipy.integrate import odeint

def compute_eps_SA(m,M_per,a_div_Q,e_per):
    return (M_per/np.sqrt(m*(m+M_per)))*pow(a_div_Q,3.0/2.0)*pow(1.0 + e_per,-3.0/2.0)

def compute_eps_oct(m1,m2,m,a_div_Q,e_per):
    return a_div_Q*np.fabs(m1-m2)/((1.0+e_per)*m)

def compute_eps_hex(m1,m2,m,a_div_Q,e_per):
    return a_div_Q**2*(m1**2 - m1*m2 + m2**2)/((1.0+e_per)**2*m**2)

def compute_eps_hex_cross(m1,m2,m,a_div_Q,e_per):
    ### Values should correspond to the `companion' binary!
    return a_div_Q**2*(m1*m2)/((1.0+e_per)**2*m**2)

def integrated_quad_function_f(n,e_per):
    L = np.arccos(-1.0/e_per)
    npi = n*np.pi
    return (npi - 3.0*L)*(npi - 2.0*L)*(npi - L)*(npi + L)*(npi + 2.0*L)*(npi + 3.0*L)

def integrated_quad_function(eps_SA,a,Q,e_per,m1,m2,ex,ey,ez,jx,jy,jz):
    
    ### First and second order in eps_SA 
    Delta_e = 0.0
    epsilon = 1.0e-2
    if e_per <= 1.0 + epsilon: ### parabolic limit   

        ArcCos = np.arccos
        Sqrt = np.sqrt
        Pi = np.pi

        Delta_e = eps_SA*(15*ez*(ey*jx - ex*jy)*np.pi)/(2.*np.sqrt(ex**2 + ey**2 + ez**2)) \
            + (3*Pi*(3*ex**2*(6*jy**2 - jz**2)*Pi + 2*ex*ez*jz*(-25*jy + 6*jx*Pi) + ez**2*(-25*jx*jy - 75*ex**2*Pi + 18*jx**2*Pi + 18*jy**2*Pi) + ey**2*(25*jx*jy + 18*jx**2*Pi - 3*(25*ez**2 + jz**2)*Pi) + ey*(2*ez*jz*(25*jx + 6*jy*Pi) + ex*(-25*jy**2 + 25*jz**2 - 36*jx*jy*Pi)))*eps_SA**2)/(8.*Sqrt(ex**2 + ey**2 + ez**2))
    else:
        f1 = integrated_quad_function_f(1.0,e_per)
        f2 = integrated_quad_function_f(2.0,e_per)
        f3 = integrated_quad_function_f(3.0,e_per)

        exn = ex + (eps_SA*(np.sqrt(1 - e_per**(-2))*(ez*(jy - 10*e_per**2*jy) + (-5 + 2*e_per**2)*ey*jz) - 3*e_per*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per)))/(4.*e_per) - (3*np.sqrt(1 - e_per**(-2))*(ez*jx - 5*ex*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        eyn = ey + (eps_SA*(np.sqrt(1 - e_per**(-2))*(ez*(jx + 8*e_per**2*jx) + (-5 + 8*e_per**2)*ex*jz) + 3*e_per*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per)))/(4.*e_per) + (3*np.sqrt(1 - e_per**(-2))*(ez*jy - 5*ey*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        ezn = ez + (np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*ey*jx + (2 - 5*e_per**2)*ex*jy)*eps_SA + 3*e_per*(ey*jx - ex*jy)*eps_SA*np.arccos(-(1/e_per)))/(2.*e_per) - (12*np.sqrt(1 - e_per**(-2))*(ex*jx - ey*jy)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        jxn = jx - ((5*ey*ez - jy*jz)*eps_SA*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per))))/(4.*e_per) + (3*np.sqrt(1 - e_per**(-2))*(5*ex*ez - jx*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        jyn = jy + ((5*ex*ez - jx*jz)*eps_SA*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per))))/(4.*e_per) - (3*np.sqrt(1 - e_per**(-2))*(5*ey*ez - jy*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        jzn = jz - (np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*(5*ex*ey - jx*jy)*eps_SA)/(2.*e_per) - (3*np.sqrt(1 - e_per**(-2))*(5*ex**2 - 5*ey**2 - jx**2 + jy**2)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)

        Delta_e = (5*eps_SA*(np.sqrt(1 - e_per**(-2))*((1 + 2*e_per**2)*eyn*ezn*jxn + (1 - 4*e_per**2)*exn*ezn*jyn + 2*(-1 + e_per**2)*exn*eyn*jzn) + 3*e_per*ezn*(eyn*jxn - exn*jyn)*np.arccos(-1.0/e_per)))/(2.*e_per*np.sqrt(exn**2 + eyn**2 + ezn**2))

        Delta_e += (3*eps_SA**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*((ey*(-18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-27*(2*jx*((7 - 4*e_per**2)*ey*jy + (-2 + 5*e_per**2)*ez*jz) + ex*(25*(-2 + e_per**2)*ey**2 - 5*ez**2 + 4*jy**2 - e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**4 + 3*(2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 2*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f2**2 - 18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-27*(5*(-5 + 2*e_per**2)*ex**3 + 4*jx*((-1 + 2*e_per**2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*e_per**2)*ey**2 + 5*(-1 + 2*e_per**2)*ez**2 + 9*jx**2 - 10*e_per**2*jx**2 - 5*jy**2 + 2*e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**5 - 3*(5*(-25 + 4*e_per**2)*ex**3 - 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(-5*(-25 + 4*e_per**2)*ey**2 - 25*(1 + 4*e_per**2)*ez**2 + 45*jx**2 + 76*e_per**2*jx**2 - 25*jy**2 + 4*e_per**2*jy**2 + 125*jz**2 - 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 2*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(9*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1**2*f2**2*f3 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(2*jx*((7 - 4*e_per**2)*ey*jy + (-2 + 5*e_per**2)*ez*jz) + ex*(25*(-2 + e_per**2)*ey**2 - 5*ez**2 + 4*jy**2 - e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**4 + 2*(2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f3**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 2*e_per**2)*ex**3 + 4*jx*((-1 + 2*e_per**2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*e_per**2)*ey**2 + 5*(-1 + 2*e_per**2)*ez**2 + 9*jx**2 - 10*e_per**2*jx**2 - 5*jy**2 + 2*e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**5 + 2*(-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1**2*f2*f3**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((2*jx*((-7 + 4*e_per**2)*ey*jy + (2 - 5*e_per**2)*ez*jz) + ex*(-25*(-2 + e_per**2)*ey**2 + 5*ez**2 - 4*jy**2 + e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**4 + (2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f2**2*f3**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 2*e_per**2)*ex**3 + 4*jx*(ey*(jy - 2*e_per**2*jy) + ez*jz) + ex*(5*(-5 + 2*e_per**2)*ey**2 + (5 - 10*e_per**2)*ez**2 - 9*jx**2 + 10*e_per**2*jx**2 + 5*jy**2 - 2*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**5 + (-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1*f2**2*f3**2))/np.pi + (ex*(18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-27*(25*(-2 + e_per**2)*ex**2*ey + 2*(7 - 3*e_per**2)*ex*jx*jy - 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(5*(-1 + e_per**2)*ez**2 + (4 - 3*e_per**2)*jx**2 + 5*(5 - 3*e_per**2)*jz**2))*np.pi**4 + 3*(25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 2*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f2**2 - 18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-27*(5*(-5 + 3*e_per**2)*ex**2*ey - 5*(-5 + 3*e_per**2)*ey**3 + 4*(1 + e_per**2)*ex*jx*jy - 4*(-1 + e_per**2)*ez*jy*jz + ey*(5*(1 + e_per**2)*ez**2 + (5 - 3*e_per**2)*jx**2 - 9*jy**2 - e_per**2*jy**2 - 25*jz**2 + 15*e_per**2*jz**2))*np.pi**5 + 3*(5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 2*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(9*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1**2*f2**2*f3 + 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(25*(-2 + e_per**2)*ex**2*ey + 2*(7 - 3*e_per**2)*ex*jx*jy - 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(5*(-1 + e_per**2)*ez**2 + (4 - 3*e_per**2)*jx**2 + 5*(5 - 3*e_per**2)*jz**2))*np.pi**4 + 2*(25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f3**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 3*e_per**2)*ex**2*ey - 5*(-5 + 3*e_per**2)*ey**3 + 4*(1 + e_per**2)*ex*jx*jy - 4*(-1 + e_per**2)*ez*jy*jz + ey*(5*(1 + e_per**2)*ez**2 + (5 - 3*e_per**2)*jx**2 - 9*jy**2 - e_per**2*jy**2 - 25*jz**2 + 15*e_per**2*jz**2))*np.pi**5 + 2*(5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1**2*f2*f3**2 + 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((-25*(-2 + e_per**2)*ex**2*ey + 2*(-7 + 3*e_per**2)*ex*jx*jy + 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(-5*(-1 + e_per**2)*ez**2 + (-4 + 3*e_per**2)*jx**2 + 5*(-5 + 3*e_per**2)*jz**2))*np.pi**4 + (25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f2**2*f3**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 3*e_per**2)*ex**2*ey + 5*(-5 + 3*e_per**2)*ey**3 - 4*(1 + e_per**2)*ex*jx*jy + 4*(-1 + e_per**2)*ez*jy*jz + ey*(-5*(1 + e_per**2)*ez**2 + (-5 + 3*e_per**2)*jx**2 + 9*jy**2 + e_per**2*jy**2 + 25*jz**2 - 15*e_per**2*jz**2))*np.pi**5 + (5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1*f2**2*f3**2))/np.pi - 12*np.sqrt(1 - e_per**(-2))*e_per**2*ez*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*f3*(f2*f3 + f1*(f2 + f3)) - e_per*(-5 + 2*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.pi**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*f1*f2*f3*(f2*f3 + f1*(9*f2 + 4*f3)) - (-5 + 2*e_per**2)*np.pi**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*f3*(f2*f3 + f1*(9*f2 + 4*f3)) + 36*np.sqrt(1 - e_per**(-2))*(-5 + 8*e_per**2)*(4*(ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy - ey*jx*jz + 7*ex*jy*jz))*np.arccos(-(1/e_per))**9*(f2**2*f3**2 + f1**2*(f2**2 + f3**2)) - np.sqrt(1 - e_per**(-2))*(1320*(-(ey*jx*jz) + ex*jy*jz) + 16*e_per**4*(55*ex*ey*ez + 55*ez*jx*jy + 21*ey*jx*jz + 45*ex*jy*jz) - e_per**2*(1225*ex*ey*ez + 1225*ez*jx*jy - 1173*ey*jx*jz + 2643*ex*jy*jz))*np.pi**2*np.arccos(-(1/e_per))**7*(f2**2*f3**2 + f1**2*(9*f2**2 + 4*f3**2)) + 2*np.sqrt(1 - e_per**(-2))*(e_per**4*(85*ex*ey*ez + 85*ez*jx*jy + 111*ey*jx*jz - 9*ex*jy*jz) + 240*(-(ey*jx*jz) + ex*jy*jz) - e_per**2*(175*ex*ey*ez + 175*ez*jx*jy + 141*ey*jx*jz + 69*ex*jy*jz))*np.pi**4*np.arccos(-(1/e_per))**5*(f2**2*f3**2 + f1**2*(81*f2**2 + 16*f3**2)) + np.arccos(-(1/e_per))**3*(e_per*(-5 + 8*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.arccos(-1.0/e_per)*f1*f2*f3*(f2*f3 + f1*(f2 + f3)) - np.sqrt(1 - e_per**(-2))*(-5 + 2*e_per**2)*(24*(-(ey*jx) + ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy + 15*ey*jx*jz - 9*ex*jy*jz))*np.pi**6*(f2**2*f3**2 + f1**2*(729*f2**2 + 64*f3**2))))))/(2.*e_per**4*np.sqrt(ex**2 + ey**2 + ez**2)*f1**2*f2**2*f3**2)
             

    return Delta_e
    
def integrated_oct_function(eps_SA,a,Q,e_per,m1,m2,ex,ey,ez,jx,jy,jz):
    M1 = m1+m2

    eps_oct = (np.fabs(m1-m2)/M1)*(a/Q)*(1.0 + e_per)

    Delta_e = 0.0
    epsilon = 1.0e-2
    if e_per <= 1.0 + epsilon: ### parabolic limit; first AND second order in eps_SA 
        ArcCos = np.arccos
        Sqrt = np.sqrt
        Pi = np.pi
        
        Delta_e += eps_oct*(-15*Pi*(3*ex**2*ez*jz*(-3759*jy + 1100*jx*Pi) + ey**3*(-707*jx**2 + 2037*jz**2 + 12*jx*jy*Pi) + 3*ez*jz*(7*(-20 + 160*ez**2 + 51*jx**2)*jy + 301*jy**3 - 4*jx*(-12 + 96*ez**2 + 35*jx**2)*Pi - 140*jx*jy**2*Pi) - ex**3*(847*jx*jy + 60*(146*ez**2 - 33*jy**2 + 3*jz**2)*Pi) + ex*(-49*jx**3*jy + 7*jx*jy*(20 - 676*ez**2 + jy**2 - 62*jz**2) + 12*jx**2*(276*ez**2 + 5*(-3*jy**2 + jz**2))*Pi + 12*(320*ez**4 - 15*jy**4 - 4*jz**2 + 3*jy**2*(4 + 5*jz**2) + 4*ez**2*(-10 + 65*jy**2 + 8*jz**2))*Pi) + ey*(49*jx**4 + 21*(20 - 43*jy**2)*jz**2 + 7*jx**2*(-20 + 121*ex**2 + 682*ez**2 - jy**2 - 91*jz**2) + 42*ez**2*(73*jy**2 - 80*jz**2) + ex**2*(-3066*jy**2 + 3549*jz**2) + 180*jx**3*jy*Pi + 2712*ex*ez*jy*jz*Pi - 6*jx*(-1281*ex*ez*jz + 662*ex**2*jy*Pi + 2*jy*(12 - 16*ez**2 - 15*jy**2 + 10*jz**2)*Pi)) + ey**2*(21*ez*jz*(-243*jy + 28*jx*Pi) + ex*(3773*jx*jy + 1992*jx**2*Pi - 12*(730*ez**2 + jy**2 + 15*jz**2)*Pi)))*eps_SA**2)/(512.*Sqrt(ex**2 + ey**2 + ez**2))
        Delta_e += eps_oct**2*(-225*Pi*(36*ey**6*Pi + 2*ey**3*ez*jz*(-4067*jx + 992*jy*Pi) + 2*ey*ez*jz*(5517*jx**3 + jx*(-2408 + 19264*ez**2 + 3945*jy**2) - 240*jx**2*jy*Pi + 16*jy*(-34 + 272*ez**2 + 55*jy**2)*Pi) + 4*ez**2*(519*jx**3*jy + 3*jx*jy*(-28 + 224*ez**2 + 191*jy**2) + 75*jx**4*Pi + 10*jx**2*(-8 + 64*ez**2 - 17*jy**2)*Pi + (16 + 1024*ez**4 + 176*jy**2 - 245*jy**4 - 128*ez**2*(2 + 11*jy**2))*Pi) - ey**4*(7679*jx*jy + 240*jx**2*Pi + 4*(-24 + 183*ez**2 - 10*jy**2 + 18*jz**2)*Pi) + ex**4*(7343*jx*jy + 4*(9*ey**2 + 5329*ez**2 + 54*(-19*jy**2 + jz**2))*Pi) + ey**2*(-1795*jx**3*jy - jx*jy*(-1008 + 3374*ez**2 + 55*jy**2 + 3802*jz**2) + 300*jx**4*Pi - 40*jx**2*(8 + 2*ez**2 - 30*jy**2 - 39*jz**2)*Pi + 4*(16 + 832*ez**4 + 125*jy**4 - 24*jz**2 - 10*jy**2*(12 + 7*jz**2) + 4*ez**2*(-58 + 123*jy**2 + 48*jz**2))*Pi) + ex**2*(-281*jx**3*jy - jx*(2237*jy**3 + 14350*ey*ez*jz + 2*jy*(336 + 6048*ey**2 - 4207*ez**2 - 1901*jz**2)) - 8*jx**2*(602*ey**2 + 1342*ez**2 - 85*jy**2 - 105*jz**2)*Pi + 8*(9*ey**4 - 2336*ez**4 - 108*jy**2 + 135*jy**4 - 608*ey*ez*jy*jz + 12*jz**2 - 65*jy**2*jz**2 + ey**2*(12 + 2573*ez**2 + 64*jy**2 + 18*jz**2) - 2*ez**2*(-146 + 871*jy**2 + 48*jz**2))*Pi) + ex**3*(10*ez*jz*(2191*jy - 1216*jx*Pi) - 7*ey*(1049*jx**2 - 2045*jy**2 + 612*jz**2 - 1280*jx*jy*Pi)) + ex*(2*ey**2*ez*jz*(9359*jy - 2656*jx*Pi) - ey**3*(2219*jx**2 - 7679*jy**2 + 2772*jz**2 + 192*jx*jy*Pi) + 2*ez*jz*((3752 - 30016*ez**2 - 8817*jx**2)*jy - 7029*jy**3 + 16*jx*(-38 + 304*ez**2 + 105*jx**2)*Pi + 2800*jx*jy**2*Pi) + ey*(281*jx**4 + 55*jy**4 + 2352*(-1 + 8*ez**2)*jz**2 - 2*jy**2*(504 + 2219*ez**2 - 3839*jz**2) + jx**2*(672 - 11690*ez**2 + 4032*jy**2 + 722*jz**2) - 480*jx**3*jy*Pi - 32*jx*jy*(-22 - 164*ez**2 + 40*jy**2 + 15*jz**2)*Pi)))*eps_SA**2)/(8192.*Sqrt(ex**2 + ey**2 + ez**2))
    else: ### first order in eps_SA only
        Delta_e = (5*a*eps_SA*np.fabs(m1 - m2)*(np.sqrt(1 - e_per**(-2))*(ey**2*ez*(14 - 31*e_per**2 + 8*e_per**4)*jy + ez*jy*(6*jx**2 - 2*jy**2 + 8*e_per**4*(-1 + 8*ez**2 + 2*jx**2 + jy**2) + e_per**2*(-4 + 32*ez**2 - 7*jx**2 + 9*jy**2)) + ey**3*(-14 + 31*e_per**2 - 8*e_per**4)*jz - ey*(2*(jx**2 - jy**2) + 8*e_per**4*(-1 + 8*ez**2 + 4*jx**2 + jy**2) + e_per**2*(-4 + 32*ez**2 + 11*jx**2 + 9*jy**2))*jz - ex**2*(ez*(14 + 45*e_per**2 + 160*e_per**4)*jy - 3*ey*(14 - 27*e_per**2 + 16*e_per**4)*jz) + 2*ex*(-2 + 9*e_per**2 + 8*e_per**4)*jx*(7*ey*ez + jy*jz)) + 3*e_per**3*(32*ez**3*jy + ez*jy*(-4 - 3*ey**2 + 5*jx**2 + 5*jy**2) - 32*ey*ez**2*jz + ey*(4 + 3*ey**2 - 15*jx**2 - 5*jy**2)*jz + ex**2*(-73*ez*jy + 3*ey*jz) + 10*ex*jx*(7*ey*ez + jy*jz))*np.arccos(-(1/e_per))))/(32.*np.sqrt(ex**2 + ey**2 + ez**2)*e_per**2*(1 + e_per)*M1*Q)
    
    return Delta_e

def integrated_hex_function(eps_SA1,a1,Q,eper,m1,m2,e1x,e1y,e1z,j1x,j1y,j1z):
    M1 = m1+m2
    ### first order in eps_SA only
    
    return (7*a1**2*(m1**2 - m1*m2 + m2**2)*eps_SA1*(np.sqrt(1 - eper**(-2))*(e1x**3*(e1z*(36 + 24*eper**2 - 1751*eper**4 - 1024*eper**6)*j1y + 6*e1y*(-24 + 70*eper**2 - 63*eper**4 + 32*eper**6)*j1z) + e1x**2*j1x*(e1y*e1z*(108 - 444*eper**2 + 2129*eper**4 + 832*eper**6) + 2*(12 - 36*eper**2 + 421*eper**4 + 128*eper**6)*j1y*j1z) - e1y*j1x*(e1y**2*e1z*(36 - 240*eper**2 - 563*eper**4 + 32*eper**6) + e1z*(eper**4*(-166 + 1660*e1z**2 + 421*j1x**2 - 101*j1y**2) + 32*eper**6*(-1 + 10*e1z**2 + 4*j1x**2 - 5*j1y**2) + 12*(j1x**2 - 3*j1y**2) + 12*eper**2*(-1 + 10*e1z**2 - 3*j1x**2 + 16*j1y**2)) + 2*e1y*(12 - 36*eper**2 + 421*eper**4 + 128*eper**6)*j1y*j1z) + e1x*(e1y**2*e1z*(-108 + 204*eper**2 - 1049*eper**4 + 128*eper**6)*j1y + e1z*j1y*(12*(-3*j1x**2 + j1y**2) + 128*eper**6*(-1 + 10*e1z**2 + 4*j1x**2 + j1y**2) - 12*eper**2*(-1 + 10*e1z**2 - 2*j1x**2 + 3*j1y**2) + eper**4*(-274 + 2740*e1z**2 + 655*j1x**2 + 421*j1y**2)) - 6*e1y**3*(-24 + 74*eper**2 - 81*eper**4 + 16*eper**6)*j1z - 4*e1y*(-6*j1x**2 + 6*j1y**2 + eper**4*(-27 + 270*e1z**2 + 269*j1x**2 - 80*j1y**2) + 8*eper**6*(-3 + 30*e1z**2 + 20*j1x**2 + j1y**2) - 3*eper**2*(-2 + 20*e1z**2 + j1x**2 + 13*j1y**2))*j1z)) + 15*eper**3*(e1y**3*(e1z*(46 + 3*eper**2)*j1x + 6*e1x*eper**2*j1z) - e1y**2*j1y*(e1x*e1z*(46 + 9*eper**2) + 14*(2 + 3*eper**2)*j1x*j1z) + e1x*j1y*(20*e1z**3*(4 + 9*eper**2) + e1z*(-(e1x**2*(46 + 135*eper**2)) + 2*(-4 + 7*j1x**2 + 7*j1y**2) + 3*eper**2*(-6 + 21*j1x**2 + 7*j1y**2)) + 14*e1x*(2 + 3*eper**2)*j1x*j1z) + e1y*(-20*e1z**3*(4 + 3*eper**2)*j1x + e1z*j1x*(e1x**2*(46 + 129*eper**2) - 2*(-4 + 7*j1x**2 + 7*j1y**2) + eper**2*(6 - 21*j1x**2 + 21*j1y**2)) - 120*e1x*e1z**2*eper**2*j1z + 2*e1x*(3*eper**2*(2 + e1x**2 - 14*j1x**2) + 14*(-j1x**2 + j1y**2))*j1z))*np.arccos(-(1/eper))))/(128.*np.sqrt(e1x**2 + e1y**2 + e1z**2)*eper**3*(1 + eper)**2*M1**2*Q**2)

def integrated_hex_function_cross(eps_SA1,a2,Q,eper,m3,m4,e1x,e1y,e1z,j1x,j1y,j1z,e2x,e2y,e2z,j2x,j2y,j2z):
    M2 = m3+m4
    ### first order in eps_SA only
    
    return (5*a2**2*m3*m4*eps_SA1*(np.sqrt(1 - eper**(-2))*(12*e1z**2*eper**2*(-5*e2y*e2z*(2 + 13*eper**2)*j1x + 5*e2x*e2z*(-2 + 31*eper**2 + 16*eper**4)*j1y + (2 - 31*eper**2 - 16*eper**4)*j1y*j2x*j2z + (2 + 13*eper**2)*j1x*j2y*j2z) + e1x*e1z*(30*e2x*e2y*(4 - 12*eper**2 + 23*eper**4)*j1x + 12*eper**2*j1y - 72*e2z**2*eper**2*j1y - 274*eper**4*j1y + 2524*e2z**2*eper**4*j1y - 128*eper**6*j1y + 1088*e2z**2*eper**6*j1y + e2y**2*(-60 + 108*eper**2 - 461*eper**4 + 128*eper**6)*j1y - e2x**2*(-60 + 72*eper**2 + 1241*eper**4 + 832*eper**6)*j1y + 60*e2y*e2z*eper**2*(2 + 13*eper**2)*j1z - 12*j1y*j2x**2 + 577*eper**4*j1y*j2x**2 + 320*eper**6*j1y*j2x**2 - 24*j1x*j2x*j2y + 72*eper**2*j1x*j2x*j2y - 138*eper**4*j1x*j2x*j2y + 12*j1y*j2y**2 - 36*eper**2*j1y*j2y**2 + 421*eper**4*j1y*j2y**2 + 128*eper**6*j1y*j2y**2 - 24*eper**2*j1z*j2y*j2z - 156*eper**4*j1z*j2y*j2z - 176*eper**4*j1y*j2z**2 - 64*eper**6*j1y*j2z**2) + 6*e1y**2*(5*e2y*(2*e2z*eper**2*(2 + 13*eper**2)*j1x + e2x*(4 - 12*eper**2 + 23*eper**4)*j1z) - j2y*((4 - 12*eper**2 + 23*eper**4)*j1z*j2x + 2*eper**2*(2 + 13*eper**2)*j1x*j2z)) - 6*e1x**2*(5*e2x*(2*e2z*eper**2*(-2 + 31*eper**2 + 16*eper**4)*j1y + e2y*(4 - 12*eper**2 + 23*eper**4)*j1z) - j2x*((4 - 12*eper**2 + 23*eper**4)*j1z*j2y + 2*eper**2*(-2 + 31*eper**2 + 16*eper**4)*j1y*j2z)) + e1y*(12*e1x*(5*e2x*e2z*eper**2*(-2 + 31*eper**2 + 16*eper**4)*j1x - 5*e2y*e2z*eper**2*(2 + 13*eper**2)*j1y - 2*eper**2*j1z + 12*e2z**2*eper**2*j1z + 9*eper**4*j1z - 54*e2z**2*eper**4*j1z + 8*eper**6*j1z - 48*e2z**2*eper**6*j1z + e2y**2*(10 - 33*eper**2 + 16*eper**4 - 8*eper**6)*j1z + e2x**2*(-10 + 27*eper**2 + 11*eper**4 + 32*eper**6)*j1z + 2*j1z*j2x**2 - 3*eper**2*j1z*j2x**2 - 13*eper**4*j1z*j2x**2 - 16*eper**6*j1z*j2x**2 - 2*j1z*j2y**2 + 9*eper**2*j1z*j2y**2 - 14*eper**4*j1z*j2y**2 - 8*eper**6*j1z*j2y**2 + 2*eper**2*j1x*j2x*j2z - 31*eper**4*j1x*j2x*j2z - 16*eper**6*j1x*j2x*j2z + 2*eper**2*j1y*j2y*j2z + 13*eper**4*j1y*j2y*j2z) + e1z*(12*eper**2*j1x - 72*e2z**2*eper**2*j1x + 166*eper**4*j1x - 1876*e2z**2*eper**4*j1x + 32*eper**6*j1x - 512*e2z**2*eper**6*j1x + e2y**2*(-60 + 288*eper**2 + 269*eper**4 - 32*eper**6)*j1x + e2x**2*(60 - 252*eper**2 + 1109*eper**4 + 448*eper**6)*j1x - 30*e2x*(e2y*(4 - 12*eper**2 + 23*eper**4)*j1y + 2*e2z*eper**2*(-2 + 31*eper**2 + 16*eper**4)*j1z) - 12*j1x*j2x**2 + 36*eper**2*j1x*j2x**2 - 421*eper**4*j1x*j2x**2 - 128*eper**6*j1x*j2x**2 + 24*j1y*j2x*j2y - 72*eper**2*j1y*j2x*j2y + 138*eper**4*j1y*j2x*j2y + 12*j1x*j2y**2 - 72*eper**2*j1x*j2y**2 - 253*eper**4*j1x*j2y**2 - 32*eper**6*j1x*j2y**2 - 24*eper**2*j1z*j2x*j2z + 372*eper**4*j1z*j2x*j2z + 192*eper**6*j1z*j2x*j2z + 176*eper**4*j1x*j2z**2 + 64*eper**6*j1x*j2z**2))) + 3*eper**3*(12*e1z**2*(-5*e2y*e2z*(4 + eper**2)*j1x + 5*e2x*e2z*(4 + 11*eper**2)*j1y - (4 + 11*eper**2)*j1y*j2x*j2z + (4 + eper**2)*j1x*j2y*j2z) + 6*e1y**2*(5*e2y*(2*e2z*(4 + eper**2)*j1x + e2x*(2 + 3*eper**2)*j1z) - j2y*((2 + 3*eper**2)*j1z*j2x + 2*(4 + eper**2)*j1x*j2z)) - 6*e1x**2*(5*e2x*(2*e2z*(4 + 11*eper**2)*j1y + e2y*(2 + 3*eper**2)*j1z) - j2x*((2 + 3*eper**2)*j1z*j2y + 2*(4 + 11*eper**2)*j1y*j2z)) + e1x*e1z*(30*e2x*e2y*(2 + 3*eper**2)*j1x - 5*e2x**2*(34 + 105*eper**2)*j1y + 60*e2y*e2z*(4 + eper**2)*j1z - 6*j2y*((2 + 3*eper**2)*j1x*j2x + 2*(4 + eper**2)*j1z*j2z) + j1y*(-40 - 90*eper**2 + 5*e2y**2*(-22 + 3*eper**2) + 20*e2z**2*(20 + 39*eper**2) + 82*j2x**2 + 213*eper**2*j2x**2 + 70*j2y**2 + 105*eper**2*j2y**2 - 32*j2z**2 - 48*eper**2*j2z**2)) + e1y*(12*e1x*(5*e2x*e2z*(4 + 11*eper**2)*j1x - 5*e2y*e2z*(4 + eper**2)*j1y - 5*e2y**2*j1z + 5*eper**2*j1z - 30*e2z**2*eper**2*j1z + 5*e2x**2*(j1z + 3*eper**2*j1z) - j1z*j2x**2 - 9*eper**2*j1z*j2x**2 + j1z*j2y**2 - 6*eper**2*j1z*j2y**2 - 4*j1x*j2x*j2z - 11*eper**2*j1x*j2x*j2z + 4*j1y*j2y*j2z + eper**2*j1y*j2y*j2z) + e1z*(-30*e2x*(e2y*(2 + 3*eper**2)*j1y + 2*e2z*(4 + 11*eper**2)*j1z) + 6*j2x*((2 + 3*eper**2)*j1y*j2y + 2*(4 + 11*eper**2)*j1z*j2z) + j1x*(40 - 400*e2z**2 + 30*eper**2 - 420*e2z**2*eper**2 - 5*e2y**2*(-34 + 3*eper**2) + 5*e2x**2*(22 + 69*eper**2) - 70*j2x**2 - 105*eper**2*j2x**2 - 82*j2y**2 - 33*eper**2*j2y**2 + 32*j2z**2 + 48*eper**2*j2z**2))))*np.arccos(-(1/eper))))/(64.*np.sqrt(e1x**2 + e1y**2 + e1z**2)*eper**3*(1 + eper)**2*M2**2*Q**2)

def delta_e3_function(a,Q,eper,m1,m2,ex,ey,ez,jx,jy,jz):
    return (a/Q)**2*(m1*m2/((m1+m2)**2))*np.sqrt(1.0 + eper)*pow(eper-1.0,5.0/2.0)*pow(eper,-3.0)*(5.0*ex*ey - jx*jy)

def delta_i3_function(a1,a2,Q,eper,m1,m2,m3,m4,e1x,e1y,e1z,j1x,j1y,j1z,e2x,e2y,e2z,j2x,j2y,j2z):

    Sqrt=np.sqrt
    G=1.0 ## will cancel out later

    M=m1+m2+m3+m4
    M1=m1+m2
    M2=m3+m4
    a=a1
    ex=e1x
    ey=e1y
    ez=e1z
    jx=j1x
    jy=j1y
    jz=j1z
    dh3x = (a**2*(5*ey*ez - jy*jz)*m1*m2*Sqrt(((-1 + eper)**3*G*M)/Q**3)*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*np.arccos(-1.0/eper)))/(2.*eper*(-1 + eper**2)**1.5*M1**2)
    dh3y = -(a**2*(5*ex*ez - jx*jz)*m1*m2*Sqrt(((-1 + eper)**3*G*M)/Q**3)*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*np.arccos(-1.0/eper)))/(2.*eper*(-1 + eper**2)**1.5*M1**2) 
    dh3z = (a**2*(5*ex*ey - jx*jy)*m1*m2*Sqrt(((-1 + eper)**3*eper**2*G*M)/Q**3))/(eper**3*M1**2)

    a=a2
    ex=e2x
    ey=e2y
    ez=e2z
    jx=j2x
    jy=j2y
    jz=j2z
    dh3x += (a**2*(5*ey*ez - jy*jz)*m1*m2*Sqrt(((-1 + eper)**3*G*M)/Q**3)*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*np.arccos(-1.0/eper)))/(2.*eper*(-1 + eper**2)**1.5*M1**2)
    dh3y += -(a**2*(5*ex*ez - jx*jz)*m1*m2*Sqrt(((-1 + eper)**3*G*M)/Q**3)*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*np.arccos(-1.0/eper)))/(2.*eper*(-1 + eper**2)**1.5*M1**2) 
    dh3z += (a**2*(5*ex*ey - jx*jy)*m1*m2*Sqrt(((-1 + eper)**3*eper**2*G*M)/Q**3))/(eper**3*M1**2)

    h3x = 0.0
    h3y = 0.0
    h3z = np.sqrt(G*M*Q*(1.0+eper))
    
    h3vec = np.array([h3x,h3y,h3z])
    dh3vec = np.array([dh3x,dh3y,dh3z])
    h3vecp = h3vec + dh3vec
    z=np.array([0.0,0.0,1.0])
    cos_di3 = np.dot(h3vecp,z)/np.sqrt(np.dot(h3vecp,h3vecp))
    #print("cos_di3",cos_di3)
    
    ArcCos=np.arccos

    di3 = ArcCos((2*eper*(a1**2*(-1 + eper)**1.5*(5*e1x*e1y - j1x*j1y)*m1*m2*M2**2 + M1**2*(a2**2*(-1 + eper)**1.5*(5*e2x*e2y - j2x*j2y)*m3*m4 + eper**2*Sqrt(1 + eper)*M2**2*Q**2)))/(M2**2*Sqrt((4*(Q - eper**2*Q)**3*((a1**2*(5*e1x*e1y - j1x*j1y)*m1*m2*M2**2 + a2**2*(5*e2x*e2y - j2x*j2y)*M1**2*m3*m4)*Sqrt(((-1 + eper)**3*eper**2*G*M)/Q**3) + eper**3*M1**2*M2**2*Sqrt((1 + eper)*G*M*Q))**2 + (1 - eper)**3*eper**4*G*M*(a1**2*(5*e1y*e1z - j1y*j1z)*m1*m2*M2**2 + a2**2*(5*e2y*e2z - j2y*j2z)*M1**2*m3*m4)**2*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*np.arccos(-1.0/eper))**2 + (1 - eper)**3*eper**4*G*M*(a1**2*(5*e1x*e1z - j1x*j1z)*m1*m2*M2**2 + a2**2*(5*e2x*e2z - j2x*j2z)*M1**2*m3*m4)**2*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*np.arccos(-1.0/eper))**2)/((1 - eper**2)**3*G*M*M2**4))))
    
    di3b = ArcCos((2*eper*(a1**2*(-1 + eper)**1.5*(5*e1x*e1y - j1x*j1y)*m1*m2*M2**2 + M1**2*(a2**2*(-1 + eper)**1.5*(5*e2x*e2y - j2x*j2y)*m3*m4 + eper**2*Sqrt(1 + eper)*M2**2*Q**2)))/(M2**2*Sqrt((4*(Q - eper**2*Q)**3*((a1**2*(5*e1x*e1y - j1x*j1y)*m1*m2*M2**2 + a2**2*(5*e2x*e2y - j2x*j2y)*M1**2*m3*m4)*Sqrt(((-1 + eper)**3*eper**2*G*M)/Q**3) + eper**3*M1**2*M2**2*Sqrt((1 + eper)*G*M*Q))**2 + (1 - eper)**3*eper**4*G*M*(a1**2*(5*e1y*e1z - j1y*j1z)*m1*m2*M2**2 + a2**2*(5*e2y*e2z - j2y*j2z)*M1**2*m3*m4)**2*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*np.arccos(-1.0/eper))**2 + (1 - eper)**3*eper**4*G*M*(a1**2*(5*e1x*e1z - j1x*j1z)*m1*m2*M2**2 + a2**2*(5*e2x*e2z - j2x*j2z)*M1**2*m3*m4)**2*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*np.arccos(-1.0/eper))**2)/((1 - eper**2)**3*G*M*M2**4))))
    
    #print("test",np.arccos(cos_di3),di3,di3b)

    return np.arccos(cos_di3)
    
def integrate_averaged(args):
    """
    Integrate inner-averaged equations of motion.
    """
    
    CONST_G = args.CONST_G
    
    m1 = args.m1
    m2 = args.m2
    m3 = args.m3
    m4 = args.m4
    
    M1 = args.M1
    M2 = args.M2
    M = args.M
   
    a1 = args.a1
    a2 = args.a2
    a3 = args.a3
    e1 = args.e1
    e2 = args.e2
    e3 = args.e3
    
    Q = args.Q

    TA1 = args.TA1
    i1 = args.i1
    AP1 = args.AP1
    LAN1 = args.LAN1

    TA2 = args.TA2
    i2 = args.i2
    AP2 = args.AP2
    LAN2 = args.LAN2

    TA3 = args.TA3
    epsilon=1.0e-5

  
    e1x,e1y,e1z,j1x,j1y,j1z = orbital_elements_to_orbital_vectors(e1,i1,AP1,LAN1)

    e2x,e2y,e2z,j2x,j2y,j2z = orbital_elements_to_orbital_vectors(e2,i2,AP2,LAN2)
    
    e3x = 1.0*e3
    e3y = 0.0
    e3z = 0.0
    j3x = 0.0
    j3y = 0.0
    j3z = 1.0
    if args.verbose==True:
        print("e1x,e1y,e1z,j1x,j1y,j1z",e1x,e1y,e1z,j1x,j1y,j1z)
        print("e3x,e3y,e3z,j3x,j3y,j3z",e3x,e3y,e3z,j3x,j3y,j3z)

    r3,v3 = orbital_vectors_to_cartesian(CONST_G,M,a3,TA3,e3x,e3y,e3z,j3x,j3y,j3z)

    N_steps = args.N_steps
    times = args.times

    ODE_args = (CONST_G,m1,m2,m3,m4,M1,M2,M,a1,a2,args)

    RHR_vec = [e1x,e1y,e1z,j1x,j1y,j1z,e2x,e2y,e2z,j2x,j2y,j2z,r3[0],r3[1],r3[2],v3[0],v3[1],v3[2]]
        
    ### numerical solution ###
    sol = odeint(RHS_function, RHR_vec, times, args=ODE_args,mxstep=args.mxstep,rtol=1.0e-13,atol=1e-13)
    
    e1x_sol = np.array(sol[:,0])
    e1y_sol = np.array(sol[:,1])
    e1z_sol = np.array(sol[:,2])
    j1x_sol = np.array(sol[:,3])
    j1y_sol = np.array(sol[:,4])
    j1z_sol = np.array(sol[:,5])
    e2x_sol = np.array(sol[:,6])
    e2y_sol = np.array(sol[:,7])
    e2z_sol = np.array(sol[:,8])
    j2x_sol = np.array(sol[:,9])
    j2y_sol = np.array(sol[:,10])
    j2z_sol = np.array(sol[:,11])
    x3_sol = np.array(sol[:,12])
    y3_sol = np.array(sol[:,13])
    z3_sol = np.array(sol[:,14])
    vx3_sol = np.array(sol[:,15])
    vy3_sol = np.array(sol[:,16])
    vz3_sol = np.array(sol[:,17])

    N_bodies = args.N_bodies
    N_orbits = args.N_orbits

    a_print = [[] for x in range(N_orbits)]
    e_print = [[] for x in range(N_orbits)]
    i_print = [[] for x in range(N_orbits)]

    e1_sol = [np.sqrt(e1x_sol[i]**2 + e1y_sol[i]**2 + e1z_sol[i]**2) for i in range(len(times))]
    j1_sol = [np.sqrt(j1x_sol[i]**2 + j1y_sol[i]**2 + j1z_sol[i]**2) for i in range(len(times))]
    i1_sol = np.array([np.arccos(j1z_sol[i]/j1_sol[i]) for i in range(len(times))])
    
    e2_sol = [np.sqrt(e2x_sol[i]**2 + e2y_sol[i]**2 + e2z_sol[i]**2) for i in range(len(times))]
    j2_sol = [np.sqrt(j2x_sol[i]**2 + j2y_sol[i]**2 + j2z_sol[i]**2) for i in range(len(times))]
    i2_sol = np.array([np.arccos(j2z_sol[i]/j2_sol[i]) for i in range(len(times))])

    r3_sol = np.array([np.sqrt(x3_sol[i]**2 + y3_sol[i]**2 + z3_sol[i]**2) for i in range(len(times))])
    v3_sol = np.array([np.sqrt(vx3_sol[i]**2 + vy3_sol[i]**2 + vz3_sol[i]**2) for i in range(len(times))])

    for i in range(N_orbits):
        if i==0:
            a_print[i] = np.array([a1 for x in range(len(times))])
            e_print[i] = np.array(e1_sol)
            i_print[i] = np.array(i1_sol)
        elif i==1:
            a_print[i] = np.array([a2 for x in range(len(times))])
            e_print[i] = np.array(e2_sol)
            i_print[i] = np.array(i2_sol)
        elif i==2:
            for j in range(len(times)):
                r3 = np.array([x3_sol[j],y3_sol[j],z3_sol[j]])
                v3 = np.array([vx3_sol[j],vy3_sol[j],vz3_sol[j]])
                a3,e3,i3 = orbital_elements_from_nbody(CONST_G,m1+m2+m3+m4,r3,v3)

                a_print[i].append(a3)
                e_print[i].append(e3)
                i_print[i].append(i3)
    for i in range(N_orbits):
        a_print[i] = np.array(a_print[i])
        e_print[i] = np.array(e_print[i])
        i_print[i] = np.array(i_print[i])
    
    Delta_a1 = 0.0
    Delta_e1 = e1_sol[-1] - e1_sol[0]
    Delta_i1 = i1_sol[-1] - i1_sol[0]

    Delta_a2 = 0.0
    Delta_e2 = e2_sol[-1] - e2_sol[0]
    Delta_i2 = i2_sol[-1] - i2_sol[0]


    data = {'times': times,'a_print':a_print,'e_print':e_print,'i_print':i_print, \
        'Delta_a1':Delta_a1,'Delta_a2':Delta_a2,'Delta_e1':Delta_e1,'Delta_e2':Delta_e2,'Delta_i1':Delta_i1,'Delta_i2':Delta_i2}
    
    if args.verbose == True:
        print("inner averaged")
        print("delta e1",Delta_e1)
        print("delta e2",Delta_e2)

    if 1==0:
 
        Hs = []
        for i,t in enumerate(times):
            e1_vec = np.array( [e1x_sol[i],e1y_sol[i],e1z_sol[i]] )
            j1_vec = np.array( [j1x_sol[i],j1y_sol[i],j1z_sol[i]] )

            e2_vec = np.array( [e2x_sol[i],e2y_sol[i],e2z_sol[i]] )
            j2_vec = np.array( [j2x_sol[i],j2y_sol[i],j2z_sol[i]] )
            
            r3 = r3_sol[i]
            r3_vec = np.array( [x3_sol[i],y3_sol[i],z3_sol[i]] )
            fquad1 = (-CONST_G*(m1*m2*M2/(M1)))*(1.0/r3)*(a1/r3)**2
            fquad2 = (-CONST_G*(m3*m4*M1/(M2)))*(1.0/r3)*(a2/r3)**2
            H = fquad1*H_quad_function(e1_vec,j1_vec,r3_vec/r3) + fquad2*H_quad_function(e2_vec,j2_vec,r3_vec/r3)

            foct1 = (-CONST_G*(m1*m2*M2/(M1)))*(np.fabs(m1-m2)/M1)*(1.0/r3)*(a1/r3)**3
            foct2 = (-CONST_G*(m3*m4*M1/(M2)))*(np.fabs(m3-m4)/M1)*(1.0/r3)*(a2/r3)**3
            if i==0:
                H0=H
            
            Hs.append(np.fabs(H0-H)/H0)
        Hs=np.array(Hs)
 
        from matplotlib import pyplot
        fig=pyplot.figure()
        plot=fig.add_subplot(1,1,1,yscale="log")
        #plot.scatter(x3_sol,y3_sol,color='k')
        plot.plot(times,Hs,color='k')
        #print("x3_sol",list(x3_sol))
        
        fig=pyplot.figure()
        plot1=fig.add_subplot(3,1,1)
        plot2=fig.add_subplot(3,1,2)
        plot3=fig.add_subplot(3,1,3)
        colors = ['k','r','b']
        #for i in range(N_orbits):
            #plot1.plot(times,a_print[i],color=colors[i])
        plot2.plot(times,e1_sol,color=colors[0])
        plot2.plot(times,e2_sol,color=colors[1])
        plot3.plot(times,(180.0/np.pi)*i1_sol,color=colors[0])
        plot3.plot(times,(180.0/np.pi)*i2_sol,color=colors[1])

        pyplot.show()

    return data

def quad_function_e_dot_vec(e_vec,j_vec,r_vec_hat):

    return -3.0*np.cross(j_vec,e_vec) - 1.5*np.dot(j_vec,r_vec_hat)*np.cross(e_vec,r_vec_hat) + 7.5*np.dot(e_vec,r_vec_hat)*np.cross(j_vec,r_vec_hat)

def quad_function_j_dot_vec(e_vec,j_vec,r_vec_hat):

    return 1.5*( -np.dot(j_vec,r_vec_hat)*np.cross(j_vec,r_vec_hat) + 5.0*np.dot(e_vec,r_vec_hat)*np.cross(e_vec,r_vec_hat) )

def H_quad_function(e_vec,j_vec,r_vec_hat):
    return 0.25*( 1.0 - 6.0*np.dot(e_vec,e_vec) + 15.0*np.dot(e_vec,r_vec_hat)**2 - 3.0*np.dot(j_vec,r_vec_hat)**2 )

def oct_function_e_dot_vec(e_vec,j_vec,r_vec_hat):
    e_p2 = np.dot(e_vec,e_vec)
    j_vec_cross_e_vec = np.cross(j_vec,e_vec)
    e_vec_dot_r_vec_hat = np.dot(e_vec,r_vec_hat)
    j_vec_dot_r_vec_hat = np.dot(j_vec,r_vec_hat)
    j_vec_cross_r_vec_hat = np.cross(j_vec,r_vec_hat)
    e_vec_cross_r_vec_hat = np.cross(e_vec,r_vec_hat)
    
    return -0.9375*( -35.0*e_vec_dot_r_vec_hat**2*j_vec_cross_r_vec_hat + 5.0*j_vec_dot_r_vec_hat**2*j_vec_cross_r_vec_hat \
        - ( 1.0 - 8.0*e_p2 )*j_vec_cross_r_vec_hat + 2.0*e_vec_dot_r_vec_hat*( 5.0*j_vec_dot_r_vec_hat*e_vec_cross_r_vec_hat + 8.0*j_vec_cross_e_vec ) )

def oct_function_j_dot_vec(e_vec,j_vec,r_vec_hat):
    e_p2 = np.dot(e_vec,e_vec)
    e_vec_dot_r_vec_hat = np.dot(e_vec,r_vec_hat)
    j_vec_dot_r_vec_hat = np.dot(j_vec,r_vec_hat)
    e_vec_cross_r_vec_hat = np.cross(e_vec,r_vec_hat)
    j_vec_cross_r_vec_hat = np.cross(j_vec,r_vec_hat)

    return 0.9375*( -10.0*e_vec_dot_r_vec_hat*j_vec_dot_r_vec_hat*j_vec_cross_r_vec_hat + e_vec_cross_r_vec_hat*( 35.0*e_vec_dot_r_vec_hat**2 \
        - 5.0*j_vec_dot_r_vec_hat**2 + 1.0 - 8.0*e_p2 ) )
    
def H_oct_function(e_vec,j_vec,r_vec_hat):
    return (5.0/16.0)*np.dot(e_vec,r_vec_hat)*( 3.0*(1.0 - 8.0*np.dot(e_vec,e_vec)) + 35.0*np.dot(e_vec,r_vec_hat)**2 - 15.0*np.dot(j_vec,r_vec_hat)**2 )
    
def hex_function_e_dot_vec(e_vec,j_vec,r_vec_hat):
    e_p2 = np.dot(e_vec,e_vec)
    j_vec_cross_e_vec = np.cross(j_vec,e_vec)
    e_vec_dot_r_vec_hat = np.dot(e_vec,r_vec_hat)
    j_vec_dot_r_vec_hat = np.dot(j_vec,r_vec_hat)
    e_vec_cross_r_vec_hat = np.cross(e_vec,r_vec_hat)
    j_vec_cross_r_vec_hat = np.cross(j_vec,r_vec_hat)
    
    return 0.9375*( 7.0*( 21.0*e_vec_dot_r_vec_hat**3*j_vec_cross_r_vec_hat - 7.0*e_vec_dot_r_vec_hat**2*j_vec_dot_r_vec_hat*e_vec_cross_r_vec_hat \
        - 7.0*e_vec_dot_r_vec_hat*j_vec_dot_r_vec_hat**2*j_vec_cross_r_vec_hat + j_vec_dot_r_vec_hat**3*e_vec_cross_r_vec_hat ) \
        + 7.0*e_vec_dot_r_vec_hat*( j_vec_cross_r_vec_hat - 10.0*( e_vec_dot_r_vec_hat*j_vec_cross_e_vec + e_p2*j_vec_cross_r_vec_hat ) ) \
        + (10.0*e_p2 - 3.0)*j_vec_dot_r_vec_hat*e_vec_cross_r_vec_hat + 10.0*j_vec_dot_r_vec_hat**2*j_vec_cross_e_vec + 2.0*(8.0*e_p2 - 1.0)*j_vec_cross_e_vec )

def hex_function_j_dot_vec(e_vec,j_vec,r_vec_hat):
    e_p2 = np.dot(e_vec,e_vec)
    j_vec_cross_e_vec = np.cross(j_vec,e_vec)
    e_vec_dot_r_vec_hat = np.dot(e_vec,r_vec_hat)
    j_vec_dot_r_vec_hat = np.dot(j_vec,r_vec_hat)
    e_vec_cross_r_vec_hat = np.cross(e_vec,r_vec_hat)
    j_vec_cross_r_vec_hat = np.cross(j_vec,r_vec_hat)

    return 0.9375*( 7.0*e_vec_dot_r_vec_hat*(21.0*e_vec_dot_r_vec_hat**2 - 7.0*j_vec_dot_r_vec_hat**2 + 1.0 - 10.0*e_p2)*e_vec_cross_r_vec_hat \
    + j_vec_dot_r_vec_hat*( -49.0*e_vec_dot_r_vec_hat**2 + 7.0*j_vec_dot_r_vec_hat**2 + 10.0*e_p2 - 3.0 )*j_vec_cross_r_vec_hat )

def hex_function_cross(e_vec,j_vec,E_vec,J_vec,r_vec_hat):
    grad_e = hex_function_cross_grad_e(e_vec,j_vec,E_vec,J_vec,r_vec_hat)
    grad_j = hex_function_cross_grad_j(e_vec,j_vec,E_vec,J_vec,r_vec_hat)
    
    e_dot_vec = ( np.cross(e_vec,grad_j) + np.cross(j_vec,grad_e) )
    j_dot_vec = ( np.cross(j_vec,grad_j) + np.cross(e_vec,grad_e) )
    return e_dot_vec, j_dot_vec
    
def hex_function_cross_grad_e(e_vec,j_vec,E_vec,J_vec,r_vec_hat):
    return (3.0/16.0)*( -12.0*e_vec + 72.0*np.dot(E_vec,E_vec)*e_vec + 100.0*np.dot(e_vec,E_vec)*E_vec - 20.0*np.dot(e_vec,J_vec)*J_vec + 50.0*np.dot(e_vec,r_vec_hat)*r_vec_hat + 60.0*np.dot(J_vec,r_vec_hat)**2*e_vec \
        - 300.0*np.dot(E_vec,E_vec)*np.dot(e_vec,r_vec_hat)*r_vec_hat - 300.0*np.dot(E_vec,r_vec_hat)**2*e_vec - 500.0*np.dot(E_vec,r_vec_hat)*( np.dot(e_vec,r_vec_hat)*E_vec + np.dot(e_vec,E_vec)*r_vec_hat ) \
        + 100.0*np.dot(J_vec,r_vec_hat)*( np.dot(e_vec,r_vec_hat)*J_vec + np.dot(e_vec,J_vec)*r_vec_hat ) + 1750.0*np.dot(e_vec,r_vec_hat)*np.dot(E_vec,r_vec_hat)**2*r_vec_hat - 350.0*np.dot(e_vec,r_vec_hat)*np.dot(J_vec,r_vec_hat)**2*r_vec_hat )

def hex_function_cross_grad_j(e_vec,j_vec,E_vec,J_vec,r_vec_hat):
    return (3.0/16.0)*( -20.0*np.dot(j_vec,E_vec)*E_vec + 4.0*np.dot(j_vec,J_vec)*J_vec + 10.0*(6.0*np.dot(E_vec,E_vec) - 1.0)*np.dot(j_vec,r_vec_hat)*r_vec_hat - 20.0*np.dot(J_vec,r_vec_hat)*( np.dot(j_vec,r_vec_hat)*J_vec + np.dot(j_vec,J_vec)*r_vec_hat ) \
        + 100.0*np.dot(E_vec,r_vec_hat)*( np.dot(j_vec,r_vec_hat)*E_vec + np.dot(j_vec,E_vec)*r_vec_hat ) + 70.0*np.dot(j_vec,r_vec_hat)*np.dot(J_vec,r_vec_hat)**2*r_vec_hat - 350.0*np.dot(j_vec,r_vec_hat)*np.dot(E_vec,r_vec_hat)**2*r_vec_hat )
    
def quad_function_r_ddot_vec(e_vec,j_vec,r_vec,r_p5_div,r_p7_div):
    e_p2 = np.dot(e_vec,e_vec)
    e_vec_dot_r_vec = np.dot(e_vec,r_vec)
    j_vec_dot_r_vec = np.dot(j_vec,r_vec)
    
    return 0.25*( -3.0*(1.0 - 6.0*e_p2)*r_vec*r_p5_div + 30.0*e_vec_dot_r_vec*e_vec*r_p5_div - 75.0*e_vec_dot_r_vec*e_vec_dot_r_vec*r_vec*r_p7_div - 6.0*j_vec_dot_r_vec*j_vec*r_p5_div + 15.0*j_vec_dot_r_vec*j_vec_dot_r_vec*r_vec*r_p7_div )

def RHS_function(RHR_vec, theta, *ODE_args):
    """
    Singly-averaged (SA) equations of motion.
    """

    ### initialization ###
    CONST_G,m1,m2,m3,m4,M1,M2,M,a1,a2,args = ODE_args
    verbose = args.verbose

    e1x = RHR_vec[0]
    e1y = RHR_vec[1]
    e1z = RHR_vec[2]

    j1x = RHR_vec[3]
    j1y = RHR_vec[4]
    j1z = RHR_vec[5]

    e2x = RHR_vec[6]
    e2y = RHR_vec[7]
    e2z = RHR_vec[8]

    j2x = RHR_vec[9]
    j2y = RHR_vec[10]
    j2z = RHR_vec[11]
    
    x3 = RHR_vec[12]
    y3 = RHR_vec[13]
    z3 = RHR_vec[14]
    
    vx3 = RHR_vec[15]
    vy3 = RHR_vec[16]
    vz3 = RHR_vec[17]

    e1_vec = np.array([e1x,e1y,e1z])
    j1_vec = np.array([j1x,j1y,j1z])
    
    e2_vec = np.array([e2x,e2y,e2z])
    j2_vec = np.array([j2x,j2y,j2z])

    r3_vec = np.array([x3,y3,z3])
    v3_vec = np.array([vx3,vy3,vz3])
    
    r3 = np.linalg.norm(r3_vec)
    r3_vec_hat = r3_vec/r3
    
    r3_p2 = r3*r3
    r3_p3 = r3_p2*r3
    r3_p5 = r3_p2*r3_p3
    r3_p7 = r3_p2*r3_p5
    r3_p3_div = 1.0/r3_p3
    r3_p5_div = 1.0/r3_p5
    r3_p7_div = 1.0/r3_p7
    

    fquad1 = (-CONST_G*(m1*m2*M2/(M1)))*(1.0/r3)*(a1/r3)**2
    fquad2 = (-CONST_G*(m3*m4*M1/(M2)))*(1.0/r3)*(a2/r3)**2

    ### ODE RHS ###
    n1 = np.sqrt(CONST_G*M1/(a1**3))
    n2 = np.sqrt(CONST_G*M2/(a2**3))
    
    e1_vec_dot = [0.0,0.0,0.0]
    e2_vec_dot = [0.0,0.0,0.0]
    j1_vec_dot = [0.0,0.0,0.0]
    j2_vec_dot = [0.0,0.0,0.0]
    
    if args.quad == True:
        #print("quad")
        common_factor1_quad = n1*(M2/M1)*(a1/r3)**3
        common_factor2_quad = n2*(M1/M2)*(a2/r3)**3
    
        e1_vec_dot += common_factor1_quad*quad_function_e_dot_vec(e1_vec,j1_vec,r3_vec_hat)
        e2_vec_dot += common_factor2_quad*quad_function_e_dot_vec(e2_vec,j2_vec,r3_vec_hat)
    
        j1_vec_dot += common_factor1_quad*quad_function_j_dot_vec(e1_vec,j1_vec,r3_vec_hat)
        j2_vec_dot += common_factor2_quad*quad_function_j_dot_vec(e2_vec,j2_vec,r3_vec_hat)

    if args.oct == True:
       # print("oct")
        common_factor1_oct = n1*(M2/M1)*(np.fabs(m1-m2)/M1)*(a1/r3)**4
        common_factor2_oct = n2*(M1/M2)*(np.fabs(m3-m4)/M2)*(a2/r3)**4
    
        e1_vec_dot += common_factor1_oct*oct_function_e_dot_vec(e1_vec,j1_vec,r3_vec_hat)
        e2_vec_dot += common_factor2_oct*oct_function_e_dot_vec(e2_vec,j2_vec,r3_vec_hat)
    
        j1_vec_dot += common_factor1_oct*oct_function_j_dot_vec(e1_vec,j1_vec,r3_vec_hat)
        j2_vec_dot += common_factor2_oct*oct_function_j_dot_vec(e2_vec,j2_vec,r3_vec_hat)
    
    if args.hex == True:
        #print("hex")
        common_factor1_hex = n1*(M2/M1)*((m1**2 - m1*m2 + m2**2)/(M1**2))*(a1/r3)**5
        common_factor2_hex = n2*(M1/M2)*((m3**2 - m3*m4 + m4**2)/(M2**2))*(a2/r3)**5
    
        e1_vec_dot += common_factor1_hex*hex_function_e_dot_vec(e1_vec,j1_vec,r3_vec_hat)
        e2_vec_dot += common_factor2_hex*hex_function_e_dot_vec(e2_vec,j2_vec,r3_vec_hat)
    
        j1_vec_dot += common_factor1_hex*hex_function_j_dot_vec(e1_vec,j1_vec,r3_vec_hat)
        j2_vec_dot += common_factor2_hex*hex_function_j_dot_vec(e2_vec,j2_vec,r3_vec_hat)

    if args.hex_cross == True:
        #print("hex cross")
        common_factor1_hex_cross = n1*(m3*m4/(M1*M2))*(a1/r3)**3*(a2/r3)**2
        common_factor2_hex_cross = n2*(m1*m2/(M1*M2))*(a2/r3)**3*(a1/r3)**2
    
        e1_vec_dot_hex_cross, j1_vec_dot_hex_cross = hex_function_cross(e1_vec,j1_vec,e2_vec,j2_vec,r3_vec_hat)
        e1_vec_dot_hex_cross *= common_factor1_hex_cross
        j1_vec_dot_hex_cross *= common_factor1_hex_cross

        e2_vec_dot_hex_cross, j2_vec_dot_hex_cross = hex_function_cross(e2_vec,j2_vec,e1_vec,j1_vec,r3_vec_hat)
        e2_vec_dot_hex_cross *= common_factor2_hex_cross
        j2_vec_dot_hex_cross *= common_factor2_hex_cross
    
        e1_vec_dot += e1_vec_dot_hex_cross
        e2_vec_dot += e2_vec_dot_hex_cross
        j1_vec_dot += j1_vec_dot_hex_cross
        j2_vec_dot += j2_vec_dot_hex_cross
    
    r3_vec_dot = v3_vec
    r3_vec_ddot = -CONST_G*M*r3_p3_div*r3_vec

    if (args.include_backreaction == True):
        common_factor1_r3 = (CONST_G*M*m1*m2/M1**2)*(a1)**2
        common_factor2_r3 = (CONST_G*M*m3*m4/M2**2)*(a2)**2

        r3_vec_ddot += common_factor1_r3*quad_function_r_ddot_vec(e1_vec,j1_vec,r3_vec,r3_p5_div,r3_p7_div) + common_factor2_r3*quad_function_r_ddot_vec(e2_vec,j2_vec,r3_vec,r3_p5_div,r3_p7_div)
    
    RHR_vec_dot = [e1_vec_dot[0],e1_vec_dot[1],e1_vec_dot[2],j1_vec_dot[0],j1_vec_dot[1],j1_vec_dot[2],e2_vec_dot[0],e2_vec_dot[1],e2_vec_dot[2],j2_vec_dot[0],j2_vec_dot[1],j2_vec_dot[2],r3_vec_dot[0],r3_vec_dot[1],r3_vec_dot[2],r3_vec_ddot[0],r3_vec_ddot[1],r3_vec_ddot[2]]

    return RHR_vec_dot

def integrate_nbody(args):

    """
    Integrate N-body equations of motion.
    """

    ### initial conditions ###   
    CONST_G = args.CONST_G
    
    m1 = args.m1
    m2 = args.m2
    m3 = args.m3
    m4 = args.m4
    
    M1 = args.M1
    M2 = args.M2
    M = args.M
   
    a1 = args.a1
    a2 = args.a2
    a3 = args.a3
    e1 = args.e1
    e2 = args.e2
    e3 = args.e3
    
    Q = args.Q

    TA1 = args.TA1
    i1 = args.i1
    AP1 = args.AP1
    LAN1 = args.LAN1

    TA2 = args.TA2
    i2 = args.i2
    AP2 = args.AP2
    LAN2 = args.LAN2

    TA3 = args.TA3

    epsilon=1.0e-5

    e1x,e1y,e1z,j1x,j1y,j1z = orbital_elements_to_orbital_vectors(e1,i1,AP1,LAN1)
    r1,v1 = orbital_vectors_to_cartesian(CONST_G,M1,a1,TA1,e1x,e1y,e1z,j1x,j1y,j1z)

    e2x,e2y,e2z,j2x,j2y,j2z = orbital_elements_to_orbital_vectors(e2,i2,AP2,LAN2)
    r2,v2 = orbital_vectors_to_cartesian(CONST_G,M2,a2,TA2,e2x,e2y,e2z,j2x,j2y,j2z)
    
    e3x = 1.0*e3
    e3y = 0.0
    e3z = 0.0
    j3x = 0.0
    j3y = 0.0
    j3z = 1.0
    r3,v3 = orbital_vectors_to_cartesian(CONST_G,M,a3,TA3,e3x,e3y,e3z,j3x,j3y,j3z)

    R_cm, V_cm = np.zeros(3), np.zeros(3)
    
    R_cm1 = R_cm + (M2/M)*r3
    V_cm1 = V_cm + (M2/M)*v3
    R_cm2 = R_cm - (M1/M)*r3
    V_cm2 = V_cm - (M1/M)*v3
    
    R1 = R_cm1 + (m2/M1)*r1
    V1 = V_cm1 + (m2/M1)*v1
    R2 = R_cm1 - (m1/M1)*r1
    V2 = V_cm1 - (m1/M1)*v1

    R3 = R_cm2 + (m4/M2)*r2
    V3 = V_cm2 + (m4/M2)*v2
    R4 = R_cm2 - (m3/M2)*r2
    V4 = V_cm2 - (m3/M2)*v2

    N_steps = args.N_steps
    
    times = args.times
    tend = args.tend
    N_bodies = args.N_bodies
    N_orbits = args.N_orbits

    R_print = [[] for x in range(N_bodies)]
    V_print = [[] for x in range(N_bodies)]

    a_print = [[] for x in range(N_orbits)]
    e_print = [[] for x in range(N_orbits)]
    i_print = [[] for x in range(N_orbits)]
    
    if (args.code == "abie"):
        integrate_nbody_ABIE(CONST_G,m1,m2,m3,m4,tend,times,N_bodies,N_orbits,R1,R2,R3,R4,V1,V2,V3,V4,R_print,V_print,a_print,e_print,i_print)
    elif (args.code == "rebound"):
        integrate_nbody_rebound(CONST_G,m1,m2,m3,m4,tend,times,N_bodies,N_orbits,R1,R2,R3,R4,V1,V2,V3,V4,R_print,V_print,a_print,e_print,i_print)

    for i in range(N_orbits):
        a_print[i] = np.array(a_print[i])
        e_print[i] = np.array(e_print[i])
        i_print[i] = np.array(i_print[i])
    
    Delta_a1 = a_print[0][-1] - a_print[0][0]
    Delta_e1 = e_print[0][-1] - e_print[0][0]
    Delta_i1 = i_print[0][-1] - i_print[0][0]
    
    Delta_a2 = a_print[1][-1] - a_print[1][0]
    Delta_e2 = e_print[1][-1] - e_print[1][0]
    Delta_i2 = i_print[1][-1] - i_print[1][0]
    
    if args.verbose == True:
        print("nbody, using ",args.code)
        print("Delta e1",Delta_e1,'; Delta a1',Delta_a1)
        print("Delta e2",Delta_e2,'; Delta a2',Delta_a2)
    
    if 1==0:
        fig=pyplot.figure()
        plot=fig.add_subplot(1,1,1)
        for i in range(N_bodies):
            plot.plot([R_print[i][j][0] for j in range(N_steps)],[R_print[i][j][1] for j in range(N_steps)])

        fig=pyplot.figure()
        plot1=fig.add_subplot(3,1,1)
        plot2=fig.add_subplot(3,1,2)
        plot3=fig.add_subplot(3,1,3)
        colors = ['k','r','b']
        for i in range(N_orbits):
            plot1.plot(times,a_print[i],color=colors[i])
            plot2.plot(times,e_print[i],color=colors[i])
            plot3.plot(times,np.array(i_print[i])*180.0/np.pi,color=colors[i])

        pyplot.show()

    data = {'times': times,'a_print':a_print,'e_print':e_print,'i_print':i_print, \
        'Delta_a1':Delta_a1,'Delta_a2':Delta_a2,'Delta_e1':Delta_e1,'Delta_e2':Delta_e2,'Delta_i1':Delta_i1,'Delta_i2':Delta_i2}
    
    return data

def get_elements_from_nbody(CONST_G,m1,m2,m3,m4,R_print,V_print):
    r1 = R_print[0][-1] - R_print[1][-1]
    v1 = V_print[0][-1] - V_print[1][-1]
    a1,e1,i1 = orbital_elements_from_nbody(CONST_G,m1+m2,r1,v1)

    r2 = R_print[2][-1] - R_print[3][-1]
    v2 = V_print[2][-1] - V_print[3][-1]
    a2,e2,i2 = orbital_elements_from_nbody(CONST_G,m3+m4,r2,v2)

    r3 = (R_print[0][-1]*m1 + R_print[1][-1]*m2)/(m1+m2) - (R_print[2][-1]*m3 + R_print[3][-1]*m4)/(m3+m4)
    v3 = (V_print[0][-1]*m1 + V_print[1][-1]*m2)/(m1+m2) - (V_print[2][-1]*m3 + V_print[3][-1]*m4)/(m3+m4)
    a3,e3,i3 = orbital_elements_from_nbody(CONST_G,m1+m2+m3+m4,r3,v3)

    return a1,e1,i1,a2,e2,i2,a3,e3,i3

def integrate_nbody_ABIE(CONST_G,m1,m2,m3,m4,tend,times,N_bodies,N_orbits,R1,R2,R3,R4,V1,V2,V3,V4,R_print,V_print,a_print,e_print,i_print):

    try:
        from ABIE.abie import ABIE
    except ImportError:
        from abie import ABIE

    sim = ABIE()

    # Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
    # sim.integrator = 'WisdomHolman'
    sim.integrator = 'GaussRadau15'
    # sim.integrator = 'RungeKutta'

    sim.CONST_G = CONST_G

    sim.acceleration_method = 'ctypes'
    # sim.acceleration_method = 'numpy'

    sim.add(mass=m1, x=R1[0], y=R1[1], z=R1[2], vx=V1[0], vy=V1[1], vz=V1[2], name='m1')
    sim.add(mass=m2, x=R2[0], y=R2[1], z=R2[2], vx=V2[0], vy=V2[1], vz=V2[2], name='m2')
    sim.add(mass=m3, x=R3[0], y=R3[1], z=R3[2], vx=V3[0], vy=V3[1], vz=V3[2], name='m3')
    sim.add(mass=m4, x=R4[0], y=R4[1], z=R4[2], vx=V4[0], vy=V4[1], vz=V4[2], name='m4')

    # sim.close_encounter_distance = 0
    # sim.max_close_encounter_events = 100
    # sim.max_collision_events = 1

#    sim.particles['star1'].primary = sim.particles['star2']

    sim.output_file = 'bin.h5'
    sim.collision_output_file = 'bin.collisions.txt'
    sim.close_encounter_output_file = 'bin.ce.txt'

    # The output frequency
    sim.store_dt = 0.01

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = 0.001

    # The size of the buffer. The buffer is full when `buffer_len` outputs are
    # generated. In this case, the collective output will be flushed to the HDF5
    # file, generating a `Step#n` HDF5 group
    sim.buffer_len = 10000

    sim.initialize()

    t=0.0
    for t in times:
        sim.integrate(t)
        
        for i in range(N_bodies):
            x = sim.particles[i].x
            y = sim.particles[i].y
            z = sim.particles[i].z
            R = np.array( [x,y,z] )
            R_print[i].append( R )

            vx = sim.particles[i].vx
            vy = sim.particles[i].vy
            vz = sim.particles[i].vz
            V = np.array( [vx,vy,vz] )
            V_print[i].append( V )

        a1,e1,i1,a2,e2,i2,a3,e3,i3 = get_elements_from_nbody(CONST_G,m1,m2,m3,m4,R_print,V_print)

        a_print[0].append(a1)
        a_print[1].append(a2)
        a_print[2].append(a3)
        e_print[0].append(e1)
        e_print[1].append(e2)
        e_print[2].append(e3)
        i_print[0].append(i1)
        i_print[1].append(i2)
        i_print[2].append(i3)

    sim.stop()

def integrate_nbody_rebound(CONST_G,m1,m2,m3,m4,tend,times,N_bodies,N_orbits,R1,R2,R3,R4,V1,V2,V3,V4,R_print,V_print,a_print,e_print,i_print):

    import rebound
    sim = rebound.Simulation()
    sim.G = CONST_G

    sim.add(m=m1, x=R1[0], y=R1[1], z=R1[2], vx=V1[0], vy=V1[1], vz=V1[2])
    sim.add(m=m2, x=R2[0], y=R2[1], z=R2[2], vx=V2[0], vy=V2[1], vz=V2[2])
    sim.add(m=m3, x=R3[0], y=R3[1], z=R3[2], vx=V3[0], vy=V3[1], vz=V3[2])
    sim.add(m=m4, x=R4[0], y=R4[1], z=R4[2], vx=V4[0], vy=V4[1], vz=V4[2])

    sim.move_to_com()

    #sim.status()
    t=0.0
    
    for t in times:
        sim.integrate(t)
        
        for i in range(N_bodies):
            x = sim.particles[i].x
            y = sim.particles[i].y
            z = sim.particles[i].z
            R = np.array( [x,y,z] )
            R_print[i].append( R )

            vx = sim.particles[i].vx
            vy = sim.particles[i].vy
            vz = sim.particles[i].vz
            V = np.array( [vx,vy,vz] )
            V_print[i].append( V )

        a1,e1,i1,a2,e2,i2,a3,e3,i3 = get_elements_from_nbody(CONST_G,m1,m2,m3,m4,R_print,V_print)
        
        a_print[0].append(a1)
        a_print[1].append(a2)
        a_print[2].append(a3)
        e_print[0].append(e1)
        e_print[1].append(e2)
        e_print[2].append(e3)
        i_print[0].append(i1)
        i_print[1].append(i2)
        i_print[2].append(i3)




def orbital_elements_from_nbody(G,m,r,v):
    E = 0.5*np.dot(v,v) - G*m/np.linalg.norm(r)
    a = -G*m/(2.0*E)
    h = np.cross(r,v)
    e = (1.0/(G*m))*np.cross(v,h) - r/np.linalg.norm(r)
    e_norm = np.linalg.norm(e)
    i = np.arccos(h[2]/np.linalg.norm(h))
        
    return a,e_norm,i

def orbital_elements_to_orbital_vectors(e,i,omega,Omega):
    
    if (e>=0 and e<1.0):
        j = np.sqrt(1.0 - e**2)
    else:
        j = 1.0
        
    ex = e*(np.cos(omega)*np.cos(Omega) - np.cos(i)*np.sin(omega)*np.sin(Omega))
    ey = e*(np.cos(i)*np.cos(Omega)*np.sin(omega) + np.cos(omega)*np.sin(Omega))
    ez = e*np.sin(i)*np.sin(omega)
    jx = j*np.sin(i)*np.sin(Omega)
    jy = -j*np.cos(Omega)*np.sin(i)
    jz = j*np.cos(i)
    return ex,ey,ez,jx,jy,jz


def orbital_vectors_to_cartesian(G,m,a,theta_bin,ex,ey,ez,jx,jy,jz):
    e = np.sqrt(ex**2+ey**2+ez**2)
    e_hat_vec = np.array((ex,ey,ez))/e
    j_hat_vec = np.array((jx,jy,jz))/np.sqrt(jx**2+jy**2+jz**2)
    q_hat_vec = np.cross(j_hat_vec,e_hat_vec)
    
    cos_theta_bin = np.cos(theta_bin)
    sin_theta_bin = np.sin(theta_bin)
    r_norm = a*(1.0-e**2)/(1.0 + e*cos_theta_bin)
    v_norm = np.sqrt(G*m/(a*(1.0-e**2)))
    r = np.zeros(3)
    v = np.zeros(3)

    for i in range(3):
        r[i] = r_norm*(e_hat_vec[i]*cos_theta_bin + q_hat_vec[i]*sin_theta_bin)
        v[i] = v_norm*(-sin_theta_bin*e_hat_vec[i] + (e+cos_theta_bin)*q_hat_vec[i])
    return r,v


def third_body_cartesian(G,m,M_per,Q,e_per,theta_0):
    a_per = Q/(e_per-1.0)

    M_tot = m+M_per

    n_per = np.sqrt(G*M_tot/(a_per**3))

    cos_true_anomaly = np.cos(theta_0)
    sin_true_anomaly = np.sin(theta_0)

    r_per = Q*(1.0 + e_per)/(1.0 + e_per*cos_true_anomaly);     
    r_dot_factor = np.sqrt(G*M_tot/(Q*(1.0 + e_per)))
   
    r_per_vec = np.zeros(3)
    r_dot_per_vec = np.zeros(3)
    e_per_hat_vec = np.array([1.0,0.0,0.0])
    q_per_hat_vec = np.array([0.0,1.0,0.0])
    j_per_hat_vec = np.array([0.0,0.0,1.0])
    
    for i in range(3):
        r_per_vec[i] = r_per*(cos_true_anomaly*e_per_hat_vec[i] + sin_true_anomaly*q_per_hat_vec[i])
        r_dot_per_vec[i] = r_dot_factor*( -sin_true_anomaly*e_per_hat_vec[i] + (e_per + cos_true_anomaly)*q_per_hat_vec[i])
    
    return r_per_vec,r_dot_per_vec

def compute_total_energy(G,m1,m2,m3,R1,V1,R2,V2,R3,V3):
    T = 0.5*m1*np.dot(V1,V1) + 0.5*m2*np.dot(V2,V2) + 0.5*m3*np.dot(V3,V3)
    V = -G*m1*m2/np.linalg.norm(R1-R2) - G*m1*m3/np.linalg.norm(R1-R3) - G*m2*m3/np.linalg.norm(R2-R3)
    return T+V

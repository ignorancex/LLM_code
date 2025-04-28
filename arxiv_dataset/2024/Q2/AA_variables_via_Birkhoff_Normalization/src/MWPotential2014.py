import sympy as sp
import celmech as cm
from scipy.special import hyp1f1

_A_bulge0 = 0.029994597188218296
_A_disk0 = 0.7574802019371595
_A_halo0 = 4.852230533527998
def _my_lower_gamma(s,z):
    return (z**s/s) * hyp1f1(s,s+1,-z)


def MWPotential2014_get_hamiltonian(ics,NL,\
                                    A_disk = _A_disk0, a_disk = 3./8., b_disk = 0.28 / 8,\
                                    A_halo = _A_halo0, r_s_halo = 16/8.,\
                                    A_bulge = _A_bulge0, alpha_bulge = 1.8, rC_bulge = 1.9/8.
                                    ):
    """
    Get a Hamiltonian for the MWPotential2014 potential from galpy.
    """

    R,z,pR,pz = sp.symbols("R,z,p_R,p_z",real=True)
    L = sp.symbols("L",positive=True)
    r = sp.sqrt(R*R + z*z)
    A1,A2,A3 = sp.symbols("A(1:4)",positive = True)
    pars = {L:NL,A1:A_disk,A2:A_halo,A3:A_bulge}

    # Miyamoto-Nagai disk
    a,b = sp.symbols("a,b",positive = True)
    z_term = a + sp.sqrt(z*z + b*b)
    phi_disk = - A1 / sp.sqrt(R * R + z_term * z_term)
    pars[a] = a_disk
    pars[b] = b_disk

    # NFW halo
    r_s = sp.symbols("r_s",positive = True)
    r_by_r_s = r / r_s
    phi_halo = - A2 * sp.log(1 + r_by_r_s)  / r
    pars[r_s] = r_s_halo

    # power-law bulge
    alpha,rC = sp.symbols("alpha,r_C",positive=True)
    r_by_rC = r/rC
    phi_bulge = A3 * 2 * sp.pi * (1/rC)**alpha * rC**2
    phi_bulge *=(sp.lowergamma(1-alpha/2,r_by_rC**2) - (1/r_by_rC) * sp.lowergamma((3-alpha)/2,r_by_rC**2))
    pars[alpha] = alpha_bulge
    pars[rC] = rC_bulge

    PE = phi_disk+phi_halo+phi_bulge
    KE = (pR * pR + pz * pz + (L/R) * (L/R))/2
    state = cm.PhaseSpaceState((R,z,pR,pz),ics)
    ham = cm.Hamiltonian(KE + PE,pars,state)
    ham._lambdify_kwargs['modules'][1].update({'lowergamma':_my_lower_gamma})
    return ham

def MWPotential2014_get_hamiltonian_full(ics,\
                                    A_disk = _A_disk0, a_disk = 3./8., b_disk = 0.28 / 8,\
                                    A_halo = _A_halo0, r_s_halo = 16/8.,\
                                    A_bulge = _A_bulge0, alpha_bulge = 1.8, rC_bulge = 1.9/8.
                                    ):
    """
    Get a Hamiltonian for the MWPotential2014 potential from galpy.
    """

    phi,R,z,L,pR,pz = sp.symbols("phi,R,z,L,p_R,p_z",real=True)
    r = sp.sqrt(R*R + z*z)
    A1,A2,A3 = sp.symbols("A(1:4)",positive = True)
    pars = {A1:A_disk,A2:A_halo,A3:A_bulge}

    # Miyamoto-Nagai disk
    a,b = sp.symbols("a,b",positive = True)
    z_term = a + sp.sqrt(z*z + b*b)
    phi_disk = - A1 / sp.sqrt(R * R + z_term * z_term)
    pars[a] = a_disk
    pars[b] = b_disk

    # NFW halo
    r_s = sp.symbols("r_s",positive = True)
    r_by_r_s = r / r_s
    phi_halo = - A2 * sp.log(1 + r_by_r_s)  / r
    pars[r_s] = r_s_halo

    # power-law bulge
    alpha,rC = sp.symbols("alpha,r_C",positive=True)
    r_by_rC = r/rC
    phi_bulge = A3 * 2 * sp.pi * (1/rC)**alpha * rC**2
    phi_bulge *=(sp.lowergamma(1-alpha/2,r_by_rC**2) - (1/r_by_rC) * sp.lowergamma((3-alpha)/2,r_by_rC**2))
    pars[alpha] = alpha_bulge
    pars[rC] = rC_bulge

    PE = phi_disk+phi_halo+phi_bulge
    KE = (pR * pR + pz * pz + (L/R) * (L/R))/2
    state = cm.PhaseSpaceState((phi,R,z,L,pR,pz),ics)
    ham = cm.Hamiltonian(KE + PE,pars,state)
    ham._lambdify_kwargs['modules'][1].update({'lowergamma':_my_lower_gamma})
    return ham

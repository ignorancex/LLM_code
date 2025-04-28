"""
A Python module which computes observable properties of variable relativistic discs.  
The model itself was presented in the paper "Extending the theory of propagating fluctuations: 
the first fully relativistic treatment and analytical Fourier-Green's functions", Andrew Mummery, 2023. 

Author: 
    Andrew Mummery, 
    Oxford University Physics,
    andrew.mummery@physics.ox.ac.uk

Paper:
    "Extending the theory of propagating fluctuations: the first fully relativistic treatment and analytical Fourier-Green's functions"
    Andrew Mummery, Oxford University Physics.
    Accepted to MNRAS, May 2023. 
    arXiv:XXXX.XXXXX

git repo:
    https://github.com/andymummeryastro/GR_prop_fluc 

"""
import numpy as np
from scipy.special import iv, kv, hyp2f1, gamma
from scipy.integrate import simps
from tqdm import tqdm### Keeps track of progress as loops take a while to run.   

def main():
    ### If ran as main will make some plots similar to section 6 of the paper.  
    import matplotlib.pyplot as plt
    import warnings 
    warnings.filterwarnings("ignore", module="matplotlib")


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['font.size'] = 20
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['figure.figsize'] = [12, 9]
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.bbox'] = 'tight'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    for a in [-0.9, 0, 0.5, 0.9]:
        f, Sh, Ss, Chs = GetVariableProperties(a, 3, 5, N_f=100, f_min=1e-5, f_max=1000, r_max=30, N_r=50)

        ax.semilogx(f, np.angle(Chs), label='Black hole spin = %.2f'%a)
        ax2.semilogx(f, abs(Chs)**2/(Sh * Ss), label='Black hole spin = %.2f'%a)
        ax3.loglog(f, (Sh * f), label='Black hole spin = %.2f'%a)
        color = next(ax4._get_lines.prop_cycler)['color']
        ax4.loglog(f[np.angle(Chs)>0], +np.angle(Chs[np.angle(Chs)>0])/f[np.angle(Chs)>0], 'o', label='Black hole spin = %.2f'%a, c=color)
        ax4.loglog(f[np.angle(Chs)<0], -np.angle(Chs[np.angle(Chs)<0])/f[np.angle(Chs)<0], 'x', c=color)


    ax.axhline(0, ls='--', c='k')
    ax.set_ylabel('Hard-soft phase lag, rad')
    ax.set_xlabel(r'Fourier frequency, Hz $(M_{\rm BH} = 10 M_\odot$)')
    ax.legend()

    ax2.set_ylabel('Hard-soft coherence')
    ax2.set_xlabel(r'Fourier frequency, Hz $(M_{\rm BH} = 10 M_\odot$)')
    ax2.set_ylim(top=1.00)
    ax2.legend()

    ax3.set_ylabel('Hard band power spectrum * frequency')
    ax3.set_xlabel(r'Fourier frequency, Hz $(M_{\rm BH} = 10 M_\odot$)')
    ax3.legend()

    ax4.set_ylabel(r'Hard-soft time delay, seconds $(M_{\rm BH} = 10 M_\odot)$')
    ax4.set_xlabel(r'Fourier frequency, Hz $(M_{\rm BH} = 10 M_\odot$)')
    ax4.legend()

    plt.show()

def GetVariableProperties(a, gammaS, gammaH, f_min=0.001, f_max=1000, N_f = 100, mu=0.5, alpha=0.1, HoR=0.1, M=10, r_max=50, N_r=100):
    """
    Returns the hard state power spectrum, soft state power spectrum, and cross spectrum between the hard and soft bands. 
    Also returns the frequencies at which these functions 

    INPUTS: 
        a = black hole spin (dimensionless).
        gammaS = emissivity index of soft band (dimensionless). 
        gammaH = emissivity index of hard band (dimensionless). 
        f_min, f_max = frequency lower limit (Hz), upper limit (Hz). 
        N_f = Number of frequncies sampled between f_min and f_max (dimensionless). Returns logarithmically spaced range. 
        mu = Stress tensor index (dimensionless). 
        alpha = Shakura-Sunyaev alpha parameter of the flow (dimensionless). 
        HoR = disc aspect ratio H/R (dimensionless). 
        M = Black hole mass (solar masses). 
        r_max = outer edge of disc (gravitational radii). 
        N_r = number of radii each integrand is sampled at (dimensionless). Integrals computed in linearly spaced radial range. 
    
    RETURNS:
        f = frequncies functions evaluated at (Hz). 
        Sh = hard state power spectrum (arbritrary). 
        Ss = soft state power spectrum (arbritrary). 
        Chs = hard-soft cross spectrum (arbritrary). 

    NOTES: 
        While the various quantities are in arbritrary units, the ratio C = |Chs|^2/(Sh * Ss) is correctly dimensionless (0 < C < 1). 

        If the user wants to modify the functional forms of the emissivity indices, or input surface density power, that is possible below this function. 
    """

    N_V = alpha * HoR**2 * 3e8**3/(6.67e-11 * M * 2e30)
    N_K = N_V/(alpha*HoR**2)
    alp = (3 - 2*mu)/4

    f = np.logspace(np.log10(f_min), np.log10(f_max), N_f)

    r1s = np.linspace(get_isco(a)+0.1, r_max, N_r)
    r2s = np.linspace(get_isco(a)+0.1, r_max, N_r)

    M = np.zeros((len(r1s), len(r2s), len(f)), dtype=np.cdouble)

    for i, r1 in enumerate(tqdm(r1s, desc="Generating local cross spectra: ", postfix="Note, iterations speed up.")):
        for j, r2 in enumerate(r2s):
            
            xp1i = np.linspace(get_isco(a) + 0.001, r1, 100)
            xp1o = np.linspace(r1, 101, 100)
            xpio = np.linspace(r1+0.01, r2, 100)
            xpoo = np.linspace(r2, 101, 100)
        
            for k, v in enumerate(f):            
                if r1 == r2:
                    M[i, j, k] = simps(integrand_in(2*r1/get_isco(a), 2*xp1i/get_isco(a), v, alp, a, N_K, N_V), 2*xp1i/get_isco(a)) + simps(integrand_out(2*r1/get_isco(a), 2*xp1o/get_isco(a), v, alp, a, N_K, N_V), 2*xp1o/get_isco(a))
                elif r1 < r2:
                    M[i, j, k] = simps(cm_integrand_in_in(2*r1/get_isco(a), 2*r2/get_isco(a), 2*xp1i/get_isco(a), v, alp, a, N_K, N_V), 2*xp1i/get_isco(a)) + simps(cm_integrand_in_out(2*r1/get_isco(a), 2*r2/get_isco(a), 2*xpio/get_isco(a), v, alp, a, N_K, N_V), 2*xpio/get_isco(a)) + simps(cm_integrand_out_out(2*r1/get_isco(a), 2*r2/get_isco(a), 2*xpoo/get_isco(a), v, alp, a, N_K, N_V), 2*xpoo/get_isco(a))
                else:
                    M[i, j, k] = np.conj(M[j, i, k])


    Sh = np.zeros(len(f), dtype=np.cdouble)
    Ss = np.zeros(len(f), dtype=np.cdouble)
    Chs = np.zeros(len(f), dtype=np.cdouble)

    for k, v in enumerate(f):    
        F1 = np.zeros(len(r1s), dtype=np.cdouble)
        F2 = np.zeros(len(r1s), dtype=np.cdouble)
        F3 = np.zeros(len(r1s), dtype=np.cdouble)

        for i, r1 in enumerate(r1s):
            F1[i] = simps(h(r1, a, gammaH) * h(r2s, a, gammaH) * M[i, :, k], r2s)
            F2[i] = simps(s(r1, a, gammaS) * s(r2s, a, gammaS) * M[i, :, k], r2s)
            F3[i] = simps(h(r1, a, gammaH) * s(r2s, a, gammaS) * M[i, :, k], r2s)        
                                
        Sh[k] = simps(F1, r1s)
        Ss[k] = simps(F2, r1s)
        Chs[k] = simps(F3, r1s)

    return f, Sh, Ss, Chs


######## Editting input power functional form will modify results. 
def input_power(x, f, a, N_K):
    fK = 1/((0.5 * get_isco(a) * x)**1.5 * 2 * np.pi) * N_K * 0.01
    return 2/np.pi * fK/(fK**2 + f**2)

######## Editting the functional form of the hard and soft emissivity profiles also modifies results.  

######## Hard band emissivity profile
def h(r, a, gammaH=3.5):
    return (r/get_isco(a))**-gammaH * (1 - (get_isco(a)/r)**0.5)

######## Soft band emissivity profile
def s(r, a, gammaS=2.0):
    return (r/get_isco(a))**-gammaS * (1 - (get_isco(a)/r)**0.5)


#### The location of the ISCO for general dimensionless spin parameter -1 < a < 1. 
def get_isco(a):
    Z_1 = 1 + (1-a**2)**(1/3) * ((1+a)**(1/3) + (1-a)**(1/3)) 
    Z_2 = np.sqrt(3*a**2 + Z_1**2)
    return (3 + Z_2 - np.sign(a) * np.sqrt((3-Z_1)*(3 + Z_1 + 2 * Z_2)))


#### The Relativistic Fourier-Greens function for x > x0. 
def FGF_GR_GT(x, x0, f, alpha=1/2, a=0):
    beta = (1 + 1j)*np.sqrt(np.pi*f)
    rI = get_isco(a)
    mu = (3-2*alpha)/4
    nu = 1/(4*alpha)
    
    fac = (2**0.5/rI**1.5 * (1-2/x0))**0.5  * (x0**(-mu/2) * np.exp(-1/x0))## final part to take w -> w_Newton
    
    R = 2**(alpha - 2)/(alpha*(alpha-1)) * np.pi**0.5 * gamma(2-alpha)/gamma(3/2 - alpha)
    
    fa = x**alpha/(2*alpha) * (1-2/x)**0.5 - x**(alpha-1)/(2*alpha*(alpha-1)) * (1-2/x)**0.5 * (hyp2f1(1, 3/2 - alpha, 2 - alpha, 2/x))
    fa += R
    fa/=fac
    fa0 = x0**alpha/(2*alpha) * (1-2/x0)**0.5 - x0**(alpha-1)/(2*alpha*(alpha-1)) * (1-2/x0)**0.5 * (hyp2f1(1, 3/2 - alpha, 2 - alpha, 2/x0))
    fa0 += R
    fa0/=fac
    
    q = x**0.25 * np.sqrt(x**-alpha * fa) * np.exp(1/(2*x)) * (1-2/x)**(5/4 - 3/(8*alpha))
    qlnp = (1-2*alpha)/(4*x) - 1/(2*x**2) + (10*alpha - 3)/(4*alpha*x**2) * 1/(1-2/x) + 1/(2*fa*fac) * 1/2 * x**(alpha-1) * np.sqrt(1-2/x)
    
    B = 2 * iv(1/(4*alpha), beta*fa0) * kv(1/(4*alpha), beta*fa)
    Bp = - 2 * iv(1/(4*alpha), beta*fa0) * (kv(1/(4*alpha)+1, beta*fa) + kv(1/(4*alpha)-1, beta*fa))
    D = (fa0/fa)**(2*nu) * (qlnp - nu/(fac*fa) * 1/2 * x**(alpha-1)*np.sqrt(1-2/x))/(qlnp + nu/(fac*fa) * 1/2 * x**(alpha-1)*np.sqrt(1-2/x))
    norm = nu * (fa0/fa)**nu * 1/q * 1/(qlnp + nu*1/(fac*fa) * 1/2 * x**(alpha-1)*np.sqrt(1-2/x))
    return norm * q * (beta/(2*fac) * 1/2 * x**(alpha-1)*np.sqrt(1-2/x) * Bp + qlnp * B) - D * np.exp(-(100 * f * (get_isco(a) * 0.5 * x0)**0.75)**2.0)

#### The Relativistic Fourier-Greens function for x < x0. 
def FGF_GR_LT(x, x0, f, alpha=1/2, a=0):
    beta = (1 + 1j)*np.sqrt(np.pi*f)
    rI = get_isco(a)
    mu = (3-2*alpha)/4
    nu = 1/(4*alpha)
    
    fac = (2**0.5/rI**1.5 * (1-2/x0))**0.5  * (x0**(-mu/2) * np.exp(-1/x0))## final part to take w -> w_Newton
    
    R = 2**(alpha - 2)/(alpha*(alpha-1)) * np.pi**0.5 * gamma(2-alpha)/gamma(3/2 - alpha)
    
    fa = x**alpha/(2*alpha) * (1-2/x)**0.5 - x**(alpha-1)/(2*alpha*(alpha-1)) * (1-2/x)**0.5 * (hyp2f1(1, 3/2 - alpha, 2 - alpha, 2/x))
    fa += R
    fa/=fac
    fa0 = x0**alpha/(2*alpha) * (1-2/x0)**0.5 - x0**(alpha-1)/(2*alpha*(alpha-1)) * (1-2/x0)**0.5 * (hyp2f1(1, 3/2 - alpha, 2 - alpha, 2/x0))
    fa0 += R
    fa0/=fac
    
    q = x**0.25 * np.sqrt(x**-alpha * fa) * np.exp(1/(2*x)) * (1-2/x)**(5/4 - 3/(8*alpha))
    qlnp = (1-2*alpha)/(4*x) - 1/(2*x**2) + (10*alpha - 3)/(4*alpha*x**2) * 1/(1-2/x) + 1/(2*fa*fac) * 1/2 * x**(alpha-1) * np.sqrt(1-2/x)
    
    B = 2 * kv(1/(4*alpha), beta*fa0) * iv(1/(4*alpha), beta*fa)
    Bp = + 2 * kv(1/(4*alpha), beta*fa0) * (iv(1/(4*alpha)+1, beta*fa) + iv(1/(4*alpha)-1, beta*fa))
    D = 0

    norm = nu * (fa0/fa)**nu * 1/q * 1/(qlnp + nu*1/(fac*fa) * 1/2 * x**(alpha-1)*np.sqrt(1-2/x))
    return norm * q * (beta/(2*fac) * 1/2 * x**(alpha-1)*np.sqrt(1-2/x) * Bp + qlnp * B) - D * np.exp(-(100 * f * (get_isco(a) * 0.5 * x0)**0.75)**2.0)


#### The various integrands which are integrated over in the function GetVariableProperties(). 
def integrand_in(x, xp, f, alpha, a, N_K, N_V):
    g = FGF_GR_GT(x, xp, f/N_V, alpha, a)
    sa = input_power(xp, f, a, N_K)
    return (2/get_isco(a)) * 1/xp**2 * abs(g)**2 * sa 

def integrand_out(x, xp, f, alpha, a, N_K, N_V):
    g = FGF_GR_LT(x, xp, f/N_V, alpha, a)
    sa = input_power(xp, f, a, N_K)
    return (2/get_isco(a)) * 1/xp**2 * abs(g)**2 * sa 

def cm_integrand_in_in(x1, x2, xp, f, alpha, a, N_K, N_V):
    g1 = FGF_GR_GT(x1, xp, f/N_V, alpha, a)
    g2 = FGF_GR_GT(x2, xp, f/N_V, alpha, a)
    sa = input_power(xp, f, a, N_K)
    return (2/get_isco(a)) * 1/xp**2 * g1 * np.conj(g2) * sa 
    
def cm_integrand_in_out(x1, x2, xp, f, alpha, a, N_K, N_V):
    g1 = FGF_GR_LT(x1, xp, f/N_V, alpha, a)
    g2 = FGF_GR_GT(x2, xp, f/N_V, alpha, a)
    sa = input_power(xp, f, a, N_K)
    return (2/get_isco(a)) * 1/xp**2 * g1 * np.conj(g2) * sa 

def cm_integrand_out_out(x1, x2, xp, f, alpha, a, N_K, N_V):
    g1 = FGF_GR_LT(x1, xp, f/N_V, alpha, a)
    g2 = FGF_GR_LT(x2, xp, f/N_V, alpha, a)
    sa = input_power(xp, f, a, N_K)
    return (2/get_isco(a)) * 1/xp**2 * g1 * np.conj(g2) * sa 


if __name__ == "__main__":
    main()

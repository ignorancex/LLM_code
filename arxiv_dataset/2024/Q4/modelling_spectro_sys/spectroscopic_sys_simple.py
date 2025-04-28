from numpy import log,pi,exp,cos,sin,tan,sqrt
import numpy as np
import random
from scipy.special import erfinv
from argparse import ArgumentParser
import warnings

random.seed(2143178)
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--redshift", help="the redshift of the galaxy mock", type=float, default=None, required=True,)
    parser.add_argument("--size", help="the length of the galaxy mock", type=int, default=None, required=True)
    parser.add_argument("--source", help="systematics measurements comes from which survey? SDSS or DESI or any. If you chose any, you need to specify the numbers (see the tutorial ipynb)", type=str, default='DESI', required=None, choices=['DESI','SDSS','any'])
    parser.add_argument("--geometry", help="are galaxies in box or survey-like geometry? box or survey", type=str, default='box', required=None, choices=['box','survey'])
    parser.add_argument("--tracer", help="the tracer type of the mock: LRG and QSO have redshift uncertainty, ELGs have redshift uncertainty and catastrophics", type=str, default='ELG',required=True, choices=['LRG','QSO','ELG'])
    parser.add_argument("--fcatas", help="the catastrophics failure rate. 0.26 for DESI ELG (means 0.26%), 1 for the upper limit of redshift surveys", type=float, default=None, required=None)
    parser.add_argument("--zerr", help="the redshift uncertainty. ~80km/s for LRGs, ~10km/s for ELGs, >100km/s for QSOs", type=float, default=None, required=None)
    parser.add_argument("--output", help="the name of the output Delta_z catalogue", type=str, default=None, required=True)
    args = None
    args = parser.parse_args()
    return args
args = parse_args()

z       = args.redshift
size    = args.size
source  = args.source
geometry= args.geometry
tracer  = args.tracer
output  = args.output

# we use inverse transform sampling to generate random numbers with Gaussian, Lorentzian and lognormal profiles from uniform random numbers
def inv_trans(uniform, func='G', sigma=None):
    if func == 'G': # Gaussian profile
        half    = int(len(uniform)/2)
        invfunc = np.append(sigma*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))
    elif func == 'L': # Lorentzian profile
        invfunc = tan((uniform-1/2)*pi)*sigma
    elif func == 'lnG': # lognormal profile
        invfunc = 3321/500-exp((erfinv(1-200/99*uniform)*243*np.sqrt(2)+657)/1000)
    return invfunc   

# the actual specroscopic systematics implementation:
## we need the extra sampling for Lorentzian and lognormal profiles to force random numbers distribute within a physical range
def spec_systematics(uniform, func='L', sigma=None, maxdv=None, dvcatas=1000, dvcatasmax=10**5.65):
    Delta_v   = inv_trans(uniform, func=func, sigma=sigma)
    if   func == 'L':
        outliers     = abs(Delta_v)>maxdv
        while np.sum(outliers)>0:
            Delta_v[outliers] =  inv_trans(np.random.rand(np.sum(outliers)), func=func, sigma=sigma) 
            outliers = abs(Delta_v)>maxdv
    elif func == 'lnG':
        dv_tot       = Delta_v.copy()
        outliers     = (dv_tot<np.log10(dvcatas))|(dv_tot>np.log10(dvcatasmax))|(~np.isfinite(dv_tot))
        while np.sum(outliers)>0:
            dv_tot[outliers] = inv_trans(np.random.rand(np.sum(outliers)), func=func, sigma=sigma) 
            outliers = (dv_tot<np.log10(dvcatas))|(dv_tot>np.log10(dvcatasmax))|(~np.isfinite(dv_tot))  
        Delta_v      = 10**np.append(dv_tot[:int(len(dv_tot)/2)],-dv_tot[int(len(dv_tot)/2):])
        random.shuffle(Delta_v)
    return Delta_v

# generate the uniform random numbers
if size%2 ==1: size = size -1 
uniform  = np.random.rand(size)
# Implement the box
if geometry == 'box':
    ## for LRGs
    if tracer == 'LRG':
        print('only redshift uncertainty vsmearing on peculiar velocity will be implemented')
        if args.zerr is not None:
            ### any redshift uncertainty you specify
            dv    = spec_systematics(uniform, func='G', sigma=args.zerr)
            if (args.zerr<20)|(args.zerr>90): warnings.warn('LRG Gaussian redshift uncertainty at 0.2<z<1.1 should be 20<zerr<90 km/s')
        else:
            ### for SDSS LRGs
            if source == 'SDSS':
                #### for different SDSS LRGs redshifts
                if   (0.2<z)&(z<=0.33):  dv = spec_systematics(uniform, func='G', sigma=21.2)
                elif (0.33<z)&(z<=0.43): dv = spec_systematics(uniform, func='G', sigma=27.9)
                elif (0.43<z)&(z<=0.51): dv = spec_systematics(uniform, func='G', sigma=39.0)
                elif (0.51<z)&(z<=0.6):  dv = spec_systematics(uniform, func='G', sigma=43.4)
                elif (0.6<z)&(z<=0.8):   dv = spec_systematics(uniform, func='G', sigma=83.5)
                elif (0.8<z)&(z<=1.0):   dv = spec_systematics(uniform, func='G', sigma=86.1)
                elif (z<0.2)|(z>1.0): raise ValueError("redshift for SDSS LRG-like mocks should be at 0.2<z<1.0")
            elif source == 'DESI':
                #### for different DESI LRGs redshifts
                if   (0.4<z)&(z<=0.6):   dv = spec_systematics(uniform, func='G', sigma=37.2)
                elif (0.6<z)&(z<=0.8):   dv = spec_systematics(uniform, func='G', sigma=66.6)
                elif (0.8<z)&(z<=1.1):   dv = spec_systematics(uniform, func='G', sigma=85.6)
                elif (z<0.4)|(z>1.2): raise ValueError("redshift for DESI LRG-like mocks should be at 0.4<z<1.1")
    elif tracer == 'ELG':
        print('redshift uncertainty vsmearing on peculiar velocity and redshift catastrophical failure will be implemented')
        if (args.zerr is None)&(args.fcatas is None):
            if source == 'SDSS':      raise ValueError("SDSS ELG-like spectroscopic systematics are not ready")
            elif source == 'DESI':
                # implement redshift uncertainty
                if   (0.8<z)&(z<=1.1):   vsmear = spec_systematics(uniform, func='L', sigma=9.3, maxdv=400)
                elif (1.1<z)&(z<=1.6):   vsmear = spec_systematics(uniform, func='L', sigma=13.4,maxdv=400)
                elif (z<0.8)|(z>1.6): raise ValueError("redshift for DESI ELG-like mocks should be at 0.8<z<1.6")
                # implement catastrophics 
                vcatas  = np.zeros(size)
                Nfail   = int(size*0.26/100)
                inds    = random.sample(range(0, size), Nfail)
                vcatas[inds] = spec_systematics(uniform[inds], func='lnG')
                dv      = vsmear + vcatas
        elif (args.zerr is not None)&(args.fcatas is not None):
            # implement redshift uncertainty with assigned redshift
            vsmear = spec_systematics(uniform, func='L', sigma=args.zerr, maxdv=200)
            if (args.zerr<9)|(args.zerr>15): warnings.warn('ELG Lorentzian redshift uncertainty at 0.8<z<1.6 should be 9<zerr<15 km/s')
            # implement catastrophics 
            vcatas  = np.zeros(size)
            Nfail   = int(size*args.fcatas/100)
            inds    = random.sample(range(0, size), Nfail)
            vcatas[inds] = spec_systematics(uniform[inds], func='lnG')
            dv      = vsmear + vcatas
        else: raise ValueError("Provide zerr and fcatas simulteneously for ELGs")
    elif tracer == 'QSO':
        print('only redshift uncertainty+velocity shift vsmearing on peculiar velocity will be implemented')
        if args.zerr is not None:
            ### any redshift uncertainty you specify
            dv    = spec_systematics(uniform, func='G', sigma=args.zerr)
            if (args.zerr<80)|(args.zerr>300): warnings.warn('QSO Lorentzian redshift uncertainty+velocity shift at 0.8<z<2.1 should be 80<zerr<300 km/s')
        else:    
            if source == 'SDSS':      raise ValueError("SDSS QSO-like spectroscopic systematics are not ready")
            elif source == 'DESI':
                # implement redshift uncertainty
                if   (0.8<z)&(z<=1.1):   dv = spec_systematics(uniform, func='L', sigma=100, maxdv=2000)
                elif (1.1<z)&(z<=1.6):   dv = spec_systematics(uniform, func='L', sigma=78, maxdv=2000)
                elif (1.6<z)&(z<=2.1):   dv = spec_systematics(uniform, func='L', sigma=273, maxdv=2000)
                elif (z<0.8)|(z>2.1): raise ValueError("redshift for DESI QSO-like mocks should be at 0.8<z<2.1")
    np.savetxt(output,dv,header='Delta-v (km/s), add it to the peculiar velocity of your galaxy mocks')
else:
    if source == 'SDSS':    raise ValueError("SDSS-like spectroscopic systematics in lightcones are not ready")
    elif source == 'DESI':  print('DESI-like spectroscopic systematics (including WEIGHT_FKP recomputing) are implemented in the subdirectory ./DESI_lighcone')


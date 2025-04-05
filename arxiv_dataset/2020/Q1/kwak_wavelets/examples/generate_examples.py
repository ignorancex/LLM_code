import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import jn_zeros

FileName = "HEPData-ins1609773-v1-Table_9.csv"
InputFile = pd.read_csv(FileName, comment='#') # Read input file with Pandas

# -- Store columns as arrays --
Mass = InputFile['M(gamma gamma) [GeV]'].values  # Diphoton invariant mass
MassLow = InputFile['M(gamma gamma) [GeV] LOW'].values  # Lowest mass value
MassHigh = InputFile['M(gamma gamma) [GeV] HIGH'].values  # Highest mass value
Nevents = InputFile['Number of events [Number of events]'].values  # No. of events
Stat_plus = InputFile['stat +'].values # Statistical uncertainty (+)
Stat_minus = InputFile['stat -'].values # Statistical uncertainty (-)

# -- Parameters --
SqrtS = 13000 # Com energy [GeV]

X = Mass
Xmin = np.min(X); Xmax = np.max(X)
Y = Nevents
Sigma = np.subtract(Stat_plus, Stat_minus)*0.5
BinStep = min(np.subtract(X[1:], X[:-1]))  # step-size
Nbins = int((Xmax-Xmin)/BinStep+1)  # Binning of the output data set


# ------------- Find the best fit parameters -------------
# Model function for the fit
def fitfunction_diphoton(x, n, a, b):
    return n * pow(1 + pow(x, 1.0/3.0), b) * pow(x, a)

X_fit = X/SqrtS  # Scale mass by SqrtS
Y_fit = np.array(Y, dtype=int)  # No. of events must be integer-type
Sigma_fit = Sigma
params, cov = curve_fit(fitfunction_diphoton, X_fit, Y_fit,
                        sigma=Sigma_fit, maxfev=2000)
# ---------------------------------------------------------


# -- Preprocess data --
# Bin the data in uniform intervals of length BinStep.
# The length of the data array must be a power of two.
pow2 = np.ceil(np.log2(Nbins))  # The length of the data must be the next largest power of two
if pow2!=np.log2(Nbins):
    Nbins = int(2**pow2)

X_array = np.zeros(Nbins, dtype=float)
Y_array = np.zeros(Nbins, dtype=int)
Sigma_array = np.zeros(Nbins, dtype=float)

Xbin = Xmin
for i in range(Nbins):
    if Xbin in X:
        ix = np.where(X==Xbin)
        Ybin = Y[ix]
        Sbin = Sigma[ix]
    X_array[i] = Xbin
    Y_array[i] = Ybin
    Sigma_array = Sbin
    Xbin += BinStep

X_scaled = X_array/SqrtS  # Scalled X-data array
Xmin_scaled = np.min(X_scaled)
Xmax_scaled = np.max(X_scaled)
BinStep_scaled = BinStep/SqrtS

# -- Find the binned hypothesis according to the scaled data and the best-fit parameters --
n, a, b = params[0], params[1], params[2]
Hypothesis_scaled = np.vectorize(fitfunction_diphoton)(X_scaled, n, a, b)



# --------------- Functions to generate the examples -------------
# -- Simple Examples (Bumps and Dips) --
def XtoY(x):
    y = (x-Xmin_scaled)/(Xmax_scaled-Xmin_scaled)
    return y

def flat_wide(x):
    y = XtoY(x)
    amplitude = 23.0
    width = 0.05
    center = 0.23
    Signal = amplitude*np.exp(-0.5*pow((y-center)/width, 2))
    SigPLusHypo = fitfunction_diphoton(x, n, a, b)+Signal
    return SigPLusHypo, Signal

def flat_narrow(x):
    y = XtoY(x)
    amplitude = 41.0
    width = 0.013
    center = 0.17
    Signal = amplitude*np.exp(-0.5*pow((y-center)/width, 2))
    SigPLusHypo = fitfunction_diphoton(x, n, a, b)+Signal
    return SigPLusHypo, Signal

def flat_bumpdip(x):
    y = XtoY(x)
    amplitude1 = -31.5
    width1 = 0.015
    center1 = 0.17
    amplitude2 = 21.0
    width2 = 0.02
    center2 = 0.21
    bump1 = amplitude1*np.exp(-0.5*pow(float(y-center1)/width1, 2))
    bump2 = amplitude2*np.exp(-0.5*pow(float(y-center2)/width2, 2))
    Signal = bump1+bump2
    SigPLusHypo = fitfunction_diphoton(x, n, a, b)+Signal
    return SigPLusHypo, Signal

def flat_osc(x):
    y = XtoY(x)
    scaling = np.sqrt(fitfunction_diphoton(x, n, a, b))**(-1)
    amplitude = 1.7*scaling
    frequency = 9.6
    start = 0.07
    cutofflength = 1.0
    if y < start:
        SigPLusHypo = fitfunction_diphoton(x, n, a, b)*1
        Signal = SigPLusHypo-fitfunction_diphoton(x, n, a, b)
        return SigPLusHypo, Signal
    else:
        theta = (y-start)*2.*np.pi*frequency
        SigPLusHypo = fitfunction_diphoton(x, n, a, b)*(1.+amplitude*np.exp(-0.5*(y/cutofflength)**2)*(1.-np.cos(theta)))
        Signal = SigPLusHypo-fitfunction_diphoton(x, n, a, b)
        return SigPLusHypo, Signal

# -- Kaluza-Klein Model --
kkamplitudefactor = 1.0
gmpl = 2.435e18  #GeV (Reduced Planck mass)

besselzeros = 20
j1zeros = jn_zeros(1, besselzeros)

def gaussian(x, sigma, mu):
    return np.exp(-(x - mu)**2 / (2.0 * sigma**2))

def xtoi(x, xmin, delta):
    i = float(x-xmin)/delta
    return int(round(i))

def KK_mg(x, m1, gamma1, n):
    y = XtoY(x)
    cutofflength = 0.3
    rho = 0.28
    widthpower = 1
    Kdiv = np.sqrt(gamma1)/np.sqrt(m1*rho*j1zeros[0]**2)
    Krc = 1./np.pi*np.log(Kdiv*gmpl/m1*j1zeros[0])
    mass = m1*j1zeros[n-1]/j1zeros[0]
    width = gamma1*(j1zeros[n-1]/j1zeros[0])**widthpower
    mgg = x*SqrtS
    KKmgg = gaussian(mgg, width, mass)*np.exp(-0.5*(y/cutofflength)**2)
    return KKmgg

def KK_signal(m1, g1, Nlist):
    signal = np.zeros(Nbins)
    for i in range(Nbins):
        x = X_scaled[i]
        y = XtoY(x)
        cutofflength = 0.3
        KK_sum = 0
        for n, nsize in enumerate(Nlist):
            KK_sum += nsize*KK_mg(x, m1, g1, n+1)
        signal[i] = KK_sum
    return signal

def kk_model(mKK1):
    background = Hypothesis_scaled
    baseline0 = [0.13, 0.5, 2.2, 5, 6, 9, 9, 6]
    baseline = np.ones(besselzeros)
    baseline[:len(baseline0)] = baseline0
    baseline *= kkamplitudefactor

    X = lambda i : mKK1*(j1zeros[i]/j1zeros[0])/SqrtS
    Xlist = [X(i) for i in range(besselzeros)]

    Nlist = []
    for ix, X in enumerate(Xlist):
        i = xtoi(X, Xmin_scaled, BinStep_scaled)
        if i < Nbins:
            Nb = background[i]*baseline[ix]
        else:
            Nb = background[Nbins-1]*baseline[ix]
        Nlist.append(Nb)

    Signal = KK_signal(mKK1, np.sqrt(mKK1), Nlist)
    SigPlusHypo = np.add(background, Signal)
    return SigPlusHypo, Signal
# ----------------------------------------------------------------


# -- Obtain the arrays for each example --
Wide_SigPlusHypo, Wide_Sig = np.vectorize(flat_wide)(X_scaled)
Narrow_SigPlusHypo, Narrow_Sig = np.vectorize(flat_narrow)(X_scaled)
BumpDip_SigPlusHypo, BumpDip_Sig = np.vectorize(flat_bumpdip)(X_scaled)
Osc_SigPlusHypo, Osc_Sig = np.vectorize(flat_osc)(X_scaled)
# -- Kaluza-Klein --
mKK1 = 18.0**2
KK_SigPlusHypo, KK_Sig = kk_model(mKK1)

# -- Sample from a Poisson distribution --
def generatePoisson(data_array, seed):
    np.random.RandomState(seed)
    pseudodata = [np.random.poisson(i) for i in data_array]
    return pseudodata

ran_Wide_SigPlusHypo = generatePoisson(Wide_SigPlusHypo, seed=123)
ran_Narrow_SigPlusHypo = generatePoisson(Narrow_SigPlusHypo, seed=124)
ran_BumpDip_SigPlusHypo = generatePoisson(BumpDip_SigPlusHypo, seed=125)
ran_Osc_SigPlusHypo = generatePoisson(Osc_SigPlusHypo, seed=126)
ran_KK_SigPlusHypo = generatePoisson(KK_SigPlusHypo, seed=127)
ran_Hypothesis = generatePoisson(Hypothesis_scaled, seed=128)

# -- Assuming the error is Gaussian --
Wide_Sigma = np.sqrt(ran_Wide_SigPlusHypo)
Narrow_Sigma = np.sqrt(ran_Narrow_SigPlusHypo)
BumpDip_Sigma = np.sqrt(ran_BumpDip_SigPlusHypo)
Osc_Sigma = np.sqrt(ran_Osc_SigPlusHypo)
KK_Sigma = np.sqrt(ran_KK_SigPlusHypo)
Hypo_Sigma = np.sqrt(ran_Hypothesis)

# -- Output data to file --
Wide_Data =     {'M(gamma gamma) [GeV]':X_array, 'Nevents':ran_Wide_SigPlusHypo,
                 'Sigma':Wide_Sigma, 'Hypothesis':Hypothesis_scaled, 'Generating Function':Wide_Sig}
Narrow_Data =   {'M(gamma gamma) [GeV]':X_array, 'Nevents':ran_Narrow_SigPlusHypo,
                 'Sigma':Narrow_Sigma, 'Hypothesis':Hypothesis_scaled, 'Generating Function':Narrow_Sig}
BumpDip_Data =  {'M(gamma gamma) [GeV]':X_array, 'Nevents':ran_BumpDip_SigPlusHypo,
                 'Sigma':BumpDip_Sigma, 'Hypothesis':Hypothesis_scaled, 'Generating Function':BumpDip_Sig}
Osc_Data =      {'M(gamma gamma) [GeV]':X_array, 'Nevents':ran_Osc_SigPlusHypo,
                 'Sigma':Osc_Sigma, 'Hypothesis':Hypothesis_scaled, 'Generating Function':Osc_Sig}
KK_Data =       {'M(gamma gamma) [GeV]':X_array, 'Nevents':ran_KK_SigPlusHypo,
                 'Sigma':KK_Sigma, 'Hypothesis':Hypothesis_scaled, 'Generating Function':KK_Sig}
Null_Data =     {'M(gamma gamma) [GeV]':X_array, 'Nevents':ran_Hypothesis,
                 'Sigma':Hypo_Sigma, 'Hypothesis':Hypothesis_scaled}

Wide_tofile = pd.DataFrame(data=Wide_Data).to_csv("Wide.csv", index=False)
Narrow_tofile = pd.DataFrame(data=Narrow_Data).to_csv("Narrow.csv", index=False)
BumpDip_tofile = pd.DataFrame(data=BumpDip_Data).to_csv("BumpDip.csv", index=False)
Osc_tofile = pd.DataFrame(data=Osc_Data).to_csv("Oscillations.csv", index=False)
KK_tofile = pd.DataFrame(data=KK_Data).to_csv("KK.csv", index=False)
Null_tofile = pd.DataFrame(data=Null_Data).to_csv("Null.csv", index=False)

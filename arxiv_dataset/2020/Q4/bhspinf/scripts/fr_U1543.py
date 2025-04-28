import numpy as np
from convert_fK2fr import *

# 4U 1543-47

fout   = '../data/fr/fr_U1543.h5'
fpCF   = '../data/CF/CF_results_U1543.p'
fr2K   = '../data/fK/fK_U1543.h5'
GRflag = True

Nproc    = 4
r_min    = 0
r_max    = 12
fcol_err = [0.1, 0.2, 0.3]
fcol_val = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
fcolL    = 0.9
fcolR    = 2.4
atol     = 1e-3
aFe      = 0.3

# Gaussian smoothing kernels
r_GR,  i_GR  = 0,  0  # gGR(r,i)
r_dGR, i_dGR = 25, 0  # d/dr[gGR(r,i)]
r_NT         = 0      # gNT(r)
r_dNT        = 0      # d/dr[gNT(r)]

fr_analysis(fout=fout, fpCF=fpCF, fr2K=fr2K, GRflag=GRflag,
            r_min=r_min, r_max=r_max,
            fcol_val=fcol_val, fcol_err=fcol_err,
            fcolL=fcolL, fcolR=fcolR, atol=atol,
            aFe=aFe, Nproc=Nproc,
            r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)

import numpy as np
import schnell as snl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

t_obs = 1
f_ref = 63
nside = 64
obs_time = t_obs*365*24*3600.
freqs = np.linspace(10., 1010., 101)

dets = [snl.GroundDetector('Hanford',     46.4, -119.4, 171.8,
                           'data/aLIGO.txt'),
        snl.GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                           'data/aLIGO.txt'),
        snl.GroundDetector('Virgo',       43.6,   10.5, 116.5,
                           'data/Virgo.txt'),
        snl.GroundDetector('KAGRA',       36.3,  137.2, 225.0,
                           'data/KAGRA.txt')]

print("HL")
mc = snl.MapCalculator(dets[:2], f_pivot=f_ref)
nl_HL = mc.get_N_ell(obs_time, freqs, nside, no_autos=True)
print("HLV")
mc = snl.MapCalculator(dets[:3], f_pivot=f_ref)
nl_HLV = mc.get_N_ell(obs_time, freqs, nside, no_autos=True)
print("HLVK")
mc = snl.MapCalculator(dets[:4], f_pivot=f_ref)
nl_HLVK = mc.get_N_ell(obs_time, freqs, nside, no_autos=True)
print("HLVKa")
nl_HLVKa = mc.get_N_ell(obs_time, freqs, nside, no_autos=False)


ls = np.arange(3*nside)
plt.figure()
plt.plot(ls, (ls+0.5)*nl_HL, 'k--', label='LIGO')
plt.plot(ls, (ls+0.5)*nl_HLV, 'k:', label=' + Virgo')
plt.plot(ls, (ls+0.5)*nl_HLVK, 'k-', label=' + KAGRA')
plt.plot(ls, (ls+0.5)*nl_HLVKa, 'k-.', label=' + auto correlations')
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$(\ell+1/2)\,N_\ell$', fontsize=16)
plt.ylim([3E-20, 1E-10])
plt.xlim([1, 100])
plt.legend(loc='upper left', fontsize='x-large', frameon=False)
plt.gca().tick_params(labelsize="large")
plt.savefig("nl_LIGO.pdf", bbox_inches='tight')
plt.show()

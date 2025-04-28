import sys

sys.path.append("../")
from analyse import *

samples_plp = loadMC("../plp/long", "plp+plp+trivial+trivial")
samples_ppisn = loadMC("../ppisn/long", "ppisn+trivial+trivial")

m1, yaplp, ybplp, ycplp = best_fit_band_norms(samples_plp, "plp+plp+trivial+trivial", CL=0.68)
m1, yappisn, ybppisn, ycppisn = best_fit_band_norms(samples_ppisn, "ppisn+trivial+trivial", CL=0.68)

fig = figure(figsize=(4,3))
fill_between(m1, yaplp/5e3, ybplp/5e3, alpha=0.2, label=r"${\rm PLP}$")
plot(m1, ycplp/5e3, 'C0')
fill_between(m1, 1.5e-2*yappisn/5e3, 1.5e-2*ybppisn/5e3, alpha=0.2, label=r"${\rm PPISN}$")
plot(m1, 1.5e-2*ycppisn/5e3, 'C1')

yscale('log')
ylim(4e-14,1)
ylabel(r"${\rm d}R/{\rm d}m_1$")
xlabel("$m_1/M_\odot$")
legend()

fig.tight_layout()
fig.savefig("mass-function-best-fit.pdf")

avg, lvar, uvar = central_value(samples_ppisn.get1DDensity('mgap'))
print(f"MBHMG = {avg:7.2f} -{np.sqrt(2*lvar):7.2f}+{np.sqrt(2*uvar):7.2f}")

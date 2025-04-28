import sys
sys.path.append("..")
from analyse import *

sh0es = [73.04, 1.04]
planck = [67.4, 0.5]

samples_plp = loadMC('../plp-h0/long.card', 'plp+plp+planck+trivial')
densityH0_PLP = samples_plp.get1DDensityGridData('h0').normalize()

samples_ppisn = loadMC('../ppisn-h0/long', 'ppisn+planck+trivial')
densityH0 = samples_ppisn.get1DDensityGridData('h0').normalize()
densityMG = samples_ppisn.get1DDensityGridData('mgap').normalize()
density=samples_ppisn.get2DDensityGridData('h0','mgap').normalize()


fig, axs = subplots(2,2, sharex='col', sharey='row', gridspec_kw={"hspace": 0, "wspace": 0}, figsize=(4,4))
axs[0,0].plot(densityH0.x, densityH0.P)
axs[0,0].plot(densityH0_PLP.x, densityH0_PLP.P, label=r'${\rm PLP}$', color='C3')
axs[1,1].plot(densityMG.P, densityMG.x)
axs[1,0].contourf(*np.meshgrid(density.x,density.y),density.P, levels=density.getContourLevels((0.95,0.68,1e-10)))

axs[1,0].axvspan(sh0es[0]-sh0es[1], sh0es[0]+sh0es[1], color='C1')
axs[1,0].axvspan(planck[0]-planck[1], planck[0]+planck[1], color='C2')

axs[0,0].axvspan(sh0es[0]-sh0es[1], sh0es[0]+sh0es[1], color='C1')
axs[0,0].axvspan(planck[0]-planck[1], planck[0]+planck[1], color='C2')
axs[0,1].legend(
    [matplotlib.lines.Line2D([0], [0], color=f'C{i}') for i in range(4)],
    [r'${\rm This\ work}$', r'${\rm SH0ES}$', r'${\rm Planck\ 2018}$', r'{\rm PLP}'],
    loc='center'
)

axs[1,0].set_xlabel(r"$H_0\,/\,{\rm km}\,{\rm s}^{-1}\,{\rm Mpc}^{-1}$")
axs[1,0].set_ylabel(r"$M_{\rm BHMG}\,/\,M_{\odot}$")
axs[0,1].set_axis_off()
axs[0,0].get_yaxis().set_ticks([])
axs[1,1].get_xaxis().set_ticks([])
axs[1,0].set_xlim(20,150)
axs[1,0].set_ylim(20,120)

fig.savefig("h0-corner.pdf")


wavg = mergenumbers(np.array([sh0es, planck]))

avg, lvar, uvar = central_value(densityH0)
sig = -(avg - wavg[0])/np.sqrt(2*uvar + wavg[1]**2)
print(f"H0(PPISN) = {avg:7.2f} -{np.sqrt(2*lvar):7.2f}+{np.sqrt(2*uvar):7.2f} ({sig:7.2f} sigma)")


avg, lvar, uvar = central_value(densityH0_PLP)
sig = -(avg - wavg[0])/np.sqrt(2*uvar + wavg[1]**2)
print(f"H0(PLP) = {avg:7.2f} -{np.sqrt(2*lvar):7.2f}+{np.sqrt(2*uvar):7.2f} ({sig:7.2f} sigma)")

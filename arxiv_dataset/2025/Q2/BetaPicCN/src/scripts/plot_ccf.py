import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.table import Table

t=Table.read(paths.data/'ccf_data1.ecsv',format='ascii.ecsv')

fig, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_xlim(-50,50)
ax.set_ylim(0,30)

ax.set_xlabel('Velocity [$km/s$]',fontsize=16)
ax.set_ylabel('CCF [a.u.]',fontsize=16)

ax.plot(t['vel'],t['N11'],label="N=$10^{11}$")
ax.plot(t['vel'],t['N12'],label="N=$10^{12}$")
ax.plot(t['vel'],t['N13'],label="N=$10^{13}$")
ax.legend()
plt.draw()

plt.tight_layout()

fig.savefig(paths.figures/'ccf.pdf')

# PARS imports
import sys, os, time, pickle
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf
# matplotlib and numpy
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Computer Modern"],
    "font.size": 20,
	"figure.figsize": (7, 5.5)
})

# pre-compute Roche model volume versus PARS's omega
# and PARS's omega versus MESA's omega
sf.calcVA()
sf.calcom()

x = np.linspace(0, sf.omax, 1000)
y = sf.omega(x)
plt.plot(x, y, lw=3)
plt.xlabel(r'$\omega_{\rm M}\sqrt{1 - \frac{L}{L_{\rm Edd}}}$')
plt.ylabel(r'$\omega$')
plt.tight_layout()
plt.savefig('../../data/omega_plot.pdf')
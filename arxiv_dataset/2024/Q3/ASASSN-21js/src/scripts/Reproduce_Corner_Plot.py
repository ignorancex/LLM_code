import emcee
import corner
from os.path import isfile
import matplotlib.pyplot as plt
import paths

path = paths.data / "mcmc_20240208_backup_2e6.h5"

reader = emcee.backends.HDFBackend(path)
samples = reader.get_chain()

print('Emcee results loaded successfully.')

ndim = 11
labels = [r'$a_{ring, outer}$', r'$i$', r'$\phi$', r'$tr_{outer}$', r'$w_{outer}$', r'$a_{ring, inner}$', r'$tr_{inner}$', 
          r'$w_{inner}$', r'$y_{ring}$', r'$v$', r'$t_0$']

burnin2 = int(len(samples) * 0.2)

thin2 = int(len(samples) * 0.01)

flat_samples = reader.get_chain(discard = burnin2, flat = True, thin = thin2)

#draw corner plot
fig = corner.corner(flat_samples, labels=labels, label_kwargs={"fontsize":15}, quantiles = [0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}, title_fmt='.3f')

plt.savefig(paths.figures / f'asassn21js_corner_plot.png', facecolor = 'white', transparent = False, dpi = 300)


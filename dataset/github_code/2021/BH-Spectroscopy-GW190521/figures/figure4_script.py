import h5py, numpy
from matplotlib import pyplot, patches
from pycbc import conversions
from pycbc.results import scatter_histograms

# Load data from posterior files
kerr_file = '../posteriors/kerr/220_330/KERR-220_330-06MS.hdf'
fp_kerr = h5py.File(kerr_file, 'r')
kerr_mass = fp_kerr['samples/final_mass'][()]
kerr_spin = fp_kerr['samples/final_spin'][()]

nongr_file = '../posteriors/nongr/NONGR-220_330-06MS.hdf'
fp_nongr = h5py.File(nongr_file, 'r')
f220, tau220 = fp_nongr['samples/f220'][()], fp_nongr['samples/tau220'][()]
f330 = fp_nongr['samples/f330'][()] * (1 + fp_nongr['samples/delta_f330'][()])
tau330 = fp_nongr['samples/tau330'][()] * \
        (1 + fp_nongr['samples/delta_tau330'][()])

mass330 = conversions.final_mass_from_f0_tau(f330, tau330, l=3, m=3)
spin330 = conversions.final_spin_from_f0_tau(f330, tau330, l=3, m=3)
# Exclude unphysical values from conversions above
def exclude_regions(masses, spins):
    conds = (masses>=0) & (spins>=-1) & (spins<=1)
    return masses[conds], spins[conds]
nongr_mass330, nongr_spin330 = exclude_regions(mass330, spin330)

# Make figure
width_ratios = [3, 1]
height_ratios = [1, 3]

params = ['mass', 'spin']
nparams = len(params)
lbls={'mass':'redshifted final mass $(1+z)M_f$ [$M_\odot$]',
      'spin':'final spin $\chi_f$'}
plot_colors = ['navy', 'lightseagreen', 'yellowgreen']

fig, axis_dict = scatter_histograms.create_axes_grid(
            params, labels=lbls,
            width_ratios=width_ratios, height_ratios=height_ratios,
            no_diagonals=False)

all_samples = [{'mass':kerr_mass, 'spin':kerr_spin},
               {'mass':nongr_mass330, 'spin':nongr_spin330}]
legend_lbls = ['Kerr 220 + 330', 'Kerr with $\delta$(330)']

# Get the minimum and maximum of the mass for the plot limits
joint_masses = numpy.concatenate([s['mass'] for s in all_samples])
mins = {'mass':numpy.min(joint_masses), 'spin':0.2}
maxs = {'mass':numpy.max(joint_masses), 'spin':1}

handles = []
for si, samples in enumerate(all_samples):
    samples_color = plot_colors[si]

    # Plot 1D histograms
    for pi, param in enumerate(params):
        ax, _, _ = axis_dict[param, param]
        rotated = nparams == 2 and pi == nparams-1
        scatter_histograms.create_marginalized_hist(ax,
                samples[param], label=lbls[param],
                color=samples_color, fillcolor=None,
                linecolor=samples_color, title=False,
                rotated=rotated, plot_min=mins[param],
                plot_max=maxs[param], percentiles=[5,95])

    # Plot 2D contours
    ax, _, _ = axis_dict[('mass','spin')]
    scatter_histograms.create_density_plot(
                'mass', 'spin', samples, plot_density=False,
                plot_contours=True, percentiles=[90],
                contour_color=samples_color,
                xmin=mins['mass'], xmax=maxs['mass'],
                ymin=mins['spin'], ymax=maxs['spin'],
                ax=ax, use_kombine=False)

    handles.append(patches.Patch(color=samples_color, label=legend_lbls[si]))

fig.legend(loc=(0.48,0.13),
        handles=handles, labels=legend_lbls)
fig.set_dpi(250)
fig.savefig('Figure4.png')

import h5py, numpy
from matplotlib import pyplot
from pycbc.results import str_utils

# Load data from posterior files
kerr_file = '../posteriors/kerr/220_330/KERR-220_330-06MS.hdf'
fp_kerr = h5py.File(kerr_file, 'r')
amp330 = fp_kerr['samples/amp330'][()]

#Make figure
def plot_percentiles(ax, samples, color):
    plotp = numpy.percentile(samples, [5, 95])
    for val in plotp:
        ax.axvline(x=val, ls='dashed', color=color, lw=2, zorder=5)

def get_interval(samples):
    values_min, values_med, values_max = numpy.percentile(samples, [5, 50, 95])
    negerror = values_med - values_min
    poserror = values_max - values_med
    return '${0}$'.format(str_utils.format_value(
            values_med, negerror, plus_error=poserror))

fig = pyplot.figure(figsize=(6,5)) 
ax = fig.add_subplot(111)

amp330_color = 'navy'
fillcolor = 'lightsteelblue'
# prior: amp prior uniform in [0, 0.5), so prior pdf is 1/0.5 = 2
# between [0, 0.5)
priorpdf = [2, 2]
ax.plot([0, 0.5], priorpdf, color='gray', linestyle=':', label='prior',
        zorder=2, lw=2)
ax.hist(amp330, label='posterior',
        edgecolor=amp330_color, facecolor=fillcolor,
        bins=50, density=True,
        histtype='stepfilled', lw=2, zorder=1)
plot_percentiles(ax, amp330, amp330_color)

ax.set_xlim(0, 0.5)
ax.set_yticks([])
ax.set_yticklabels([])
ax.legend(loc='upper right')
ax.set_xlabel('amplitude ratio $A_{330}/A_{220}$', fontsize=14)

fig.savefig('Figure3.png', dpi=300, bbox_inches='tight')

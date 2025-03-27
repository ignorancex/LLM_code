from __future__ import print_function, division
import matplotlib
from matplotlib import rcParams
from matplotlib import rc
import matplotlib.pyplot as plt

# Setup plot styles
fontsize = 22
xylabelfs = fontsize + 2
linewidth = 2
tick_size = (8, 4)  # major, minor
tick_width = (1, 0.5)  # major, minor
#rc('font', family='Times')
rc('font', family='Times New Roman')
#rc('font',**{'family':'serif','serif':['Times New Roman']})
#rc('font',**{'family':'serif'})
# enforce true type instead of type3
#rcParams.update({'pdf.fonttype': 42})
#rc('text', usetex=True)
rcParams.update({'font.size': fontsize})
# added on 22/6/2018:
#matplotlib.rcParams['mathtext.fontset'] = 'stix'

rc('xtick', labelsize=fontsize)
rc('ytick', labelsize=fontsize)
rc('axes', linewidth=linewidth)
rc('patch', linewidth=linewidth)
rc('lines', linewidth=linewidth)
plt.rcParams['xtick.major.size'] = tick_size[0]
plt.rcParams['xtick.major.width'] = tick_width[0]
plt.rcParams['xtick.minor.size'] = tick_size[1]
plt.rcParams['xtick.minor.width'] = tick_width[1]
plt.rcParams['xtick.major.pad'] = '10'
if True:
    plt.rcParams['ytick.major.size'] = tick_size[0]
    plt.rcParams['ytick.major.width'] = tick_width[0]
    plt.rcParams['ytick.minor.size'] = tick_size[1]
    plt.rcParams['ytick.minor.width'] = tick_width[1]
#plt.rcParams['ytick.major.pad'] = '20'
rcParams.update({'lines.solid_capstyle': u'round'})
# from http://colorbrewer2.org/
blues = ['#bdd7e7', '#6baed6', '#2171b5', 'darkblue']
blues5 = ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']
blues6 = ['#eff3ff', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
greens = ['#edf8fb', '#b2e2e2', '#66c2a4', '#238b45', 'g']
greens4 = ['#edf8e9', '#bae4b3', '#74c476', '#238b45']
greens5 = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
reds5 = ['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000']
PuBuGn6 = ['#f6eff7', '#d0d1e6', '#a6bddb', '#67a9cf', '#1c9099', '#016c59']
YlGnBl6 = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494']
YlGnBl5 = ['#a1dab4', '#41b6c4', '#2c7fb8', '#253494',
           'blue']  # first was '#ffffcc']
oranges = ['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#e6550d', '#a63603']
oranges3 = ['#fee6ce', '#fdae6b', '#e6550d']
oranges4 = ['#feedde', '#fdbe85', '#fd8d3c', '#d94701']
YlGn5 = ['#ffffcc', '#c2e699', '#78c679', '#31a354', '#006837']
purples4 = ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#6a51a3']
magentas1 = ['#980043']
myblue1 = '#4CA0FF'

log10M_string = '\\log M\\,[h^{\\!\\!{-1}}\\mathrm{M}_{\\!\\!\\odot}\\!]'
log10M_string2 = '\\log M'
hMsun_string = 'h^{\\!\\!{-1}}\\mathrm{M}_{\\!\\!\\odot}'
log10Mmin_string = '\\log_{10}\\,M_\\mathrm{min}\\,[h^{\\!\\!{-1}}\\mathrm{M}_{\\!\\!\\odot}\\!]'

# use matplotlib 1.5 style rather than matplotlib 2 style
if matplotlib.__version__.startswith('2'):
    if True:
        plt.style.use('classic')

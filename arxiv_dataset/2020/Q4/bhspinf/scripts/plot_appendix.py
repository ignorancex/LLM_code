import numpy as np
import pickle
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator, NullFormatter, FixedLocator, FixedFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, SymLogNorm
from matplotlib.transforms import Affine2D
from pdf_trans import *
from gs_stats import *

#====================================================================================================
# COLLECT THE RESULTS
source = r"GRO J1655--40"

fpCF = '../data/CF/CF_results_J1655.p'
with open(fpCF, 'rb') as fp:
    fpick = pickle.load(fp)
    rCF   = fpick['r_grid']
    f_rCF = fpick['f_rCF'].pdf(rCF)

fh5 = '../data/fK/fK_J1655.h5'
f   = h5py.File(fh5, 'r')
# inputs
a        = f.get('a')[()]
b        = f.get('b')[()]
Nf       = f.get('Nf')[()]
NM       = f.get('NM')[()]
ND       = f.get('ND')[()]
Ni       = f.get('Ni')[()]
fcol_val = f.get('fcol_val')[()]
fcol_err = f.get('fcol_err')[()]
fcol_min = f.get('fcol_min')[()]
fcol_max = f.get('fcol_max')[()]
M_min    = f.get('M_min')[()]
M_max    = f.get('M_max')[()]
D_min    = f.get('D_min')[()]
D_max    = f.get('D_max')[()]
i_min    = f.get('i_min')[()]
i_max    = f.get('i_max')[()]
# pdfs
fcolCF   = f.get('fcolCF')[()]
fcol     = f.get('fcol')[:]
f_fcol   = f.get('f_fcol')
M        = f.get('M')[:]
D        = f.get('D')[:]
inc      = f.get('inc')[:]
# f_r'(r') and f_r(r)
Nm       = f.get('Nm')[()]
Nr       = f.get('Nr')[()]
r_min    = f.get('r_min')[()]
r_max    = f.get('r_max')[()]
r        = f.get('r')[:]
rP       = f.get('rP')[:]
f_r      = f.get('f_r')[:]
f_rP     = f.get('f_rP')[:]
C_m      = f.get('C_m')[:]
Tm_rP    = f.get('Tm_rP')[:,:]
lcf_rP   = f.get('lcf_rP')[:]
lcf_r    = f.get('lcf_r')[:]
# ~T_n(r')
Nk       = f.get('Nk')[()]
NT       = f.get('NT')[()]
TP       = f.get('TP')[:]
Tn_rP    = f.get('Tn_rP')[:,:]
A_nk     = f.get('A_nk')[:,:]
lcTn_rP  = f.get('lcTn_rP')[:,:]
# f_K'(K') and f_K(K)
Nn       = f.get('Nn')[()]
NK       = f.get('NK')[()]
K_min    = f.get('K_min')[()]
K_max    = f.get('K_max')[()]
K        = f.get('K')[:]
KP       = f.get('KP')[:]
c_n      = f.get('c_n')[:]
Tn_KP    = f.get('Tn_KP')[:,:]
lcf_KP   = f.get('lcf_KP')[:]
lcf_K    = f.get('lcf_K')[:]
# recf_r
recf_r   = f.get('recf_r')[:]
f.close()

# A_nm coefficients
A_nm = A_nk[:,0:Nm]

# Arrays of m-values and n-values
mArr = np.arange(Nm)
nArr = np.arange(Nn)

# Low resolution version
fh5 = '../data/fK/fK_J1655_lores.h5'
f   = h5py.File(fh5, 'r')
Nn_lores      = f.get('Nn')[()]
rP_lores      = f.get('rP')[:]
lcf_rP_lores  = f.get('lcf_rP')[:]
KP_lores      = f.get('KP')[:]
lcf_KP_lores  = f.get('lcf_KP')[:]
TP_lores      = f.get('TP')[:]
lcTn_rP_lores = f.get('lcTn_rP')[:,:]
r_lores       = f.get('r')[:]
recf_r_lores  = f.get('recf_r')[:]
f.close()

#----------------------------------------------------------------------------------------------------
# PLOT THE RESULTS
dout     = '../ms/figures/'
fout_frP = dout + 'frP_J1655.png'    # f_r'(r') and its approximation: \sum C_m T_m(r')
fout_Tnr = dout + 'Tnr_J1655.png'    # ~T_n(r') and its approximation: \sum A_nm T_m(r')
fout_Anm = dout + 'Anm_J1655.eps'    # Coefficients A_nm
fout_fKP = dout + 'fKP_J1655.png'    # f_K'(K') and its approximation: \sum c_n T_n(K')
fout_fr  = dout + 'recfr_J1655.eps'  # f_r(r)

# NOTE: There is a bug in matplotlib that prevents the rainbow_text() function from working if saving to EPS.
# https://github.com/matplotlib/matplotlib/issues/2831

# Take a list of *strings* and *colors* and place them next to each other, ...
# ...with text strings[i] being shown in colors[i].
def rainbow_text(x, y, strings, colors, orientation='horizontal', ax=None, **kwargs):
    '''
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    '''
    if ax is None:
        ax = plt.gca()
    #t = ax.transData
    #t = ax.transFigure
    t = ax.transAxes
    canvas = ax.figure.canvas

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        if orientation == 'horizontal':
            t = text.get_transform() + Affine2D().translate(ex.width, 0)
        else:
            t = text.get_transform() + Affine2D().translate(0, ex.height)

#....................................................................................................
# PLOT: f_r'(r'); \sum C_m T_m(r'); C_m
print("PLOTTING...f_r'(r'); \sum C_m T_m(r'); C_m")

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 5.6
Lgap, Rgap, Bgap, Tgap = 0.175, 0.025, 0.175, 0.05
left, right, bottom, top = Lgap, 1.0-Rgap, Bgap, 1.0-Tgap
xbox = 1.0 - (Lgap + Rgap)
ybox = 1.0 - (Bgap + Tgap)
fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 10, 10, 5
fs_in, fs_sm_in, lw_in, pad_in, tlmaj_in, tlmin_in = 20, 16, 2, 5, 5, 2.5
xlab_pos = (0.5, -0.125)
ylab_pos = (1.75*xlab_pos[1]*(ybox/xbox)*(ysize/xsize), 0.5)
thin  = lw
thick = lw * 2.0
alpha = 0.5

# Axes limits and tick increments
x_min, x_max = a, b
y_min, y_max = None, 2.5#-0.1*np.max(f_rP), 1.1*np.max(f_rP)
if (y_min is None): y_min = -0.1 * y_max
xinc_maj, xinc_min = 0.5, 0.1
yinc_maj, yinc_min = 1, 0.2
xmajorLocator = MultipleLocator(xinc_maj)
xminorLocator = MultipleLocator(xinc_min)
ymajorLocator = MultipleLocator(yinc_maj)
yminorLocator = MultipleLocator(yinc_min)
x_min_in, x_max_in = 0, Nm-1
y_min_in, y_max_in = -0.75, 0.75
xinc_maj_in, xinc_min_in = 10, 5
yinc_maj_in, yinc_min_in = 0.5,  0.25
xmajorLocator_in = MultipleLocator(xinc_maj_in)
xminorLocator_in = MultipleLocator(xinc_min_in)
ymajorLocator_in = MultipleLocator(yinc_maj_in)
yminorLocator_in = MultipleLocator(yinc_min_in)

# Setup the figure
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
ax = fig.add_subplot(111)
ax.set_xlabel(r"${\rm Rescaled\ Inner\ Disc\ Radius},\ r_{\rm in}^{\prime}$", fontsize=fs, ha='center', va='top')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.tick_params('both', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax.tick_params('both', direction='in', length=tlmin, width=lw, which='minor')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

# Align the axes labels (using the axes coordinate system)
ax.xaxis.set_label_coords(xlab_pos[0], xlab_pos[1], transform=ax.transAxes)#fig.transFigure
#ax.yaxis.set_label_coords(ylab_pos[0], ylab_pos[1], transform=ax.transAxes)#fig.transFigure

# Label the y-axis
#ax.set_ylabel(r"$f_{r^{\prime}}^{\rm CF}(r_{\rm in}^{\prime}) \simeq \sum\limits_{m=0}^{M} C_{m} T_{m}(r_{\rm in}^{\prime})$", fontsize=fs, ha='center', va='bottom')
strings = [r"$f_{r^{\prime}}^{\rm CF}(r_{\rm in}^{\prime}) \simeq $", r"$\sum\limits_{m=0}^{M} C_{m} T_{m}(r_{\rm in}^{\prime})$"]
colors  = ['k', 'C1']
rainbow_text(ylab_pos[0], 0.05, strings, colors, ax=ax, orientation='vertical', size=fs, ha='center', va='bottom')

# Plotting
ax.plot(rP, f_rP,   linewidth=thick, linestyle='solid',  color='k',  zorder=2)
ax.plot(rP, lcf_rP, linewidth=thick, linestyle='dashed', color='C1', zorder=3)
# ...and the approximation to f_r'(r') if Nm were thirded
'''
for m in np.arange(Nm):
    lcf_rP = np.zeros(NK)
    for i in np.arange(NK):
        lcf_rP[i] = np.sum(C_m[0:m+1] * Tm_rP[0:m+1,i])
    if (m == Nm/3-1):
        ax.plot(rP, lcf_rP, linewidth=thick, linestyle='dotted', color='C1', zorder=1)
'''
ax.plot(rP_lores, lcf_rP_lores, linewidth=thick, linestyle='dotted', color='C1', zorder=1)

# f_r = 0 line
ax.plot([x_min,x_max], [0,0], linewidth=thin, linestyle='dotted', color='lightgray', zorder=4)

# Inset Plot
#inW, inH, inGap = 0.33, 0.33, 0.05
inW, inH, inGap = 0.225, 0.225, 0.0225
inL = right - inW - inGap
inB = top   - inH - inGap * xsize/ysize
ax_inset = fig.add_axes([inL, inB, inW, inH])
ax_inset.set_xlabel(r"$m$",     fontsize=fs_sm, color='C1', labelpad=0)
ax_inset.set_ylabel(r"$C_{m}$", fontsize=fs_sm, color='C1', labelpad=-pad_in)
ax_inset.set_xlim(x_min_in, x_max_in)
ax_inset.set_ylim(y_min_in, y_max_in)
ax_inset.tick_params('both', direction='in', labelsize=fs_sm_in, length=tlmaj_in, width=lw, which='major', color='C1', labelcolor='C1', pad=pad_in)
ax_inset.tick_params('both', direction='in', labelsize=fs_sm_in, length=tlmin_in, width=lw, which='minor', color='C1', labelcolor='C1')
ax_inset.xaxis.set_major_locator(xmajorLocator_in)
ax_inset.xaxis.set_minor_locator(xminorLocator_in)
ax_inset.yaxis.set_major_locator(ymajorLocator_in)
ax_inset.yaxis.set_minor_locator(yminorLocator_in)
ax_inset.plot(mArr, C_m, linewidth=thin, linestyle='solid', color='C1', zorder=1)
ax_inset.scatter(mArr, C_m, s=25, marker='o', linewidths=thin*0.5, facecolors='C1', edgecolors='k', zorder=10)
ax_inset.minorticks_off()  # <-- This must come after plotting
ax_inset.spines['top'].set_color('C1')
ax_inset.spines['bottom'].set_color('C1')
ax_inset.spines['left'].set_color('C1')
ax_inset.spines['right'].set_color('C1')

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout_frP, bbox_inches=0, dpi=dpi)
plt.close()

#----------------------------------------------------------------------------------------------------
# PLOT: f_K'(K'); \sum c_n T_n(K'); c_n
print("PLOTTING...f_K'(K'); \sum c_n T_n(K'); c_n")

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 5.6
Lgap, Rgap, Bgap, Tgap = 0.175, 0.025, 0.175, 0.05
left, right, bottom, top = Lgap, 1.0-Rgap, Bgap, 1.0-Tgap
xbox = 1.0 - (Lgap + Rgap)
ybox = 1.0 - (Bgap + Tgap)
fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 10, 10, 5
fs_in, fs_sm_in, lw_in, pad_in, tlmaj_in, tlmin_in = 20, 16, 2, 5, 5, 2.5
xlab_pos = (0.5, -0.125)
ylab_pos = (1.75*xlab_pos[1]*(ybox/xbox)*(ysize/xsize), 0.5)
thin  = lw
thick = lw * 2.0
alpha = 0.5

# Axes limits and tick increments
x_min, x_max = a, b
y_min, y_max = None, 4.0#1.1*np.max(lcf_KP)
if (y_min is None): y_min = -0.1 * y_max
#xinc_maj, xinc_min = 0.5, 0.1
#yinc_maj, yinc_min = 1, 0.25
#xmajorLocator = MultipleLocator(xinc_maj)
#xminorLocator = MultipleLocator(xinc_min)
#ymajorLocator = MultipleLocator(yinc_maj)
#yminorLocator = MultipleLocator(yinc_min)
x_min_in, x_max_in = 0, Nn-1
y_min_in, y_max_in = -0.75, 0.75
xinc_maj_in, xinc_min_in = 10, 5
yinc_maj_in, yinc_min_in = 0.5,  0.25
xmajorLocator_in = MultipleLocator(xinc_maj_in)
xminorLocator_in = MultipleLocator(xinc_min_in)
ymajorLocator_in = MultipleLocator(yinc_maj_in)
yminorLocator_in = MultipleLocator(yinc_min_in)

# Setup the figure
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
ax = fig.add_subplot(111)
ax.set_xlabel(r"${\rm Rescaled\ Disc\ Flux\ Normalization},\ K_{\rm flux}^{\prime}$", fontsize=fs, ha='center', va='top')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.tick_params('both', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax.tick_params('both', direction='in', length=tlmin, width=lw, which='minor')
#ax.xaxis.set_major_locator(xmajorLocator)
#ax.xaxis.set_minor_locator(xminorLocator)
#ax.yaxis.set_major_locator(ymajorLocator)
#ax.yaxis.set_minor_locator(yminorLocator)

# Align the axes labels (using the axes coordinate system)
ax.xaxis.set_label_coords(xlab_pos[0], xlab_pos[1], transform=ax.transAxes)#fig.transFigure
#ax.yaxis.set_label_coords(ylab_pos[0], ylab_pos[1], transform=ax.transAxes)#fig.transFigure

# Label the y-axis
#ax.set_ylabel(r"$f_{K^{\prime}}(K_{\rm flux}^{\prime}) \simeq \sum\limits_{n=0}^{N} c_{n} T_{n}(K_{\rm flux}^{\prime})$", fontsize=fs, ha='center', va='bottom')
strings = [r"$f_{K^{\prime}}(K_{\rm flux}^{\prime}) \simeq$", r"$\sum\limits_{n=0}^{N} c_{n} T_{n}(K_{\rm flux}^{\prime})$"]
colors  = ['k', 'C1']
rainbow_text(ylab_pos[0], 0.05, strings, colors, ax=ax, orientation='vertical', size=fs, ha='center', va='bottom')

# Plotting
ax.plot(KP, lcf_KP, linewidth=thick, linestyle='dashed', color='C1', zorder=2)
# ...and the approximation to f_K'(K') if Nn were thirded
'''
for n in np.arange(Nn):
    lcf_KP = np.zeros(Nr)
    for i in np.arange(Nr):
        lcf_KP[i] = np.sum(c_n[0:n+1] * Tn_KP[0:n+1,i])
    if (n == Nn/3-1):
        ax.plot(KP, lcf_KP, linewidth=thick, linestyle='dotted', color='C1', zorder=1)
'''
ax.plot(KP_lores, lcf_KP_lores, linewidth=thick, linestyle='dotted', color='C1', zorder=1)

# f_K = 0 line
ax.plot([x_min,x_max], [0,0], linewidth=thin, linestyle='dotted', color='lightgray', zorder=3)

# Inset Plot
inW, inH, inGap = 0.225, 0.225, 0.025
inL = right - inW - inGap
inB = top   - inH - inGap * xsize/ysize
ax_inset = fig.add_axes([inL, inB, inW, inH])
ax_inset.set_xlabel(r"$n$",     fontsize=fs, color='C1', labelpad=0)
ax_inset.set_ylabel(r"$c_{n}$", fontsize=fs, color='C1', labelpad=-pad_in)
ax_inset.set_xlim(x_min_in, x_max_in)
ax_inset.set_ylim(y_min_in, y_max_in)
ax_inset.tick_params('both', direction='in', labelsize=fs_sm_in, length=tlmaj_in, width=lw, which='major', color='C1', labelcolor='C1', pad=pad_in)
ax_inset.tick_params('both', direction='in', labelsize=fs_sm_in, length=tlmin_in, width=lw, which='minor', color='C1', labelcolor='C1')
ax_inset.xaxis.set_major_locator(xmajorLocator_in)
ax_inset.xaxis.set_minor_locator(xminorLocator_in)
ax_inset.yaxis.set_major_locator(ymajorLocator_in)
ax_inset.yaxis.set_minor_locator(yminorLocator_in)
ax_inset.plot(nArr, c_n, linewidth=thin, linestyle='solid', color='C1', zorder=1)
ax_inset.scatter(nArr, c_n, s=50, marker='o', linewidths=thin*0.5, facecolors='C1', edgecolors='k', zorder=10)
ax_inset.minorticks_off()  # <-- This must come after plotting
ax_inset.spines['top'].set_color('C1')
ax_inset.spines['bottom'].set_color('C1')
ax_inset.spines['left'].set_color('C1')
ax_inset.spines['right'].set_color('C1')

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout_fKP, bbox_inches=0, dpi=dpi)
plt.close()

#----------------------------------------------------------------------------------------------------
# PLOT: A_nm
print("PLOTTING...A_nm")

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 5.6
Lgap, Bgap, Tgap = 0.175, 0.175, 0.05
Rgap = Tgap + Bgap - Lgap
xbox = (1.0 - (Lgap + Rgap)) * (ysize/xsize)  # Square
ybox =  1.0 - (Bgap + Tgap)                   # Square
fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 10, 10, 5
fs_in, fs_sm_in, lw_in, pad_in, tlmaj_in, tlmin_in = 20, 16, 2, 5, 5, 2.5
xlab_pos = (0.5, -0.1375)
ylab_pos = (1.5*xlab_pos[1]*(ybox/xbox)*(ysize/xsize), 0.5)
thin  = lw
thick = lw * 2.0
alpha = 0.75

# Axes limits and tick increments
x_min, x_max = 0, Nn-1
y_min, y_max = 0, Nm-1
xinc_maj, xinc_min = 5, 1
yinc_maj, yinc_min = 5, 1
xmajorLocator = MultipleLocator(xinc_maj)
xminorLocator = MultipleLocator(xinc_min)
ymajorLocator = MultipleLocator(yinc_maj)
yminorLocator = MultipleLocator(yinc_min)

'''
# Colormap
cmap     = cm.get_cmap('seismic')
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap     = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
# Bounds and normalization
yx_min, yx_max, Ncb = -1.5, 1.5, 16
yx_edges = np.linspace(yx_min, yx_max, Ncb)
yx_cents = 0.5 * (yx_edges[:-1] + yx_edges[1:])
yx_norm  = BoundaryNorm(yx_edges, cmap.N)
'''
# Colormap, Bounds, and Normalization
cmap = cm.get_cmap('RdBu_r')
yx_min, yx_max = -1, 1
yx_norm = SymLogNorm(linthresh=0.001, linscale=1, vmin=yx_min, vmax=yx_max)

# Setup the figure
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = plt.axes([Lgap, Bgap, xbox, ybox])
ax.set_xlabel(r"$n \rightarrow$", fontsize=fs, ha='left', va='top')
ax.set_ylabel(r"$m$"+"\n"+r"$\downarrow$", fontsize=fs, ha='center', va='top', rotation=0)
ax.set_xlim(-0.5, Nn-1+0.5)
ax.set_ylim(-0.5, Nm-1+0.5)
ax.tick_params('both', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad, labeltop=False, labelbottom=True)
ax.tick_params('both', direction='out', length=tlmin, width=lw, which='minor')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
#ax.xaxis.set_label_position('top')

# Align the axes labels (using the axes coordinate system)
ax.xaxis.set_label_coords(0, xlab_pos[1], transform=ax.transAxes)#fig.transFigure
ax.yaxis.set_label_coords(ylab_pos[0], 1, transform=ax.transAxes)#fig.transFigure

# Plot A_nm
A_nm_mrev = A_nm[:,::-1]  # Reverse the "m" dimension
A_mn_mrev = A_nm_mrev.T   # A_nm --> A_mn (as needed for pcolormesh)
n_edges = np.linspace(-0.5, Nn-1+0.5, Nn+1)
m_edges = np.linspace(-0.5, Nm-1+0.5, Nm+1)[::-1]  # Reversed!
x_mesh, y_mesh = np.meshgrid(n_edges, m_edges)
im = ax.pcolormesh(x_mesh, y_mesh, A_mn_mrev, cmap=cmap, norm=yx_norm)
plt.gca().invert_yaxis()  # This inverts the y-axis to be consistent with the reversed "m" dimension

# Colorbar
cmaj  = 2
CBgap = 0.033
Lcb = Lgap + xbox + CBgap
Bcb = Bgap
Wcb = 0.05
Hcb = 1.0 - (Bgap + Tgap)
cbar_ax = fig.add_axes([Lcb, Bcb, Wcb, Hcb])
#cb = plt.colorbar(im, cax=cbar_ax, cmap=cmap, norm=yx_norm, ticks=yx_cents[::cmaj], boundaries=yx_edges, orientation='vertical', spacing='proportional', format='%.1f')
cbticks = [-1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1]
cbticklabels = [r"$-1$", r"$-10^{-1}$", r"$-10^{-2}$", r"$-10^{-3}$", r"$0$", r"$+10^{-3}$", r"$+10^{-2}$", r"$+10^{-1}$", r"$+1$"]
cb = plt.colorbar(im, cax=cbar_ax, cmap=cmap, norm=yx_norm, ticks=cbticks, orientation='vertical')
cbar_ax.tick_params('y', direction='out', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
cbar_ax.tick_params('y', direction='out', length=tlmin, width=lw, which='minor')
cbar_ax.set_yticklabels(cbticklabels)
cbar_ax.set_ylabel(r"$A_{nm}$", fontsize=fs, ha='center', va='center', rotation=0)
#cbar_ax.yaxis.set_label_position('right')
cbar_ax.yaxis.set_label_coords(4.5, 0.475, transform=cbar_ax.transAxes)#fig.transFigure
cbar_ax.yaxis.tick_right()
cbar_ax.minorticks_off()

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout_Anm, bbox_inches=0, dpi=dpi)
plt.close()

#----------------------------------------------------------------------------------------------------
# PLOT: ~T_n(r'); \sum A_nk T_k(r')
print("PLOTTING...~T_n(r'); \sum A_nk T_k(r')")

# Plotting preferences
dpi = 300
xsize = 8.4
fs, fs_sm, fs_vsm, lw, pad, tlmaj, tlmin = 16, 12, 10, 2, 5, 5, 2.5
xlab_pos = (0.5, -0.7)
ylab_pos = (0.25*xlab_pos[1]*(ybox/xbox)*(ysize/xsize), 3.125)
#xlab_pos = (0.5, -0.75)
#ylab_pos = (0.225*xlab_pos[1]*(ybox/xbox)*(ysize/xsize), 0.5)
thin  = lw * 0.5
thick = lw * 1.0
alpha = 0.5
nullfmt = NullFormatter()

# x-axis labels
x_formatter = FixedFormatter([r"$-1$", r"$-0.5$", r"$0$", r"$0.5$", r"$1$"])
x_locator   = FixedLocator([-1, -0.5, 0, 0.5, 1])

# Number of T_n's to plot (one per panel)
Ncols = 4
Nrows = int(Nn/Ncols)

# Panel size
#Xpad, Ypad = -0.750, -0.125
Lgap, Rgap, Bgap, Tgap, Igap = 0.075, 0.0, 0.10, 0.025, 0.015
left, right, bottom, top = Lgap, 1.0-Rgap, Bgap, 1.0-Tgap
Lx    = 1.0 - (Lgap + Rgap)
ysize = 0.67 * xsize * (Lx + (Bgap + Tgap))
xbox  = (1.0 - (Lgap + Rgap) - (Ncols - 1) * Igap) / Ncols
ybox  = (1.0 - (Bgap + Tgap) - (Nrows - 1) * Igap) / Nrows

# Initialize the figure
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Axes linewidths
plt.rcParams['axes.linewidth'] = lw*0.5

# Label title
#plt.suptitle(r"$\widetilde{T}_{n}(r_{\rm in}^{\prime}) \simeq \sum\limits_{m=0}^{M} A_{nm} T_{m}(r_{\rm in}^{\prime})$", fontsize=fs, ha='center', va='center', y=(1.0-0.5*Tgap))
#xloc = Lgap - 0.5 * (Igap * ysize/xsize)
#yloc = Bgap + 0.5 * (ybox * Nrows + Igap * (Nrows - 1))
#fig.text(xloc, yloc, r"$\widetilde{T}_{n}(r_{\rm in}^{\prime}) \simeq \sum\limits_{m=0}^{M} A_{nm} T_{m}(r_{\rm in}^{\prime})$", fontsize=fs, va='center', ha='right', rotation='vertical')

# Loop through columns (left --> right)
left = Lgap
for j in np.arange(Ncols):

    # Loop through rows (bottom --> top)
    bottom = Bgap
    for i in reversed(np.arange(Nrows)):

        # Define the axes object
        ax = plt.axes([left, bottom, xbox, ybox])

        # Axes and Ticks
        ax.set_xlim(a, b)
        ax.set_ylim(-2, 2)
        ax.tick_params('both', direction='in', which='major', length=tlmaj*0.5, width=lw*0.5, labelsize=fs_sm, pad=pad, rotation=45)
        ax.tick_params('both', direction='in', which='minor', length=tlmin*0.5, width=lw*0.5)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_major_locator(x_locator)

        # Label the x-axis
        if (i == (Nrows-1)): ax.set_xlabel(r"$r_{\rm in}^{\prime}$", fontsize=fs, labelpad=pad)
        else: ax.xaxis.set_major_formatter(nullfmt)  # No labels on the x-axis

        # Label the y-axis
        ax.yaxis.set_major_formatter(nullfmt)  # No labels on the y-axis
        dy = 0.125
        dx = dy * (ybox / xbox) * (ysize / xsize)
        ax.text(1.0-dx, 1.0-dy, r"$\widetilde{T}_{"+str(j*Nrows+i)+"}$", color='k', transform=ax.transAxes, fontsize=fs_vsm, rotation=0, ha='right', va='top')
        if ((i == Nrows-1) and (j == 0)):
            #ax.set_ylabel(r"$\widetilde{T}_{n}(r_{\rm in}^{\prime}) \simeq \sum\limits_{m=0}^{M} A_{nm} T_{m}(r_{\rm in}^{\prime})$", fontsize=fs, ha='center', va='bottom')
            strings = [r"$\widetilde{T}_{n}(r_{\rm in}^{\prime}) \simeq$", r"$\sum\limits_{k=0}^{K} A_{nk} T_{k}(r_{\rm in}^{\prime})$"]
            colors  = ['k', 'C1']
            rainbow_text(ylab_pos[0], ylab_pos[1], strings, colors, ax=ax, orientation='vertical', size=fs, ha='center', va='bottom')

        # Align the axes labels
        ax.xaxis.set_label_coords(xlab_pos[0], xlab_pos[1])
        #if (j == 0): ax.yaxis.set_label_coords(Ypad, 0.5)
        #if (j == 1): ax.yaxis.set_label_coords(1.0+np.abs(Ypad), 0.5)

        # Plotting
        ax.plot(TP, Tn_rP[j*Nrows+i,:],   linewidth=thick, linestyle='solid',  color='k',  zorder=1)
        ax.plot(TP, lcTn_rP[j*Nrows+i,:], linewidth=thick, linestyle='dashed', color='C1', zorder=2)
        #if (j*Nrows+i < Nn_lores):
        #    ax.plot(TP_lores, lcTn_rP_lores[j*Nrows+i,:], linewidth=thick, linestyle='dotted', color='C2', zorder=3)

        #print(np.min(Tn_rP[j*Nrows+i,:]), np.max(Tn_rP[j*Nrows+i,:]))

        # Tn_rP = 0 line
        ax.plot([a,b], [0,0], linewidth=thin, linestyle='dotted', color='lightgray', zorder=4)

        # Turn off minor ticks
        ax.minorticks_off()  # <-- This must come after plotting
        ax.tick_params(axis='y', which='both', left=False, right=False)

        # Increment the bottom location
        bottom = bottom + (ybox + Igap)
    
    # Increment the left location
    left = left + (xbox + Igap * ysize/xsize)

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout_Tnr, bbox_inches=0, dpi=dpi)

# Axes linewidths <-- set this back to where it was
plt.rcParams['axes.linewidth'] = lw
plt.close()

#----------------------------------------------------------------------------------------------------
# PLOT: f_r(r)
print("PLOTTING...f_r(r)")

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 5.6
Lgap, Rgap, Bgap, Tgap = 0.15, 0.05, 0.175, 0.10
left, right, bottom, top = Lgap, 1.0-Rgap, Bgap, 1.0-Tgap
xbox = 1.0 - (Lgap + Rgap)
ybox = 1.0 - (Bgap + Tgap)
fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 10, 10, 5
xlab_pos = (0.5, -0.125)
ylab_pos = (1.33*xlab_pos[1]*(ybox/xbox)*(ysize/xsize), 0.5)
thin  = lw
thick = lw * 2.0
alpha = 0.5

# Axes limits and tick increments
x_min, x_max = r_min, r_max
y_min, y_max = -0.1*np.max(lcf_r), 1.1*np.max(lcf_r)
#xinc_maj, xinc_min = 1, 0.25
#yinc_maj, yinc_min = 0.2, 0.05
#xmajorLocator = MultipleLocator(xinc_maj)
#xminorLocator = MultipleLocator(xinc_min)
#ymajorLocator = MultipleLocator(yinc_maj)
#yminorLocator = MultipleLocator(yinc_min)

# Setup
fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
ax = fig.add_subplot(111)
ax.set_xlabel(r"${\rm Inner\ Disc\ Radius,}\ r_{\rm in} \equiv R_{\rm in} / R_{\rm g}$", fontsize=fs, ha='center', va='top')
ax.set_ylabel(r"${\rm Marginal\ Density},\ f_{r}(r_{\rm in})$", fontsize=fs, ha='center', va='bottom')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.tick_params('both', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax.tick_params('both', direction='in', length=tlmin, width=lw, which='minor')
#ax.xaxis.set_major_locator(xmajorLocator)
#ax.xaxis.set_minor_locator(xminorLocator)
#ax.yaxis.set_major_locator(ymajorLocator)
#ax.yaxis.set_minor_locator(yminorLocator)

# Align the axes labels (using the axes coordinate system)
ax.xaxis.set_label_coords(xlab_pos[0], xlab_pos[1], transform=ax.transAxes)#fig.transFigure
ax.yaxis.set_label_coords(ylab_pos[0], ylab_pos[1], transform=ax.transAxes)#fig.transFigure

#....................................................................................................
# Plot f_rCF
ax.plot(rCF, f_rCF, linewidth=thick, linestyle='solid', color='k', zorder=2)

# Plot the "recovered" f_r...
ax.plot(r, recf_r, linewidth=thick, linestyle='dashed', color='C1', zorder=3)
# ...and the approximation if Nn were thirded
'''
for n in np.arange(Nn):
    recf_r = np.zeros(Nr)
    for i in np.arange(Nr):
        recf_r[i] = np.sum(c_n[0:n+1] * Tn_KP[0:n+1,i])
    if (n == Nn/3-1):
        ax.plot(r, recf_r, linewidth=thick, linestyle='dotted', color='C1', zorder=2)
'''
ax.plot(r_lores, recf_r_lores, linewidth=thick, linestyle='dotted', color='C1', zorder=1)

# f_r = 0 line
ax.plot([x_min,x_max], [0,0], linewidth=thin, linestyle='dotted', color='lightgray', zorder=4)

#....................................................................................................

# Print out the source name
dy = 0.1
dx = dy * (ybox / xbox) * (ysize / xsize)
ax.text(1.0-dx, 1.0-dy, source, transform=ax.transAxes, color='k', fontsize=fs, ha='right', va='top')

'''
# Print out the aCF measurement...
aCF_ml, f_aCF_ml, aCF_valm, f_aCF_valm, aCF_valp, f_aCF_valp = confidence_interval(x=aCF, y=f_aCF, sigma=0.683)
aCF_errm  = aCF_ml   - aCF_valm
aCF_errp  = aCF_valp - aCF_ml
aCF_txt = r"$a^{\rm CF} = " + '{:.2f}'.format(aCF_ml) + "^{+" + '{:.2f}'.format(aCF_errp) + "}_{-" + '{:.2f}'.format(aCF_errm) + "}$"
ax.text(1.0-dx, 0.5, aCF_txt, transform=ax.transAxes, color='C1', fontsize=fs, ha='right', va='center')
'''
'''
# ...and the revised BH spin measurement
a_ml, f_a_ml, a_valm, f_a_valm, a_valp, f_a_valp = confidence_interval(x=a, y=f_a, sigma=0.683)
a_errm  = a_ml   - a_valm
a_errp  = a_valp - a_ml
a_txt = r"$a = " + '{:.2f}'.format(a_ml) + "^{+" + '{:.2f}'.format(a_errp) + "}_{-" + '{:.2f}'.format(a_errm) + "}$"
ax.text(1.0-dx, 0.5, a_txt, transform=ax.transAxes, color='C0', fontsize=fs, ha='right', va='center')
'''

# Plot the black hole spin values on the top axis
Nticks_a = 3
a_tick_values = np.linspace(1, 0, Nticks_a)  # a values from 0 --> 1
a_tick_locations = np.zeros(Nticks_a)
for i in np.arange(Nticks_a):
    a_tick_locations[i] = (spin2risco(a=a_tick_values[i]) - x_min) / (x_max - x_min)  # rin values
a_tick_labels = [r"$1$", r"$0.5$", '']
ax_top = ax.twiny()
ax_top.tick_params(axis='x', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax_top.set_xticks(a_tick_locations)
ax_top.set_xticklabels(a_tick_labels)
a_tickmin_locations = [(spin2risco(a=0.9) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.8) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.7) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.6) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.4) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.3) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.2) - x_min) / (x_max - x_min), \
                       (spin2risco(a=0.1) - x_min) / (x_max - x_min)]
ax_top.tick_params(axis='x', direction='in', length=tlmin, width=lw, which='minor', labelsize=fs_sm, pad=pad)
ax_top.set_xticks(a_tickmin_locations, minor=True)
ax_top.set_xticklabels(['0.9','','0.7','','','0.3','',''], minor=True)
ax_top.set_xlabel(r"$\leftarrow a$", fontsize=fs, ha='right', va='top', labelpad=0, x=1)

# Draw the x-grid
a_list = np.linspace(0, 1.0, 11).tolist()
for a_draw in a_list:
    rin_draw = spin2risco(a=a_draw)
    ax.plot([rin_draw,rin_draw], [y_min,y_max], linewidth=lw*0.5, linestyle='solid', color='lightgray', zorder=0)

# Do this here to avoid the code snippet above messing with the y-axis for some reason
#ax.yaxis.set_major_locator(ymajorLocator)
#ax.yaxis.set_minor_locator(yminorLocator)

# Update the y-axis to scientific notation
#ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# !!! Seems to fix the issue of the different fonts !!!
plt.rcParams.update({'font.family': 'custom'})

# Save the figure
fig.savefig(fout_fr, bbox_inches=0, dpi=dpi)
plt.close()

#----------------------------------------------------------------------------------------------------

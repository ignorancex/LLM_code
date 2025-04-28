import aplpy
import astropy.units as u
from astropy.constants import G
import sys
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib import rc
sys.path.append('/home/mvaldivi/velocity_tools')
# definition of constants and fuctions

ra_yso = (3 +(47+41.591/60)/60)*15
dec_yso = 32+(51+43.672/60)/60 # Tobin et al 2016 catalog
dist_B5 = 302 * u.pc # pc, Zucker et al 2018?

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
rc('font',**{'family':'serif'})#,'sans-serif':['Helvetica']})
rc('text', usetex=True)

def plot_aplpy_subfig(image, figure, subplotindex, stretch, vmin, vmax, cmap, vmid=0.01, label_col='k', distance=302., barsize=5000, showframeScalebar=False):
    """
    This code works for both single plot figures (subplotindex=(1,1,1)) and for subplots
    
    Args:
        image (ImageHDU): either the name of the file in .fits containing the image to plot
                            or an HDU
        figure (matplotlib.figure.Figure): figure where to put the image
        subplotindex (tuple): 3 element array with nrows, ncols, and index where to put the
                                image
        stretch (string): name of the stretch for the colorbar. Can be 'linear', 'log', 'arcsinh', etc
        vmin, vmax (float): minimum and maximum values to plot in the colorbar
        cmap (string): name of the matplotlib colormap to use for the image
    
    Returns: 
        aplpy.FITSFigure: aplpy instance which is placed in the figure
    """
    
    fig = aplpy.FITSFigure(image, figure=figure, subplot=subplotindex)
    if stretch=='arcsinh':
        fig.show_colorscale(stretch=stretch, cmap=cmap, vmin=vmin, vmax=vmax, vmid=vmid)
    else:
        fig.show_colorscale(stretch=stretch, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.add_colorbar()
    # set properties of the axis
    fig.set_system_latex(True)
    # fig.ticks.set_color(label_col)
    fig.tick_labels.set_xformat('hh:mm:ss')
    fig.tick_labels.set_yformat('dd:mm:ss')
    fig.axis_labels.set_xtext(r'Right Ascension (J2000)')
    fig.axis_labels.set_ytext(r'Declination (J2000)')
    # set beam
    fig.add_beam(color=label_col)
    fig.beam.set_color('k')
    #set scalebar
    ang_size = (barsize / distance) * u.arcsec
    fig.add_scalebar(ang_size, label=str(barsize)+' au', color=label_col, corner='top right', frame=showframeScalebar)
    fig.scalebar.set_linewidth(2)
    #set bad color
    fig.set_nan_color('0.85')
    
    return fig

def get_vc_r(velfield_file, centerra, centerdec, distance, region_file=None):
	
	"""
    Returns the centroid velocity and projected separation in the sky for the
    centroid velocity from Per-emb-50

    Given a region and a velocity field for the vicinity of a protostar,
    obtains the projected radius and the central velocity of each pixel in the
    region. The velocity field must be masked to contain only the relevant
    pixels.

    Args:
        velfield_file (string): path to the .fits file containing the velocity
        field
        region_file (string): path to the .reg (ds9) region file where the
        streamer is contained
        centerra, centerdec (Units: deg): central position of the protostar in degrees (J2000)
        distance (Units: pc): distance to the protostar in pc

    Returns:
        type: description

    """
	from regions import Regions
	from astropy.wcs import WCS
	from astropy.io import fits
	import velocity_tools.coordinate_offsets as c_offset
	# load region file and WCS structures
	if region_file is not None: regions = Regions.read(region_file)
	wcs_Vc = WCS(velfield_file)
	hd_Vc = fits.getheader(velfield_file)
	results = c_offset.generate_offsets(hd_Vc, centerra, centerdec, pa_angle=0*u.deg, inclination=0*u.deg)
	rad_au = (results.r * distance*u.pc).to(u.au, equivalencies=u.dimensionless_angles())
	if region_file is not None:
		mask_Vc = (regions[0].to_pixel(wcs_Vc)).to_mask()
		Vc_cutout = mask_Vc.cutout(fits.getdata(velfield_file))
		rad_cutout = mask_Vc.cutout(rad_au)
		gd = (mask_Vc.data == 1)
	else:
		Vc_cutout = fits.getdata(velfield_file)
		gd = ~np.isnan(Vc_cutout)
		rad_cutout = rad_au
	v_los = Vc_cutout[gd]*u.km/u.s
	r_proj = rad_cutout[gd]
	return r_proj, v_los



def primary_beam_alma(freq, diameter=12):
	freq0 = 300 * u.GHz
	fwhm0_12 = 21 * u.arcsec
	fwhm0_7 = 35 * u.arcsec
	if diameter==12:
		return fwhm0_12 / freq0 * freq
	elif diameter==7:
		return fwhm0_7 / freq0 * freq
	else:
		print('Please select either 12 or 7 to calculate the primary beam FWHM for a 12m or 7m antenna')
		return
	
	
def distancepix(x, y, x0, y0):
    #supports np.array
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


def v_kepler(mass, radius):
    vel = np.sqrt(G * mass / radius)
    return vel

def v_infall_rot(radius, j0):
    vel = j0 / radius
    return vel

def v_kepler_array(masses, radius, inclination, v_lsr=10.2*u.km/u.s):
	'''
	Returns the curves, both in the positive and negative values of the radius array given.
	Returns velocity in km/s
	Args:
		masses: array of masses with units
		radius: array of radii with units
	'''
	radius_neg = -1 * radius
	velocity_peri = []
	velocity_neg_peri = []
	for m in masses:
		velocity = v_kepler(m, radius).to(u.km/u.s) * np.sin(inclination*np.pi/180)
		velocity_pos = velocity + v_lsr
		velocity_neg = -1*velocity + v_lsr
		velocity_peri.append(velocity_pos)
		velocity_neg_peri.append(velocity_neg)
	return np.array(velocity_peri), np.array(velocity_neg_peri)

from matplotlib.colors import ListedColormap
from io import StringIO   # StringIO behaves like a file object
planckRGB = StringIO(u"  0   0 255\n  0   2 255\n  0   5 255\n  "
                                 + "0   8 255\n  0  10 255\n  0  13 255\n  "
                                 + "0  16 255\n  0  18 255\n  "
                                 + "0  21 255\n  0  24 255\n  0  26 255\n  "
                                 + "0  29 255\n  0  32 255\n  0  34 255\n  "
                                 + "0  37 255\n  0  40 255\n  0  42 255\n  "
                                 + "0  45 255\n  0  48 255\n  0  50 255\n  "
                                 + "0  53 255\n  0  56 255\n  0  58 255\n  "
                                 + "0  61 255\n  0  64 255\n  0  66 255\n  "
                                 + "0  69 255\n  0  72 255\n  0  74 255\n  "
                                 + "0  77 255\n  0  80 255\n  0  82 255\n  "
                                 + "0  85 255\n  0  88 255\n  0  90 255\n  "
                                 + "0  93 255\n  0  96 255\n  0  98 255\n  "
                                 + "0 101 255\n  0 104 255\n  0 106 255\n  "
                                 + "0 109 255\n  0 112 255\n  0 114 255\n  "
                                 + "0 117 255\n  0 119 255\n  0 122 255\n  "
                                 + "0 124 255\n  0 127 255\n  0 129 255\n  "
                                 + "0 132 255\n  0 134 255\n  0 137 255\n  "
                                 + "0 139 255\n  0 142 255\n  0 144 255\n  "
                                 + "0 147 255\n  0 150 255\n  0 152 255\n  "
                                 + "0 155 255\n  0 157 255\n  0 160 255\n  "
                                 + "0 162 255\n  0 165 255\n  0 167 255\n  "
                                 + "0 170 255\n  0 172 255\n  0 175 255\n  "
                                 + "0 177 255\n  0 180 255\n  0 182 255\n  "
                                 + "0 185 255\n  0 188 255\n  0 190 255\n  "
                                 + "0 193 255\n  0 195 255\n  0 198 255\n  "
                                 + "0 200 255\n  0 203 255\n  0 205 255\n  "
                                 + "0 208 255\n  0 210 255\n  0 213 255\n  "
                                 + "0 215 255\n  0 218 255\n  0 221 255\n"
                                 + "  6 221 254\n 12 221 253\n 18 222 252\n"
                                 + " 24 222 251\n 30 222 250\n 36 223 249\n"
                                 + " 42 223 248\n 48 224 247\n 54 224 246\n"
                                 + " 60 224 245\n 66 225 245\n 72 225 244\n"
                                 + " 78 225 243\n 85 226 242\n 91 226 241\n"
                                 + " 97 227 240\n103 227 239\n109 227 238\n"
                                 + "115 228 237\n121 228 236\n127 229 236\n"
                                 + "133 229 235\n139 229 234\n145 230 233\n"
                                 + "151 230 232\n157 230 231\n163 231 230\n"
                                 + "170 231 229\n176 232 228\n182 232 227\n"
                                 + "188 232 226\n194 233 226\n200 233 225\n"
                                 + "206 233 224\n212 234 223\n218 234 222\n"
                                 + "224 235 221\n230 235 220\n236 235 219\n"
                                 + "242 236 218\n248 236 217\n255 237 217\n"
                                 + "255 235 211\n255 234 206\n255 233 201\n"
                                 + "255 231 196\n255 230 191\n255 229 186\n"
                                 + "255 227 181\n255 226 176\n255 225 171\n"
                                 + "255 223 166\n255 222 161\n255 221 156\n"
                                 + "255 219 151\n255 218 146\n255 217 141\n"
                                 + "255 215 136\n255 214 131\n255 213 126\n"
                                 + "255 211 121\n255 210 116\n255 209 111\n"
                                 + "255 207 105\n255 206 100\n255 205  95\n"
                                 + "255 203  90\n255 202  85\n255 201  80\n"
                                 + "255 199  75\n255 198  70\n255 197  65\n"
                                 + "255 195  60\n255 194  55\n255 193  50\n"
                                 + "255 191  45\n255 190  40\n255 189  35\n"
                                 + "255 187  30\n255 186  25\n255 185  20\n"
                                 + "255 183  15\n255 182  10\n255 181   5\n"
                                 + "255 180   0\n255 177   0\n255 175   0\n"
                                 + "255 172   0\n255 170   0\n255 167   0\n"
                                 + "255 165   0\n255 162   0\n255 160   0\n"
                                 + "255 157   0\n255 155   0\n255 152   0\n"
                                 + "255 150   0\n255 147   0\n255 145   0\n"
                                 + "255 142   0\n255 140   0\n255 137   0\n"
                                 + "255 135   0\n255 132   0\n255 130   0\n"
                                 + "255 127   0\n255 125   0\n255 122   0\n"
                                 + "255 120   0\n255 117   0\n255 115   0\n"
                                 + "255 112   0\n255 110   0\n255 107   0\n"
                                 + "255 105   0\n255 102   0\n255 100   0\n"
                                 + "255  97   0\n255  95   0\n255  92   0\n"
                                 + "255  90   0\n255  87   0\n255  85   0\n"
                                 + "255  82   0\n255  80   0\n255  77   0\n"
                                 + "255  75   0\n251  73   0\n247  71   0\n"
                                 + "244  69   0\n240  68   0\n236  66   0\n"
                                 + "233  64   0\n229  62   0\n226  61   0\n"
                                 + "222  59   0\n218  57   0\n215  55   02\n"
                                 + "211  54   0\n208  52   0\n204  50   0\n"
                                 + "200  48   0\n197  47   0\n193  45   0\n"
                                 + "190  43   0\n186  41   0\n182  40   0\n"
                                 + "179  38   0\n175  36   0\n172  34   0\n"
                                 + "168  33   0\n164  31   0\n161  29   0\n"
                                 + "157  27   0\n154  26   0\n150  24   0\n"
                                 + "146  22   0\n143  20   0\n139  19   0\n"
                                 + "136  17   0\n132  15   0\n128  13   0\n"
                                 + "125  12   0\n121  10   0\n118   8   0\n"
                                 + "114   6   0\n110   5   0\n107   3   0\n"
                                 + "103   1   0\n100   0   0")
planck_map = ListedColormap(np.loadtxt(planckRGB)/255.)
planck_map.set_bad("0.85")
import numpy as np
import astropy.io.ascii as io
import astropy.io.fits as pyfits
from astropy import constants as const
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from cycler import cycler

cmap_names = ["YlGn_r", "RdPu_r", "Blues_r"]

cmap_dict = {"thmin70": "Blues_r", "thmin45": "RdPu_r", "thmin20": "YlGn_r"}
fiducial_angle = 15

g_DataDir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'Data'))
g_FigureDir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'Figures'))


def rho2nh():
    z, abun, mass = np.genfromtxt(
        "{}/elements.txt".format(g_DataDir), unpack=True, usecols=(1, 3, 4))
    x = 0
    y = 0
    xx = 0
    for i in range(len(abun)):
        x += ((10.0 ** abun[i]) * mass[i])
        xx += ((10.0 ** abun[i]) * z[i])
    y = (10.0 ** abun[0])
    x *= const.m_p.cgs.value
    return (y/x, xx/y)


def save_paper_figure(savename, fig=None, figure_dir=g_FigureDir, **savefig_kwargs):
    """wrapper to save a paper figure in the main figures directory"""
    if fig == None:
        fig = plt.gcf()

    full_savename = "{}/{}".format(figure_dir, savename)
    fig.savefig(full_savename, **savefig_kwargs)


def get_filenames_for_alpha(system_args=["dummy", "0.5"], override=None):
    # CHOOSE YOUR FIGHTER
    # by which I mean, choose which value of alpha to plot
    if len(system_args) > 1:
        Alpha = system_args[1]
    else:
        Alpha = "0.5"

    if override is not None:
        Alpha = override

    speclist_dir = "{}/speclists/".format(g_DataDir)

    if Alpha == "-1":
        AlphaString = "combo"
        speclist_column = 0
        data_dir = data_dir = "{}/allspec/".format(g_DataDir)
        speclist = np.genfromtxt("{}/speclist_combo.txt".format(speclist_dir),
                                 dtype=str, usecols=(speclist_column,), unpack=True)
        Alpha = float(Alpha)
    elif Alpha == "-1.5":
        AlphaString = "alpha1.5"
        speclist_column = 0
        data_dir = "{}/{}/".format(g_DataDir, AlphaString)
        speclist = np.genfromtxt("{}/speclist_alpha1.5_9.txt".format(
            speclist_dir), dtype=str, usecols=(speclist_column,), unpack=True)
        Alpha = 1.5
    else:
        AlphaString = "alpha{}".format(Alpha)
        # gives 0,1,2 for 0.5,1,2 as desired
        speclist_column = int((float(Alpha) + 0.01) // 0.5) - 1
        # folder containing spectra for this value of alpha
        data_dir = "{}/{}/".format(g_DataDir, AlphaString)
        # load list of spectra with column set by value of alpha
        speclist = np.genfromtxt("{}/speclist.txt".format(speclist_dir),
                                 dtype=str, usecols=(speclist_column,), unpack=True)
        Alpha = float(Alpha)

    return (Alpha, AlphaString, data_dir, speclist)


def get_data_dir_for_alpha(alpha=0.5):
    if alpha == 0.5:
        AlphaString = "alpha{:.1f}".format(alpha)
    else:
        AlphaString = "alpha{:.0f}".format(int(alpha))
    # folder containing spectra for this value of alpha
    data_dir = "{}/{}/".format(g_DataDir, AlphaString)
    return (data_dir)


def get_norm_mappable(cmap):
    norm = matplotlib.colors.Normalize(vmin=5, vmax=90)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    return (norm, mappable)


def read_fits_spectrum(f="h19_c3_19061908_1525_a.fits"):
    """Read an observed spectrum from a fits file

    Args:
        f (str, optional): filename. Defaults to "h19_c3_19061908_1525_a.fits".

    Returns:
        w, fl: wavelength and flux arrays 
    """
    with pyfits.open("{}".format(f)) as hdulist:
        hdr = hdulist[0].header
        wfirst = hdr["COEFF0"]
        wlast = wfirst + (hdr["COEFF1"] * hdr["NAXIS1"])
        w = np.logspace(wfirst, wlast, hdr["NAXIS1"], endpoint=False)
        fl = hdulist[0].data[0]

    return (w, fl)


def get_line_lum(wavelength, velocity, f, fcont, d=100.0):
    select = (np.fabs(velocity) < 1e4)
    integrand = f - fcont
    integrand = integrand[select]
    integral = np.fabs(np.trapz(integrand, wavelength[select]))
    pc = const.pc.cgs.value
    lum = integral * 4.0 * np.pi * (d * d * pc * pc)
    return (lum)


def get_ew(wavelength, velocity, f, fcont):
    """Calculate equivalent width

    Args:
        wavelength (array-like): wavelength array
        velocity (array-like): velocity array
        f (array-like): flux
        fcont (array-like): continuum flux 

    Returns:
        integral (float): equivalent width
    """
    integrand = (1.0 - f/fcont)
    select = (np.fabs(velocity) < 1e4)
    integrand = integrand[select]
    integral = np.trapz(integrand, wavelength[select])

    return (integral)


def calc_blueshift(f, data_dir):
    """Calculate blueshift

    Args:
        f (string): filename
        fcont (data_dir): directory

    Returns:
        blueshifts (array-like): array of blueshifts for each angle
        ews (array-like): array of EWs for each angle
        angles (array-like): array of angles in degrees
    """
    try:
        s = io.read("{}/{}".format(data_dir, f))
    except io.core.InconsistentTableError:
        print("Error for", f)

    wavelength = s["Lambda"]
    velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs
    velocity = velocity.to(u.km/u.s)

    pf_root = f[:-5]
    pf = read_pf("{}/{}".format(data_dir, pf_root))
    theta1 = float(pf["SV.thetamin(deg)"])

    angles = np.arange(5, int(theta1)+5, 5)

    blueshifts = np.zeros_like(angles, dtype=float)
    ews = np.zeros_like(angles, dtype=float)

    for i in range(len(angles)):
        colname = "A{:02d}P0.50".format(angles[i])

        fcont = fit_continuum(wavelength, s[colname])

        # get blueshift
        blueshifts[i] = get_blueshift(velocity/(u.km / u.s), s[colname], fcont)
        ews[i] = get_ew(wavelength, velocity/(u.km / u.s), s[colname], fcont)

    return (blueshifts, ews, angles)


def get_blueshift(velocity, flux, fcont):
    """Calculate blueshift from a velocity, flux and continuum flux

    Args:
        velocity (array-like): velocity array in km/s
        flux (array-like): flux array
        fcont (array-like): continuum flux array

    Returns:
        velocity (float): blueshift in km.s 
    """
    line_flux = flux-fcont
    line_flux = line_flux[(np.fabs(velocity) < 1e4)]
    v_use = velocity[(np.fabs(velocity) < 1e4)]

    # interpolate on wavelengths
    interp_func = interp1d(v_use, line_flux)
    velocity_new = np.linspace(np.min(v_use), np.max(v_use), 100000)
    fnew = interp_func(velocity_new)

    cdf = np.cumsum(fnew) / np.sum(fnew)
    halfway = np.argmin(np.fabs(cdf - 0.5))

    return (-velocity_new[halfway])

# def order(mi,ma):
#     if mi>ma:
#         mi,ma=ma,mi
#     return mi,ma

# @jit(nopython=True)


def fit_continuum(w, f, lines=[1215, 1240, 1400, 1640, 1753, 1909, 2800, 1550, 1030, 4863, 6563],
                  window_length=9, polyorder=5, mask_size=70):
    f_filter = savgol_filter(f, window_length, polyorder)
    mask = np.ones_like(w, dtype=bool)
    for l in lines:
        mask *= (np.fabs(l - w) > mask_size)

    interp_func = interp1d(w[mask], f_filter[mask],
                           kind="slinear", fill_value="extrapolate")
    fcont = interp_func(w)
    return (fcont)

# def getind(lst,val):
#     diff=abs(lst-val)
#     closest=min(diff)
#     diff=list(diff)
#     return diff.index(closest)

# def shape(w,det,bls,lmin,lmax,l0, return_arg = False):
#     xmin,xmax=order(getind(w,lmax),getind(w,lmin))
#     reduced=det[xmin:xmax]
#     argmax = list(reduced).index(np.max(reduced))
#     argmax = np.argmax(reduced)
#     ws=w[xmin:xmax][argmax]

#     print (xmin, xmax, argmax)

#     vpeak = const.c.cgs * (ws-l0)/l0

#     vpeak = vpeak.cgs.value / 1e5

#     if return_arg:
#         return vpeak - bls, ws, vpeak, argmax, xmin, xmax
#     else:
#         return vpeak - bls, ws, vpeak


def get_skew(velocity, flux, blueshift):
    """Calculate the skew of an emission line

    Args:
        velocity (array-like): velocity array in km/s
        flux (array-like): flux array
        blueshift (float): blueshift in km.s 

    Returns:
        vpeak (float): peak velocity in km/s
        fpeak (float): flux at peak in same units as flux array
        skew (float): skew value in km.s
    """
    line_flux = flux
    line_flux = line_flux[(np.fabs(velocity) < 1e4)]
    v_use = velocity[(np.fabs(velocity) < 1e4)]

    # interpolate on wavelengths
    interp_func = interp1d(v_use, line_flux)
    velocity_new = np.linspace(np.min(v_use), np.max(v_use), 100000)
    fnew = interp_func(velocity_new)

    argmax = np.argmax(fnew)
    vpeak = velocity_new[argmax]
    fpeak = fnew[argmax]
    skew = vpeak - blueshift

    return (vpeak, fpeak, skew)


def get_flux_at_wavelength(lambda_array, flux_array, w):
    """
    Find the flux at wavelength w

    Parameters:
        lambda_array: array-like    
            array of wavelengths in angstroms. 1d 

        flux_array: array-like 
            array of fluxes same shape as lambda_array 

        w: float 
            wavelength in angstroms to find

    Returns:
        f: float 
            flux at point w
    """

    i = np.abs(lambda_array - w).argmin()

    return flux_array[i]


def measure_blueshifts(subfolder="/composites33/", fits_structure=True):
    """measure blueshifts in observational data from a file 

    Args:
        folder (str, optional): 
        folder name within the global data directory. 
        Defaults to composites33.

    Returns:
        blueshifts, ews, skew, vpeak:
            arrays holding the blueshifts, EWs, skews and peak velocities
            for each spectrum in the folder.
    """
    # list all files
    folder = "{}/{}/".format(g_DataDir, subfolder)

    if fits_structure:
        files = os.listdir(folder)
        N = len(files)
    else:
        N = 33
        wavelength = np.genfromtxt("{}/jm_33comps_norm.wav".format(folder))
        fluxes = np.genfromtxt(
            "{}/jm_33comps_norm.dat".format(folder), unpack=True)

    # blank arrays
    blueshifts = np.zeros(N)
    ews = np.zeros(N)
    skew = np.zeros(N)
    lambda0 = 1550
    vpeak = np.zeros(N)

    for i in range(N):
        if fits_structure:
            f = files[i]
            wavelength, fl = read_fits_spectrum(folder+f)
        else:
            fl = fluxes[i]

        fcont = fit_continuum(wavelength, fl)
        velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs
        velocity = velocity.to(u.km/u.s)

        blueshifts[i] = get_blueshift(velocity/(u.km / u.s), fl, fcont)
        ews[i] = get_ew(wavelength, velocity/(u.km / u.s), fl, fcont)
        vpeak[i], fpeak, skew[i] = get_skew(
            velocity/(u.km / u.s), fl/fcont, blueshifts[i])
        #skew[i], _, vpeak[i] = shape(wavelength,fl-fcont,blueshifts[i],1500,1600,1550)

    return (blueshifts, ews, skew, vpeak)


def gradient_fill(x, y, fill_color=None, ax=None, mappable=None, alpha=None, zorder=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    kwargs2 = kwargs.copy()
    kwargs2["label"] = None
    kwargs2["lw"] = 0
    line, = ax.plot(x, y, **kwargs2)
    if fill_color is None:
        fill_color = line.get_color()

    if zorder is None:
        zorder = line.get_zorder()
    if alpha is None:
        alpha = line.get_alpha()
        alpha = 1.0 if alpha is None else alpha

    if mappable is None:
        Nz = 10000
        z = np.empty((Nz, 1, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(fill_color)
        rgb = fill_color[:3]
        z[:, :, :3] = rgb

        z[:, :, -1] = np.linspace(0, 0.85, Nz)[:, None] ** 1.5

    else:
        xx, yy = np.meshgrid(x, y)
        z = mappable.to_rgba(xx[::-1])
        z[:, :, -1] = alpha

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, 0, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, 0], xy, [xmax, 0], [xmin, 0]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)
    line, = ax.plot(x, y, color=fill_color, **kwargs, zorder=zorder)
    return line, im


def set_plot_defaults(tex="True"):
    """set some nice plot defaults and use times font
    """
    # FIGURE
    plt.rcParams["text.usetex"] = tex

    # FONT
    if tex == "True":
        plt.rcParams['font.serif'] = ['Times']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    plt.rcParams['font.family'] = 'serif'

    plt.rcParams['font.size'] = 18
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["lines.linewidth"] = 2.2
    # TICKS
    plt.rcParams['xtick.top'] = 'True'
    plt.rcParams['xtick.bottom'] = 'True'
    plt.rcParams['xtick.minor.visible'] = 'True'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.left'] = 'True'
    plt.rcParams['ytick.right'] = 'True'
    plt.rcParams['ytick.minor.visible'] = 'True'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['ytick.minor.size'] = 3


def set_cmap_cycler(cmap_name="viridis", N=None):
    '''
    set the cycler to use a colormap
    '''
    if cmap_name == "default" or N is None:
        my_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    else:
        _, colors = get_mappable(N, cmap_name=cmap_name)
        # if type(style) == str:
        my_cycler = (cycler(color=colors))

    plt.rc('axes', prop_cycle=my_cycler)


def get_mappable(N, vmin=0, vmax=1, cmap_name="Spectral", return_func=False):
    my_cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = my_cmap(np.linspace(0, 1, num=N))

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_name)
    if return_func:
        fcol = colour_func(norm, cmap_name)
        return (mappable, colors, mappable.to_rgba)
    else:
        return (mappable, colors)


def read_pywind(filename, return_inwind=False, mode="2d", complete=True):
    """
    read a py_wind output file using np array reshaping and manipulation

    Parameters            
        filename : file or str
            File, filename to read, e.g. root.ne.dat  

        return_inwind: Bool
            return the array which tells you whether you
            are partly, fully or not inwind.

        mode: string 
            can be used to control different coord systems 

    Returns          
        x, z, value: masked arrays
            value is the quantity you are concerned with, e.g. ne
    """
    # first, simply load the filename
    # d = np.loadtxt(filename, comments="#", dtype = "float", unpack = True)
    d = io.read(filename)

    return wind_to_masked(d, "var", return_inwind=return_inwind, mode=mode)


def read_pf(root):
    '''
    reads a Python .pf file and returns a dictionary

    Parameters
        root : file or str
            File, filename to read.  

        new:
            True means the Created column exists in the file 

    Returns
        pf_dict
            Dictionary object containing parameters in pf file
    '''
    OrderedDict_present = True
    try:
        from collections import OrderedDict
    except ImportError:
        OrderedDict_present = False

    if not ".pf" in root:
        root = root + ".pf"

    params, vals = np.loadtxt(root, dtype=str, unpack=True)

    if OrderedDict_present:
        pf_dict = OrderedDict()
    else:
        pf_dict = dict()    # should work with all version of python, but bad for writing
        print("Warning, your dictionary object is not ordered.")

    old_param = None
    old_val = None

    for i in range(len(params)):

        # convert if it is a float
        try:
            val = float(vals[i])

        except ValueError:
            val = vals[i]

        if params[i] == old_param:

            if isinstance(pf_dict[params[i]], list):
                pf_dict[params[i]].append(val)

            else:
                pf_dict[params[i]] = [old_val, val]

        else:
            pf_dict[params[i]] = val

        old_param = params[i]
        old_val = val

    return pf_dict


def get_flux_at_wavelength(lambda_array, flux_array, w):
    '''
    Find the flux at wavelength w

    Parameters:
        lambda_array: array-like    
            array of wavelengths in angstroms. 1d 

        flux_array: array-like 
            array of fluxes same shape as lambda_array 

        w: float 
            wavelength in angstroms to find

    Returns:
        f: float 
            flux at point w
    '''

    i = np.abs(lambda_array - w).argmin()

    return flux_array[i]


def wind_to_masked(d, value_string, return_inwind=False, mode="2d", ignore_partial=True):
    '''
    turn a table, one of whose colnames is value_string,
    into a masked array based on values of inwind 

    Parameters:
        d: astropy.table.table.Table object 
            data, probably read from .complete wind data 

        value_string: str 
            the variable you want in the array, e.g. "ne"

        return_inwind: Bool
            return the array which tells you whether you
            are partly, fully or not inwind.
    Returns:
        x, z, value: Floats 
            value is the quantity you are concerned with, e.g. ne
    '''
    # this tuple helpd us decide whether partial cells are in or out of the wind
    if ignore_partial:
        inwind_crit = (0, 1)
    else:
        inwind_crit = (0, 2)

    if mode == "1d":
        inwind = d["inwind"]
        x = d["r"]
        values = d[value_string]

        # create an inwind boolean to use to create mask
        inwind_bool = (inwind >= inwind_crit[0]) * (inwind < inwind_crit[1])
        mask = ~inwind_bool

    # finally we have our mask, so create the masked array
        masked_values = np.ma.masked_where(mask, values)

    # return the arrays later, z is None for 1d
        z = None

    elif mode == "2d":
        # our indicies are already stored in the file- we will reshape them in a sec
        zindices = d["j"]
        xindices = d["i"]

        # we get the grid size by finding the maximum in the indicies list 99 => 100 size grid
        zshape = int(np.max(zindices) + 1)
        xshape = int(np.max(xindices) + 1)

        # now reshape our x,z and value arrays
        x = d["x"].reshape(xshape, zshape)
        z = d["z"].reshape(xshape, zshape)

        values = d[value_string].reshape(xshape, zshape)

        # these are the values of inwind PYTHON spits out
        inwind = d["inwind"].reshape(xshape, zshape)

        # create an inwind boolean to use to create mask
        inwind_bool = (inwind >= inwind_crit[0]) * (inwind < inwind_crit[1])
        mask = ~inwind_bool

        # finally we have our mask, so create the masked array
        masked_values = np.ma.masked_where(mask, values)

    else:
        print("Error: mode {} not understood!".format(mode))

    # return the transpose for contour plots.
    if return_inwind:
        return x, z, masked_values, inwind_bool
    else:
        return x, z, masked_values

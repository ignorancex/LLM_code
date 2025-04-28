import batman, catwoman
import juliet

from scipy.stats import norm,beta
import numpy as np

from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

__all__ = ['load_plt_params', 'load_parula', 'pipeline_dictionary',
           'convolve_model', 'convolve_model_xy', 'truncate_colormap',
           'avg_lightcurves', 'get_MAD_sigma']

def load_plt_params():
    """ Load in plt.rcParams and set (based on paper defaults).
    """
    params = Table.read('rcParams.txt', format='csv')
    for i, name in enumerate(params['name']):
        try:
            plt.rcParams[name] = float(params['value'][i])
        except:
            plt.rcParams[name] = params['value'][i]
    return params

def load_parula():
    """ Load in custom parula colormap.
    """
    colors = np.load('parula_colors.npy')
    return colors

def pipeline_dictionary():
    """ Loads in the custom colors for the paper figures.
    """
    pipeline_dict = {}
    pipelines = Table.read('pipelines.csv', format='csv')

    # Sets the initials key for each pipeline
    for i, name in enumerate(pipelines['initials']):
        pipeline_dict[name] = {}
        pipeline_dict[name]['color'] = pipelines['color'][i]
        pipeline_dict[name]['name'] = pipelines['name'][i]
        pipeline_dict[name]['filename'] = pipelines['filename'][i]
        pipeline_dict[name]['author'] = pipelines['author'][i]
        pipeline_dict[name]['contact'] = pipelines['contact'][i]

    return pipeline_dict

def convolve_model(filename, R=300):
    model = np.loadtxt(filename)

    R0=3000.0 #cross-section resolution
    xker = np.arange(1000)-500
    sigma = (R0/R)/(2.* np.sqrt(2.0*np.log(2.0)))
    yker = np.exp(-0.5 * (xker / sigma)**2.0)
    yker /= yker.sum()
    model_to_plot=np.convolve(model[:,1],yker,mode='same') #convolving
    return model[:,0], model_to_plot


def convolve_model_xy(y, R=300):
    R0=3000.0 #cross-section resolution
    xker = np.arange(1000)-500
    sigma = (R0/R)/(2.* np.sqrt(2.0*np.log(2.0)))
    yker = np.exp(-0.5 * (xker / sigma)**2.0)
    yker /= yker.sum()
    model_to_plot=np.convolve(y,yker,mode='same') #convolving
    return model_to_plot

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,
                                                                                            a=minval,
                                                                                            b=maxval),cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def avg_lightcurves(i, data, err, idx_oot, per=5):
    """
    Creates averaged spectroscopic light curves across 'per' number of
    channels.
    """

    flux  = np.zeros((per*2+1, len(data['time'])))
    model = np.zeros((per*2+1, len(data['time'])))
    error = np.zeros((per*2+1, len(data['time'])))
    fnorm = np.zeros((per*2+1, len(data['time'])))
    wrange = np.zeros(per*2+1)

    for j in range(i-per, i+per+1):
        flux[j-(i-per)]  = data['lc_w{}'.format(j)]
        model[j-(i-per)] = data['combined_model_w{}'.format(i)]

        eind = np.where(err[1] <=
                        data['w{}'.format(j)][0].value)[0][0]

        error[j-(i-per)] = err[3][:,eind]
        fnorm[j-(i-per)] = err[2][:,eind]
        wrange[j-(i-per)] = data['w{}'.format(j)][0].value

    wrange = np.sort(wrange)
    wmed = wrange[5]
    low,upp = wrange[5]-wrange[0], wrange[-1]-wrange[5]
    lim = np.round(np.nanmedian([low,upp]),3)

    e = (np.sqrt(np.nansum(error,axis=0))/len(error))/np.nanmax(fnorm)
    f = np.nanmean(flux, axis=0)
    m = np.nanmax(model[-2:], axis=0)

    f /= np.nanmedian(f[idx_oot])
    m /= np.nanmedian(m[idx_oot])

    return f, e, m, wmed, lim, wrange[0], wrange[-1], model, flux

def get_MAD_sigma(x, median):
    """
    Wrapper function for transitspectroscopy.utils.get_MAD_sigma to estimate
    the noise properties of the light curves.

    Parameters
    ----------
    x : np.ndarray
    median : np.ndarray
    """
    mad = np.nanmedian( np.abs ( x - median ) )

    return 1.4826*mad

def transform_uniform(x,a,b):
    return a + (b-a)*x

def transform_loguniform(x,a,b):
    la=np.log(a)
    lb=np.log(b)
    return np.exp(la + x*(lb-la))

def transform_normal(x,mu,sigma):
    return norm.ppf(x,loc=mu,scale=sigma)

def transform_beta(x,a,b):
    return beta.ppf(x,a,b)

def ibin(winput, w, d, derr):

    doutput = np.zeros( len(winput) )
    doutput_err = np.zeros( len(winput) )
    for i in range( len(winput) ):

        if i == 0:

            dw = ( winput[i+1] - winput[i] ) * 0.5

        else:

            dw = ( winput[i] - winput[i-1] ) * 0.5

        idx = np.where(np.abs(w-winput[i])<dw)[0]

        doutput[i] = np.mean(d[idx])
        doutput_err[i] = np.sqrt( np.mean( derr[idx]**2 ) ) / np.sqrt( len(idx) )
        
    return doutput, doutput_err

def get_lc(params, t, reg, posteriors, method = 'batman', nsamples = 1000):

    if method == 'batman':

        full_mean_model = 0.
        mean_model = 0.
        q1, q2, p = posteriors['q1_SOSS'], posteriors['q2_SOSS'], posteriors['p_p1']
        idx = np.random.choice(np.arange(len(q1)), nsamples, replace = False)
        for i in idx:

        # This is eq (1) and (5) in the juliet paper:

            mflux, theta = posteriors['mflux_SOSS'][i], posteriors['theta0_SOSS'][i]
            u1, u2 = juliet.utils.reverse_ld_coeffs('quadratic', q1[i], q2[i])

            transit_model = get_bm_transit(params, t, p[i], u1, u2)

            full_mean_model += (transit_model / (1. + mflux) ) + reg*theta
            mean_model += transit_model

        return mean_model / len(idx), full_mean_model / len(idx)

    else:

        full_mean_model = 0.
        mean_model = 0.
        q1, q2, p1, p2 = posteriors['q1_SOSS'], posteriors['q2_SOSS'], posteriors['p1_p1'], posteriors['p2_p1']
        idx = np.random.choice(np.arange(len(q1)), nsamples, replace = False)
        for i in idx:

            mflux, theta = posteriors['mflux_SOSS'][i], posteriors['theta0_SOSS'][i]
            u1, u2 = juliet.utils.reverse_ld_coeffs('quadratic', q1[i], q2[i])

            transit_model = get_cw_transit(params, t, p1[i], p2[i], u1, u2)

            full_mean_model += (transit_model / (1. + mflux) ) + reg*theta

            mean_model += get_cw_transit(params, t, p1[i], p2[i], u1, u2) 

        return mean_model / len(idx), full_mean_model / len(idx)

def get_bm_transit(params, t, p, u1, u2):

    params.t0 = 2459771.335647
    params.per = 4.0552842
    params.rp = 0.1 
    params.a = 11.39
    params.inc = 87.7367563737318
    params.ecc = 0.
    params.w = 90

    params.limb_dark = 'quadratic'
    params.u = [u1, u2] 
    params.rp = p

    model = batman.TransitModel(params, t)
    return model.light_curve(params)

def get_cw_transit(params, t, p1, p2, u1, u2):

    params.t0 = 2459771.335647
    params.per = 4.0552842
    params.phi = 90.
    params.rp = 0.1
    params.rp2 = 0.1
    params.a = 11.39
    params.inc = 87.7367563737318
    params.ecc = 0.
    params.w = 90

    params.limb_dark = 'quadratic'
    params.u = [u1, u2] 
    params.rp = p1
    params.rp2 = p2

    model = catwoman.TransitModel(params, t)
    return model.light_curve(params)

def correct_lightcurve(t, f, ferr, reg, posteriors):

    # Generate mean systematics model:
    model = np.zeros(len(t))
    for i in range(len(posteriors['sigma_w_SOSS'])):

        # This is eq (1) and (5) in the juliet paper:
        mflux, theta = posteriors['mflux_SOSS'][i], posteriors['theta0_SOSS'][i] 
        model += (1. / (1. + mflux) ) + reg*theta

    model = model / len( posteriors['sigma_w_SOSS'] )

    return model, f / model, ferr / model

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='cornflowerblue', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)

def plot_ellipse(x, y, ax):

    confidence_ellipse(x, y, ax, n_std=1,
                       edgecolor='cornflowerblue', facecolor = 'cornflowerblue', alpha = 0.8)
    confidence_ellipse(x, y, ax, n_std=2,
                       edgecolor='cornflowerblue', facecolor = 'cornflowerblue', alpha = 0.5)
    confidence_ellipse(x, y, ax, n_std=3,
                       edgecolor='cornflowerblue', facecolor = 'cornflowerblue', alpha = 0.3)

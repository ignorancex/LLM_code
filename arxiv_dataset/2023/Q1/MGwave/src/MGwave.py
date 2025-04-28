#####
#
# This code has been developed by Scott Lucchini and Emil Pellet at UW - Madison.
# Please see the guide at https://github.com/DOnghiaGroup/MGwave and direct any
# questions to lucchini@wisc.edu.
#
# This code is licensed under MIT license copyright 2021 Scott Lucchini.
#
###

###
#
# The main function is run_wavelet which will calculate the histogram, wavelet
# transform, extremal points, and significance for each extremum.
# The starlet transform is used for the wavelet transformation.
# Significance determination follows steps outlined in Slezak et al. (1993)
# section 3.1 (https://ui.adsabs.harvard.edu/abs/1993ApJ...409..517S/abstract) and
# Starck et al. (1998) section 2.1.5 (http://www.multiresolution.com/cupbook.pdf)
#
#####

import sys
import time
from functools import partial

import numpy as np
import scipy.signal
import seaborn as sns
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt

import starlet

# ------------------------------------------------------------------------------
# ----------------------------  AUXILIARY FUNCTIONS ----------------------------

def peaks(array, min_distance=1, edges=0):
    """
    Uses peak_local_max to locate local maxima, and then uses a custom algorithm
    to make sure that points are separated by at least min_distance

    Parameters
    ----------
    array : ndarray
        The data in which to search for the maxima
    min_distance : int
        The minimum distance between detected extrema
    edges : int
        The distance from the edge of the data (array) to exclude

    Returns
    -------
    ndarray
        A list of maxima and their values

    """

    # min_distance = 2*(min_distance+1)
    plm = peak_local_max(array, exclude_border=edges)#,min_distance = min_distance - 1)
    dim = array.ndim
    x = np.zeros((len(plm),dim+1))
    x[:,:dim] = plm
    x[:,dim] = np.array([array[tuple(p)] for p in plm])
    x = x.tolist()
    x.sort(key = lambda x: x[-1], reverse=True)
    x = np.asarray(x)

    for i in range(len(x)):
        mask = (np.sum((x[i,:-1] - x[:,:-1])**2, axis = 1) > min_distance**2) | (np.sum((x[i,:-1] - x[:,:-1])**2, axis = 1) == 0)
        x = x[mask]
        if (len(x)-1 <= i):
            break

    return x

# B3-Spline function defined in http://jstarck.free.fr/Chapter_Starlet2011.pdf on pg. 9
def b3(x):
    return (1/12.)*(np.power(np.abs(x-2),3) - 4*np.power(np.abs(x-1),3) + 6*np.power(np.abs(x),3) -
                    4*np.power(np.abs(x+1),3) + np.power(np.abs(x+2),3))

# Continuous wavelet function defined in Starck et al. (1998) equation 1.42
def psi(x,phi=b3):
    return 2*phi(2*x) - phi(x)

# https://www.him.uni-bonn.de/fileadmin/him/Starck_Lecture_1_IntroCosmo_Wavelets.pdf Slide 74
def psi2d(x,y,phi=b3):
    return 4*phi(2*x)*phi(2*y) - phi(x)*phi(y)

# gaussian function
def gaus(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))

def convolve(x1,f1,x2,f2):
    """
    Convolve two 1D functions while taking the padding into consideration for the x values.
    x1 and x2 must have the same bin size i.e. deltax must be the same.

    Parameters
    ----------
    x1 : ndarray
        x values for the first function
    f1 : ndarray
        y values for the first function. Must have same length as x1
    x2 : ndarray
        x values for the second function
    f2 : ndarray
        y values for the second function. Must have same length as x2

    Returns
    -------
    ndarray
        The list of x values for the convolved function
    ndarray
        The list of y values for the convolved function

    """

    result = np.convolve(f1,f2,'full')
    delta1 = x1[1] - x1[0]
    delta2 = x2[1] - x2[0]
    if (delta1 - delta2 > 1e-10):
        raise Exception("Deltax must be the same for both arrays.")
    start = x1[0] + x2[0]
    end = x1[-1] + x2[-1]
    newx = np.array(range(len(result)))*delta1 + start
    if (newx[-1] - end > delta1):
        raise Exception("End of x array doesn't match.")
    return newx, result

def nconvolve(x1,f1,x2,f2,n):
    """
    Convolve f1 with f2 n times while taking x values and padding into account.

    Parameters
    ----------
    x1 : ndarray
        x values for the first function
    f1 : ndarray
        y values for the first function. Must have same length as x1
    x2 : ndarray
        x values for the second function
    f2 : ndarray
        y values for the second function. Must have same length as x2
    n : int
        Number of times to convolve the functions

    Returns
    -------
    ndarray
        The list of x values for the convolved function
    ndarray
        The list of y values for the convolved function

    """

    xarr = [x1]
    Harr = [f1]
    for ni in range(n):
        try:
            x,H = convolve(xarr[-1],Harr[-1],x2,f2)
        except Exception as e:
            print("Error for nconvolve ni={}".format(ni))
            raise e
        xarr.append(x)
        Harr.append(H)
    return xarr[-1],Harr[-1]

def find_total_n(x, step):
    """
    Count the total number of stars that went into calculating the wavelet
    coefficient for that point (using the starlet transform).
    This function uses the same convolution as the starlet transform.

    Parameters
    ----------
    x : ndarray
        Histogram data used for wavelet transformation
    step : int
        Scale parameter from wavelet transformation

    Returns
    -------
    ndarray
        An array with the same shape as x with the value of each pixel representing
        the number of stars contributing to the wavelet transformation at that point

    """

    x = x.astype(np.float32)
    ndim = len(x.shape)
    kernel = starlet.get_kernel(1,1,1,step)

    if ndim == 2:
        import cv2
        result = cv2.sepFilter2D(x, cv2.CV_32F, kernelX = kernel, kernelY = kernel)
        return result

    else:
        result = x
        import scipy.ndimage
        for dim in range(ndim):
            result = scipy.ndimage.filters.convolve1d(result, kernel, axis = dim, mode='reflect', cval = 0.0)
    return result

def F(H,x,c,typ='max'):
    """
    Perform an integration of the PDF (H) from -inf to c (the wavelet coefficient).
    Taken from Slezak et al. (1993) equation 9

    Parameters
    ----------
    H : ndarray
        y values for the PDF of the distribution of wavelet coefficients (obtained
        from convolving the result from get_pdf_single_event)
    x : int
        x values for the PDF
    c : int
        The wavelet coefficient
    typ : string ('max' | 'min'); default='max'
        'max' if c is a local maximum, 'min' if c is a local minimum

    Returns
    -------
    float
        A value between 0 and 1 representing the probabilty that the wavelet coefficient
        is significant

    """

    cdf = np.cumsum(H)
    cdf /= cdf[-1]
    i = np.digitize(c,x)-1
    i = max(i,0)
    # i = min(i,len(x)-1)
    if (typ == 'max'):
        return cdf[i]
    if (typ == 'min'):
        return 1-cdf[i]

def reduce(w,n,wavelet,j):
    """
    This function reduces the wavelet coefficient according to equation 2.17
    in Starck et al. (1998)
    """

    if (isinstance(w,(list,np.ndarray))):
        mask = n > 0
        result = np.zeros(len(w))
        result[mask] = w[mask]/np.sqrt(n[mask])/np.std(wavelet) #*2**j
        return result
    else:
        return w/np.sqrt(n)/np.std(wavelet) #*2**j

def get_local_extrema(wt,xedges,yedges,j):
    """
    Returns locations of local maxima and minima in the image wt (the
    wavelet transformed image). xedges and yedges are the locations of the bins
    used in the original histogram.

    Parameters
    ----------
    wt : ndarray
        Wavelet transformed image
    xedges : ndarray
        Locations of x bins from histogram calculation
    yedges : ndarray
        Locations of y bins from histogram calculation
    j : int
        Scale parameter used in wavelet transformation

    Returns
    -------
    indexmax : ndarray
        List of the maxima as indices of the wt image
    indexmin : ndarray
        List of the minima as indices of the wt image
    maxima : ndarray
        List of the maxima in physical units (transformed using xedges and yedges)
    minima : ndarray
        List of the minima in physical units (transformed using xedges and yedges)

    """

    indexmax = peaks(wt.T,min_distance=2**j,edges = 2**j)
    indexmin = peaks(-1*wt.T,min_distance=2**j,edges = 2**j)
    indexmin[:,2] *= -1

    maxima = np.zeros(np.shape(indexmax),dtype=np.float64)
    minima = np.zeros(np.shape(indexmin),dtype=np.float64)

    for m_p,m in zip([indexmax,indexmin],[maxima,minima]):
        m[:,2] = m_p[:,2]
        for i,xy in enumerate([xedges,yedges]):
            m[:,i] = m_p[:,i]*(xy[1] - xy[0])
            m[:,i] += xy[0]

    return indexmax,indexmin,maxima,minima

def get_all_points(wt,xedges,yedges):
    """
    Obtains lists of all positive and negative values in the image wt, so that
    we can calculate the significance of each point in the image.

    Parameters
    ----------
    wt : ndarray
        Wavelet transformed image
    xedges : ndarray
        Locations of x bins from histogram calculation
    yedges : ndarray
        Locations of y bins from histogram calculation

    Returns
    -------
    indexmax : list
        List of the maxima as indices of the wt image
    indexmin : list
        List of the minima as indices of the wt image
    maxima : list
        List of the maxima in physical units (transformed using xedges and yedges)
    minima : list
        List of the minima in physical units (transformed using xedges and yedges)

    """

    pos = np.where(wt.T >= 0)
    indexmax = np.array(list(zip(pos[0],pos[1],wt.T[pos])))
    neg = np.where(wt.T < 0)
    indexmin = np.array(list(zip(neg[0],neg[1],wt.T[neg])))

    maxima = np.zeros(np.shape(indexmax),dtype=np.float64)
    minima = np.zeros(np.shape(indexmin),dtype=np.float64)

    for m_p,m in zip([indexmax,indexmin],[maxima,minima]):
        m[:,2] = m_p[:,2]
        for i,xy in enumerate([xedges,yedges]):
            m[:,i] = m_p[:,i]*(xy[1] - xy[0])
            m[:,i] += xy[0]

    return indexmax,indexmin,maxima,minima

def get_pdf_single_event(wavelet,gridsize=200):
    """
    Calculate the histogram of a wavelet function using seaborn kernel density
    estimator.

    This histogram represents the PDF for a single event (described in Slezak et al. (1993)
    section 3.1)

    Parameters
    ----------
    wavelet : ndarray
        y values for the wavelet function (generated from B3-spline)
    gridsize : int, optional
        Gridsize to be used in seaborn kdeplot function. (default: 200)

    Returns
    -------
    x : ndarray
        x values for the PDF of the wavelet function
    y : ndarray
        y values for the PDF of the wavelet function

    """

    fig = plt.figure()
    ax = sns.kdeplot(wavelet,bw_adjust=0.2,gridsize=int(gridsize))
    H1 = ax.lines[0].get_ydata()
    x1 = ax.lines[0].get_xdata()
    H1 = H1/np.sum(H1)
    plt.close()
    return x1,H1

def get_cl(sg,epsilons = [0.8413,0.9772,0.9987]):
    """
    Calculate the confidence levels for the detected peaks output from the wavelet
    transformation.

    Parameters
    ----------
    sg : ndarray
        Significance values for the extrema (determined from get_significance)
    epsilons : ndarray, optional
        Values for the significance cutoffs for confidence levels. Default values
        are the integrals of a normal distribution (N(0,1)) from -inf to 1, 2, and 3.
        (default: [0.8413,0.9772,0.9987])

    Returns
    -------
    ndarray
        List of confidence levels for input significance values

    """
    cl = np.zeros(len(sg))
    for i in range(len(epsilons)):
        cl[sg > epsilons[i]] = i+1
    return cl

def phi_norm(scl,x):
    """
    Normalized scaling function (B3-spline) to given scale.

    Parameters
    ----------
    scl : int
        Scale of the wavelet transformation
    x : float | ndarray
        x value(s) to evaluate the function at

    Returns
    -------
    float | ndarray
        y values of the normalized scaling function. Same size as x

    """
    lnsp = np.linspace(-5*scl,5*scl,int(1e3))
    return b3(x/scl)/np.trapz(b3(lnsp/scl),lnsp)

# ------------------------------------------------------------------------------
# ------------------------------  MAIN FUNCTIONS -------------------------------

def get_significance(wt,max_i,min_i,ns,j,nmin={'max':3,'min':4},verbose=0,return_pdfs=False,reduce_wavelet=False):
    """
    Use the F(c) integral (F function) to determine significance values for each
    extrema found in the wavelet image wt.

    Parameters
    ----------
    wt : ndarray
        The wavelet transformed image
    max_i : list
        List of local maxima as indices of the wt image (not physical units)
    min_i : list
        List of local minima as indices of the wt image (not physical units)
    ns : dict
        Dictionary with each key corresponding to a list of number of stars used
        to calculate each extremal wavelet coefficient. Required keys: "max", "min"
    j : int
        The wavelet transformation band that the wt image corresponds to
    nmin : dict | int, optional
        If int, specifies the minimum number of events to consider for significance.
        You can specify different values for minima vs maxima by passing a dictionary
        with keys "min" and "max". (default: {'max':3,'min':4})
    verbose : bool, optional
        Will print output updating on the progress of the function (default: False)
    return_pdfs : bool, optional
        Will also return array of PDFs for each n value in addition to significances
        (default: False)
    reduce_wavelet : bool, optional
        Perform the reduction of the PDFs and wavelet coefficients outlined in
        Starck et al. (1998) section 2.1.5. (default: False)

    Returns
    -------
    sig : dict
        Dictionary with keys "min" and "max" which have values of an array where
        the first two columns are the x and y positions of the extrema and the third
        column is the significance value

    -- only returned if return_pdfs == True --
    pdfs : list
        List of tuples containing an array of x values and y values for a given
        number of events, n. Use n as the index of the pdfs list.

    """

    # Get the PDF for a single event using the continuous wavelet (psi(x))
    scl = 2**j
    psix = np.linspace(-4*scl,4*scl,int(1e6))
    phi_norm_partial = partial(phi_norm,scl)
    wavelet = psi2d(0,psix,phi_norm_partial)

    sigpsi = np.std(wavelet)
    x1,H1 = get_pdf_single_event(wavelet,gridsize=1000)
    deltax = x1[1]-x1[0]

    # Gather the list of all the n values that we will need and autoconvolve the
    #   above PDF n times
    if (verbose):
        print("Convolving PDFs...",end="",flush=True)
        stime = time.time()
    all_ns = np.concatenate((ns['max'],ns['min']))
    unique_ns = np.int_(np.unique(np.concatenate(([1],all_ns))))
    pdfs = [0]*(max(unique_ns)+1)
    pdfs[1] = (x1,H1)
    for i,un in enumerate(unique_ns):
        if (un <= 1):
            continue

        # If n is less than or equal to 100, convolve the PDF
        if (un <= 100):
            prevn = unique_ns[i-1]
            deltan = un - prevn
            pdfs[un] = nconvolve(*pdfs[prevn],x1,H1,deltan)

        # If n is larger than 100, it is well approximated by a Gaussian with a
        #   mean of zero and a standard deviation equal to the standard deviation of
        #   the wavelet multiplied by the square root of the number of events.
        # This drastically reduces computation time
        else:
            gauss_sig = sigpsi*np.sqrt(un)
            gauss_xs = np.linspace(-5*gauss_sig,5*gauss_sig,int(10*gauss_sig/deltax))
            pdfs[un] = (gauss_xs,gaus(gauss_xs,0,gauss_sig))

    # Reduce the wavelet PDF according to equation 2.12 in Starck et al. (1998)
    #   by dividing by the standard deviation of the PDF (standard deviation of
    #   the wavelet multiplied by the square root of the number of events).
    if (reduce_wavelet):
        for un in unique_ns:
            if (un == 0):
                continue
            x,H = pdfs[un]
            x = x/(np.sqrt(un)*sigpsi)
            H = H/np.trapz(H,x)
            pdfs[un] = (x,H)
    if (verbose):
        print("done ({:3.3f} s)".format(time.time() - stime))

    sig = {}
    try:
        int(nmin)
        nmin = {'max':nmin,'min':nmin}
    except:
        pass

    if (verbose):
        print("Calculating significance...",end="",flush=True)
        stime = time.time()

    # For each extrema, perform the integration of the PDF (F(c)) to determine the
    #   significance of the wavelet coefficient.
    for typ,mm in zip(['max','min'],[max_i,min_i]):

        sig[typ] = np.zeros((len(mm),3))
        v = mm[:,2]
        if (reduce_wavelet):
            v = reduce(mm[:,2],ns[typ],wavelet,j)
        for i,m in enumerate(mm):
            n = int(ns[typ][i])
            sig[typ][i][:2] = [int(m[0]),int(m[1])]
            if (n >= nmin[typ]):
                xi,fi = pdfs[n]
                sig[typ][i][2] = F(fi,xi,v[i],typ)


    if (verbose):
        print("done ({:3.3f} s)".format(time.time() - stime))

    if (return_pdfs):
        return sig,pdfs
    else:
        return sig

def _sim(x,x_error,y,y_error,bands,plot_range,bins,k):
    xx = np.random.normal(x,x_error)
    yy = np.random.normal(y,y_error)
    if (plot_range is None):
        h,xedges,yedges = np.histogram2d(xx,yy,bins=bins)
    else:
        h,xedges,yedges = np.histogram2d(xx,yy,bins=bins,range=plot_range)
    wt,scale = starlet.starlet_transform(h.T,num_bands=max(bands)+1,gen2=False)
    output = []
    for i in range(len(bands)):
        j = bands[i]
        if (len(scale) <= bands[i]):
            raise RuntimeException("Band {} not fount in wavelet transform scales.".format(bands[i]))
        peak_max = np.array(peaks(wt[j].T,min_distance=2**j))
        peak_min = np.array(peaks(-1*wt[j].T,min_distance=2**j))
        if (len(peak_max) == 0):
            peak_info = [[],[]]
        else:
            peak_info = [peak_max[:,0],peak_min[:,0]]
        if (len(peak_min) == 0):
            peak_info.extend([[],[]])
        else:
            peak_info.extend([peak_max[:,1],peak_min[:,1]])
        output.append(peak_info)
    return output

def simulate(x,x_error,y,y_error,maximum,minimum,N=10,min_d=2,bands=[4],plot_range=None,bins=[100,100],multiprocessing=False,verbose=False):
    '''
    Takes x and y data and their associated errors and performs Monte Carlo simulations.

    Parameters
    ----------
    x : list (n,)
        x coordinate values of the data points
    x_error : list (n,)
        Errors on the x values for each data point
    y : list (n,)
        y coordinate values of the data points
    y_error : list (n,)
        Errors on the y values for each data point
    N : int, optional
        Number of simulations to execute. (default: 10)
    min_d : float, optional
        The number of pixels allowed for the peak to move between detections.
        (default: 2)
    bands : list (x,), optional
        Which wavelet bands to compute. Band values correspond to scales of 2**(band).
        (default: [4]).
    plot_range : list (2,2) | None, optional
        The x and y limits of the histogram in physical units. If None, np.histogram2d
        will automatically set ranges to include all data. (default: None)
    bins : list(2,), optional
        Bin size for the histogram calculation. (default: [100,100])
    multiprocessing : int | boolean, optional
        Whether or not to use python's multiprocessing module to run simulations
        on multiple cores. If False | 0, will linearly loop through all simulations
        on a single core. If True | 1, will automatically determine how many cores
        to use to asynchronously run simulations. Can specify exactly how many cores
        to use by passing in an int > 1. (default: False)
    verbose : boolean, optional
        Whether or not to print progress and statistics. (default: False)

    Returns
    -------
    count_sim : dict
        Dictionary with keys "min" and "max" which have values of an array indexed
        by bands and by the scale which contains a numpy array of the number of
        times each extremum is detected.

    '''

    count_max = [np.zeros(len(maximum[i][:])) for i in range(len(bands))]
    count_min = [np.zeros(len(minimum[i][:])) for i in range(len(bands))]
    max_x_sim = [[] for i in range(len(bands))]
    min_x_sim = [[] for i in range(len(bands))]
    max_y_sim = [[] for i in range(len(bands))]
    min_y_sim = [[] for i in range(len(bands))]

    sim_partial = partial(_sim,x,x_error,y,y_error,bands,plot_range,bins)

    sstime = time.time()
    if (verbose):
        print("Running sims...",end="")
    if (multiprocessing):
        import multiprocessing as mp
        from tqdm import tqdm_notebook as tqdm
        ncores = None
        if (multiprocessing > 1):
            ncores = multiprocessing
        with mp.Pool(ncores) as p:
            if (verbose):
                results = list(tqdm(p.imap(sim_partial, range(N)),total=N,position=0,leave=True))
            else:
                results = p.map(sim_partial, range(N))
    # output is the result of each run of the simulation
    else:
        results = []
        for k in range(N):
            if (verbose):
                print("\rRunning sims...{}/{} ".format(k+1,N),end="")
            results.append(sim_partial(k))

    if (verbose):
        print("done ({:3.3f} s)".format(time.time()-sstime))
        print("Saving results...",end="")
    stime2 = time.time()
    for output in results:
        # one peak_info for each band
        for i,peak_info in enumerate(output):
            max_x_sim[i].extend(peak_info[0])
            min_x_sim[i].extend(peak_info[1])
            max_y_sim[i].extend(peak_info[2])
            min_y_sim[i].extend(peak_info[3])

    for i in range(len(bands)):
        max_x_sim[i] = np.array(max_x_sim[i])
        min_x_sim[i] = np.array(min_x_sim[i])
        max_y_sim[i] = np.array(max_y_sim[i])
        min_y_sim[i] = np.array(min_y_sim[i])
        for ii in range(max([len(maximum[i][:]),len(minimum[i][:])])):
            if (ii < len(maximum[i][:])):
                count_max[i][ii] += np.count_nonzero(((max_x_sim[i] - maximum[i][ii][0])**2 + (max_y_sim[i] - maximum[i][ii][1])**2) <= min_d**2)
            if (ii < len(minimum[i][:])):
                count_min[i][ii] += np.count_nonzero(((min_x_sim[i] - minimum[i][ii][0])**2 + (min_y_sim[i] - minimum[i][ii][1])**2) <= min_d**2)
    count_sim = {'max':count_max,'min':count_min}
    if (verbose):
        print("done ({:3.3f} s)".format(time.time()-stime2))
    return count_sim

def run_wavelet(x=None,y=None,plot_range=None,bins=100,
        hist=None,xedges=None,yedges=None,
        bands=[3,4,5],nmin={'max':3,'min':4},
        allpoints=False,verbose=False,extrema=True,extra_output=False,
        reduce_wavelet=False,x_error=None,y_error=None,run_simulations=False,N=10,multiprocessing=False):
    """
    Takes x and y data, performs the histogram, wavelet transform, and significance
    calculation for each extremum.

    Note that the "bands", "bins", and "plot_range" parameters need to be coordinated
    such that you are detecting structures of the desired size. For a given band,
    structures of size 2**(band) in pixels will be detected. Therefore, consider
    your number of bins and plot_range to determine the physical size of the detected
    structures.

    Must supply either:
        (1) x and y (in this case, the code will create the histogram), OR
        (2) hist, xedges, and yedges (a histogram that the user has created)

    Parameters
    ----------
    x : list (n,) | None, optional
        x coordinate values of the data points
    y : list (n,) | None, optional
        y coordinate values of the data points
    plot_range : list (2,2) | None, optional
        The x and y limits of the histogram in physical units. If None, np.histogram2d
        will automatically set ranges to include all data. (default: None)
    bins : list (2,) | int, optional
        The number of bins in x and y when performing the initial histogram. If
        an int is passed in, the same number of bins are used for x and y. (default: 100)
    hist : array (n,n) | None, optional
        The 2D image output from e.g. np.histogram2d. If x and y are supplied, this parameter
        is ignored.
    xedges : array (n,) | None, optional
        The list of x edges for the supplied 2D histogram. If hist is supplied, this must
        be supplied.
    yedges : array (n,) | None, optional
        The list of y edges for the supplied 2D histogram. If hist is supplied, this must
        be supplied.
    bands : list (x,), optional
        Which wavelet bands to compute. Band values correspond to scales of 2**(band).
        (default: [3,4,5])
    nmin : dict | int, optional
        If int, specifies the minimum number of events to consider for significance.
        You can specify different values for minima vs maxima by passing a dictionary
        with keys "min" and "max". (default: {'max':3,'min':4})
    allpoints : bool, optional
        Whether to calculate the significance for every point in the image. If True,
        the function will return an array the same size as the histogram image with
        significance values for each pixel. If False, it will only calculate significance
        values for the extremal points. (default: False)
    verbose : bool, optional
        Will print output updating on the progress of the function (default: False)
    extrema : bool, optional
        Will calculate the significance of each extremal point (default: True)
    extra_output : bool, optional
        Modifies the returned maximum, minimum, indexmax, and indexmin arrays. If
        False, they include the extrema locations and significance values. If True,
        they also include the wavelet coefficients, and number of stars used to
        calculate the wavelet. (default: False)
    reduce_wavelet : bool, optional
        Perform the reduction of the PDFs and wavelet coefficients outlined in
        Starck et al. (1998) section 2.1.5. (default: False)
    x_error : list (n,) | None, optional
        Errors on the x values for each data point supplied in x. Must be supplied
        if run_simulations == True. (default: None)
    y_error : list (n,) | None, optional
        Errors on the y values for each data point supplied in y. Must be supplied
        if run_simulations == True. (default: None)
    run_simulations : boolean, optional
        Whether or not to run Monte Carlo simulations to propogate errors supplied
        in x_error and y_error through wavelet transformation. (default: False)
    N : int, optional
        Number of Monte Carlo simulations to run. (default: 10)
    multiprocessing : int | boolean, optional
        Whether or not to use python's multiprocessing module to run simulations
        on multiple cores. If False | 0, will linearly loop through all simulations
        on a single core. If True | 1, will automatically determine how many cores
        to use to asynchronously run simulations. Can specify exactly how many cores
        to use by passing in an int > 1. (default: False)

    Returns
    -------
    img : list
        A list containing the output from np.histogram2d (histogram, xedges, and yedges)
    wt : list (len(bands),)
        The wavelet transformed images at each band supplied in parameter "bands".
        Each item in the list is an ndarray with size (bins[0],bins[1]) (the output
        from the starlet transform)

    -- only returned if allpoints == True --
    significance_image : list (len(bands),)
        Each item in the list is an ndarray with size (bins[0],bins[1]) with each
        pixel value equal to the significance value of the wavelet coefficient
        corresponding to that pixel

    -- only returned if extrema == True --
    maximum : list (len(bands),)
        List of the local maxima for each wavelet band. Columns are x-position in
        physical coordinates, y-position in physical coordinates, significance value,
        percent of Monte Carlo sims where this peak appeared (only if run_simulations == True),
        (the following included if extra_output == True) unreduced wavelet coefficient,
        number of stars used to calculate the wavelet (result of find_total_n)
    minimum : list (len(bands),)
        As above except for local minima
    indexmax : list (len(bands),)
        As maximum except the first two columns for each wavelet band are the
        x-position and y-position in indices of the wt image
    indexmin : list (len(bands),)
        As above except for local minima


    """

    if (isinstance(bins,int)):
        bins = [bins,bins]
    if (isinstance(bands,int)):
        bands = [bands]
    if (run_simulations):
        if ((x_error is None) or (y_error is None)):
            raise Exception("Can't run simulations without x_error and y_error specified.")

    # Perform the histogram and the wavelet transform
    if ((x is not None) & (y is not None)):
        if (plot_range is None):
            h,xedges,yedges = np.histogram2d(x,y,bins=bins)
        else:
            h,xedges,yedges = np.histogram2d(x,y,bins=bins,range=plot_range)
    elif ((hist is not None) & (xedges is not None) & (yedges is not None)):
        h = hist
        bins = [len(xedges)-1,len(yedges)-1]
        plot_range = [[xedges[0],xedges[-1]],[yedges[0],yedges[-1]]]
    else:
        raise RuntimeException("Please supply either x and y, or hist, xedges, and yedges.")
    wt,scale = starlet.starlet_transform(h.T,num_bands=max(bands)+1,gen2=False)
    mrs = []
    max_out = []
    min_out = []
    wt2 = []
    max_out_i = []
    min_out_i = []

    for s in range(len(bands)):
        j = bands[s]
        wt2.append(wt[j])

    # Loop through each of the output bands
    if (extrema | allpoints):
        for s in range(len(bands)):

            totalstime = time.time()

            if (len(scale) <= bands[s]):
                raise RuntimeException("Band {} not found in wavelet transform scales.".format(bands[s]))
            j = bands[s]

            if (verbose):
                print("band = {}".format(bands[s]))

            if (verbose):
                print("Finding extrema...",end="",flush=True)
                stime = time.time()

            max_i,min_i,maxima,minima = get_local_extrema(wt[j],xedges,yedges,j=j)
            if (allpoints):
                pos_i,neg_i,pos,neg = get_all_points(wt[j],xedges,yedges)
            else:
                pos_i,neg_i,pos,neg = max_i,min_i,maxima,minima

            ns = {}
            all_ns = find_total_n(h,2**j)
            ns['max'] = np.array([all_ns[tuple(np.int_(p[:2]))] for p in pos_i])
            ns['min'] = np.array([all_ns[tuple(np.int_(p[:2]))] for p in neg_i])
            if (verbose):
                print("done ({:.3f} s)".format(time.time() - stime))

            if verbose:
                print("len(pos) = {}, len(neg) = {}".format(len(pos),len(neg)))

            mrs.append({'max':np.zeros((bins[0],bins[1])),'min':np.zeros((bins[0],bins[1]))})
            if (allpoints == True):
                sig = get_significance(wt[j],pos_i,neg_i,ns,j=j,nmin=nmin,verbose=verbose,reduce_wavelet=reduce_wavelet)

                for m,typ in zip([pos_i,neg_i],['max','min']):
                    for k in range(len(m[:,0])):
                        # mrs[s][typ][int(m[k,0]),int(m[k,1])] = out[typ][k]
                        mrs[s][typ][int(m[k,0]),int(m[k,1])] = sig[typ][k][2]
                    # mrs[s][typ][mrs[s][typ] == 0] = np.nan

            if (extrema == True):
                if (verbose):
                    print("Evaluating extrema...",end="",flush=True)
                    stime = time.time()

                ns_extrema = {}
                all_ns_extrema = find_total_n(h,2**j)
                ns_extrema['max'] = np.array([all_ns_extrema[tuple(np.int_(p[:2]))] for p in max_i])
                ns_extrema['min'] = np.array([all_ns_extrema[tuple(np.int_(p[:2]))] for p in min_i])

                if (allpoints == True):
                    sig_extrema = {'max':[],'min':[]}
                    for mkey,mi in zip(['max','min'],[max_i,min_i]):
                        for i in range(len(mi)):
                            # mask = ((mi[i][0]**2 + mi[i][1]**2) < 2**j)
                            # sig_extrema[mkey].append(np.mean(mrs[s][mkey][mask]))
                            sig_extrema[mkey].append([mi[i][0],mi[i][1],mrs[s][mkey][(int(mi[i][1]),int(mi[i][0]))]])
                        sig_extrema[mkey] = np.array(sig_extrema[mkey])
                else:
                    sig_extrema = get_significance(wt[j],max_i,min_i,ns_extrema,j=j,nmin=nmin,verbose=False,reduce_wavelet=reduce_wavelet)

                psix = np.linspace(-4,4,int(1e5))
                wavelet = psi(psix)

                if (extra_output):
                    for mout,miout,m,mi,t in zip([max_out,min_out],[max_out_i,min_out_i],[maxima,minima],[max_i,min_i],['max','min']):
                        # This should be the same whether we've reduced or not
                        if (reduce_wavelet):
                            m[:,2] = reduce(m[:,2],ns_extrema[t],wavelet,j)
                            mi[:,2] = reduce(mi[:,2],ns_extrema[t],wavelet,j)

                        mout.append(np.zeros((len(m),5)))
                        mout[s][:,:2] = m[:,:2]
                        mout[s][:,2] = sig_extrema[t][:,2]
                        mout[s][:,3] = m[:,2]
                        mout[s][:,4] = ns_extrema[t]

                        miout.append(np.zeros((len(m),5)))
                        miout[s][:,:2] = mi[:,:2]
                        miout[s][:,2] = sig_extrema[t][:,2]
                        miout[s][:,3] = mi[:,2]
                        miout[s][:,4] = ns_extrema[t]
                else:
                    for mout,miout,m,mi,t in zip([max_out,min_out],[max_out_i,min_out_i],[maxima,minima],[max_i,min_i],['max','min']):

                        mout.append(np.zeros((len(m),3)))
                        mout[s][:,:2] = m[:,:2]
                        mout[s][:,2] = sig_extrema[t][:,2]

                        miout.append(np.zeros((len(mi),3)))
                        miout[s][:,:2] = mi[:,:2]
                        miout[s][:,2] = sig_extrema[t][:,2]

                if (run_simulations):
                    for mout,m in zip([max_out,min_out,max_out_i,min_out_i],[maxima,minima,max_i,min_i]):
                        mout[s] = np.insert(mout[s],3,np.zeros(len(m)),axis=1)

                if (verbose):
                    print("done ({:3.3f} s)".format(time.time() - stime))

            if (verbose):
                print("band = {} done ({:3.3f} s)".format(bands[s],time.time() - totalstime))
                print("")

    if (run_simulations):
        if ((x is None) | (y is None)):
            print("Warning: Cannot run simulations without supplying x,y and x_error,y_error. Skipping.")
        if (verbose):
            time_start = time.time()
            print('Running Monte Carlo Simulations')
        count_sim = simulate(x=x,x_error=x_error,y=y,y_error=y_error,maximum=max_out_i,minimum=min_out_i, \
                                N=N,min_d=2**j/2,bands=bands,plot_range=plot_range,bins=bins, \
                                multiprocessing=multiprocessing,verbose=verbose)
        if (verbose):
            print('done ({:3.3f} s)'.format(time.time()-time_start))
        max_out = np.array(max_out,dtype='object')
        min_out = np.array(min_out,dtype='object')
        for i in range(len(bands)):
            max_out[i][:,3] = ((count_sim['max'][i])/N)
            min_out[i][:,3] = ((count_sim['min'][i])/N)
            max_out_i[i][:,3] = ((count_sim['max'][i])/N)
            min_out_i[i][:,3] = ((count_sim['min'][i])/N)

    output = [h,xedges,yedges],wt2
    if (allpoints == True):
        output = output + tuple([mrs])
    if (extrema == True):
        output = output + (np.array(max_out,dtype='object'),np.array(min_out,dtype='object'),np.array(max_out_i,dtype='object'),np.array(min_out_i,dtype='object'))
    return output

#####
#
# This code has been developed by Scott Lucchini and Emil Pellet at UW - Madison.
# Please see the guide at https://github.com/DOnghiaGroup/MGwave and direct any
# questions to lucchini@wisc.edu.
#
# This code is licensed under MIT license copyright 2021 Scott Lucchini.
#
###

import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import MGwave as wave


def runwt(R,phi,z,V_r,V_phi,rlim,philim,zlim,rmin,rmax,deltar,phicent,bands,wt_range,bins,V_rerr,V_phierr,multiprocessing,montecarlo,i):
    '''
    Run the wavelet transformation for a given radial bin. Automatically masks
    the data and runs the wavelet transformation with significance calculation
    and optionally Monte Carlo simulations.

    Usually run automatically from run_worms.

    '''
    r_bin = rmin + i*deltar
    zrmask = (abs(z)<zlim) & (R > r_bin-rlim) & (R < r_bin+rlim)
    if (phicent == 180):
        phimask = ((phi<phicent)&(phi>phicent-philim)) | ((phi>-phicent)&(phi<-phicent+philim))
    else:
        phimask = (phi > phicent-philim) & (phi < phicent+philim)
    mask = zrmask & phimask
    vr = V_r[mask]
    vphi = V_phi[mask]
    if ((V_rerr is not None) & (V_phierr is not None)):
        vr_err = V_rerr[mask]
        vphi_err = V_phierr[mask]
    else:
        vr_err = None
        vphi_err = None

    runsimulations = False
    Nsims = None
    if (montecarlo):
        runsimulations = True
        Nsims = montecarlo

    img,wt,maximum,minimum,indexmax,indexmin = wave.run_wavelet(vr,vphi, bands=bands, bins=bins,
                                                                plot_range=wt_range, allpoints=False,
                                                                extrema=True, verbose=False, extra_output=True,
                                                                reduce_wavelet=False,x_error=vr_err,y_error=vphi_err,
                                                                run_simulations=runsimulations,N=Nsims,
                                                                multiprocessing=multiprocessing)
    h,xe,ye=img

    return maximum,minimum,wt

def runwt_phi(R,phi,z,V_r,V_phi,rlim,philim,zlim,rcent,phimin,phimax,deltaphi,bands,wt_range,bins,V_rerr,V_phierr,multiprocessing,montecarlo,i):
    '''
    Run the wavelet transformation for a given radial bin. Automatically masks
    the data and runs the wavelet transformation with significance calculation
    and optionally Monte Carlo simulations.

    Usually run automatically from run_worms.

    '''
    phi_bin = phimin + i*deltaphi
    zrmask = (abs(z)<zlim) & (R > rcent-rlim) & (R < rcent+rlim)
    phiedges = np.array([phi_bin - philim, phi_bin + philim])
    # if ((phiedges[0] < 180) & (phiedges[1] > 180)):
    #     phimask = ((phi<180)&(phi>phiedges[0])) | ((phi>-180)&(phi<360-phiedges[1]))
    # else:
    #     phimask = (phi > phiedges[0]) & (phi < phiedges[1])
    if (np.any(phiedges > 180) | np.any(phiedges < -180)):
        raise Exception("Phi bin edges must not cross +/- 180 degrees.")
    phimask = (phi > phiedges[0]) & (phi < phiedges[1])
    mask = zrmask & phimask
    vr = V_r[mask]
    vphi = V_phi[mask]
    if ((V_rerr is not None) & (V_phierr is not None)):
        vr_err = V_rerr[mask]
        vphi_err = V_phierr[mask]
    else:
        vr_err = None
        vphi_err = None

    runsimulations = False
    Nsims = None
    if (montecarlo):
        runsimulations = True
        Nsims = montecarlo

    img,wt,maximum,minimum,indexmax,indexmin = wave.run_wavelet(vr,vphi, bands=bands, bins=bins,
                                                                plot_range=wt_range, allpoints=False,
                                                                extrema=True, verbose=False, extra_output=True,
                                                                reduce_wavelet=False,x_error=vr_err,y_error=vphi_err,
                                                                run_simulations=runsimulations,N=Nsims,
                                                                multiprocessing=multiprocessing)
    h,xe,ye=img

    return maximum,minimum,wt

def run_worms(R,phi,z,V_r,V_phi,rlim=0.1,philim=1.5,zlim=0.5,
                rmin=None,rmax=None,deltar=None,phicent=180,
                rcent=8.15,phimin=None,phimax=None,deltaphi=None,
                bands=[3],wt_range=[[-150,150],[50,350]],bins=[300,300],V_rerr=None,V_phierr=None,
                multiprocessing=False,montecarlo=False):
    '''
    Run the wavelet transformation on data automatically binning into a sequence
    of radial bins to track the evolution of moving groups throughout radius.
    Must supply r range (using rmin, rmax, deltar, phicent) OR
        phi range (using rcent, phimin, phimax, deltaphi)

    Parameters
    ----------
    R : ndarray
        Galactocentric radius values for all stars
    phi : ndarray
        Galactocentric aximuth values for all stars
    z : ndarray
        z height above the Galactic plane for all stars
    V_r : ndarray
        Radial velocity in Galactocentric coordinates for all stars
    V_phi : ndarray
        Azimuthal velocity in Galactocentric coordinates for all stars
    rlim : float, optional
        Width of each radial bin (i.e. deltar = 0.01 and rlim = 0.1
        will produce radial bins of [(5.0,5.1), (5.01,5.11), (5.02,5.22)])
    philim : float, optional
        Azimuthal size of the bins in degrees. (default: 1.5)
    zlim : float, optional
        Cutoff for vertical height of stars in kpc. (default: 0.5)
    rmin : float, optional
        Minimum Galactocentric radius. If supplied, must also supply rmax and deltar
        and must not supply phimin, phimax, or deltaphi. (default: None)
    rmax : float, optional
        Maximum Galactocentric radius. If supplied, must also supply rmin and deltar
        and must not supply phimin, phimax, or deltaphi. (default: None)
    deltar : float, optional
        Step size between radial bins (i.e. deltar = 0.01 and rlim = 0.1
        will produce radial bins of [(5.0,5.1), (5.01,5.11), (5.02,5.22)]).
        If supplied, must also supply rmin and rmax and must not supply 
        phimin, phimax, or deltaphi. (default: None)
    phicent : float, optional
        Azimuthal value to center all the bins on in degrees. (default: 180)
    phimin : float
        Minimum phi value. If supplied, must also supply phimax and deltaphi
        and must not supply rmin, rmax, or deltar. (default: None)
    phimax : float
        Maximum phi value. If supplied, must also supply phimax and deltaphi
        and must not supply rmin, rmax, or deltar. (default: None)
    deltaphi : float
        Step size between phi bins (i.e. deltaphi = 0.5 and philim = 3
        will produce phi bins of [(0,3), (0.5,3.5), (1,4)]).
        If supplied, must also supply phimin and phimax and must not supply 
        rmin, rmax, or deltar. (default: None)
    rcent : float
        Radial center for all phi bins. (default: 8.15)
    bands : list, optional
        Bands in which to compute the wavelet transform. Band values correspond
        to scales of 2**(band). (default: [3])
    wt_range : list (2,2), optional
        Range of V_r and V_phi in the histogram and wavelet transformation.
        (default: [[-150,150],[50,350]])
    bins : list(2,), optional
        Number of bins in V_r and V_phi for the histogram. (default: [300,300])
    V_rerr : ndarray | None, optional
        Errors in V_r for all stars. Must be supplied if multiprocessing >= 1.
        (default: None)
    V_phierr : ndarray | None, optional
        Errors in V_phi for all stars. Must be supplied if multiprocessing >= 1.
        (default: None)
    multiprocessing : int | boolean, optional
        Whether or not to use python's multiprocessing module. If montecarlo == True,
        multiple cores will be used to run Monte Carlo simulations for each radial bin.
        If montecarlo == False, multiple cores will be used to run the wavelet
        transformations at different radii.
        If False | 0, will linearly loop through all steps on a single core. If
        True | 1, will automatically determine how many cores to use to asynchronously
        loop. Can specify exactly how many cores to use by passing in an int > 1.
        (default: False)
    montecarlo : boolean, optional
        Whether or not to run Monte Carlo simulations for each radial bin. Requires
        V_rerr and V_phierr. (default: False)

    Returns
    -------
    allmaxs : list
        A list containing the "maximum" array (output from run_wavelet) for each
        radial bin.
    allmins : list
        A list containing the "minimum" array (output from run_wavelet) for each
        radial bin.
    allwts : list
        A list containing the "wt" array (output from run_wavelet) for each
        radial bin.

    '''

    if ((rmin is not None) & (rmax is not None) & (deltar is not None)):
        imax = int((rmax-rmin)/deltar) + 1
        runwt_partial = partial(runwt,R,phi,z,V_r,V_phi,rlim,philim,zlim,rmin,rmax,deltar,phicent, \
                            bands,wt_range,bins,V_rerr,V_phierr,multiprocessing,montecarlo)
    elif ((phimin is not None) & (phimax is not None) & (deltaphi is not None)):
        imax = int((phimax-phimin)/deltaphi) + 1
        runwt_partial = partial(runwt_phi,R,phi,z,V_r,V_phi,rlim,philim,zlim,rcent,phimin,phimax,deltaphi, \
                            bands,wt_range,bins,V_rerr,V_phierr,multiprocessing,montecarlo)
    if (multiprocessing and (montecarlo == False)):
        ncores = None
        if (multiprocessing > 1):
            ncores = multiprocessing
        with mp.Pool(ncores) as p:
            output = list(tqdm(p.imap(runwt_partial, range(imax)),total=imax,position=0,leave=True))
            # output = p.map(runwt_partial, range(imax))
    else:
        output = []
        for i in tqdm(range(imax)):
            # print("\rRunning {}/{}...".format(i+1,imax),end="")
            output.append(runwt_partial(i))
        print("done")

    allmaxs = [o[0] for o in output]
    allmins = [o[1] for o in output]
    allwts = [o[2] for o in output]

    return allmaxs,allmins,allwts

def plot_worms(allmaxs,rvals,ax=None,wormdelta=5,minmcval=None,plot_range=[[-120,120],[170,270]],sun_r=8.34,colorbar=True,cbar_nticks=9):
    '''
    Plot the detected maxima across all radii. Colored by radius, optionally requiring
    all points to be within "wormdelta" of another point, or have a Monte Carlo
    simulation value greater than "minmcval".

    Parameters
    ----------
    allmaxs : ndarray
        A list containing the "maximum" array (output from run_wavelet) for each
        radial bin. Output from rum_worms.
    rvals : ndarray
        A list of all the radius values used to calculate allmaxs
    ax : matplotlib.pyplot.axis | None, optional
        A matplotlib axis object onto which to plot the results. If None, it will
        be generated.
    wormdelta : int, optional
        The maximum separation in pixels between detected maxima at adjacent radii.
        Setting this to a low enough value will remove outlier points that are not
        members of worms. Setting this too low may truncate some worms.
        (default: 5)
    minmcval : float | None, optional
        The minimum percentage of Monte Carlo simulations that a maximum must appear
        in. A typical value is ~0.8. Requires that run_worms was run with montecarlo == True.
        If None, will not provide any additional mask. (default: None)
    plot_range : list(2,2), optional
        V_r and V_phi axes limits for the plot. (default: [[-120,120],[170,270]])
    sun_r : float | None, optional
        Radius value of the sun in kpc. Detected maxima at this radius will be
        plotted with square markers. If None, points for all radii will have
        circle markers. (default: 8.34)
    colorbar : boolean, optional
        Whether or not to display the colorbar. (default: True)

    '''

    if ax is None:
        fig,ax = plt.subplots(figsize=(7,5))
    cmap = 'turbo'
    rmin = rvals[0]
    rmax = rvals[-1]
    deltar = rvals[1]-rvals[0]
    norm = mpl.colors.Normalize(rmin,rmax)
    sun_i = 0
    if (sun_r is not None):
        # sun_i = np.where(np.abs(rvals-sun_r) == min(np.abs(rvals-sun_r)))[0][0]
        sun_i = np.digitize(sun_r,rvals)-1

    for i in range(len(allmaxs)):
        mx = allmaxs[i][0]
        cl = wave.get_cl(mx[:,2])
        clmask = cl >= 2
        if (minmcval is not None):
            mc = mx[:,3]
            mcmask = mc >= minmcval
            mask = clmask & mcmask
        else:
            mask = clmask
        zvals = np.ones(len(mx))*(rmin + i*deltar)

        wormmask = []
        if (i==len(allmaxs)-1):
            prevmx = allmaxs[i-1][0]
            prevcl = wave.get_cl(prevmx[:,2])
            if (minmcval is not None):
                prevmx = prevmx[(prevcl >= 2) & (prevmx[:,3] >= minmcval)]
            else:
                prevmx = prevmx[(prevcl >= 2)]
            for p in mx[mask]:
                wormmask.append(len(prevmx[(prevmx[:,0]-p[0])**2 + (prevmx[:,1]-p[1])**2 < wormdelta**2]) > 0)
        elif (i==0):
            nextmx = allmaxs[i+1][0]
            nextcl = wave.get_cl(nextmx[:,2])
            if (minmcval is not None):
                nextmx = nextmx[(nextcl >= 2) & (nextmx[:,3] >= minmcval)]
            else:
                nextmx = nextmx[(nextcl >= 2)]
            for p in mx[mask]:
                wormmask.append(len(nextmx[(nextmx[:,0]-p[0])**2 + (nextmx[:,1]-p[1])**2 < wormdelta**2]) > 0)
        else:
            prevmx = allmaxs[i-1][0]
            prevcl = wave.get_cl(prevmx[:,2])
            nextmx = allmaxs[i+1][0]
            nextcl = wave.get_cl(nextmx[:,2])
            if (minmcval is not None):
                prevmx = prevmx[(prevcl >= 2) & (prevmx[:,3] >= minmcval)]
                nextmx = nextmx[(nextcl >= 2) & (nextmx[:,3] >= minmcval)]
            else:
                prevmx = prevmx[(prevcl >= 2)]
                nextmx = nextmx[(nextcl >= 2)]
            for p in mx[mask]:
                prevtest = len(prevmx[(prevmx[:,0]-p[0])**2 + (prevmx[:,1]-p[1])**2 < wormdelta**2]) > 0
                nexttest = len(nextmx[(nextmx[:,0]-p[0])**2 + (nextmx[:,1]-p[1])**2 < wormdelta**2]) > 0
                wormmask.append(prevtest | nexttest)

        if ((sun_i is not None) & (i == sun_i)):
            ax.scatter(mx[mask][wormmask][:,0],mx[mask][wormmask][:,1],s=35,
                   c=zvals[mask][wormmask],cmap=cmap,norm=norm,marker='s',edgecolors='k',linewidths=0.7)
        else:
            ax.scatter(mx[mask][wormmask][:,0],mx[mask][wormmask][:,1],s=7,
                   c=zvals[mask][wormmask],cmap=cmap,norm=norm)

    ax.set_xlim(plot_range[0])
    ax.set_xlabel(r"V$_r$ (km s$^{-1}$)")
    ax.set_ylim(plot_range[1])
    ax.set_ylabel(r"V$_\phi$ (km s$^{-1}$)")
    if (minmcval is not None):
        ax.set_title(r"MC >= {:.2f}; worm$\Delta$ <= {:.0f}".format(minmcval,wormdelta))
    else:
        ax.set_title(r"worm$\Delta$ <= {:.0f}".format(wormdelta))
    if (colorbar):
        cbar_ticks = np.array(range(cbar_nticks))*(rmax-rmin)/float(cbar_nticks-1) + rmin
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),label='R (kpc)',ticks=cbar_ticks)
    # plt.show()

def plot_contours(allmaxs,allwts,rvals,clevel,plot_rlims=None,ax=None,wt_range=[[-150,150],[50,350]],bins=[300,300],
                alpha=1,plot_range=[[-120,120],[170,270]],sun_r=8.34,plot_solar_groups=True,colorbar=True,cbar_nticks=9):
    '''
    Plot the detected maxima across all radii. Colored by radius, optionally requiring
    all points to be within "wormdelta" of another point, or have a Monte Carlo
    simulation value greater than "minmcval".

    Parameters
    ----------
    allmaxs : ndarray
        A list containing the "maximum" array (output from run_wavelet) for each
        radial bin. Output from rum_worms.
    allwts : ndarray
        A list containing the "wt" array (output from run_wavelet) for each
        radial bin. Output from rum_worms.
    rvals : ndarray
        A list of all the radius values used to calculate allmaxs
    clevel : float
        Contour level to draw as a percent of the maximum value (i.e. a value between
        0 and 1).
    plot_rlims : list(2,) | None, optional
        Minimum and maximum radius values to plot. If None, will plot the entire
        range computed in run_worms. (default: None)
    ax : matplotlib.pyplot.axis | None, optional
        A matplotlib axis object onto which to plot the results. If None, it will
        be generated.
    wt_range : list(2,2), optional
        The total size of the histograms. (The same as what was passed into wt_range
        in run_worms). (default: [[-150,150],[50,350]])
    bins : list(2,), optional
        The number of bins used in run_worms. (default: [300,300])
    alpha : float, optional
        Alpha opacity value for the plotted contours. (default: 1)
    plot_range : list(2,2), optional
        V_r and V_phi axes limits for the plot. (default: [[-120,120],[170,270]])
    sun_r : float | None, optional
        Radius value of the sun in kpc. Detected maxima at this radius will be
        plotted with square markers. If None, points for all radii will have
        circle markers. (default: 8.34)
    plot_solar_groups : boolean, optional
        If True, will plot the detected maxima at the solar radius (the detected
        moving groups in the solar neighborhood). (default: True)
    colorbar : boolean, optional
        Whether or not to display the colorbar. (default: True)

    '''

    if (ax is None):
        fig,ax = plt.subplots(figsize=(7,5))

    cmap = mpl.cm.get_cmap('turbo')
    rmin = rvals[0]
    rmax = rvals[-1]
    deltar = rvals[1]-rvals[0]
    norm = mpl.colors.Normalize(rmin,rmax)
    sun_i = np.where(np.abs(rvals-sun_r) == min(np.abs(rvals-sun_r)))[0][0]

    dxy = [float(p[1]-p[0])/bins[i] for i,p in enumerate(wt_range)]
    xe,ye = [np.array(range(int((p[1]-p[0])/dxy[i])))*dxy[i]+p[0] for i,p in enumerate(wt_range)]

    if (plot_rlims is None):
        plot_rlims = [rmin,rmax]

    for i,r in enumerate(rvals):

        if ((r < plot_rlims[0]) | (r > plot_rlims[1])):
            continue

        wt = allwts[i][0]

        wtlim = max(np.abs([max(np.reshape(wt,(-1,))),min(np.reshape(wt,(-1,)))]))
        ax.contour(xe,ye,wt,levels=np.array([clevel])*wtlim,linewidths=0.7,colors=[cmap(norm(r))],alpha=alpha)

        if ((plot_solar_groups) & (i == sun_i)):
            mx = allmaxs[i][0]
            cl = wave.get_cl(mx[:,2])
            clmask = cl >= 2
            zvals = np.ones(len(mx))*sun_r
            ax.scatter(mx[clmask][:,0],mx[clmask][:,1],s=35,
                   c=zvals[clmask],cmap=cmap,norm=norm,marker='s',edgecolors='k',linewidths=0.7)

    ax.set_title("Contours = {:.0f}%; R = [{:.2f},{:.2f}] kpc".format(clevel*100,*plot_rlims))
    ax.set_xlim(plot_range[0])
    ax.set_xlabel(r"V$_r$ (km s$^{-1}$)")
    ax.set_ylim(plot_range[1])
    ax.set_ylabel(r"V$_\phi$ (km s$^{-1}$)")
    if (colorbar):
        cbar_ticks = np.array(range(cbar_nticks))*(rmax-rmin)/float(cbar_nticks-1) + rmin
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),label='R (kpc)',ticks=cbar_ticks)
    # plt.show()

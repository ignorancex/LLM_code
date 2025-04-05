import numpy as np
import isodist
import mwdust
from isodist import FEH2Z, Z2FEH
from scipy.interpolate import interp1d
import multiprocessing
import tqdm
import sys
import apogee.select as apsel
from galpy.util import bovy_coords, _rotate_to_arbitrary_vector
from scipy.optimize import newton
from scipy.special import erfinv
from scipy.stats import norm
import densprofiles
import os

def join_on_id(dat1,dat2,joinfield='APOGEE_ID'):
    '''
    Takes two recarrays and joins them based on a ID string 
    '''
    #find common fields
    names1 = [dat1.dtype.descr[i][0] for i in range(len(dat1.dtype.descr))]
    names2 = [dat2.dtype.descr[i][0] for i in range(len(dat2.dtype.descr))]
    namesint = np.intersect1d(names1,names2)
    if joinfield not in namesint:
        return NameError('Field '+joinfield+' is not present in both arrays.')
    #work out which fields get appended from dat2
    descr2 = dat2.dtype.descr
    fields_to_append = []
    names_to_append = []
    for i in range(len(names2)):
        if names2[i] not in namesint:
            fields_to_append.append(descr2[i])
            names_to_append.append(names2[i])
        else:
            continue   
    # Faster way to join structured arrays (see https://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays)
    newdtype= dat1.dtype.descr+fields_to_append
    newdata= np.empty(len(dat1),dtype=newdtype)
    for name in dat1.dtype.names:
        newdata[name]= dat1[name]
    for f in names_to_append:
        newdata[f]= np.zeros(len(dat1))-9999.
    dat1= newdata
    
    hash1= dict(zip(dat1[joinfield],
                    np.arange(len(dat1))))
    hash2= dict(zip(dat2[joinfield],
                    np.arange(len(dat2))))
    common, indx1, indx2 = np.intersect1d(dat1[joinfield],dat2[joinfield],return_indices=True)
    for f in names_to_append:
        dat1[f][indx1]= dat2[f][indx2]
    return dat1

#effective selection function utilities
def generate_lowfeh_isogrid(mag=None):
    zs = np.arange(0.0001,0.0031,0.0001)
    base = os.environ['ISODIST_DATA']
    isoname = 'parsec1.2-'
    if mag is None:
        mag = '2mass-spitzer-wise-old'
    isolist = []
    for i in tqdm.tqdm(range(len(zs))): 
        file = os.path.join(base,isoname+mag,isoname+mag+'-Z-%5.4f.dat.gz' % zs[i])
        alliso = np.genfromtxt(file, dtype=None, names=True, skip_header=11)
        ages = np.unique(alliso['logAge'])
        for age in ages:
            mask = alliso['logAge'] == round(age,2)
            iso = alliso[mask]
            iso = iso[np.argsort(iso['Mini'])]
            deltam = iso['int_IMF'][1:]-iso['int_IMF'][:-1]
            iso = np.lib.recfunctions.append_fields(iso[1:], 'deltaM', deltam)
            isolist.append(iso)
    fulliso = np.concatenate([entry for entry in isolist])
    return fulliso
                       

def generate_isogrid():
    """
    generate a recarray with all the entries from PARSEC isochrones in isodist
    """
    zs = np.arange(0.0005,0.0605, 0.0005)
    zlist = []
    for i in range(len(zs)):
        zlist.append(format(zs[i],'.4f'))
    iso = isodist.PadovaIsochrone(type='2mass-spitzer-wise', Z=zs, parsec=True)

    logages = []
    mets = []
    js = []
    hs = []
    ks = []
    loggs = []
    teffs = []
    imf = []
    deltam = []
    M_ini = []
    M_act = []
    logL
    iso_logages = iso._logages
    iso_Zs = iso._ZS
    for i in tqdm.tqdm(range(len(iso_logages))):
        for j in range(len(iso_Zs)):
            thisage = iso_logages[i]
            thisZ = iso_Zs[j]
            thisiso = iso(thisage, Z=thisZ)
            so = np.argsort(thisiso['M_ini'])
            loggs.extend(thisiso['logg'][so][1:])
            logages.extend(thisiso['logage'][so][1:])
            mets.extend(np.ones(len(thisiso['H'][so])-1)*thisZ)
            js.extend(thisiso['J'][so][1:])
            hs.extend(thisiso['H'][so][1:])
            ks.extend(thisiso['Ks'][so][1:])
            teffs.extend(thisiso['logTe'][so][1:])
            imf.extend(thisiso['int_IMF'][so][1:])
            deltam.extend(thisiso['int_IMF'][so][1:]-thisiso['int_IMF'][so][:-1])
            M_ini.extend(thisiso['M_ini'][so][1:])
            M_act.extend(thisiso['M_act'][so][1:])
            logL.extend(thisiso['logL'][so][1:])
    logages = np.array(logages)
    mets = np.array(mets)
    js = np.array(js)
    hs = np.array(hs)
    ks = np.array(ks)
    loggs = np.array(loggs)
    teffs = 10**np.array(teffs)
    imf = np.array(imf)
    deltam = np.array(deltam)
    M_ini = np.array(M_ini)
    M_act = np.array(M_act)
    logL = np.array(logL)
    rec = np.recarray(len(deltam), dtype=[('logageyr', float),
                                          ('Z', float),
                                          ('J', float),
                                          ('H', float),
                                          ('K', float),
                                          ('logg', float),
                                          ('teff', float),
                                          ('int_IMF', float),
                                          ('deltaM', float),
                                          ('M_ini', float),
                                          ('M_act', float),
                                          ('logL', float)])

    rec['logageyr'] = logages
    rec['Z'] = mets
    rec['J'] = js
    rec['H'] = hs
    rec['K'] = ks
    rec['logg'] = loggs
    rec['teff'] = teffs
    rec['int_IMF'] = imf
    rec['deltaM'] = deltam
    rec['M_ini'] = M_ini
    rec['M_act'] = M_act
    rec['logL'] = logL
    return rec

def sampleiso(N, iso, return_inds=False, return_iso=False, lowfeh=True):
    """
    Sample isochrone recarray iso weighted by lognormal chabrier (2001) IMF
    """
    if lowfeh:
        logagekey = 'logAge'
        zkey = 'Zini'
        jkey, hkey, kkey = 'Jmag', 'Hmag', 'Ksmag'
    else:
        logagekey = 'logageyr'
        zkey = 'Z'
        jkey, hkey, kkey = 'J', 'H', 'K'
    weights = iso['deltaM']*(10**(iso[logagekey]-9)/iso[zkey])
    sort = np.argsort(weights)
    tinter = interp1d(np.cumsum(weights[sort])/np.sum(weights), range(len(weights[sort])), kind='linear')
    randinds = np.round(tinter(np.random.rand(N))).astype(np.int64)
    if return_inds:
        return randinds, iso[jkey][sort][randinds], iso[hkey][sort][randinds], iso[kkey][sort][randinds]
    elif return_iso:
        return iso[sort][randinds]
    else:
        return iso[jkey][sort][randinds], iso[hkey][sort][randinds], iso[kkey][sort][randinds]
    

    
def average_mass(iso, lowfehgrid=True):
    """
    find the average mass for a given slice of the isochrone recarray
    """
    if lowfehgrid:
        agekey = 'logAge'
        Zkey = 'Zini'
        Mkey = 'Mini'
    else:
        agekey = 'logageyr'
        Zkey = 'Z_ini'
        Mkey = 'M_ini'
    weights = iso['deltaM']*(10**(iso[agekey]-9)/iso[Zkey])
    return np.sum(iso[Mkey]*weights)/np.sum(weights)

def mass_ratio(iso, lowfehgrid=True, minjk=0.3, maxjk=9999.):
    """
    find the mass ratio between stars in the cuts adopted for giants in APOGEE, and the rest of the isochrones
    """
    if lowfehgrid:
        agekey = 'logAge'
        Zkey = 'Zini'
        Mkey = 'Mini'
        jkey, kkey= 'Jmag', 'Ksmag'
    else:
        agekey = 'logageyr'
        Zkey = 'Z_ini'
        Mkey = 'M_ini'
        jkey, kkey= 'J', 'K'
    weights = iso['deltaM']*(10**(iso[agekey]-9)/iso[Zkey])
    mask = (iso[jkey]-iso[kkey] > minjk) & (iso[jkey]-iso[kkey] < maxjk) & (iso['logg'] < 3.0) & (iso['logg'] > 1.0) & (iso['logAge'] > 10)
    return np.sum(iso[Mkey][mask]*weights[mask])/np.sum(iso[Mkey]*weights)

def APOGEE_iso_samples(nsamples, rec, fehrange=[-1,-1.5], lowfehgrid = True):
    """
    get samples from the isochrones in the apogee selection (including Log(g) cut for giants, and minimum mass)
    """
    trec = np.copy(rec)
    if lowfehgrid:
        logagekey = 'logAge'
        zkey = 'Zini'
        mkey = 'Mini'
        jkey, hkey, kkey = 'Jmag', 'Hmag', 'Ksmag'
    else:
        logagekey = 'logageyr'
        zkey = 'Z'
        mkey ='M_ini'
        jkey, hkey, kkey = 'J', 'H', 'K'
    mask = (trec[jkey]-trec[kkey] > 0.5)  & (Z2FEH(trec[zkey]) > fehrange[0]) & (Z2FEH(trec[zkey]) < fehrange[1]) & (trec[mkey] > 0.75) & (trec['logg'] < 3.) & (trec['logg'] > 1.) & (trec[logagekey] >= 10.)
    niso = sampleiso(nsamples,trec[mask], return_iso=True, lowfeh=lowfehgrid)
    mask = (trec[jkey]-trec[kkey] > 0.3001)  & (Z2FEH(trec[zkey]) > fehrange[0]) & (Z2FEH(trec[zkey]) < fehrange[1]) & (trec[mkey] > 0.75) & (trec['logg'] < 3.) & (trec['logg'] > 1.) & (trec[logagekey] >= 10.)
    p3niso = sampleiso(nsamples,trec[mask], return_iso=True, lowfeh=lowfehgrid)
    return niso, p3niso

def plot_model(model, params, minmax=[-50,50], nside=150):
    """
    plot a given model in x-y,x-z and y-z projection
    """
    xyzgrid = np.mgrid[minmax[0]:minmax[1]:nside*1j,minmax[0]:minmax[1]:nside*1j,minmax[0]:minmax[1]:nside*1j]
    shape = np.shape(xyzgrid.T)
    xyzgrid = xyzgrid.T.reshape(np.product(shape[:3]),shape[3])
    rphizgrid = bovy_coords.rect_to_cyl(xyzgrid[:,0], xyzgrid[:,1], xyzgrid[:,2])
    rphizgrid = np.dstack([rphizgrid[0],rphizgrid[1],rphizgrid[2]])[0]
    rphizgrid = rphizgrid.reshape(nside,nside,nside,3).T
    denstxyz = model(rphizgrid[0],rphizgrid[1],rphizgrid[2], params=params)
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(10,3.4)
    ax[0].contour(np.rot90(np.log10(np.sum(denstxyz, axis=0))), extent=[minmax[0],minmax[1],minmax[0],minmax[1]], cmap=plt.cm.cividis)
    ax[1].contour(np.rot90(np.log10(np.sum(denstxyz, axis=1))), extent=[minmax[0],minmax[1],minmax[0],minmax[1]], cmap=plt.cm.cividis)
    ax[2].contour(np.rot90(np.log10(np.sum(denstxyz, axis=2))), extent=[minmax[0],minmax[1],minmax[0],minmax[1]], cmap=plt.cm.cividis)
    xdat, ydat, zdat = bovy_coords.cyl_to_rect(Rphiz[:,0], Rphiz[:,1], Rphiz[:,2])
    ax[0].set_xlabel(r'y')
    ax[0].set_ylabel(r'z')
    ax[1].set_xlabel(r'x')
    ax[1].set_ylabel(r'z')
    ax[2].set_xlabel(r'x')
    ax[2].set_ylabel(r'y')
    for axis in ax:
        axis.set_ylim(minmax[0],minmax[1])
        axis.set_xlim(minmax[0],minmax[1])
    fig.tight_layout()
    
def pdistmod_model(densfunc, params, effsel, returnrate=False):
    """
    return the expected distmod distribution for a given model
    """
    rate = (densfunc(Rgrid[goodindx],phigrid[goodindx],zgrid[goodindx],params=params))*effsel[goodindx]*ds**3
    pdt = np.sum(rate,axis=0)
    pd = pdt/np.sum(pdt)/(distmods[1]-distmods[0])
    if returnrate:
        return pd, pdt, rate
    return pd, pdt

def check_fit(mask, samp, effsel, model, distmods, sample=False):
    """
    return the posterior distribution of distance modulus distribution for posterior parameter samples
    """
    print(sum(mask))
    pds = np.empty((200,len(distmods)))
    if sample:
        for ii,params in tqdm.tqdm_notebook(enumerate(samp[np.random.randint(len(samp), size=200)]), total=200):
            pd, pdt,rate = pdistmod_model(model, params, effsel, returnrate=True)
            pds[ii] = pd
        return pds
    else:
        pd, pdt, rate = pdistmod_model(model, np.median(samp,axis=0), effsel, returnrate=True)
        return pd


#Maximum likelihood utilities

# Likelihood calculation

def Rphizgrid(apo,distmods):
    """
    Generates a grid of R, phi, z for each location in apo at the distance moduli supplied
    """
    ds = 10**(distmods/5.-2.)
    Rgrid = np.zeros((len(apo._locations),len(ds)))
    phigrid = np.zeros((len(apo._locations),len(ds)))
    zgrid = np.zeros((len(apo._locations),len(ds)))
    for i in range(len(apo._locations)):
        glon,glat = apo.glonGlat(apo._locations[i])
        glon = np.ones(len(ds))*glon[0]
        glat = np.ones(len(ds))*glat[0]
        xyz = bovy_coords.lbd_to_XYZ(glon,glat,ds, degree=True)
        rphiz = bovy_coords.XYZ_to_galcencyl(xyz[:,0], xyz[:,1], xyz[:,2], Xsun=8., Zsun=0.02)
        Rgrid[i] = rphiz[:,0]
        phigrid[i] = rphiz[:,1]
        zgrid[i] = rphiz[:,2]
    return Rgrid, phigrid, zgrid

def xyzgrid(apo,distmods):
    """
    Generates a grid of x, y, z for each location in apo at the distance moduli supplied
    """
    ds = 10**(distmods/5.-2.)
    xgrid = np.zeros((len(apo._locations),len(ds)))
    ygrid = np.zeros((len(apo._locations),len(ds)))
    zgrid = np.zeros((len(apo._locations),len(ds)))
    for i in range(len(apo._locations)):
        glon,glat = apo.glonGlat(apo._locations[i])
        glon = np.ones(len(ds))*glon[0]
        glat = np.ones(len(ds))*glat[0]
        xyz = bovy_coords.lbd_to_XYZ(glon,glat,ds,degree=True)
        xgrid[i] = xyz[:,0]
        ygrid[i] = xyz[:,1]
        zgrid[i] = xyz[:,2]
    return zgrid, ygrid, zgrid


def tdens(densfunc, Rgrid, phigrid, zgrid, params=None):
    """
    returns the densities at the supplied grid given the supplied densfunc
    """
    if params is None:
        dens = densfunc(Rgrid,phigrid,zgrid)
    else:
        dens = densfunc(Rgrid,phigrid,zgrid,params=params)
    return dens

def effvol(densfunc, effsel, Rgrid, phigrid, zgrid, params=None):
    """
    returns the effective volume given a density function, an effective selection function 
    (including D**2 deltaD factor...) and the grid that it was evaluated on.
    """
    if params is None:
        effdens = tdens(densfunc,Rgrid,phigrid,zgrid)
    else:
        effdens = tdens(densfunc,Rgrid,phigrid,zgrid,params=params)
    return np.sum(effdens*effsel)
    
def loglike(params, densfunc, effsel, Rgrid, phigrid, zgrid, dataR, dataphi, dataz):
    """
    log-likelihood for the inhomogeneous Poisson point process
    """
    if not check_prior(densfunc, params):
        return -np.inf
    #the next parts are usually 0!
    logprior = log_prior(densfunc, params)
    logdatadens = np.log(tdens(densfunc, dataR, dataphi, dataz, params=params))
    logeffvol = np.log(effvol(densfunc,effsel,Rgrid,phigrid,zgrid,params=params))
    #log likelihood
    loglike = np.sum(logdatadens)-len(dataR)*logeffvol
    if not np.isfinite(loglike):
        return -np.inf
    return logprior + loglike

def check_prior(densfunc, params):
    """
    check the (uninformative?) prior for the given density model and parameters.
    """
    if densfunc is densprofiles.spherical:
        if params[0] < 0.:return False
        else: return True
    if densfunc is densprofiles.axisymmetric:
        if params[0] < 0.:return False
        elif params[1] < 0.1:return False
        elif params[1] > 1.:return False
        else: return True
    if densfunc is densprofiles.triaxial_norot:
        if params[0] < 0.:return False
        elif params[1] < 0.1:return False
        elif params[1] > 1.:return False
        elif params[2] < 0.1:return False
        elif params[2] > 1.:return False
        else: return True
    if densfunc is densprofiles.triaxial_single_angle_aby:
        if params[0] < 0.:return False
        elif params[1] < 0.1:return False
        elif params[1] > 10.:return False
        elif params[2] < 0.1:return False
        elif params[2] > 10.:return False
        elif params[3] < 0.:return False
        elif params[3] > 1.:return False
        elif params[4] < 0.:return False
        elif params[4] > 1.:return False
        elif params[5] < 0.:return False
        elif params[5] > 1.:return False
        else:return True
    if densfunc is densprofiles.triaxial_single_angle_zvecpa:
        if params[0] < 0.:return False
        elif params[1] < 0.1:return False
        elif params[1] > 1.:return False
        elif params[2] < 0.1:return False
        elif params[2] > 1.:return False
        elif params[3] <= 0.:return False
        elif params[3] >= 1.:return False
        elif params[4] <= 0.:return False
        elif params[4] >= 1.:return False
        elif params[5] <= 0.:return False
        elif params[5] >= 1.:return False
        else:return True
    if densfunc is densprofiles.triaxial_single_angle_zvecpa_plusexpdisk:
        if params[0] < 0.:return False
        elif params[1] < 0.1:return False
        elif params[1] > 1.:return False
        elif params[2] < 0.1:return False
        elif params[2] > 1.:return False
        elif params[3] <= 0.:return False
        elif params[3] >= 1.:return False
        elif params[4] <= 0.:return False
        elif params[4] >= 1.:return False
        elif params[5] <= 0.:return False
        elif params[5] >= 1.:return False
        elif params[6] < 0.:return False
        elif params[6] > 1.:return False
        else:return True
    if densfunc is densprofiles.triaxial_single_cutoff_zvecpa_plusexpdisk:
        if params[0] < 0.:return False
        if params[1] < 0.:return False
        elif params[2] < 0.1:return False
        elif params[2] > 1.:return False
        elif params[3] < 0.1:return False
        elif params[3] > 1.:return False
        elif params[4] <= 0.:return False
        elif params[4] >= 1.:return False
        elif params[5] <= 0.:return False
        elif params[5] >= 1.:return False
        elif params[6] <= 0.:return False
        elif params[6] >= 1.:return False
        elif params[7] < 0.:return False
        elif params[7] > 1.:return False
        else:return True
    if densfunc is densprofiles.triaxial_broken_angle_zvecpa_plusexpdisk:
        if params[0] < 0.:return False
        if params[1] < 0.:return False
        if params[2] < 0.:return False
        elif params[3] < 0.1:return False
        elif params[3] > 1.:return False
        elif params[4] < 0.1:return False
        elif params[4] > 1.:return False
        elif params[5] <= 0.:return False
        elif params[5] >= 1.:return False
        elif params[6] <= 0.:return False
        elif params[6] >= 1.:return False
        elif params[7] <= 0.:return False
        elif params[7] >= 1.:return False
        elif params[8] < 0.:return False
        elif params[8] > 1.:return False
        else:return True
    if densfunc is densprofiles.triaxial_einasto_zvecpa:
        if params[0] < 0.:return False
        elif params[1] < 0.5:return False
        elif params[2] < 0.1:return False
        elif params[2] > 1.:return False
        elif params[3] < 0.1:return False
        elif params[3] > 1.:return False
        elif params[4] <= 0.:return False
        elif params[4] >= 1.:return False
        elif params[5] <= 0.:return False
        elif params[5] >= 1.:return False
        elif params[6] <= 0.:return False
        elif params[6] >= 1.:return False
        elif params[7] <= 0.:return False
        elif params[7] >= 1.:return False
        else:return True
    if densfunc is densprofiles.triaxial_einasto_zvecpa_plusexpdisk:
        if params[0] < 0.:return False
        elif params[1] < 0.5:return False
        elif params[2] < 0.1:return False
        elif params[2] > 1.:return False
        elif params[3] < 0.1:return False
        elif params[3] > 1.:return False
        elif params[4] <= 0.:return False
        elif params[4] >= 1.:return False
        elif params[5] <= 0.:return False
        elif params[5] >= 1.:return False
        elif params[6] <= 0.:return False
        elif params[6] >= 1.:return False
        elif params[7] <= 0.:return False
        elif params[7] >= 1.:return False
        elif params[8] <= 0.:return False
        elif params[8] >= 1.:return False
        else:return True
    return True

def log_prior(densfunc, params):
    """
    check the (informative) prior for the given density model and parameters.
    """
    if densfunc is densprofiles.triaxial_single_angle_zvecpa:
        prior = norm.pdf(params[0], loc=2.5, scale=1)
        return np.log(prior)
    if densfunc is densprofiles.triaxial_einasto_zvecpa:
        prior = norm.pdf(params[0], loc=20, scale=10)
        return np.log(prior)
    return 0.

        
       
def mloglike(*args, **kwargs):
    """
    return the negative log-likehood
    """
    return -loglike(*args,**kwargs)


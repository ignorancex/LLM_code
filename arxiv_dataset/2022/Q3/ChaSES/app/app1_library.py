# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:rapids]
#     language: python
#     name: conda-env-rapids-py
# ---

# +
import numpy as np

import alphashape

from scipy.stats import skellam
from scipy.special import ndtri
from scipy.ndimage import gaussian_filter

from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats 
from astropy.io import fits

from collections import Counter

# from cuml.cluster import DBSCAN
# from cuml.metrics.cluster import silhouette_samples

from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import silhouette_samples

# from sklearn.cluster import DBSCAN, OPTICS #, KMeans, SpectralClustering
# from sklearn.preprocessing import StandardScaler

from glob import glob

from urllib.request import urlretrieve

import os
import sys

from mpmath import mp

import extended_library as ext_lib


# -

def db_sort(db, n_min=1):
    
    '''
    delete clusters < n_min; clusters ordered so that 0 - largest cluster
    '''
    
    n_clusters = np.max(db) + 1    
    if n_clusters == 0:
        return db

    a = list(Counter(db).items())
    a = np.array([_ for _ in a if _[0]!=-1])
    a = a[a[:, 1].argsort()][::-1]
    small = [_[0] for _ in a if _[1]<n_min]
    a = dict(zip(a[:, 0], np.arange(len(a))))
    for i in small:
        a[i] = -1
    a[-1] = -1

    db = np.array([a[i] for i in db])

    return db


def get_data(evt2_fn, ccd):
        
    evt2_data, head = ext_lib.process_fits(evt2_fn)

    if len(evt2_data)==0:
        # print(f'{obsid}_{ccd} empty')
        # msg.text = f'{obsid}_{ccd} empty'
        return 'empty'

    xy = ext_lib.xy_filter_evt2(evt2_data)[f'ccd_{ccd}']

    scaled_xy = ext_lib.scale(*xy.T)
    
    scaled_xy['head'] = head
    
    return scaled_xy


# +
# def get_data_old(obsid, ccd, fits_dir='', holes=True):
    
#     hls = '_holes' if holes else ''
        
#     evt2_data, head = ext_lib.process_fits(f'{fits_dir}/{obsid}/{ccd}/{obsid}_{ccd}{hls}_evt2_05_8keV.fits')

#     if len(evt2_data)==0:
#         print(f'{obsid}_{ccd} empty')
#         # msg.text = f'{obsid}_{ccd} empty'
#         return 'empty'

#     xy = ext_lib.xy_filter_evt2(evt2_data)[f'ccd_{ccd}']

#     scaled_xy = ext_lib.scale(*xy.T)
    
#     scaled_xy['head'] = head
    
#     return scaled_xy
# -

def get_hull(clusters, alpha):

    xs = []
    ys = []
    
    areas = []

    for c in clusters:

        # edges, hull_vertices = cxo_lib.concave_hull(c, 0.2)
        # pgon = Polygon(zip(*hull_vertices)) 
        # dens = 100 * (len(c) / (pgon.area * len(X)) - 1)
        
        ashape = alphashape.alphashape(c, alpha)
        
        x, y = ashape.exterior.xy
        
        area = ashape.area

        # x, y = cxo_lib.concave_hull(c, alpha)[1]

        xs.append([[x.tolist()]])
        ys.append([[y.tolist()]])
        areas.append(area)

    return {'xs': xs, 'ys': ys, 'area': areas}   


def nbins_sigma_func(X, nbins, sigma):

    # k = scipy.stats.gaussian_kde([x,y], sigma)
    # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))    
    # d = zi.reshape(xi.shape).T

    H, xe, ye = np.histogram2d(*X.T, bins=nbins)  

    H = gaussian_filter(H, sigma=sigma)            

    mean, median, std = sigma_clipped_stats(H, sigma=3.0)
    bkg_dens = median * nbins**2

    return H, bkg_dens


# +
# obsid = 88
# ccd = 0

# loc = 'hug'

# holes = True

# fff = {'local': '/home/ivv101/oyk/Extended_sources/2022/Chandra-ACIS-clusters-app/data',
#  'hug': 'https://huggingface.co/datasets/oyk100/Chandra-ACIS-clusters-data/resolve/main'}

# local_fits_dir = fff['local']

# fits_dir = fff[loc]

# hls = '_holes' if holes else ''

# args_func = {
#             'eps': 2.7, 
#             'min_samples': 46
#         }

# dat = process_ccd(obsid, ccd, holes=True, n_lim=True, n_max='all', 
#                 args_func=args_func, nbins=100, sigma=3, alpha=1, fits_dir=fits_dir, local_fits_dir=local_fits_dir, loc=loc)
# -

def create_fits(obsid, ccd, fits_dir, hls, cache=False):
    
    obsid_dir = f'{fits_dir}/{obsid}'
    fn_ccd = f'{obsid_dir}/{ccd}/{obsid}_{ccd}{hls}_evt2_05_8keV.fits'
        
    if cache and os.path.isfile(fn_ccd):
        return
    
    fn = glob(f'{obsid_dir}/*fits*')[0]

    with fits.open(fn) as hdul:

        # hdul.info()
        X = hdul[1].data
        head = hdul[1].header
    
    cols = ['ccd_id', 'x', 'y', 'energy']
    ccds = np.sort(np.unique(X['ccd_id'])).tolist()

    mask = (500 < X['energy']) & (X['energy'] < 8000)

    X = np.array(X)[mask][cols]

    X = X[X['ccd_id']==int(ccd)]
    
    bt = fits.BinTableHDU(X, head)
    bt.name = 'EVENTS'
    
    bt.writeto(fn_ccd, overwrite=True)


def get_signif_sigma(k, mu, N, max_dps=2000):
        
    mp.pretty = True
    
    mp.dps = 25
    
    p = 0
    
    while p==0 and mp.dps<=max_dps:
    
        cdf = mp.gammainc(k, mu, regularized=True)
    
        p = 1 - cdf**N
    
        mp.dps *= 2
    
    sigma = np.sqrt(2) * mp.erfinv(1-p)
    
    dps = mp.dps 
    
    # mp.dps = 3
    
    return float(p), float(sigma)


# +
def process_ccd(obsid, ccd, holes=True, n_lim=True, n_max='all', 
                args_func={}, nbins=100, sigma=3, alpha=1, local_fits_dir=''):
    
    
    '''
        holes=True not implemented for query or local custom (ciao...)
    
    '''
    
    hls = '_holes' if holes else ''        
    evt2_fn = f'{obsid}_{ccd}{hls}_evt2_05_8keV.fits'
    
    evt2_fn_local = f'{local_fits_dir}/{obsid}/{ccd}/{evt2_fn}'
    
#     if os.path.isfile(evt2_fn_local):
#         print('pass')
#         pass 
    
#     elif loc=='hug':        
#         url = f'{fits_dir}/{obsid}/{ccd}/{evt2_fn}'    
#         # print('url')
#         os.system(f'mkdir -p {local_fits_dir}/{obsid}/{ccd}')
#         # print('done mkdir')
#         urlretrieve(url, evt2_fn_local)   
        
#     elif loc=='local':
        
#         evt2_fn = f'{obsid}_{ccd}_evt2_05_8keV.fits' # no holes
#         evt2_fn_local = f'{local_fits_dir}/{obsid}/{ccd}/{evt2_fn}'
        
#         if not os.path.isfile(evt2_fn_local):
            
#             create_fits(obsid, ccd, local_fits_dir, cache=True) 
        
    # elif loc=='query':        
    #     status, url, evt2_fn_local = ext_lib.get_evt2_file(obsid, f'{local_fits_dir}/{obsid}')        
    #     if status != 'ok':
    #         sys.exit(status)   
            
    scaled_xy = get_data(evt2_fn_local, ccd)    
    
    # scaled_xy = get_scaled_xy(obsid, ccd, holes=holes, fits_dir=fits_dir, local_fits_dir=local_fits_dir, loc=loc)    

    # scaled_xy = get_data_old(obsid, ccd, fits_dir=fits_dir, holes=holes)
    
    if scaled_xy=='empty':
        return 'empty'
    
    # import pickle as pkl
    # pkl.dump(scaled_xy, open('/home/ivv101/oyk/Extended_sources/2022/app_cache/scaled_xy.pkl', 'wb'))

    X = scaled_xy['X'].copy()

    if len(X)==0:
        return 'empty'

    len_X_orig = len(X)

    if n_lim and n_max!='all':
        np.random.shuffle(X)
        X = X[:np.min([n_max, len_X_orig])]

    args = args_func.copy()    

    if 'eps' in args:
        eps0 = float(1 / np.sqrt(len(X)))                       
        args['eps'] *= eps0

    db = DBSCAN(**args).fit_predict(X)                
    db = db_sort(db, n_min=4)

    n_clusters = db.max() + 1

    noise = X[db==-1]
    clusters = [X[db==_].tolist() for _ in range(n_clusters)] 

    # cp.cuda.stream.get_current_stream().synchronize()

    # hulls, center of mass, area, silhouette, #cluster, n-n_bkg/area, significance 

    data = {}

    xs_ys_areas = get_hull(clusters, alpha)        
    data.update(xs_ys_areas)
        
    try:
        silhs = silhouette_samples(X, db)   
        
        if silhouette_samples.__module__.split('.')[0] == 'cuml':
        
            silhs = [np.mean(silhs[db==_]).get().tolist() for _ in range(n_clusters)]  
        else:
            silhs = [np.mean(silhs[db==_]).tolist() for _ in range(n_clusters)]  
                        
    except Exception as e:
        print(e)
        print(f'no silhs for {obsid}/{ccd}')
        silhs = [-1]*n_clusters
        
    data['silhouette'] = silhs    

    H, _ = nbins_sigma_func(X, nbins, sigma) # using bkg_dens defined below
                
    com = np.transpose([np.mean(c, 0).tolist() for c in clusters])  
    data['x_scaled'], data['y_scaled'] = com
    
    com = ext_lib.unscale(*com, scaled_xy['pars'])
    data['x'], data['y'] = com

    data['n-n_bkg'] = []
    data['signif.'] = []
    data['sigmas'] = []
        
    bkg_dens = (len(X) - len(np.concatenate(clusters))) / (1 - np.sum(data['area']))
    
    for c, a in zip(clusters, data['area']):
        
        n = len(c)
        n_bkg = bkg_dens * a
                
        p, sigma = get_signif_sigma(n, n_bkg, 1/a)
        
        data['n-n_bkg'].append(n - n_bkg)
        data['signif.'].append(p)
        data['sigmas'].append(sigma)
        
    # data['signif.'] = [1 - skellam.cdf(x, len(X) * a, bkg_dens * a) for x, a in zip(data['n-n_bkg'], data['area'])]
    # data['sigmas'] = [ndtri(1-_/2) for _ in data['signif.']]
    
#     import pickle as pkl
    
#     pkl.dump([X, clusters, data], open('tmp.pkl', 'wb'))

    h = scaled_xy['head']

    w = WCS(naxis=2)
    w.wcs.crpix = [h['TCRPX11'], h['TCRPX12']]
    w.wcs.cdelt = [h['TCDLT11'], h['TCDLT12']]
    w.wcs.crval = [h['TCRVL11'], h['TCRVL12']]
    w.wcs.ctype = [h['TCTYP11'], h['TCTYP12']]
    w.wcs.cunit = [h['TCUNI11'], h['TCUNI12']]
    w.wcs.radesys = 'ICRS'
    try:
        w.wcs.mjdobs = h['MJD-OBS']
    except:
        w.wcs.mjdobs = h['MJD_OBS']
    w.wcs.dateobs = h['DATE-OBS']

    data['ra'], data['dec'] = w.wcs_pix2world(com.T, 1).T

    data.update(dict(zip(['X', 'len_X_orig', 'db', 'n_clusters', 'bkg_dens', 'clusters', 'H'], 
                         [X, len_X_orig, db, n_clusters, bkg_dens, clusters, H])))

    return data, scaled_xy['pars'] # no filtering by sigma_min


# -

class loop_class:
    
    sp = ''.join([' ']*100)
        
    def __init__(self, lst):

        self.t0 = timer()
        self.n = len(lst) 
        self.tt = []
                
    def __call__(self):
        t = timer()
        self.tt.append(t)
        
        k = len(self.tt)
        
        perc = 100 * k / self.n
    
        rem = int((self.n - k) * (t - self.t0) / k)
        
        print(f'\r{self.sp}', end='')

        if k < self.n:    
            msg = f'\r{k}/{self.n}: {perc:.1f}%, {timedelta(seconds=rem)} remaining'    
        else:
            msg = f'\r{self.n} done, {timedelta(seconds=int(t - self.t0))} total'

        print(msg, end='', flush=True)    


class friz_class:
    
    def __init__(self, history=False, inactive=False): 
                        
        self.data = {}        
        self.pref = '' if not inactive else 'INACTIVE'
        
        self.comment = ''
        
        self.inactive = inactive
                
        if history:        
            self.history = []
        
    def freeze(self, model):
        self.data[model] = True if not self.inactive else False
        self.history.append(f'{model} freeze {self.pref}')
        
    def unfreeze(self, model):
        
        if model in self.data and self.data[model]==True:
            self.data[model] = False
            self.history.append(f'{model} unfreeze {self.pref}')
            return True
        else:
            self.history.append(f'{model} passed {self.pref}')
            return False    



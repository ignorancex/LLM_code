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
#     display_name: Python [conda env:p39] *
#     language: python
#     name: conda-env-p39-py
# ---

# +
import glob
import numpy as np
import os
import pandas as pd
import re
 
from astropy.io import fits
from astropy.table import Table
  
import requests
from bs4 import BeautifulSoup
import urllib.request 

import magic

from sklearn.cluster import DBSCAN, OPTICS #, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler

from scipy.ndimage import gaussian_filter

from kneed import KneeLocator # https://github.com/arvkevi/kneed

import warnings

from scipy.spatial import ConvexHull, distance_matrix #, Delaunay, distance

from shapely.geometry import Polygon

from pca import pca # https://github.com/erdogant/pca

# from ciao_contrib.runtool import new_pfiles_environment, dmcoords

# import json
# import pickle as pkl

# from timeit import default_timer as timer

import matplotlib.pyplot as plt
from PIL import Image

import hdbscan
# -

remove_nans = lambda x : x[~np.isnan(np.sum(x, 1))]


# # from expanded_data

def get_evt2_file(obsid, path='.', search_mask='evt2'):
    '''
    We assume that there exists a single evt2 file in primary directory in CXC database
    '''
    
    status = 'ok'
    
    # folders organized by last digit of obsid
    last = str(obsid)[-1]
    primary_url = f'https://cxc.cfa.harvard.edu/cdaftp/byobsid/{last}/{str(obsid)}/primary'
    
    _ = glob.glob(f'{path}/*{int(obsid):05d}*{search_mask}*')
    if _: return status, f'{primary_url}/{os.path.basename(_[0])}', _[0]
    
    html_text = requests.get(primary_url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    
    evt2_list = [_.get('href') for _ in soup.find_all('a') if re.search(search_mask, _.get('href'))]
    if len(evt2_list) != 1:
        print(f'Error: there are {len(evt2_list)} evt2 files: {evt2_list}')
        status = f'{len(evt2_list)} evt2 files: {evt2_list}'
        
    evt2_filename = evt2_list[0]
                
    urllib.request.urlretrieve(f'{primary_url}/{evt2_filename}', f'{path}/{evt2_filename}')
    
    return status, f'{primary_url}/{evt2_filename}', f'{path}/{evt2_filename}'


def process_fits(fn):
    
    with fits.open(fn) as _:
        head = _[1].header
        evt2_data = _[1].data
    
    return evt2_data, head


def xy_filter_evt2(evt2_data):
    
    X = evt2_data
    
    cols = ['ccd_id', 'x', 'y', 'energy']
    
    mask = (500 < X['energy']) & (X['energy'] < 8000)
    
    X = Table(X)[mask][cols].to_pandas()
    
    ccds = np.unique(X['ccd_id'].tolist())
    
    xy = {}
    
    for ccd, data in X.groupby('ccd_id'):
    
        xy[f'ccd_{ccd}'] = remove_nans(data[['x', 'y']].values).astype(None) 
    
    return xy


def process_obsid_data(obsid, evt2_dir, evt2_size_limit=np.inf):

    status, url, fn = get_evt2_file(obsid, evt2_dir)
    
    if status != 'ok':
        return status, 0, 0
        
    if os.stat(fn).st_size > evt2_size_limit * 2**20:        
        return f'{obsid} too big\n', 0, 0

    evt2_data, evt2_head = process_fits(fn)
    
    evt2_info = {
        'exp': evt2_head['EXPOSURE'] / 1000,
        'obsid': obsid,
        'url': url        
    }

    xy = xy_filter_evt2(evt2_data)
            
    return status, xy, evt2_info


detect = lambda fn : magic.Magic(mime_encoding=True).from_file(fn)


def get_OBSIDs(fn, encoding=''):

    # fn = f'All_OBSIDs_ACIS-{name}.csv'
    
    if encoding=='':
        encoding = detect(fn)
    
    df = pd.read_csv(fn, engine='python', encoding=encoding)

    fl = df['Galactic l'].apply(lambda _ : f'{_:.1f}')
    fb = df['Galactic b'].apply(lambda _ : f'{_:.1f}')

    df['Galactic lb label'] = [l + '_' + b for l, b in zip(fl, fb)]

    df = df.set_index('Obs ID')

    gb = df.groupby('Galactic lb label')

    gbval = list(gb.groups.values())

    OBSIDs = []

    # for same 'Galactic l' and 'Galactic b' we choose obsid with largest exposure

    for _ in gbval:
        if len(_) == 1:
            OBSIDs.append(_[0])
        else:
            OBSIDs.append(df.loc[_]['Exposure '].idxmax())

    return OBSIDs


# # from expanded_clusters

# +
def find_clusters(xys, n_jobs=1, algorithm='dbscan'):
    '''
    returns indices of clusters (-1(noise))
    '''
    
    X = StandardScaler().fit_transform(xys)
    
    eps = 4 * np.sqrt(4 / len(X))
    
    if algorithm == 'optics':   

        db = OPTICS(eps=float(eps), n_jobs=n_jobs).fit_predict(X) 

        return db
    
    ncl=[]
    nsmpl=[]
                    
    for i in range(1, 20):

        pts = 1 + pow(len(X), i / 40) # pow(10, i*math.log10(len(X))/40.0)

        if algorithm == 'dbscan':

            db = DBSCAN(eps=eps, min_samples=pts, n_jobs=n_jobs).fit(X)

        elif algorithm == 'hdbscan':   

            db = hdbscan.HDBSCAN(cluster_selection_epsilon=float(eps), 
                                 min_samples=int(pts),
                                 core_dist_n_jobs=1).fit(X) 

        else:

            return 'incorrect algorithm'

        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

        ncl.append(n_clusters_)
        nsmpl.append(pts)
        
    index_max = np.argmax(ncl) 
    
    if index_max == len(ncl) - 1:
        return 'bad knee'
    
#     print(index_max, len(nsmpl), len(ncl), nsmpl[index_max:], ncl[index_max:], ncl)
        
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            kn = KneeLocator(nsmpl[index_max:], ncl[index_max:], curve='convex', direction='decreasing')
        except:
            return 'bad knee'
        
    if algorithm == 'dbscan':
        
        db = DBSCAN(eps=eps, min_samples=np.floor(kn.knee), n_jobs=n_jobs).fit_predict(X)
        
    elif algorithm == 'hdbscan':   

        db = hdbscan.HDBSCAN(cluster_selection_epsilon=float(eps), 
                             min_samples=int(np.floor(kn.knee)),
                             core_dist_n_jobs=1).fit_predict(X)     

    return db


# -

def find_clusters_minpts(xys, n_jobs=1, algorithm='dbscan', minpts=False):
    '''
    returns indices of clusters (-1(noise))
    '''
    
    if algorithm != 'hdbscan' or not minpts:
        find_clusters(xys, n_jobs=n_jobs, algorithm=algorithm)
        return
    
    X = StandardScaler().fit_transform(xys)
    
    lst = np.sort(list(set(np.logspace(np.log10(3), np.log10(len(X)/100), 50).astype(int))))
    
    zzz = []

    for i in lst:

        db = hdbscan.HDBSCAN(
            min_cluster_size=int(i),
            core_dist_n_jobs=1).fit(X) 

        _ = np.sum(db.labels_==-1)

        zzz.append([int(i), _])

    min_cluster_size = zzz[np.argmin(np.transpose(zzz)[1])][0]
    
    db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                         core_dist_n_jobs=1).fit(X) 
    
    return db


# +
def find_clusters_obsid(data_file, pkl_data_size_limit, algorithm='dbscan', minpts=False):
    
#     t = timer()           
    
    sz = os.stat(data_file).st_size / 2**20
    
    if sz > pkl_data_size_limit:
        
#         print(f'too big: {sz:.2f} MB')
        
        status = 'too big'
    
        # print(status)    
        return status, {}
                        
    data = pkl.load(open(data_file, 'rb'))
    
    status = data['status']
    
    if status != 'ok':
        print(status)     
        return status, {}
        
    clusters = {}        
    
    for ccd, ccd_data in data['xy'].items():
        
        # print(ccd_data)
        
#         if np.isnan(np.sum(ccd_data)):
            
#             clusters[ccd] = 'nan'
#             status = 'nan'
            
#         else:    

        clusters[ccd] = find_clusters_minpts(ccd_data, n_jobs=1, algorithm=algorithm, minpts=minpts)

        if isinstance(clusters[ccd], str): 
            status = 'bad knee'   
        
    return status, clusters  
        
#     print(f'{timer() - t:.2f} sec')        
# -





# # from expanded_analysis

def rot2d(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], 
                    [np.sin(phi),  np.cos(phi)]])


def scale(xp, yp, decimals=None):
    
    pts = np.array([xp, yp]).T
    
    hull = ConvexHull(pts)
    hull = pts[hull.vertices]

    areas = np.array([])
    phis = np.array([])

    for i, vert in enumerate(hull):

        nxt = i + 1
        if nxt == len(hull):
            nxt = 0

        edge = hull[nxt] - vert   

        phi = np.arctan2(edge[1], edge[0])
        phis = np.append(phis, phi)
        
        xy = np.dot(rot2d(-phi), (hull - vert).T)

        area = (max(xy[0]) - min(xy[0])) * (max(xy[1]) - min(xy[1]))
        areas = np.append(areas, area)

    ind = np.argmin(areas)  

    x, y = np.dot(rot2d(-phis[ind]), (pts - hull[ind]).T)

    xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)

    dx, dy = xmax - xmin, ymax - ymin

    if dx < dy:
        xs = (x - xmin + (dy - dx) / 2) / dy
        ys = (y - ymin) / dy
        dxdy = True
    else:        
        xs = (x - xmin) / dx
        ys = (y - ymin + (dx - dy) / 2) / dx
        dxdy = False

    if decimals != None:
        xs = np.around(xs, decimals=decimals)
        ys = np.around(ys, decimals=decimals)
                
    X = np.transpose([xs, ys])    
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    
    out = {
        'pars': {
            'phi': phis[ind],
            'offset': hull[ind].tolist(),
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'dx<dy': dxdy,
            'X.min(0)': X.min(0),
            'dX(0)': X.max(0) - X.min(0),
            
        },
        'x': xs,
        'y': ys,
        'X': X
    }
        
    return out


def scale_pars(xp, yp, pars, decimals=None):
    
    nms = ['phi', 'offset', 'xmin', 'xmax', 'ymin', 'ymax', 'dx<dy', 'X.min(0)', 'dX(0)']
    
    phi, offset, xmin, xmax, ymin, ymax, dxdy, Xmin0, dX0 = [pars[_] for _ in nms]
        
    pts = np.array([xp, yp]).T
    
    x, y = np.dot(rot2d(-phi), (pts - offset).T)

    dx, dy = xmax - xmin, ymax - ymin

    if dxdy:
        xs = (x - xmin + (dy - dx) / 2) / dy
        ys = (y - ymin) / dy
    else:        
        xs = (x - xmin) / dx
        ys = (y - ymin + (dx - dy) / 2) / dx

    if decimals != None:
        xs = np.around(xs, decimals=decimals)
        ys = np.around(ys, decimals=decimals)
        
    X = np.transpose([xs, ys])    
    X = (X - Xmin0) / dX0
        
    return X


def unscale(xs, ys, pars):
        
    nms = ['phi', 'offset', 'xmin', 'xmax', 'ymin', 'ymax', 'dx<dy', 'X.min(0)', 'dX(0)']
    
    phi, offset, xmin, xmax, ymin, ymax, dxdy, Xmin0, dX0 = [pars[_] for _ in nms]
    
    x, y = (np.transpose([xs, ys]) * dX0 + Xmin0).T
                                                             
    dx, dy = xmax - xmin, ymax - ymin
                
    if dxdy:
        xp = x * dy + xmin - (dy - dx) / 2
        yp = y * dy + ymin
    else:
        xp = x * dx + xmin
        yp = y * dx + ymin - (dx - dy) / 2
            
    xy = np.dot(rot2d(phi), np.array([xp, yp])).T
    
    return (xy + offset).T


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return np.array(list(edges))


def find_ConvexHull(cluster):
    
    hull = ConvexHull(cluster)
        
    vert = hull.vertices
    
    return  cluster[np.append(vert, vert[:1])]


# +
def find_lasso(cluster, alpha=0.02):
         
    try:
        
        cs = scale(*c.T)
        cs = np.transpose([cs['x'], cs['y']])
        
        lasso = alpha_shape(cs, alpha=alpha, only_outer=True)
        
        lasso_dict = dict(zip(*lasso.T))

        out = [list(lasso_dict.keys())[0]]

        nxt = -1

        while nxt != out[0]:

            nxt = lasso_dict.pop(out[-1])
            out.append(nxt)

        out = np.array(out).astype(int)

        if len(lasso_dict) > 0:
            
            print(f'Warning: non-continuous cluster: {len(lasso_dict)} remaining edges')
            
#             return 'convex_hull', find_ConvexHull(cluster)
            
        return 'alpha_shape', cluster[out]
        
    except:
                        
        return 'convex_hull', find_ConvexHull(cluster)


# +
def find_bkg_density(xys, cl, clusters_idx):
    
    ccd_npoints = len(xys)
    ccd_area_px = ConvexHull(xys).volume   
    all_sources_area_px = np.sum([_['area_px'] for _ in cl])
    noise_npoints = np.sum(clusters_idx==-1)
    
#     print(noise_npoints, ccd_area_px, all_sources_area_px)

    bkg_density_px = noise_npoints / (ccd_area_px - all_sources_area_px)
    
    return bkg_density_px


# +
def make_images(data, an, obsid, img_dir, debug_mode=False, is_interactive=False):
    
    arr = np.concatenate(list(data['xy'].values())).T
    
    xx, yy = arr

    xmin, xmax, ymin, ymax = xx.min(), xx.max(), yy.min(), yy.max()

    w, h = xmax - xmin, ymax - ymin

    bounds = xmin, xmax, ymin, ymax

    data = np.histogram2d(*arr, bins=250)[0]
    data = gaussian_filter(data, sigma=0.5)

    fig, ax = plt.subplots(facecolor='k', figsize=(10, 10*h/w))

    cmap = 'Blues_r'
    plt.pcolormesh(np.log(data + 1).T, cmap=cmap, shading='flat')

    if debug_mode & (an['status'] == 'ok'):
        for ccd, v in an['data'].items():
            for i, k in enumerate(v['clusters']):

                x, y = np.transpose(k['lasso'][1])
                
                x = 250 * (x - xmin) / (xmax - xmin)
                y = 250 * (y - ymin) / (ymax - ymin)

                col = 'y' if k['status'] == 'ok' else 'r'   

#                 lab = ''.join([f'{_} = {lassos_info[ccd][i][_]:.2f}\n' for _ in ['roundness', 'area', 'area_psf', 'area_px']])     

                ax.plot(x, y, c=col)

    #             ax.fill(x, y, c=col, alpha=0)

#     ax.set_aspect(1)
    plt.axis('off') 
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 
    
    _ = '_debug' if debug_mode else ''    
    plt.savefig(f'{img_dir}/{obsid}{_}.jpeg', bbox_inches='tight', pad_inches = 0) 

#     mplcursors.cursor(hover=True).connect(
#         "add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

    if is_interactive: plt.show()
    
    plt.close()

#     fig, ax = plt.subplots(facecolor='k', figsize=(10, 10*h/w))

#     [ax.scatter(*xy_unscaled[ccd], s=0.1, c=[[56/255, 117/255, 174/255]]) for ccd in xy_unscaled.keys()]
    
#     if debug_mode:
#         for ccd, v in lassos_unscaled.items():
#             for i, lasso in enumerate(v):

#                 col = 'y' if lassos_info[ccd][i]['is_good'] else 'r'    

#                 ax.plot(*lasso[1].T, c=col)

#         ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], c='r')

#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)

#     plt.axis('off') 
#     ax.get_xaxis().set_visible(False) 
#     ax.get_yaxis().set_visible(False) 

#     plt.savefig(f'{webdata_dir}/{obsid}_alt.jpeg', bbox_inches='tight', pad_inches = 0) 

#     if is_interactive: plt.show()
        
#     plt.close()
    
    return bounds # make_images


# +
def get_web_entry(data, an, bounds, w_jpg, h_jpg, debug_mode):

    xmin, xmax, ymin, ymax = bounds
    
    web = {}
    
    obsid = str(data['info']['obsid'])
    
    web['obsid'] = obsid
    
    web['box_x'] = w_jpg
    web['box_y'] = h_jpg
    
    web['exp'] = f"{data['info']['exp']:.2f}"
        
    web['url'] = data['info']['url']
        
    web['lassos'] = {}
        
    for ccd, v in an['data'].items():
                
        for i, info in enumerate(v['clusters']):
            
            if (info['status'] != 'ok') & (debug_mode == False):
                continue
            
            xs, ys = info['lasso'][1].T
            
            xs = web['box_x'] * (xs - xmin) / (xmax - xmin) + 0.5
            ys = web['box_y'] * (ys - ymin) / (ymax - ymin) + 0.5
            
            coords = np.ravel([[int(x), int(y)] for x, y in zip(xs, web['box_y'] - ys)]).tolist()
            
            webinfo = {

                'tot': info['tot'],
                'bkg': info['bkg'],
                'area': f"{info['area']:,.2f}",
                # 'area': '{:,.2f}'.format(info['area']),
                'x': '{:,.2f}'.format(info['centroid_px'][0]),
                'y': '{:,.2f}'.format(info['centroid_px'][1]),
                'ra': info['ra'],
                'dec': info['dec'],
                'src': info['src'],
                'SN': '{:,.2f}'.format(info['SN']),
                'ccd_id': str(ccd),
                'bin': 4,
                'coords': coords
            }
            
#             if debug_mode: 
                
#                 webinfo['is_good'] = info['is_good']
#                 webinfo['shape'] = info['shape']
                
#                 debug_pars = ['roundness', 'area', 'area_psf', 'area_px', 'kurtosis', 'centrality', 'ks', 'kuiper']
                
#                 [webinfo.update({_: '{:,.2f}'.format(info[_])}) for _ in debug_pars] 
                        
            web['lassos'][f'{ccd}_{i}'] = webinfo

    return web
# -



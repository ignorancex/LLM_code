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
with open('bokeh_modules.py', 'wt') as _:
    _.write(
'''
from bokeh.embed import file_html, json_item, autoload_static, components
from bokeh.events import Tap
from bokeh.io import curdoc, output_notebook, export_png, export
from bokeh.layouts import layout, column, row, gridplot
from bokeh.models import ColumnDataSource, CustomJS, Slider, Legend, \
        Button, CheckboxButtonGroup, RadioButtonGroup, RadioGroup, CheckboxGroup, Label, Spacer, Title, Div, \
        PanTool, WheelZoomTool, SaveTool, ResetTool, HoverTool, TapTool, \
        BasicTicker, Scatter, CustomJSHover, FileInput, Toggle, TableColumn, DataTable, TextAreaInput, \
        TabPanel, Tabs, DateFormatter, LogColorMapper, LinearColorMapper, ColorBar
from bokeh.plotting import figure, output_file, show, save
from bokeh.resources import CDN
from bokeh.themes import Theme
from bokeh.util.compiler import TypeScript
from bokeh.document import Document
''')
import bokeh_modules as bk 
import importlib
importlib.reload(bk)
import bokeh.palettes as bkp

# bk.output_notebook()
# -
import numpy as np
# import math

# +
# from sklearn.cluster import DBSCAN
# import hdbscan
# from sklearn.cluster import OPTICS
# from sklearn.neighbors import NearestNeighbors
# import sklearn.datasets as data
from sklearn import metrics
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# from kneed import KneeLocator

# from distinctipy import distinctipy

# import gzip, json

# import glob
# import re
# import pickle as pkl

import scipy.stats

# # from cuml.cluster import HDBSCAN

from timeit import default_timer as timer
from datetime import timedelta

# import cv2

import itertools  
from collections import Counter

# import multiprocessing as mp

# # from bokeh.plotting import figure, show
# # from bokeh.io import curdoc, output_notebook
# # from bokeh.io import export_png
# # # from bokeh.models import HoverTool
# # from bokeh.layouts import column, row

# # %run extended_library.ipynb

# +
# from html2image import Html2Image
# hti = Html2Image()

# +
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager

# from selenium.webdriver.chrome.options import Options 
# -

rgb2hex = lambda r,g,b: f'#{r:02x}{g:02x}{b:02x}'
hex2rgb = lambda hx: (int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16))
arrmap = lambda f, x : [f(_) for _ in x]


def find_clusters(X, 
                  func, 
                  silhouette=False,
                  calinski_harabasz=False,
                  davies_bouldin=False,
                  **args):
    
    db = func(**args).fit_predict(X)

    noise = X[db==-1]
    
    clusters = [X[db==_] for _ in range(db.max() + 1)] 
        
    out = {
        'db': db,
        'noise': noise, 
        'clusters': clusters
    } 
    
    # # The silhouette score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. 
    # # Scores around zero indicate overlapping clusters.
    # # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.    

    # # The calinski_harabasz score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    # # The score is fast to compute.

    # # a lower Davies-Bouldin index relates to a model with better separation between the clusters.    

    
    if silhouette:
        try:
            out['silhouette'] = metrics.silhouette_score(X, db, metric='euclidean') 
        except:
            out['silhouette'] = np.nan
            
    if calinski_harabasz:
        try:
            out['calinski_harabasz'] = metrics.calinski_harabasz_score(X, db) 
        except:
            out['calinski_harabasz'] = np.nan
            
    if davies_bouldin:
        try:
            out['davies_bouldin'] = metrics.davies_bouldin_score(X, db) 
        except:
            out['davies_bouldin'] = np.nan
                    
    return out


def find_clusters_mp(X, dict_arg): 

    return find_clusters(X, 
                         dict_arg['func'], 
                         **dict_arg['args'], 
                         **dict_arg['args_func'])


# +
# # %%capture cap

def plot_blur(X,
              sigma=0.1, 
              nbins=100, 
              palette='Spectral11', # ['Viridis256', 'Spectral11']
              border_fill_color=None,
              title=None):

    x, y = X.T
    
    k = scipy.stats.gaussian_kde([x,y], sigma)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))    
    d = zi.reshape(xi.shape).T

    # p = figure(tooltips = [("x", "$x"), ("y", "$y"), ("value", "@image")])    
    p = bk.figure(title=title)
    
    p.x_range.range_padding = p.y_range.range_padding = 0
    
    p.image(image=[d], x=0, y=0, dw=1, dh=1, palette=palette, level='image')
    
    p.grid.visible = False
    
    p.toolbar_location = None
    p.axis.visible = False
    
    p.border_fill_color = border_fill_color
    p.title.background_fill_color = 'white'
    
    return p


# -

def plot_clusters(X, db, title='', border_fill_color=None, palette=bkp.Category20[20], start=0):
    
    p = bk.figure(title=title)
    
    p.x_range.range_padding = p.y_range.range_padding = 0
    
    
    palette = itertools.cycle(palette)    
    palette = itertools.islice(palette, start, None)
                
    colors = np.array(['black']*len(db), dtype='U7')
    
    a = np.array(list(Counter(db).items()))
    a = a[a[:, 1].argsort()][::-1]
    
    for i in a[:, 0]: 
        if i==-1: continue
        
        colors[db==i] = next(palette)
        
    x, y = X.T    
    
    source = bk.ColumnDataSource(data={'x': x, 'y': y, 'color': colors})

    # colors_dark = [(np.array(hex2rgb(_)) * 0.8).round().astype(int) for _ in colors]
    # colors_dark = [rgb2hex(*_) for _ in colors_dark]
    
    p.scatter('x', 'y', color='color', source=source)
        
    p.grid.visible = False    
    p.toolbar_location = None
    p.axis.visible = False
    
    p.border_fill_color = border_fill_color
    p.title.background_fill_color = 'white'
    
    return p    


def make_png_json(p):
    
    p['roots'] = p['doc']['roots']
    p['title'] = p['doc']['title']
    
    doc = bk.Document.from_json(p)
    
    return bk.export.get_screenshot_as_png(doc)


# +
def generate_grid_pic(X, 
                      noise_clusters_list, 
                      args_list, 
                      param=None, 
                      ni_nj=None, 
                      obsid_ccd='obsid_ccd',
                      width=250, 
                      height=250):
    
    if ni_nj == None:
        ni = len(noise_clusters_list)
        nj = 1
    else:
        ni, nj = ni_nj
        
    if param != None:

        pal = list(bkp.RdPu[9])[::-1]
        silhs = np.array([_[param] for _ in noise_clusters_list])
        
        print(silhs)

        mn = np.nanmin(silhs)
        mx = np.nanmax(silhs) 

        n = len(pal)

        silhs2 = np.floor(n * (silhs - mn) / (mx - mn))
        silhs2 = np.nan_to_num(silhs2, nan=-1).astype(int)
        silhs2[silhs2==n] = n - 1
        silhs2[silhs2==-1] = n

        pal.append('white')

        pp = [plot_clusters(X, 
                            c['db'], 
                            title=a['title'][:-7] + f", par={c[param]:.2f}", 
                            border_fill_color=pal[s]) for c, s, a in zip(noise_clusters_list, silhs2, args_list)]
    else:
        pp = [plot_clusters(X, 
                            c['db'], 
                            title=a['title']) for c, a in zip(noise_clusters_list, args_list)]
        
    grid = bk.gridplot(np.resize(pp, (ni, nj)).tolist(), width=width, height=height)    
    
#     html = bk.file_html(grid, bk.CDN, obsid_ccd)
    
#     hti = Html2Image(output_path='tmp',
#                      custom_flags=['--no-sandbox', '--disable-gpu'])
    
#     hti.screenshot(html, save_as=f'{obsid_ccd}_grid.png')

    # return pp
   
    return grid #bk.export.get_screenshot_as_png(grid) 


# -

def generate_grid_pic_mp(args, dict_args): 

    return generate_grid_pic(*args, **dict_args)


# +
def generate_blur_pics(X, obsid_ccd):
    
#     options = Options()
#     options.headless = True

#     # options = Options()
#     options.add_argument('--no-sandbox')
#     # options.add_argument('--headless')
#     # options.add_argument('--disable-dev-shm-usage')
#     options.add_argument('--remote-debugging-port=9222')

    blur_pics = [plot_blur(X, 
                        nbins=200, 
                        border_fill_color=None,   
                        title=f'{obsid_ccd}, {len(X)} pts', 
                        palette=_) for _ in ['Viridis256', 'Spectral11']]
    
    p = bk.row(blur_pics)
    
    return p # bk.export.get_screenshot_as_png(p)
    
    # _ = bk.json_item(p)

    # return make_png_json(_)
    
    # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)            
    # out = bk.export.get_screenshot_as_png(bk.row(blur_pics), driver=driver)
    # driver.quit()
    
#     while True:
#         try:
            
#             break
#         except BaseException as e:
#             print('Failedddd: ' + str(e))
#             pass

#     html = bk.file_html(p, bk.CDN, obsid_ccd)
    
#     hti = Html2Image(output_path='tmp',
#                  custom_flags=['--no-sandbox', '--disable-gpu'])
    
    
    # logging.getLogger('html2image').setLevel(logging.WARNING)
    # logging.getLogger('Html2Image').setLevel(logging.WARNING)
    
    # hti.screenshot(html, save_as=f'{obsid_ccd}_blur.png')
    
    # return html

# +
import numpy as np
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from collections import Counter
import itertools


def concave_hull(coords, alpha):  # coords is a 2D numpy array

    # i removed the Qbb option from the scipy defaults.
    # it is much faster and equally precise without it.
    # unless your coords are integers.
    # see http://www.qhull.org/html/qh-optq.htm
    tri = Delaunay(coords, qhull_options="Qc Qz Q12").vertices

    ia, ib, ic = (
        tri[:, 0],
        tri[:, 1],
        tri[:, 2],
    )  # indices of each of the triangles' points
    pa, pb, pc = (
        coords[ia],
        coords[ib],
        coords[ic],
    )  # coordinates of each of the triangles' points

    a = np.sqrt((pa[:, 0] - pb[:, 0]) ** 2 + (pa[:, 1] - pb[:, 1]) ** 2)
    b = np.sqrt((pb[:, 0] - pc[:, 0]) ** 2 + (pb[:, 1] - pc[:, 1]) ** 2)
    c = np.sqrt((pc[:, 0] - pa[:, 0]) ** 2 + (pc[:, 1] - pa[:, 1]) ** 2)

    s = (a + b + c) * 0.5  # Semi-perimeter of triangle

    area = np.sqrt(
        s * (s - a) * (s - b) * (s - c)
    )  # Area of triangle by Heron's formula

    filter = (
        a * b * c / (4.0 * area) < 1.0 / alpha
    )  # Radius Filter based on alpha value

    # Filter the edges
    edges = tri[filter]

    # now a main difference with the aforementioned approaches is that we dont
    # use a Set() because this eliminates duplicate edges. in the list below
    # both (i, j) and (j, i) pairs are counted. The reasoning is that boundary
    # edges appear only once while interior edges twice
    edges = [
        tuple(sorted(combo)) for e in edges for combo in itertools.combinations(e, 2)
    ]

    count = Counter(edges)  # count occurrences of each edge

    # keep only edges that appear one time (concave hull edges)
    edges = [e for e, c in count.items() if c == 1]

    # these are the coordinates of the edges that comprise the concave hull
    edges = [(coords[e[0]], coords[e[1]]) for e in edges]

    # use this only if you need to return your hull points in "order" (i think
    # its CCW)
    ml = MultiLineString(edges)
    poly = polygonize(ml)
    hull = unary_union(list(poly))
    hull_vertices = hull.exterior.coords.xy

    return edges, hull_vertices


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

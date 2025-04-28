# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:p39] *
#     language: python
#     name: conda-env-p39-py
# ---

# +
# from IPython.display import display,Javascript
# Javascript('document.title="{}"'.format('my_jupy_env'))
# -

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


# +
import os

# runs only once, safe to execute when in different folder
try:
    notebook_dir # does a exist in the current namespace
except NameError:
    notebook_dir = os.getcwd()

if notebook_dir[-4:] != '/app':
    notebook_dir += '/app'

os.chdir(notebook_dir)    
    
print(notebook_dir)

# +
with open('bokeh_modules.py', 'wt') as _:
    _.write(r'''
from bokeh.embed import file_html, json_item, autoload_static, components
from bokeh.events import Tap
from bokeh.io import curdoc, output_notebook, export_png, export
from bokeh.layouts import layout, column, row, gridplot
from bokeh.models import ColumnDataSource, CustomJS, Slider, Legend, \
        Button, CheckboxButtonGroup, RadioButtonGroup, RadioGroup, CheckboxGroup, Label, Spacer, Title, Div, \
        PanTool, WheelZoomTool, SaveTool, ResetTool, HoverTool, TapTool, \
        BasicTicker, Scatter, CustomJSHover, FileInput, Toggle, TableColumn, DataTable, TextAreaInput, \
        TabPanel, Tabs, DateFormatter, LogColorMapper, LinearColorMapper, ColorBar, Select, PreText, \
        HTMLTemplateFormatter, NumberFormatter, ScientificFormatter, TextInput
from bokeh.plotting import figure, output_file, show, save
from bokeh.resources import CDN
from bokeh.themes import Theme
from bokeh.util.compiler import TypeScript
from bokeh.util.hex import hexbin
from bokeh.transform import linear_cmap, log_cmap
''')
import bokeh_modules as bk 
import importlib
importlib.reload(bk)
import bokeh.palettes as bkp

if is_interactive():
    bk.output_notebook()

# +
import numpy as np
import pandas as pd

import itertools  

import pickle as pkl

import json
from glob import glob
import re

from astropy.io import fits

from urllib.request import urlopen, urlretrieve

# import logging

from PIL import Image

import sys

# import cupy as cp
# from cuml.cluster import DBSCAN
# from cuml.metrics.cluster import silhouette_score
# from cuml.metrics.cluster import silhouette_samples

import extended_library as ext_lib
import cxo_cluster_4_library as cxo_lib

import app1_library as app1_lib


# -

def get_table(data):

    '''
    data is dict
    '''

    # props = ['color', 'silhouette', 'area', 'n-n_bkg', 'signif.', 'sigmas', 'x', 'y', 'ra', 'dec']
    
    props = ['color', 'silhouette', 'area', 'n-n_bkg', 'sigmas', 'x', 'y', 'ra', 'dec']

    tbl_data = {k: data[k] for k in props}

    tbl_data['n-n_bkg'] = [int(_) for _ in tbl_data['n-n_bkg']]

    color_template = '''             
        <p style="color:<%= 
            (function f(){                    
                    return(color)
                }()) %>;"> 
            <%= "&#9608;" %>
        </p>
        '''

    cols = []
    for _ in tbl_data.keys():

        if _ == 'color':
            frmt = bk.HTMLTemplateFormatter(template=color_template)
        elif _ == 'n-n_bkg':
            frmt = bk.NumberFormatter()       
        elif _ in ['area', 'signif.']:
            frmt = bk.ScientificFormatter(precision=2, power_limit_low=-2)   
        elif _ in ['ra', 'dec']:
            frmt = bk.NumberFormatter(format='0.0000')                  
        else:
            frmt = bk.NumberFormatter(format='0.00')    

        c = bk.TableColumn(field=_, title=_, formatter=frmt)    

        cols.append(c)  

    return tbl_data, cols   


# +
hug_dir = 'https://huggingface.co/datasets/oyk100/ChaSES-data/resolve/main'

hug_tree = json.load(urlopen(f'{hug_dir}/obsids.json'))

# +
upper_dir = '/'.join(notebook_dir.split('/')[:-1])

cache_dir = f'{upper_dir}/cache' 

print(cache_dir)

os.system(f'mkdir -p {cache_dir}')

fits_dir = {
    'local': f'{upper_dir}/data', #'/mnt/data_ssd/holes'
    'hug': hug_dir
}

n_max = 20_000   

# +
subdirs_search = lambda d: np.sort([int(_) for _ in next(os.walk(d))[1] if _[0]!='.']).astype(str).tolist()

def get_obsid_folders(loc):
    
    if loc == 'local':
        return subdirs_search(fits_dir['local'])
    else:            
        return list(hug_tree.keys())

    
def get_ccd_folders(obsid):
    
    if str(obsid) in get_obsid_folders('hug'):
        
        ccds = hug_tree[str(obsid)]
    
    else:
        
        json_fn = f'{cache_dir}/{obsid}_ccds.json'   
        
        if os.path.isfile(json_fn):            
            ccds = json.load(open(json_fn, 'rt'))
        else:   
            
            obsid_dir = f'{fits_dir["local"]}/{obsid}'
        
            fn = glob(f'{obsid_dir}/*fits*')[0]
            
            X = fits.getdata(fn, 1)
            
            ccds = np.sort(np.unique(X['ccd_id'])).tolist()
            
            json.dump(ccds, open(json_fn, 'wt'))
                
    return [str(_) for _ in ccds]


# -

if notebook_dir.split('/')[-1] != 'app':
    
    sys.exit(f'incorrect notebook_dir! {notebook_dir}')


def image_url_pars(fn):
    
    img =  Image.open(fn)
    
    # img = img.resize((100,400))
    
    w, h = img.getdata().size
    
    if w > h:
    
        h /= w
        w = 1    
        x, y = 0, (1 + h) / 2
        
    else:
        
        w /= h
        h = 1    
        x, y = (1 - w) / 2, 1
        
    return x, y, w, h


# +
frz = app1_lib.friz_class(history=True, inactive=False)

DEBUG_WINDOW = False

len_pre_VISIBLE = False

scaled_pars = {}

def modify_doc(doc):    
    
    divTemplate = bk.Div(text="""
            <style>
            .bk.sent-later {
                font-size: 11px;
                font-color: green;
                border: 1px solid green;
                background-color: cyan;
            }
            </style>
    """)
    
    msg = bk.PreText(text=' ')
    
    debug_info_window = bk.PreText(text=f'check when deploying!!! {notebook_dir}\n', width=400, height=500, visible=DEBUG_WINDOW)    
    debug_info_window.css_classes.append('sent-later')
    
    holes_cb_group = bk.CheckboxButtonGroup(labels=['holes'], active=[0], width=70, margin=(0, -1, 0, 0))
    n_max_cb_group = bk.CheckboxButtonGroup(labels=['n_max'], active=[0], width=70, margin=(0, -1, 0, 0))
    cache_cb_group = bk.CheckboxButtonGroup(labels=['cache'], active=[0], width=70, margin=(0, -1, 0, 0), visible=False)
    
    cb_group = bk.row([holes_cb_group, n_max_cb_group, cache_cb_group])
        
    data_loc_rbg = bk.RadioButtonGroup(labels=['pre-computed', 'local'], active=1, width=200)    
    data_loc_active = lambda :  data_loc_rbg.labels[data_loc_rbg.active]
            
    query_input = bk.TextInput(value='', placeholder='obsid', width=100)
    query_button = bk.Button(label='query', button_type='success', width=100)
        
    len_pre = bk.PreText(text='', visible=len_pre_VISIBLE)
    
    eps_slider = bk.Slider(start=2, end=4, value=2.86, step=.1, title='eps', width=400)
    
    slider_min_samples = bk.Slider(start=30, end=60, value=46, step=2, title='min_samples', width=400)
    
    tbl_source = bk.ColumnDataSource()
    tbl = bk.DataTable(source=tbl_source, visible=False) 
    save_table_button = bk.Button(label='save table', width=100, visible=False, button_type='success') 
    
    save_regions_button = bk.Button(label='save regions', width=100, visible=False, button_type='success') 
    
    sigma_slider = bk.Slider(start=0, end=3, value=1, step=0.01, title='sigma', width=100)
    nbins_slider = bk.Slider(start=1, end=200, value=100, step=1, title='nbins', width=100)
    
    pallettes_dict = {'Viridis256': bkp.Viridis256, 'Spectral11': bkp.Spectral11}    
    pals = list(pallettes_dict.keys())    
    select_pallette = bk.Select(title='pallette', 
                           value=pals[0], 
                           options=pals,
                           width=200)
    
    # tools=['hover']
    tools=['save']
        
    p = bk.figure(title='', tools=tools)
    # p.toolbar.autohide = True
    
    img = p.image(image=[], x=0, y=0, dw=1, dh=1, 
                  palette=pallettes_dict[select_pallette.value], level='image')
    
    obsids = get_obsid_folders('local')
    
    obsid_ini = '755' if '755' in obsids else obsids[0]
    
    ccds = get_ccd_folders(obsid_ini)
    
    ccd_ini = '7' if obsid_ini=='755' else ccds[0]
    
    p.title = f'{obsid_ini}/{ccd_ini}'
        
    jpg_img = p.image_url(url=[], x=0, y=1, w=1, h=1, visible=True, global_alpha=1)
    
    clus_source = bk.ColumnDataSource({'xs': [], 'ys': [], 'color': []})
    clus_source_filtered = bk.ColumnDataSource({'xs': [], 'ys': [], 'color': []})
    text_source = bk.ColumnDataSource(dict(x=[], y=[], text=[]))
    
    apply_button = bk.Button(label='Apply', button_type='success', width=100)  
    
    min_sigma_slider = bk.Slider(start=0, end=10, value=5, step=1, title='min sigma', width=100)
        
    select_obsid = bk.Select(title='obsid', 
                         value=obsid_ini, 
                         options=obsids,
                         width=100)    

    select_ccd = bk.Select(title='ccd',                            
                           value=ccd_ini,
                           options=[''] + ccds,
                           width=50)
    
    
    save_div = bk.Div(text='', visible=False)
    
    title = bk.Div(text='<b>Search for extended sources in Chandra ACIS images</b>', styles={'font-size': '200%', 'color': 'black'})

    # cite = bk.Div(text=q['cite_text'], width=1000, margin=(5, 5, 0, 5))

    # description = bk.Div(text=q['description_text'], style={'font-size': '150%'}, width=1000)

    ackn = bk.Div(text='Support for this work was provided by NASA through Chandra X-ray Observatory Award AR8-19008X.')

    # contact = bk.Div(text=q['contact_text'])

    # version = bk.Div(text=version_text)
    
    X_source = bk.ColumnDataSource({'X': [], 'db': []})

    
    def show_jpg_img(obsid, path=fits_dir['local']):
    
        status, url, fn_jpg = ext_lib.get_evt2_file(obsid, 
                                                    path=f'{path}/{obsid}',
                                                    search_mask='full_img2.jpg')       
        jpg_img.data_source.data['url'] = [url]
        
        x, y, w, h = image_url_pars(fn_jpg)
        
        jpg_img.glyph.x = x
        jpg_img.glyph.y = y
        jpg_img.glyph.w = w
        jpg_img.glyph.h = h
        
        jpg_img.visible = True
        img.visible = False
        
        apply_button.disabled = True
    
    def all_done_func():
        if msg.text == 'working ...':
            msg.text = ' '
                
    def clear_clusters(): 
        
        empty = {'xs': [], 'ys': [], 'color': []}            
        clus_source.data = empty            
        clus_source_filtered.data = empty  
        text_source.data = dict(x=[], y=[], text=[])
        tbl.visible = False 
        save_table_button.visible = False 
        save_regions_button.visible = False
        img.data_source.data['image'] = []
    
    def select_obsid_callback_aux():
        
        global frz                                
        if frz.unfreeze('select_obsid'): return
    
        clear_clusters()
            
        obsid = select_obsid.value
                
        frz.freeze('select_ccd')
        select_ccd.value = ''
        frz.unfreeze('select_ccd')
        
        obsid_dir = f"{fits_dir['local']}/{obsid}"
        
        os.system(f'mkdir -p {obsid_dir}')
        
        if obsid in get_obsid_folders('hug'):
                                    
            holes_cb_group.disabled = False            
            holes_cb_group.active = [0]
                        
        else:
            
            holes_cb_group.disabled = True
            holes_cb_group.active = []
                                                            
            status, url, evt2_filename = ext_lib.get_evt2_file(obsid, path=obsid_dir) 
            
            if status != 'ok': 
                
                # os.system(f'rm -rf {obsid_dir}')                
                debug_info_window.text += f'get_evt2_file error: {status}\n'                
                return
                    
        ccds = get_ccd_folders(obsid)
        
        select_ccd.options = [''] + ccds
                    
        show_jpg_img(obsid)  
                
        p.title.text = f'{obsid}'
        
    def select_obsid_callback(attr, old, new):
        msg.text = 'working ...'
        doc.add_next_tick_callback(select_obsid_callback_aux)
        doc.add_next_tick_callback(all_done_func)
                    
    def get_ccd_evt2(obsid, ccd):   
        
        holes = True if holes_cb_group.active else False        
        hls = '_holes' if holes else ''
        
        evt2_fn = f'{obsid}_{ccd}{hls}_evt2_05_8keV.fits'          
        evt2_fn_full = f"{fits_dir['local']}/{obsid}/{ccd}/{evt2_fn}"
        
        if os.path.isfile(evt2_fn_full): 
            pass
        
        elif obsid in get_obsid_folders('hug'):
                                        
            os.system(f"mkdir -p {fits_dir['local']}/{obsid}/{ccd}")
                
            url = f"{fits_dir['hug']}/{obsid}/{ccd}/{evt2_fn}"  
                
            urlretrieve(url, evt2_fn_full) 
                    
        else:
            
            os.system(f"mkdir -p {fits_dir['local']}/{obsid}/{ccd}")
            
            app1_lib.create_fits(obsid, int(ccd), fits_dir['local'], hls, cache=bool(cache_cb_group.active))
            
        # return evt2_fn, evt2_fn_full    
                        
    def show_ccd(obsid, ccd):
        
        holes = True if holes_cb_group.active else False        
        hls = '_holes' if holes else ''
                
        evt2_fn = f'{obsid}_{ccd}{hls}_evt2_05_8keV.fits'    
        evt2_fn_full = f"{fits_dir['local']}/{obsid}/{ccd}/{evt2_fn}"
        
        scaled_xy = app1_lib.get_data(evt2_fn_full, ccd)
        
        X = scaled_xy['X'].copy()
        
        # X_source.data['X'] = X
        
        X_source.data = {'X': X, 'db': [0]*len(X)}  
        
        H, bkg_dens = app1_lib.nbins_sigma_func(X, nbins_slider.value, sigma_slider.value)
        
        img.data_source.data['image'] = [H.T]
        
        jpg_img.visible = False
        img.visible = True
        
        apply_button.disabled = False
        
    show_ccd(obsid_ini, ccd_ini)
                   
    def select_ccd_callback_aux():
                
        global frz        
        if frz.unfreeze('select_ccd'): return
    
        save_div.visible = False
    
        clear_clusters()
    
        obsid = select_obsid.value
        ccd = select_ccd.value
        
        if ccd == '':    
            show_jpg_img(obsid)
            p.title.text = obsid            
            return
        
        p.title.text = f'{obsid}/{ccd}'
        
        get_ccd_evt2(obsid, ccd)
        
        show_ccd(obsid, ccd)
        
    def select_ccd_callback(attr, old, new):
        msg.text = 'working ...'
        doc.add_next_tick_callback(select_ccd_callback_aux)
        doc.add_next_tick_callback(all_done_func)
   
    def data_loc_rbg_callback(attr, old, new):
                
        global frz              
        if frz.unfreeze('data_loc_rbg'): return
            
        obsids = get_obsid_folders(data_loc_active())
                               
        select_obsid.options = obsids   
        select_obsid.value = obsids[0]
        
    def query_button_callback_aux():
                
        global frz        
        if frz.unfreeze('query_button'): return
       
        frz.freeze('data_loc_rbg')
        data_loc_rbg.active = 1
        frz.unfreeze('data_loc_rbg')
                    
        tbl.visible = False
        save_table_button.visible = False 
        save_regions_button.visible = False
        
        obsid = query_input.value.rstrip()
                        
        local_obsids = get_obsid_folders('local')
                        
        if obsid not in local_obsids:
            local_obsids = np.sort(local_obsids + [obsid]).tolist()  
                    
        select_obsid.options = local_obsids   
        
        frz.freeze('select_obsid')
        select_obsid.value = obsid
        frz.unfreeze('select_obsid')
        
        select_obsid_callback_aux()
        
    def query_button_callback():
        msg.text = 'downloading obsid ...'
        doc.add_next_tick_callback(query_button_callback_aux)
        doc.add_next_tick_callback(all_done_func)    
                
    def holes_cb_group_callback_aux():  
        
        global frz        
        if frz.unfreeze('holes_cb_group'): return
        
        select_ccd_callback_aux()
        
    def holes_cb_group_callback(attr, old, new):
        msg.text = 'working ...'
        doc.add_next_tick_callback(holes_cb_group_callback_aux)
        doc.add_next_tick_callback(all_done_func)    
                    
    query_button.on_click(query_button_callback)
    select_obsid.on_change('value', select_obsid_callback)  
    select_ccd.on_change('value', select_ccd_callback)  
    data_loc_rbg.on_change('active', data_loc_rbg_callback)
    holes_cb_group.on_change('active', holes_cb_group_callback)
            
    p.x_range.start = p.y_range.start = 0
    p.x_range.end = p.y_range.end = 1    
    p.x_range.range_padding = p.y_range.range_padding = 0
    
    opacity_slider = bk.Slider(start=0, end=1, value=0.5, step=.01, title='opacity', width=100)
        
    clus = p.multi_polygons(xs='xs', ys='ys', color='color', source=clus_source_filtered, fill_alpha=opacity_slider.value)   
    
    # hov = p.select_one(bk.HoverTool)
    
    # TOOLTIPS = [
    #     ('index', '$index'),
    #     ('(x,y)', '@center_of_mass'),
    #     ('silhouette', '@silhouette')
    # ]
    
    # hov.tooltips = TOOLTIPS

    # hov.renderers = [clus]

    txts = p.text(x='x', y='y', text='text', angle=0, text_color='white', source=text_source)
            
    opacity_slider.js_link('value', clus.glyph, 'fill_alpha')
    
    alpha_slider = bk.Slider(start=0, end=10, value=1, step=.01, title='alpha', width=100)    
        
    def alpha_slider_callback(attr, old, new):
        
        data = clus_source.data.copy()  
        
        hull_dict = app1_lib.get_hull(data['clusters'], new)
        
        data.update(hull_dict)        
        
        data = pd.DataFrame(data)
            
        clus_source.data = data.copy()
                        
        clus_source_filtered.data = data[data['sigmas'] >= min_sigma_slider.value]
            
    alpha_slider.on_change('value_throttled', alpha_slider_callback)
        
    p.grid.visible = False
    
    # p.toolbar_location = None
    p.axis.visible = False
    
    p.border_fill_color = 'white' #border_fill_color
    p.title.background_fill_color = 'white'
        
    def select_pallette_callback(attr, old, new):
        
        img.glyph.color_mapper.palette = pallettes_dict[new]
                     
    select_pallette.on_change('value', select_pallette_callback) 
                        
    def nbins_sigma_slider_callback(attr, old, new):
        
        # frz.history.append(X_source.data['X'])
                
        H, bkg_dens = app1_lib.nbins_sigma_func(X_source.data['X'], nbins_slider.value, sigma_slider.value)
                      
        img.data_source.data['image'] = [H.T]
                
    sigma_slider.on_change('value', nbins_sigma_slider_callback)                       
    nbins_slider.on_change('value', nbins_sigma_slider_callback)   
        
    def min_sigma_slider_callback(attr, old, new):
        
        data = pd.DataFrame(clus_source.data)
        
        new_data = data[data['sigmas'] >= new]
        
        clus_source_filtered.data = new_data
        
        text_source.data = {
            'x': new_data['x_scaled'],
            'y': new_data['y_scaled'],
            'text': np.arange(len(new_data['clusters'])).tolist()            
        }
        
        tbl_source.data, tbl.columns = get_table(new_data) 
                
    min_sigma_slider.on_change('value', min_sigma_slider_callback)     
    
    pal_reverse_checkbox = bk.CheckboxGroup(labels=['reverse'], active=[])
    
    def pal_reverse_checkbox_callback(attr, old, new):
        
        _ = pallettes_dict[select_pallette.value]
                
        img.glyph.color_mapper.palette = _[::-1] if len(new)==1 else _
            
    pal_reverse_checkbox.on_change('active', pal_reverse_checkbox_callback)
        
    def save_table_button_callback():
        
        global frz
                
        data = tbl_source.data
        
        obsid = select_obsid.value
        ccd = select_ccd.value
        
        pd.DataFrame(data).reset_index(drop=True).to_csv(f'{cache_dir}/{obsid}_{ccd}_table.csv')  
        
        save_div.visible = True
        save_div.text = f'table saved as cache/{obsid}_{ccd}_table.csv'
                
    save_table_button.on_click(save_table_button_callback) 
        
    def save_regions_button_callback():
        
        global scaled_pars
        global frz
        
        obsid = select_obsid.value
        ccd = select_ccd.value
        
        polygons = 'physical\n'
        
        data = clus_source_filtered.data

        for i, (xx, yy) in enumerate(zip(data['xs'], data['ys'])):

            xy = ext_lib.unscale(xx[0][0], yy[0][0], scaled_pars).T

            polygons += 'polygon(' + ','.join(np.ravel(xy).astype(str)) + ') # text={' + str(i) + '}\n'

        with open(f'{cache_dir}/{obsid}_{ccd}_regions.reg', 'wt') as _:
            _.write(polygons)
            
        save_div.visible = True
        save_div.text = f'regions saved as cache/{obsid}_{ccd}_regions.reg'    
                
    save_regions_button.on_click(save_regions_button_callback)
           
    def apply_button_callback_aux():
        
        global scaled_pars
        
        global frz        
        if frz.unfreeze('apply_button'): return
    
        save_div.visible = False
        save_div.text = ''
                        
        obsid = select_obsid.value
        ccd = select_ccd.value
        
        if ccd == '':
            return
                               
        holes = True if holes_cb_group.active else False        
        n_lim = True if n_max_cb_group.active else False
        
        args_func = {
            'eps': eps_slider.value, 
            'min_samples': slider_min_samples.value
        }
                        
        res, scaled_pars = app1_lib.process_ccd(obsid, ccd, 
                                   holes=holes, 
                                   n_lim=n_lim, 
                                   n_max=n_max, 
                                   args_func=args_func, 
                                   nbins=nbins_slider.value, 
                                   sigma=sigma_slider.value,
                                   alpha=alpha_slider.value,
                                   local_fits_dir=fits_dir['local'])    
                
        if res=='empty':
            img.data_source.data['image'] = []  
            
            clear_clusters()
                   
            msg.text = f'{obsid}_{ccd} empty'
            
            debug_info_window.text += f'emptyyyy\n'
            
            return 'empty'
        
        # scaled_pars = res['scale_pars']
        
        new_data = res
        
        X = new_data['X']
        len_X_orig = new_data['len_X_orig']
        db = new_data['db']
        n_clusters = new_data['n_clusters']
        bkg_dens = new_data['bkg_dens']
        clusters = new_data['clusters']
        H = new_data['H']
        
        debug_info_window.text += f'{n_clusters}\n'
        
        img.data_source.data['image'] = [H.T]
                                            
        len_pre.text = f'{len(X)} events' if len_X_orig==len(X) else f'{len(X)}/{len_X_orig} events'     
                    
        X_source.data = {'X': X, 'db': db}  
        
        hls = '_holes' if holes_cb_group.active else '' 
        
        # pkl.dump(X, open(f'{cache_dir}/X_{obsid}_{ccd}{hls}.pkl', 'wb'))
                                   
        n_clusters_text = f'\n{n_clusters} clusters'
        if n_clusters==1:
            n_clusters_text = n_clusters_text[:-1]
        
        len_pre.text += n_clusters_text
                       
        palette = itertools.cycle(bkp.Category20[20]) 
        
        # start = 0
        # palette = itertools.islice(palette, start, None)

        colors = np.array(['']*n_clusters, dtype='U7')
            
        for i in range(n_clusters): 
        
            colors[i] = next(palette)
            
        colors[0] = cxo_lib.rgb2hex(0, 255, 255) # turquoise   
        
        # msg.text = f'len_X: {len(X)}, bkg: {bkg_dens}'
                                    
        new_data.update({'clusters': clusters, 'color': colors.tolist()})
        
        excl = ['X', 'len_X_orig', 'db', 'n_clusters', 'bkg_dens', 'H']
        
        h = [_ for _ in new_data.keys() if _ not in excl]                
        new_data = pd.DataFrame(new_data, columns=h).fillna('')
                                
        clus_source.data = new_data.copy()
        
        new_data = new_data[new_data['sigmas'] >= min_sigma_slider.value]
        
        clus_source_filtered.data = new_data
        
        text_source.data = {
            'x': new_data['x_scaled'],
            'y': new_data['y_scaled'],
            'text': np.arange(len(new_data['clusters'])).tolist()            
        }
                        
        tbl_source.data, tbl.columns = get_table(new_data) 
        
        tbl.visible = True
        save_table_button.visible = True
        save_regions_button.visible = True
        
        return 'success'
    
    def apply_button_callback():
        msg.text = 'working ...'
        doc.add_next_tick_callback(apply_button_callback_aux)
        doc.add_next_tick_callback(all_done_func)
                        
    apply_button.on_click(apply_button_callback)
                
    row1 = bk.row([select_obsid, select_ccd, len_pre], width=800)
    
    input_div = bk.Div(text='<input id="file-input" type="file">')
    
    settings_column = bk.column([msg, bk.row(apply_button), #process_all_button), 
                                 bk.row(data_loc_rbg, bk.Spacer(width=100), bk.row(query_input, query_button)), 
                                 row1, cb_group, eps_slider, slider_min_samples, 
                                 bk.row(save_table_button, save_regions_button), save_div, tbl, divTemplate, debug_info_window])
    
    layout = bk.row(bk.column([title, ackn, p, bk.row(select_pallette, pal_reverse_checkbox, min_sigma_slider), 
                                bk.row(opacity_slider, alpha_slider), bk.row(nbins_slider, sigma_slider)]), settings_column)
    
    doc.add_root(layout)
    
    doc.title = 'ChaSES'
       
    # bk.export_png(p, filename=f'{save_dir}/{obsid_ccd}{hls}_blur.png')
# -
if is_interactive():
    
    if os.uname()[1]=='bobr970':
        bk.show(modify_doc, notebook_url='localhost:1111', port=8910) # change
    else:
        bk.show(modify_doc)
else:
    modify_doc(bk.curdoc())

# +
# print(frz.history)
# print(frz.comment)
# -


